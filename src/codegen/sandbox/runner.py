"""Run model-written programs against test inputs inside containers.

The host keeps a pool of long-running containers. Each container runs
`worker.py`, which takes one job at a time (code + test inputs) and returns what
the program printed and how each run ended. Expected outputs never enter the
container; comparing outputs is the caller's job.

Two backends run the same worker:
- docker (laptops): no network, no host files mounted, read-only root
  filesystem with a small /tmp, memory and process-count caps, no Linux
  capabilities, unprivileged user.
- apptainer (TACC, which has no Docker): read-only image, no host folders,
  private /tmp, its own PID namespace (required: the worker kills every process
  it can see after each job). Network isolation and the process-count cap use
  the strongest option the node supports; see `Sandbox.isolation`. Memory is
  capped per test by the worker's rlimits on both backends.

Known limit: the program runs as the same user as the worker inside the
container, so it can kill the worker. The host notices and restarts the
container, and that job is reported as `sandbox_error`.

Usage:
    with Sandbox(workers=8) as sb:
        result = sb.run(code, ["3\\n1 2 3\\n", "1\\n5\\n"])
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE = "codegen-sandbox"
IMAGE_DIR = Path(__file__).parent
# Apptainer image of the Dockerfile's base (same digest), pulled by tacc/pilot.sh.
SIF = os.environ.get("CODEGEN_SANDBOX_SIF", os.path.expandvars("$WORK/sif/python311.sif"))
HOST_MARGIN = 15.0  # seconds on top of the budget before the host gives up on a container


@dataclass(frozen=True)
class Limits:
    timeout: float = 4.0  # seconds per test
    budget: float = 30.0  # seconds for all tests of one program; later tests are skipped
    memory_mb: int = 1024  # per test process
    output_bytes: int = 4_000_000  # per stream; the largest expected output in any split is 1.4 MB


@dataclass(frozen=True)
class TestRun:
    __test__ = False  # not a pytest test class
    # exited | runtime_error | timeout | memory | output_limit | skipped
    outcome: str
    stdout: str
    stderr: str  # last 2000 characters
    seconds: float


@dataclass(frozen=True)
class RunResult:
    # ok | syntax_error | sandbox_error
    status: str
    tests: tuple[TestRun, ...] = ()
    error: str = ""


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def apptainer_available() -> bool:
    return shutil.which("apptainer") is not None and Path(SIF).is_file()


def sandbox_available() -> bool:
    return docker_available() or apptainer_available()


def build_image(tag: str = IMAGE) -> None:
    """Build the sandbox image. Cached by Docker, so cheap when nothing changed."""
    subprocess.run(["docker", "build", "-q", "-t", tag, str(IMAGE_DIR)], check=True, capture_output=True)


def syntax_error(code: str) -> str | None:
    """Compile without running. Returns the error message, or None if the code compiles."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. SyntaxWarning for invalid escapes
            compile(code, "solution.py", "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        return traceback.format_exception_only(type(e), e)[-1].strip()
    return None


def _docker_run_cmd(name: str, image: str, limits: Limits) -> list[str]:
    memory = f"{limits.memory_mb + 512}m"  # test process + worker + /tmp
    return [
        "docker", "run", "-i", "--rm", "--init", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=256m,mode=1777",
        "--memory", memory, "--memory-swap", memory,
        "--pids-limit", "64", "--cpus", "1",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "10001:10001",
        image,
    ]


def _apptainer_cmd(flags: list[str], sif: str, command: list[str]) -> list[str]:
    return [
        "apptainer", "exec", "--containall", "--no-home", "--cleanenv", "--pid",
        "--no-mount", "bind-paths,hostfs,cwd",  # site-configured binds would expose $WORK etc.
        "--bind", f"{IMAGE_DIR / 'worker.py'}:/sandbox/worker.py:ro",
        *flags, sif, *command,
    ]


_PROBE_PIDS = ["python", "-c", "import os; print(sum(p.isdigit() for p in os.listdir('/proc')))"]
_PROBE_NET = [
    "python", "-c",
    "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    print('open')\n"
    "except OSError:\n    print('closed')",
]


def _probe(cmd: list[str]) -> str | None:
    """stdout of a short container run, or None if apptainer rejected the flags."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _apptainer_setup(sif: str) -> tuple[list[str], list[str], dict]:
    """Pick flags and a command prefix that this node supports. Returns
    (apptainer flags, prefix for the worker command, description)."""
    pids = _probe(_apptainer_cmd([], sif, _PROBE_PIDS))
    if pids is None or int(pids) > 10:
        # Without a PID namespace the worker's clean-up would kill the user's
        # other processes on the node (vLLM, the trainer). Refuse.
        raise RuntimeError(f"apptainer --pid does not isolate processes (saw {pids} pids)")

    net_options = [
        ("apptainer --net --network none", ["--net", "--network", "none"], []),
        ("unshare -rn", [], ["unshare", "-rn"]),
    ]
    flags, prefix, network = [], [], "none (not isolated)"
    for name, f, p in net_options:
        if _probe(_apptainer_cmd(f, sif, p + _PROBE_NET)) == "closed":
            flags, prefix, network = f, p, name
            break
    else:
        if _probe(_apptainer_cmd([], sif, _PROBE_NET)) == "closed":
            network = "blocked by the node"
        warnings.warn(f"sandbox network isolation unavailable; network: {network}")

    pids_limit = _probe(_apptainer_cmd(flags + ["--pids-limit", "64"], sif, prefix + ["true"])) is not None
    if pids_limit:
        flags = flags + ["--pids-limit", "64"]
    return flags, prefix, {"backend": "apptainer", "network": network, "pids_limit": pids_limit}


class _Container:
    """One long-running container that handles one job at a time."""

    def __init__(self, start_cmd):
        self.start_cmd = start_cmd  # name -> argv
        self._start()

    def _start(self) -> None:
        self.name = f"codegen-sbx-{uuid.uuid4().hex[:12]}"
        self.proc = subprocess.Popen(
            self.start_cmd(self.name),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8",
            start_new_session=True,  # own process group, so kill() takes the whole container
        )

    def call(self, job: dict, deadline: float) -> dict | None:
        """Send a job and wait for its result. None if the container died or hung."""
        timer = threading.Timer(deadline, self.kill)
        timer.start()
        try:
            self.proc.stdin.write(json.dumps(job) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except OSError:  # broken pipe: the container is gone
            line = ""
        finally:
            timer.cancel()
        if not line:
            self.kill()
            self._start()
            return None
        return json.loads(line)

    def kill(self) -> None:
        if self.start_cmd(self.name)[0] == "docker":
            subprocess.run(["docker", "kill", self.name], capture_output=True)
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()


class Sandbox:
    def __init__(
        self, workers: int = 4, limits: Limits = Limits(), image: str = IMAGE, build: bool = True,
        backend: str = "auto",
    ):
        """`backend` is "docker", "apptainer", or "auto" (Docker if it is running)."""
        if backend == "auto":
            backend = "docker" if docker_available() else "apptainer"
        if backend == "docker":
            if build:
                build_image(image)
            self.isolation = {"backend": "docker", "network": "docker --network none", "pids_limit": True}
            start_cmd = lambda name: _docker_run_cmd(name, image, limits)
        elif backend == "apptainer":
            flags, prefix, self.isolation = _apptainer_setup(SIF)
            start_cmd = lambda name: _apptainer_cmd(flags, SIF, prefix + ["python", "-s", "/sandbox/worker.py"])
        else:
            raise ValueError(f"unknown sandbox backend {backend!r}")
        self.limits = limits
        self._containers = [_Container(start_cmd) for _ in range(workers)]
        self._free: queue.Queue[_Container] = queue.Queue()
        for c in self._containers:
            self._free.put(c)

    def run(self, code: str, inputs: list[str], limits: Limits | None = None) -> RunResult:
        """Run `code` once per input. `limits.memory_mb` above the Sandbox's own is
        capped by the container's memory limit."""
        limits = limits or self.limits
        if (err := syntax_error(code)) is not None:
            return RunResult("syntax_error", error=err)

        job = {"code": code, "inputs": list(inputs), **asdict(limits)}
        container = self._free.get()
        try:
            reply = container.call(job, limits.budget + HOST_MARGIN)
        finally:
            self._free.put(container)

        if reply is None:
            return RunResult("sandbox_error", error="container died or stopped responding")
        if "error" in reply:
            return RunResult("sandbox_error", error=reply["error"])
        if len(reply["tests"]) != len(inputs):
            return RunResult("sandbox_error", error="worker returned the wrong number of results")
        return RunResult("ok", tuple(TestRun(**t) for t in reply["tests"]))

    def run_many(self, jobs: list[tuple[str, list[str]]], limits: Limits | None = None) -> list[RunResult]:
        """Run (code, inputs) jobs in parallel across the pool. Results keep job order."""
        with ThreadPoolExecutor(len(self._containers)) as ex:
            return list(ex.map(lambda job: self.run(job[0], job[1], limits), jobs))

    def close(self) -> None:
        for c in self._containers:
            c.kill()

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
