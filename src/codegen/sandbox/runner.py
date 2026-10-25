"""Run model-written programs against test inputs inside Docker containers.

The host keeps a pool of long-running containers. Each container runs
`worker.py`, which takes one job at a time (code + test inputs) and returns what
the program printed and how each run ended. Expected outputs never enter the
container; comparing outputs is the caller's job.

Container restrictions: no network, no host files mounted, read-only root
filesystem with a small /tmp, memory and process-count caps, no Linux
capabilities, unprivileged user.

Known limit: the program runs as the same user as the worker inside the
container, so it can kill the worker. The host notices and restarts the
container, and that job is reported as `sandbox_error`.

Usage:
    with Sandbox(workers=8) as sb:
        result = sb.run(code, ["3\\n1 2 3\\n", "1\\n5\\n"])
"""

from __future__ import annotations

import json
import queue
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


class _Container:
    """One long-running container that handles one job at a time."""

    def __init__(self, image: str, limits: Limits):
        self.image, self.limits = image, limits
        self._start()

    def _start(self) -> None:
        self.name = f"codegen-sbx-{uuid.uuid4().hex[:12]}"
        self.proc = subprocess.Popen(
            _docker_run_cmd(self.name, self.image, self.limits),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8",
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
        subprocess.run(["docker", "kill", self.name], capture_output=True)
        self.proc.kill()
        self.proc.wait()


class Sandbox:
    def __init__(self, workers: int = 4, limits: Limits = Limits(), image: str = IMAGE, build: bool = True):
        if build:
            build_image(image)
        self.limits = limits
        self._containers = [_Container(image, limits) for _ in range(workers)]
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
