"""Runs inside the sandbox container. Standard library only.

Reads one JSON job per line on stdin and writes one JSON result per line on stdout:

    job:    {"code": str, "inputs": [str], "timeout": s, "budget": s,
             "memory_mb": int, "output_bytes": int}
    result: {"tests": [{"outcome": str, "stdout": str, "stderr": str, "seconds": float}]}
            or {"error": str} if the worker itself failed

Each test runs `python solution.py` in a fresh process with the test input on
stdin. The worker never sees expected outputs; comparison happens outside the
container. After every job, leftover processes and files are removed so one
program cannot affect the next.
"""

import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

STDERR_TAIL = 2000
CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "PYTHONHASHSEED": "0",  # deterministic set/dict-of-str iteration order
    "PYTHONINTMAXSTRDIGITS": "0",  # tests contain integers with 9,000+ digits
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}


def _set_limit(kind: int, value: int) -> None:
    hard = resource.getrlimit(kind)[1]
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, hard if hard == resource.RLIM_INFINITY else value))


def _child_limits(timeout: float, memory_bytes: int, output_bytes: int):
    def apply() -> None:
        _set_limit(resource.RLIMIT_CPU, math.ceil(timeout) + 1)  # backup for the wall clock
        _set_limit(resource.RLIMIT_AS, memory_bytes)
        _set_limit(resource.RLIMIT_STACK, memory_bytes // 4)  # deep recursion is common
        _set_limit(resource.RLIMIT_FSIZE, output_bytes)  # caps stdout/stderr files
        _set_limit(resource.RLIMIT_CORE, 0)

    return apply


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _tail(path: str, n: int) -> str:
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - n))
        return f.read().decode("utf-8", errors="replace")


def run_test(job_dir: str, stdin_text: str, timeout: float, job: dict) -> dict:
    memory_bytes = job["memory_mb"] * 1024 * 1024
    output_bytes = job["output_bytes"]
    paths = {k: os.path.join(job_dir, f".{k}") for k in ("in", "out", "err")}
    with open(paths["in"], "w", encoding="utf-8") as f:
        f.write(stdin_text)

    with open(paths["in"], "rb") as fin, open(paths["out"], "wb") as fout, open(paths["err"], "wb") as ferr:
        start = time.monotonic()
        proc = subprocess.Popen(
            [sys.executable, "-s", "solution.py"],
            cwd=job_dir, stdin=fin, stdout=fout, stderr=ferr, env=CHILD_ENV,
            start_new_session=True,  # own process group, so children die with it
            preexec_fn=_child_limits(timeout, memory_bytes, output_bytes),
        )
        try:
            proc.wait(timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        _kill_group(proc.pid)
        proc.wait()
        seconds = time.monotonic() - start

    rc = proc.returncode
    stderr = _tail(paths["err"], STDERR_TAIL)
    # Only stdout counts as a flood: stderr is just truncated (judges ignore it).
    flooded = os.path.getsize(paths["out"]) >= output_bytes
    if timed_out or rc == -signal.SIGXCPU:
        outcome = "timeout"
    elif flooded:
        outcome = "output_limit"
    elif "MemoryError" in stderr or rc == -signal.SIGKILL:  # SIGKILL here = cgroup OOM killer
        outcome = "memory"
    elif rc != 0:
        outcome = "runtime_error"
    else:
        outcome = "exited"

    with open(paths["out"], "rb") as f:
        stdout = f.read(output_bytes).decode("utf-8", errors="replace")
    return {"outcome": outcome, "stdout": stdout, "stderr": stderr, "seconds": round(seconds, 4)}


def run_job(job: dict) -> dict:
    job_dir = tempfile.mkdtemp(dir="/tmp")
    with open(os.path.join(job_dir, "solution.py"), "w", encoding="utf-8") as f:
        f.write(job["code"])
    deadline = time.monotonic() + job["budget"]
    tests = []
    for stdin_text in job["inputs"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            tests.append({"outcome": "skipped", "stdout": "", "stderr": "", "seconds": 0.0})
            continue
        tests.append(run_test(job_dir, stdin_text, min(job["timeout"], remaining), job))
    return {"tests": tests}


def clean_up() -> None:
    """Kill every process except init and this worker, and empty /tmp."""
    me = os.getpid()
    for _ in range(3):  # a process may fork while we iterate
        for name in os.listdir("/proc"):
            if name.isdigit() and int(name) not in (1, me):
                try:
                    os.kill(int(name), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    for name in os.listdir("/tmp"):
        path = os.path.join("/tmp", name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.unlink(path)
            except OSError:
                pass


def main() -> None:
    for line in sys.stdin:
        try:
            result = run_job(json.loads(line))
        except Exception as e:  # report instead of dying, so the host can carry on
            result = {"error": repr(e)}
        finally:
            clean_up()
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
