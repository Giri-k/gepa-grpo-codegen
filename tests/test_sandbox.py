from dataclasses import replace

import pytest

from codegen.sandbox import Limits, Sandbox, sandbox_available
from codegen.sandbox.runner import syntax_error

FAST = Limits(timeout=1.0, budget=10.0, memory_mb=256, output_bytes=10_000)

# Syntax checks run on the host; everything else needs a sandbox backend.
needs_docker = pytest.mark.skipif(not sandbox_available(), reason="no sandbox backend (Docker or Apptainer)")


@pytest.fixture(scope="module")
def sandbox():
    with Sandbox(workers=2, limits=FAST) as sb:
        yield sb


@pytest.fixture(scope="module")
def single():
    """One container, for tests where two jobs must land in the same container."""
    with Sandbox(workers=1, limits=FAST) as sb:
        yield sb


def outcomes(result):
    return [t.outcome for t in result.tests]


def test_syntax_error_detected_without_running():
    assert "SyntaxError" in syntax_error("print(")
    assert syntax_error("print(1)") is None


@needs_docker
def test_passing_program(sandbox):
    r = sandbox.run("n = int(input())\nprint(n * 2)", ["3\n", "10\n"])
    assert r.status == "ok"
    assert outcomes(r) == ["exited", "exited"]
    assert [t.stdout for t in r.tests] == ["6\n", "20\n"]


@needs_docker
def test_reads_multiple_lines(sandbox):
    code = "import sys\ndata = sys.stdin.read().split()\nprint(sum(map(int, data[1:])))"
    assert sandbox.run(code, ["3\n1 2 3\n"]).tests[0].stdout == "6\n"


@needs_docker
def test_syntax_error_result(sandbox):
    r = sandbox.run("print(", ["1\n"])
    assert r.status == "syntax_error" and r.tests == () and "SyntaxError" in r.error


@needs_docker
def test_runtime_error_keeps_output_and_traceback(sandbox):
    t = sandbox.run("print(1)\nraise ValueError('boom')", [""]).tests[0]
    assert t.outcome == "runtime_error"
    assert t.stdout == "1\n"
    assert "ValueError: boom" in t.stderr


@needs_docker
def test_infinite_loop_times_out(sandbox):
    t = sandbox.run("while True:\n    pass", [""]).tests[0]
    assert t.outcome == "timeout" and t.seconds < 3


@needs_docker
def test_budget_skips_remaining_tests(sandbox):
    r = sandbox.run("while True:\n    pass", [""] * 4, replace(FAST, budget=1.5))
    assert outcomes(r)[0] == "timeout"
    assert outcomes(r)[-1] == "skipped"


@needs_docker
def test_memory_limit(sandbox):
    t = sandbox.run("a = bytearray(512 * 1024 * 1024)\nprint('allocated')", [""]).tests[0]
    assert t.outcome == "memory" and t.stdout == ""


@needs_docker
def test_output_flood_is_cut_off(sandbox):
    t = sandbox.run("while True:\n    print('x' * 1000)", [""]).tests[0]
    assert t.outcome == "output_limit"
    assert len(t.stdout) <= FAST.output_bytes


@needs_docker
def test_huge_integers_allowed(sandbox):
    t = sandbox.run("print(int('9' * 5000) % 7)", [""]).tests[0]
    assert t.outcome == "exited"


@needs_docker
def test_deep_recursion_allowed(sandbox):
    code = "import sys\nsys.setrecursionlimit(10**6)\ndef f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\nprint(f(50000))"
    t = sandbox.run(code, [""]).tests[0]
    assert t.outcome == "exited" and t.stdout == "50000\n"


@needs_docker
def test_no_network(sandbox):
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\nprint('connected')"
    t = sandbox.run(code, [""]).tests[0]
    assert t.outcome == "runtime_error" and "connected" not in t.stdout


@needs_docker
def test_host_files_not_visible(sandbox):
    code = "import os\nprint(os.path.exists('/Users'), os.path.exists('/home/runner'))"
    assert sandbox.run(code, [""]).tests[0].stdout == "False False\n"


@needs_docker
def test_filesystem_read_only_except_working_dir(sandbox):
    assert sandbox.run("open('/sandbox/x', 'w')", [""]).tests[0].outcome == "runtime_error"
    t = sandbox.run("open('f.txt', 'w').write('hi')\nprint(open('f.txt').read())", [""]).tests[0]
    assert t.outcome == "exited" and t.stdout == "hi\n"


@needs_docker
def test_escaped_processes_killed_between_jobs(single):
    spawn = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1000)'], start_new_session=True)\n"
        "print('started')"
    )
    count = (
        "import os\nn = 0\n"
        "for p in filter(str.isdigit, os.listdir('/proc')):\n"
        "    try:\n"
        "        n += b'time.sleep(1000)' in open(f'/proc/{p}/cmdline', 'rb').read()\n"
        "    except OSError:\n"
        "        pass\n"
        "print(n)"
    )
    assert single.run(spawn, [""]).tests[0].stdout == "started\n"
    assert single.run(count, [""]).tests[0].stdout == "0\n"


@needs_docker
def test_files_do_not_leak_between_jobs(single):
    single.run("open('/tmp/leak', 'w').write('x')", [""])
    assert single.run("import os\nprint(os.path.exists('/tmp/leak'))", [""]).tests[0].stdout == "False\n"


@needs_docker
def test_fork_bomb_contained(single):
    r = single.run("import os\nwhile True:\n    os.fork()", [""])
    assert outcomes(r)[0] in ("timeout", "runtime_error")
    assert single.run("print('alive')", [""]).tests[0].stdout == "alive\n"


@needs_docker
def test_worker_killed_is_restarted(single):
    r = single.run("import os, signal\nos.kill(os.getppid(), signal.SIGKILL)", [""])
    assert r.status == "sandbox_error"
    assert single.run("print('alive')", [""]).tests[0].stdout == "alive\n"


@needs_docker
def test_run_many_keeps_order(sandbox):
    results = sandbox.run_many([(f"print({i})", [""]) for i in range(6)])
    assert [r.tests[0].stdout for r in results] == [f"{i}\n" for i in range(6)]
