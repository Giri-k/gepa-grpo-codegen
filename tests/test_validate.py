import pytest

from codegen.data.apps import Problem
from codegen.data.validate import label, validate
from codegen.sandbox import Limits, Sandbox, docker_available

FAST = Limits(timeout=1.0, budget=10.0, memory_mb=256, output_bytes=10_000)
needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker is not running")

PY2 = "n = input()\nprint n * 2"
GOOD = "print(int(input()) * 2)"


def problem(uid, solutions, outputs=("6\n", "20\n")):
    return Problem(
        uid=uid, source_split="taco", source_id=0, difficulty="EASY", question="Print 2n.",
        starter_code="", fn_name=None, inputs=["3\n", "10\n"], outputs=list(outputs),
        n_solutions=len(solutions), url="", solutions=tuple(solutions),
    )


def ok(**fails):
    return {"status": "ok", "passed": 0, "fails": fails, "case_fix": False}


@pytest.mark.parametrize(
    "attempts, expected",
    [
        ([ok(wrong_answer=1), ok()], "valid"),
        ([{"status": "syntax_error", "passed": 0, "fails": {}, "case_fix": False}], "no_usable_solution"),
        ([], "no_usable_solution"),
        ([ok(timeout=1, skipped=3), ok(wrong_answer=2)], "too_slow"),
        ([ok(wrong_answer=1, runtime_error=1)], "wrong_answer"),
        ([ok(runtime_error=2)], "runtime_error"),
    ],
)
def test_label(attempts, expected):
    assert label(attempts) == expected


@needs_docker
def test_validate_labels_and_stops_at_first_passing_solution():
    problems = [
        problem("p-valid", [PY2, GOOD, "this would not even run"]),
        problem("p-wrong", ["print(1)", "print(2)"]),
        problem("p-slow", ["while True:\n    pass"]),
        problem("p-case", ["print('yes')"], outputs=("YES\n", "YES\n")),
        problem("p-py2", [PY2]),
    ]
    with Sandbox(workers=2, limits=FAST) as sb:
        v = validate(problems, sb, max_solutions=5, progress=lambda _: None)

    assert v["p-valid"]["label"] == "valid"
    assert len(v["p-valid"]["attempts"]) == 2  # third solution never tried
    assert v["p-valid"]["attempts"][0]["status"] == "syntax_error"
    assert v["p-wrong"]["label"] == "wrong_answer" and len(v["p-wrong"]["attempts"]) == 2
    assert v["p-slow"]["label"] == "too_slow"
    assert v["p-case"]["label"] == "wrong_answer" and v["p-case"]["attempts"][0]["case_fix"]
    assert not v["p-wrong"]["attempts"][0]["case_fix"]
    assert v["p-py2"]["label"] == "no_usable_solution"
