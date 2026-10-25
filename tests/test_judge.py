from dataclasses import replace

import pytest

from codegen.data.apps import Problem
from codegen.judge import evaluate, evaluate_many, tokens_match, values_match
from codegen.sandbox import Limits, Sandbox, docker_available

FAST = Limits(timeout=1.0, budget=10.0, memory_mb=256, output_bytes=10_000)
needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker is not running")

DOUBLE = Problem(
    uid="t-double", source_split="test", source_id=0, difficulty="introductory",
    question="Print 2n.", starter_code="", fn_name=None,
    inputs=["3\n", "10\n", "0\n", "-4\n"], outputs=["6\n", "20\n", "0\n", "-8\n"],
    n_solutions=1, url="",
)
SOLUTION_CALL = Problem(
    uid="t-call", source_split="test", source_id=1, difficulty="introductory",
    question="Return a + b.", starter_code="class Solution:\n    def add(self, a: int, b: int) -> int:\n",
    fn_name="add", inputs=[[1, 2], [5, 5]], outputs=[3, [10]],  # second one APPS-style wrapped
    n_solutions=1, url="",
)


def answer(code: str) -> str:
    return f"Here is my solution:\n```python\n{code}\n```"


@pytest.fixture(scope="module")
def sandbox():
    with Sandbox(workers=2, limits=FAST) as sb:
        yield sb


# --- comparison rules (no Docker needed) ---

@pytest.mark.parametrize(
    "got, expected, ok",
    [
        ("6\n", "6", True),
        ("1 2 3", "1  2\n3\n\n", True),  # whitespace ignored
        ("0.3333333", "0.333333333333", True),  # within 1e-6
        ("2.5", "2.5000000000", True),
        ("1e-7", "0.0", True),
        ("0.334", "0.333333", False),
        ("2.0", "2", False),  # integer answers stay exact
        ("Yes", "YES", False),  # case-sensitive
        ("6 7", "6", False),
        ("", "6", False),
        ("nan", "1.5", False),
    ],
)
def test_tokens_match(got, expected, ok):
    assert tokens_match(got, expected) is ok


@pytest.mark.parametrize(
    "got, expected, ok",
    [
        (5, 5, True),
        (5, [5], True),  # APPS wraps return values in a list
        ([[1, 2, 4]], [[1, 2, 4]], True),
        ([1, 2, 4], [[1, 2, 4]], True),
        (0.1 + 0.2, 0.3, True),
        (True, 1, False),
        ([2, 1], [1, 2], False),
        ("a", "a", True),
    ],
)
def test_values_match(got, expected, ok):
    assert values_match(got, expected) is ok


# --- end to end through the sandbox ---

@needs_docker
def test_correct_answer(sandbox):
    e = evaluate(DOUBLE, answer("print(int(input()) * 2)"), sandbox)
    assert e.status == "ok" and e.passed_all and e.reward == 1.0
    assert e.feedback == "Passed 4/4 tests."


@needs_docker
def test_partly_correct_answer_gets_fraction(sandbox):
    e = evaluate(DOUBLE, answer("print(abs(int(input())) * 2)"), sandbox)  # wrong only for -4
    assert e.verdicts == ("passed", "passed", "passed", "wrong_answer")
    assert e.reward == 0.75 and not e.passed_all
    assert "Test 4: wrong answer" in e.feedback and "expected: '-8'" in e.feedback and "got: '8'" in e.feedback


@needs_docker
def test_runtime_error_feedback(sandbox):
    e = evaluate(DOUBLE, answer("n = int(input())\nprint(10 // n)"), sandbox)
    assert e.verdicts[2] == "runtime_error"
    assert "ZeroDivisionError" in e.feedback


@needs_docker
def test_timeout_and_skipped(sandbox):
    e = evaluate(DOUBLE, answer("while True:\n    pass"), sandbox, replace(FAST, budget=1.5))
    assert e.verdicts[0] == "timeout" and e.verdicts[-1] == "skipped"
    assert e.reward == 0.0
    assert "time limit exceeded" in e.feedback and "time budget ran out" in e.feedback


@needs_docker
def test_no_code_and_syntax_error(sandbox):
    e = evaluate(DOUBLE, "I would sort the array first.", sandbox)
    assert e.status == "no_code" and e.reward == 0.0 and e.verdicts == ()
    e = evaluate(DOUBLE, answer("print("), sandbox)
    assert e.status == "syntax_error" and e.reward == 0.0 and "SyntaxError" in e.feedback


@needs_docker
def test_call_based_solution_class(sandbox):
    code = "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b"
    e = evaluate(SOLUTION_CALL, answer(code), sandbox)
    assert e.passed_all, e.feedback


@needs_docker
def test_call_based_plain_function_and_prints_ignored(sandbox):
    code = "def add(a, b):\n    print('debug')\n    return a + b"
    assert evaluate(SOLUTION_CALL, answer(code), sandbox).passed_all


@needs_docker
def test_call_based_wrong_answer(sandbox):
    code = "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a - b"
    e = evaluate(SOLUTION_CALL, answer(code), sandbox)
    assert e.verdicts == ("wrong_answer", "wrong_answer")
    assert "expected: '3'" in e.feedback and "got: '-1'" in e.feedback


@needs_docker
def test_evaluate_many_keeps_order(sandbox):
    items = [(DOUBLE, answer("print(int(input()) * 2)")), (DOUBLE, "no code"), (DOUBLE, answer("print(0)"))]
    rewards = [e.reward for e in evaluate_many(items, sandbox)]
    assert rewards == [1.0, 0.0, 0.25]
