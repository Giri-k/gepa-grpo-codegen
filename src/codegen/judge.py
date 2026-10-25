"""Score a model answer against a problem's tests. This is the shared evaluator
used by GEPA, GRPO and the final evaluation.

    model text -> extract_code -> sandbox run -> compare each output -> Evaluation

Comparison rules for stdio problems:
- whitespace does not matter: outputs are compared token by token
- where the expected token is a decimal (has "." or an exponent), the printed
  number may differ by 1e-6, absolute or relative
- everything else must match exactly, case included

Call-based problems: the function's return value is compared as JSON. APPS
sometimes wraps the expected value in a one-element list, so both forms pass.

Problems that accept "any valid answer" can't be judged against a single
expected output; they are found and removed by validating reference solutions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from codegen.data.apps import Problem
from codegen.extract import extract_code
from codegen.sandbox import Limits, RunResult, Sandbox

FLOAT_TOL = 1e-6
_DECIMAL = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][+-]?\d+)$")
_RESULT_TAG = "@@RESULT@@"
# One line, so line numbers in tracebacks shift by exactly one.
_CALL_PRELUDE = (
    "from typing import *; import sys, math, heapq, bisect, itertools, functools, collections; "
    "from collections import *\n"
)
_CALL_RUNNER = """
import json as _json, sys as _sys
_args = _json.loads(_sys.stdin.read())
_fn = getattr(Solution(), {name!r}) if "Solution" in globals() else globals()[{name!r}]
print({tag!r} + _json.dumps(_fn(*_args), default=list))
"""
FEEDBACK_MAX_FAILURES = 3
FEEDBACK_MAX_CHARS = 200


@dataclass(frozen=True)
class Evaluation:
    uid: str
    # ok | no_code | unfinished_think | syntax_error | sandbox_error
    status: str
    # One per test when status == "ok": passed | wrong_answer | runtime_error |
    # timeout | memory | output_limit | skipped
    verdicts: tuple[str, ...]
    n_tests: int
    code: str | None
    feedback: str  # plain-text summary of what went wrong, for GEPA's reflection
    # What each test printed; only filled when asked for (keep_stdout=True).
    stdouts: tuple[str, ...] = field(default=(), repr=False)

    @property
    def passed(self) -> int:
        return self.verdicts.count("passed")

    @property
    def reward(self) -> float:
        """GRPO reward: fraction of tests passed (1.0 only when all pass)."""
        return self.passed / self.n_tests if self.n_tests else 0.0

    @property
    def passed_all(self) -> bool:
        """pass@1 for this answer."""
        return self.n_tests > 0 and self.passed == self.n_tests


def _float_close(got: str, expected: str) -> bool:
    if not _DECIMAL.match(expected):
        return False
    try:
        return math.isclose(float(got), float(expected), rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)
    except ValueError:
        return False


def tokens_match(got: str, expected: str) -> bool:
    """Compare printed output with expected output (stdio rules above)."""
    g, e = got.split(), expected.split()
    if g == e:
        return True
    return len(g) == len(e) and all(a == b or _float_close(a, b) for a, b in zip(g, e))


def values_match(got, expected) -> bool:
    """Compare a returned value with the expected one (call-based rules above)."""

    def same(a, b) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return a is b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b or math.isclose(a, b, rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b

    return same(got, expected) or (
        isinstance(expected, list) and len(expected) == 1 and same(got, expected[0])
    )


def call_program(code: str, fn_name: str) -> str:
    """Wrap a function/Solution-class answer so it reads args as JSON and prints the result."""
    return _CALL_PRELUDE + code + "\n" + _CALL_RUNNER.format(name=fn_name, tag=_RESULT_TAG)


def _returned_value(stdout: str):
    """The value the call runner printed, or raise ValueError if there is none."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_TAG):
            return json.loads(line[len(_RESULT_TAG):])
    raise ValueError("no return value")


def _short(text, limit: int = FEEDBACK_MAX_CHARS) -> str:
    text = text if isinstance(text, str) else json.dumps(text)
    text = text.strip()
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _judge_tests(problem: Problem, run: RunResult) -> tuple[list[str], list[str]]:
    """Per-test verdicts, plus feedback lines for the first few failures."""
    verdicts, notes = [], []
    for i, (t, expected) in enumerate(zip(run.tests, problem.outputs), start=1):
        got = t.stdout
        if t.outcome == "exited":
            if problem.test_kind == "call":
                try:
                    got = _returned_value(t.stdout)
                    ok = values_match(got, expected)
                except ValueError:
                    got, ok = "<nothing returned>", False
            else:
                ok = tokens_match(got, expected)
            verdict = "passed" if ok else "wrong_answer"
        else:
            verdict = t.outcome
        verdicts.append(verdict)

        if verdict == "passed" or len(notes) >= FEEDBACK_MAX_FAILURES:
            continue
        if verdict == "skipped":  # every later test is skipped too
            notes.append(f"Tests {i}-{len(run.tests)}: not run, the total time budget ran out.")
            notes += [""] * FEEDBACK_MAX_FAILURES  # fill the quota so nothing more is added
            continue
        if verdict == "wrong_answer":
            notes.append(
                f"Test {i}: wrong answer.\n  input: {_short(problem.inputs[i - 1])}\n"
                f"  expected: {_short(expected)}\n  got: {_short(got)}"
            )
        elif verdict == "runtime_error":
            last = t.stderr.strip().splitlines()[-1] if t.stderr.strip() else "no error message"
            notes.append(f"Test {i}: runtime error: {last}\n  input: {_short(problem.inputs[i - 1])}")
        elif verdict == "timeout":
            notes.append(f"Test {i}: time limit exceeded.\n  input: {_short(problem.inputs[i - 1])}")
        elif verdict == "memory":
            notes.append(f"Test {i}: memory limit exceeded.")
        elif verdict == "output_limit":
            notes.append(f"Test {i}: printed far too much output.")
    return verdicts, [n for n in notes if n]


_STATUS_FEEDBACK = {
    "no_code": "No Python code block was found in the answer.",
    "unfinished_think": "The answer ended while still reasoning; no final code was given.",
}


def evaluate_code_many(
    items: list[tuple[Problem, str]], sandbox: Sandbox, limits: Limits | None = None,
    keep_stdout: bool = False,
) -> list[Evaluation]:
    """Evaluate (problem, program code) pairs. Runs execute in parallel."""
    jobs = []
    for problem, code in items:
        if problem.test_kind == "call":
            jobs.append((call_program(code, problem.fn_name), [json.dumps(x) for x in problem.inputs]))
        else:
            jobs.append((code, problem.inputs))
    runs = sandbox.run_many(jobs, limits)

    results = []
    for (problem, code), run in zip(items, runs):
        base = dict(uid=problem.uid, n_tests=problem.n_tests, code=code)
        if run.status != "ok":
            results.append(Evaluation(status=run.status, verdicts=(), feedback=f"Code did not run: {run.error}", **base))
            continue
        verdicts, notes = _judge_tests(problem, run)
        summary = f"Passed {verdicts.count('passed')}/{problem.n_tests} tests."
        stdouts = tuple(t.stdout for t in run.tests) if keep_stdout else ()
        results.append(Evaluation(
            status="ok", verdicts=tuple(verdicts), feedback="\n".join([summary] + notes), stdouts=stdouts, **base
        ))
    return results


def evaluate_many(
    items: list[tuple[Problem, str]], sandbox: Sandbox, limits: Limits | None = None
) -> list[Evaluation]:
    """Evaluate (problem, model output text) pairs. Runs execute in parallel."""
    extractions = [extract_code(text) for _, text in items]
    ran = iter(evaluate_code_many(
        [(p, ex.code) for (p, _), ex in zip(items, extractions) if ex.code is not None], sandbox, limits
    ))
    return [
        next(ran) if ex.code is not None else Evaluation(
            uid=p.uid, status=ex.method, verdicts=(), n_tests=p.n_tests, code=None,
            feedback=_STATUS_FEEDBACK[ex.method],
        )
        for (p, _), ex in zip(items, extractions)
    ]


def evaluate(problem: Problem, text: str, sandbox: Sandbox, limits: Limits | None = None) -> Evaluation:
    return evaluate_many([(problem, text)], sandbox, limits)[0]
