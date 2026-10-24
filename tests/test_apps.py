import json

import pytest

from codegen.data.apps import UnusableProblem, parse_problem


def raw(io, **kw):
    base = {
        "id": 7,
        "question": "q",
        "solutions": json.dumps(["print(1)", "print(2)"]),
        "input_output": json.dumps(io) if isinstance(io, dict) else io,
        "difficulty": "interview",
        "url": "u",
        "starter_code": "",
    }
    return base | kw


def test_stdio_string_tests():
    p = parse_problem(raw({"inputs": ["1 2\n"], "outputs": ["3\n"]}), "train")
    assert p.uid == "train-0007"
    assert p.test_kind == "stdio" and p.fn_name is None
    assert p.inputs == ["1 2\n"] and p.outputs == ["3\n"]
    assert p.n_solutions == 2 and p.solutions == ()


def test_stdio_list_of_lines_joined():
    p = parse_problem(raw({"inputs": [["82", "28"]], "outputs": [["54"]]}), "test")
    assert p.inputs == ["82\n28"] and p.outputs == ["54"]


def test_call_based_keeps_raw_values():
    io = {"fn_name": "solve", "inputs": [[[1, 2], 3]], "outputs": [[5]]}
    p = parse_problem(raw(io, starter_code="def solve(a, b):"), "train")
    assert p.test_kind == "call" and p.fn_name == "solve"
    assert p.inputs == [[[1, 2], 3]] and p.outputs == [[5]]
    assert p.starter_code == "def solve(a, b):"


def test_huge_integers_parse():
    big = "9" * 9000
    io = '{"fn_name": "f", "inputs": [[%s]], "outputs": [%s]}' % (big, big)
    p = parse_problem(raw(io), "train")
    assert p.outputs == [int(big)]


def test_keep_solutions():
    p = parse_problem(raw({"inputs": ["1"], "outputs": ["1"]}), "train", keep_solutions=1)
    assert p.solutions == ("print(1)",)


def test_missing_solutions_counts_zero():
    p = parse_problem(raw({"inputs": ["1"], "outputs": ["1"]}, solutions=""), "test")
    assert p.n_solutions == 0


@pytest.mark.parametrize(
    "io",
    [
        "",
        "{not json",
        {"inputs": [], "outputs": []},
        {"inputs": ["1", "2"], "outputs": ["1"]},
        {"inputs": [5], "outputs": ["5"]},
        {"fn_name": "f", "inputs": ["x"], "outputs": [1]},
    ],
    ids=["empty", "bad-json", "no-tests", "mismatch", "stdio-non-str", "call-non-list"],
)
def test_unusable(io):
    with pytest.raises(UnusableProblem):
        parse_problem(raw(io), "train")
