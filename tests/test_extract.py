import pytest

from codegen.extract import extract_code, looks_like_program

SOLUTION = "n = int(input())\nprint(n * 2)"


def test_single_python_block():
    text = f"Here is my solution:\n\n```python\n{SOLUTION}\n```\n\nIt doubles n."
    result = extract_code(text)
    assert result.code == SOLUTION and result.method == "fenced"


@pytest.mark.parametrize("tag", ["python", "py", "Python3", "PYTHON", "py3"])
def test_python_tag_variants(tag):
    assert extract_code(f"```{tag}\n{SOLUTION}\n```").code == SOLUTION


def test_last_python_block_wins():
    text = (
        "First attempt:\n```python\nprint('wrong')\n```\n"
        "Wait, that misses the edge case. Corrected:\n"
        f"```python\n{SOLUTION}\n```"
    )
    assert extract_code(text).code == SOLUTION


def test_sample_input_and_output_blocks_skipped():
    text = (
        "For the input\n```\n3\n1 2 3\n```\nthe answer is\n```\n6\n```\n"
        f"Solution:\n```\n{SOLUTION}\n```"
    )
    assert extract_code(text).code == SOLUTION


def test_tagged_block_preferred_over_later_untagged():
    text = f"```python\n{SOLUTION}\n```\nExample run:\n```\nx = 5\n```"
    assert extract_code(text).code == SOLUTION


def test_non_python_language_ignored():
    text = "```cpp\nint main() { return 0; }\n```"
    assert extract_code(text).method == "no_code"


def test_thinking_removed():
    text = (
        "<think>\nMaybe:\n```python\nprint('draft')\n```\nNo, better to double it.\n</think>\n"
        f"```python\n{SOLUTION}\n```"
    )
    assert extract_code(text).code == SOLUTION


def test_only_closing_think_tag():
    # The chat template may put <think> in the prompt, so the output starts mid-thought.
    text = f"let me think...\n```python\nprint('draft')\n```\n</think>\n```python\n{SOLUTION}\n```"
    assert extract_code(text).code == SOLUTION


def test_unfinished_thinking_is_no_answer():
    text = "<think>\nThe idea is to use DP...\n```python\nprint('draft')\n```\nthen"
    result = extract_code(text)
    assert result.code is None and result.method == "unfinished_think"


def test_unclosed_fence_taken():
    text = "```python\nn = int(input())\nfor i in range(n):\n    print(i"
    result = extract_code(text)
    assert result.method == "unclosed_fence"
    assert result.code.startswith("n = int(input())")


def test_raw_code_without_fences():
    assert extract_code(SOLUTION + "\n") == extract_code(SOLUTION)
    assert extract_code(SOLUTION).method == "raw"


def test_prose_only_is_no_code():
    result = extract_code("I think the answer is to sort the array and take the median.")
    assert result.code is None and result.method == "no_code"


def test_empty_block_is_no_code():
    assert extract_code("```python\n\n```").method == "no_code"


def test_crlf_line_endings():
    text = f"```python\r\n{SOLUTION.replace(chr(10), chr(13) + chr(10))}\r\n```"
    assert extract_code(text).code == SOLUTION


def test_indented_code_kept_intact():
    code = "def solve():\n    for i in range(3):\n        print(i)\n\nsolve()"
    assert extract_code(f"```python\n{code}\n```").code == code


def test_syntax_error_in_tagged_block_is_still_returned():
    # The evaluator should report the syntax error rather than us hiding it.
    assert extract_code("```python\nprint(\n```").code == "print("


@pytest.mark.parametrize(
    "code, expected",
    [
        (SOLUTION, True),
        ("print(6)", True),
        ("import sys", True),
        ("6", False),
        ("3\n1 2 3", False),  # not valid Python
        ("YES", False),  # bare name
        ("", False),
    ],
)
def test_looks_like_program(code, expected):
    assert looks_like_program(code) is expected
