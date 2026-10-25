"""Extract the Python program from a model's text answer.

Rules, in order:
1. Drop <think>...</think> reasoning (Qwen3 thinking mode). Code drafted while
   thinking is not the answer. An unclosed <think> means the answer never came.
2. Prefer the last fenced block tagged as Python (```python, ```py, ```python3).
   Models often show a first attempt and then a corrected one.
3. Otherwise take the last untagged fenced block that looks like a program,
   skipping blocks like sample input ("3\\n1 2 3") or sample output ("4").
4. A final fence with no closing ``` (answer cut off at the token limit) is
   still taken. Running it decides whether it works.
5. With no fences at all, use the whole answer only if it parses as a program.

Validity is not checked here beyond rules 3 and 5. A syntax error in the chosen
block is reported by the evaluator, not hidden by picking another block.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

PYTHON_TAGS = {"python", "py", "python3", "py3"}

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
# A fence line: ``` plus an optional language tag, then the rest of the line.
_FENCE = re.compile(r"^[ \t]*```[ \t]*([\w+-]*)[^\n]*$", re.MULTILINE)


@dataclass(frozen=True)
class Extraction:
    code: str | None
    # fenced | unclosed_fence | raw | no_code | unfinished_think
    method: str


def strip_thinking(text: str) -> str | None:
    """Remove reasoning blocks. Returns None if reasoning never finished."""
    text = _THINK_CLOSED.sub("", text)
    if "<think>" in text:
        return None
    # Some chat templates put <think> in the prompt, so only "</think>" appears.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text


def looks_like_program(code: str) -> bool:
    """True if `code` parses and does more than state literal values."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        not (isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call))
        for node in tree.body
    )


def fenced_blocks(text: str) -> list[tuple[str, str, bool]]:
    """Return (language tag, body, closed) for each fenced block, in order."""
    blocks = []
    fences = list(_FENCE.finditer(text))
    i = 0
    while i < len(fences):
        opening = fences[i]
        tag = opening.group(1).lower()
        body_start = opening.end() + 1
        if i + 1 < len(fences):
            closing = fences[i + 1]
            blocks.append((tag, text[body_start : closing.start()], True))
            i += 2
        else:
            blocks.append((tag, text[body_start:], False))
            i += 1
    return blocks


def extract_code(text: str) -> Extraction:
    text = text.replace("\r\n", "\n")
    answer = strip_thinking(text)
    if answer is None:
        return Extraction(None, "unfinished_think")

    blocks = fenced_blocks(answer)
    tagged = [b for b in blocks if b[0] in PYTHON_TAGS]
    untagged = [b for b in blocks if not b[0] and looks_like_program(b[1])]
    for candidates in (tagged, untagged):
        if candidates:
            _, body, closed = candidates[-1]
            code = body.strip("\n")
            if code.strip():
                return Extraction(code, "fenced" if closed else "unclosed_fence")

    if not blocks and looks_like_program(answer.strip()):
        return Extraction(answer.strip("\n"), "raw")
    return Extraction(None, "no_code")
