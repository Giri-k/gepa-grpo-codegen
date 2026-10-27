"""Prompts shared by every condition.

A prompt is a system prompt plus a fixed user message. GEPA optimizes only the
system prompt (`original.txt` is the starting point and the baseline); the user
message, which holds the problem and the required answer format, never changes,
so the extractor and judge see the same format in all four conditions.
"""

from __future__ import annotations

from pathlib import Path

from codegen.data.apps import Problem

PROMPTS_DIR = Path(__file__).parent

STDIO_INSTRUCTION = (
    "Write a complete Python 3 program that reads the input from standard input and writes the answer "
    "to standard output. Put the final program in a single ```python code block."
)
CALL_INSTRUCTION = (
    "Complete the starter code below in Python 3. Your code will be tested by calling `{fn_name}` "
    "directly, so do not read from standard input or print the answer. Put the final code, including "
    "the starter code, in a single ```python code block.\n\n```python\n{starter}\n```"
)


def load_system_prompt(name: str = "original") -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text().strip()


def user_message(problem: Problem) -> str:
    if problem.test_kind == "call":
        instruction = CALL_INSTRUCTION.format(fn_name=problem.fn_name, starter=problem.starter_code.strip("\n"))
    else:
        instruction = STDIO_INSTRUCTION
        if problem.starter_code.strip():
            instruction += f"\n\nUse this starter code:\n\n```python\n{problem.starter_code.strip(chr(10))}\n```"
    return f"{problem.question.strip()}\n\n{instruction}"


def messages(problem: Problem, system_prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message(problem)}]
