"""Download and parse the APPS dataset into normalized `Problem` records.

We read the raw JSONL files from the Hugging Face repo directly, because the
repo's `apps.py` loading script no longer runs under recent `datasets` versions.

APPS quirks handled here:
- `input_output` is a JSON string. It may be empty, invalid JSON, or have
  mismatched/empty `inputs`/`outputs`. Such problems are unusable.
- stdin/stdout tests store each input/output either as a string or as a list
  of lines. Lists are joined with "\\n" so every stdio test is a string.
- Call-based tests (`fn_name` present) store each input as a list of
  positional args. Outputs are kept raw: APPS often wraps the expected return
  value in a one-element list, which the evaluator must account for.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

HF_REPO = "codeparrot/apps"
# Pinned so every machine (laptop, TACC) builds splits from identical bytes.
HF_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
SHA256 = {
    "train": "45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae",
    "test": "5b003a65ac40feb47dd5eaec267a767a6fc435bdcfa68ff715fe869f948e760c",
}
DIFFICULTIES = ("introductory", "interview", "competition")

# Some call-based tests hold integers with thousands of digits, which Python
# 3.11+ refuses to parse by default.
sys.set_int_max_str_digits(0)


@dataclass(frozen=True)
class Problem:
    uid: str  # e.g. "test-0042"; unique across both APPS splits
    source_split: str
    source_id: int
    difficulty: str
    question: str
    starter_code: str
    fn_name: str | None
    inputs: list  # stdio: list[str]; call-based: list[list] of positional args
    outputs: list  # stdio: list[str]; call-based: raw expected values
    n_solutions: int
    url: str
    solutions: tuple[str, ...] = field(default=(), repr=False)

    @property
    def test_kind(self) -> str:
        return "call" if self.fn_name else "stdio"

    @property
    def n_tests(self) -> int:
        return len(self.inputs)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "Problem":
        return cls(**{**d, "solutions": tuple(d.get("solutions", ()))})


class UnusableProblem(ValueError):
    """Raised when a raw APPS record has no usable tests."""


def make_uid(split: str, source_id: int) -> str:
    return f"{split}-{source_id:04d}"


def raw_path(data_dir: Path, split: str) -> Path:
    return Path(data_dir) / f"{split}.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: str, attempts: int = 5) -> None:
    """Download `url` to `dest` and verify its checksum.

    Skips the download if `dest` already matches. An interrupted download is
    resumed from its `.part` file with an HTTP Range request.
    """
    dest = Path(dest)
    if dest.exists() and sha256_file(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(req) as resp:
                # 206 = server honored the Range header; 200 = full file, start over.
                mode = "ab" if resp.status == 206 else "wb"
                with open(part, mode) as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            break
        except OSError as e:
            if attempt == attempts:
                raise
            print(f"download of {dest.name} interrupted ({e}); resuming")
    actual = sha256_file(part)
    if actual != sha256:
        part.unlink()
        raise RuntimeError(f"{dest.name} checksum mismatch: expected {sha256}, got {actual}")
    part.replace(dest)


def download_apps(data_dir: Path, splits=("train", "test")) -> None:
    """Fetch the pinned raw JSONL files, skipping any already present and verified."""
    for split in splits:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{split}.jsonl"
        fetch(url, raw_path(data_dir, split), SHA256[split])


def verify_checksum(data_dir: Path, split: str) -> None:
    actual = sha256_file(raw_path(data_dir, split))
    if actual != SHA256[split]:
        raise RuntimeError(
            f"{split}.jsonl checksum mismatch: expected {SHA256[split]}, got {actual}"
        )


def _join_lines(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    raise UnusableProblem(f"stdio test value has type {type(value).__name__}")


def parse_problem(
    raw: dict, split: str, keep_solutions: int = 0, uid: str | None = None
) -> Problem:
    """Normalize one raw APPS-format record (APPS or TACO).

    Raises UnusableProblem if it has no usable tests.
    """
    if not raw.get("input_output"):
        raise UnusableProblem("empty input_output")
    try:
        io = json.loads(raw["input_output"])
    except json.JSONDecodeError as e:
        raise UnusableProblem("invalid input_output JSON") from e
    inputs, outputs = io.get("inputs") or [], io.get("outputs") or []
    if not inputs:
        raise UnusableProblem("no tests")
    if len(inputs) != len(outputs):
        raise UnusableProblem("inputs/outputs length mismatch")

    fn_name = io.get("fn_name") or None
    if fn_name:
        if not all(isinstance(args, list) for args in inputs):
            raise UnusableProblem("call-based input is not an arg list")
    else:
        inputs = [_join_lines(v) for v in inputs]
        outputs = [_join_lines(v) for v in outputs]

    # APPS stores solutions as a JSON string, TACO as a list.
    solutions = raw.get("solutions") or []
    if isinstance(solutions, str):
        try:
            solutions = json.loads(solutions)
        except json.JSONDecodeError:
            solutions = []

    return Problem(
        uid=uid or make_uid(split, raw["id"]),
        source_split=split,
        source_id=raw["id"],
        difficulty=raw["difficulty"],
        question=raw["question"],
        starter_code=raw.get("starter_code") or "",
        fn_name=fn_name,
        inputs=inputs,
        outputs=outputs,
        n_solutions=len(solutions),
        url=raw.get("url") or "",
        solutions=tuple(solutions[:keep_solutions]),
    )


def iter_raw(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_problems(
    data_dir: Path, split: str, keep_solutions: int = 0, verify: bool = True
) -> tuple[list[Problem], Counter]:
    """Load one APPS split. Returns (usable problems, counts of skip reasons)."""
    if verify:
        verify_checksum(data_dir, split)
    problems, skipped = [], Counter()
    for raw in iter_raw(raw_path(data_dir, split)):
        try:
            problems.append(parse_problem(raw, split, keep_solutions))
        except UnusableProblem as e:
            skipped[str(e)] += 1
    return problems, skipped
