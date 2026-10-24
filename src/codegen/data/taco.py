"""Download and parse TACO-verified into the same `Problem` records as APPS.

TACO-verified (likaixin/TACO-verified) is the subset of BAAI/TACO whose tests
were checked against reference solutions. It uses APPS's record format, so
parsing is shared with `apps.parse_problem`. Differences from APPS:
- one 1.9 GB JSON array instead of JSONL, so we stream it with ijson
- `solutions` is a list, not a JSON string
- difficulty labels are EASY / MEDIUM / MEDIUM_HARD / HARD / VERY_HARD /
  UNKNOWN_DIFFICULTY
- it contains copies of APPS problems, including APPS-test ones, so it must be
  deduplicated against the eval set before training on it (see splits.py)
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import ijson

from codegen.data.apps import Problem, UnusableProblem, fetch, parse_problem, sha256_file

HF_REPO = "likaixin/TACO-verified"
HF_REVISION = "9e68575e01938f016199eda6f07d1a7ce96b35e1"
FILENAME = "taco_verified.json"
SHA256 = "d0593d49df5466f53bbe701a830ade4fda1b2c8b57d47a0b76399c783892287e"
SOURCE = "taco"

sys.set_int_max_str_digits(0)


def make_uid(source_id: int) -> str:
    return f"{SOURCE}-{source_id:05d}"


def raw_path(data_dir: Path) -> Path:
    return Path(data_dir) / FILENAME


def download_taco(data_dir: Path) -> None:
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{FILENAME}"
    fetch(url, raw_path(data_dir), SHA256)


def verify_checksum(data_dir: Path) -> None:
    actual = sha256_file(raw_path(data_dir))
    if actual != SHA256:
        raise RuntimeError(f"{FILENAME} checksum mismatch: expected {SHA256}, got {actual}")


def iter_raw(path: Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def load_problems(
    data_dir: Path, keep_solutions: int = 0, verify: bool = True
) -> tuple[list[Problem], Counter]:
    """Load TACO-verified. Returns (usable problems, counts of skip reasons)."""
    if verify:
        verify_checksum(data_dir)
    problems, skipped = [], Counter()
    for raw in iter_raw(raw_path(data_dir)):
        try:
            problems.append(
                parse_problem(raw, SOURCE, keep_solutions, uid=make_uid(raw["id"]))
            )
        except UnusableProblem as e:
            skipped[str(e)] += 1
    return problems, skipped
