"""Near-duplicate detection over problem statements (word-shingle Jaccard)."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Formatting that differs between copies of the same problem: APPS writes math as
# LaTeX ($a_1, \ldots, a_n$, \leq), while TACO uses Unicode symbols or replaces
# formulas with <image>. Stripping it lets copies match on their prose.
_FORMATTING = re.compile(r"\\[a-z]+|<image>|\$")


def normalize(text: str) -> list[str]:
    return _NON_ALNUM.sub(" ", _FORMATTING.sub(" ", text.lower())).split()


def shingles(text: str, k: int = 5) -> frozenset[int]:
    words = normalize(text)
    if len(words) < k:
        words = words + [""] * (k - len(words))
    # blake2b, not hash(): Python's str hash is salted per process.
    return frozenset(
        int.from_bytes(hashlib.blake2b(" ".join(words[i : i + k]).encode(), digest_size=8).digest(), "big")
        for i in range(len(words) - k + 1)
    )


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicate_pairs(
    texts: dict[str, str], threshold: float = 0.8, k: int = 5, max_df: int = 100
) -> list[tuple[str, str, float]]:
    """Return (id_a, id_b, jaccard) for all pairs with jaccard >= threshold.

    Candidates come from an inverted index on shingles. Shingles shared by more
    than `max_df` statements (boilerplate like "the first line of the input
    contains") are ignored for candidate generation only, not for scoring.
    """
    sh = {uid: shingles(t, k) for uid, t in texts.items()}
    index: dict[int, list[str]] = defaultdict(list)
    for uid, s in sh.items():
        for h in s:
            index[h].append(uid)
    candidates: set[tuple[str, str]] = set()
    for uids in index.values():
        if 1 < len(uids) <= max_df:
            candidates.update(combinations(sorted(uids), 2))
    pairs = []
    for a, b in candidates:
        score = jaccard(sh[a], sh[b])
        if score >= threshold:
            pairs.append((a, b, score))
    return sorted(pairs)


def clusters(ids: list[str], pairs: list[tuple[str, str, float]]) -> list[list[str]]:
    """Group ids into connected components of the near-duplicate graph."""
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
