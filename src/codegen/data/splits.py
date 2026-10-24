"""Build the frozen experiment splits from APPS and TACO-verified.

- eval: stratified sample of APPS-test, used only for the final 2x2 comparison.
- dev, gepa_val, gepa_train, grpo_train: disjoint subsets of a training pool
  drawn from APPS-train and TACO-verified.

Training-pool rules:
- only test kinds in `SplitConfig.kinds` (default stdio, the format of ~99% of
  APPS-test)
- at least `min_tests` tests. Program-style APPS-train intro/interview problems
  have exactly one test, usually the example printed in the statement, which a
  model can pass by echoing the example output.
- a reference solution, so the tests can be validated later

Leakage rules:
- pool problems that near-duplicate *any* APPS-test problem are dropped (TACO
  contains APPS-test problems verbatim)
- each near-duplicate cluster in the pool is reduced to one problem (the one
  with the most tests), so no problem appears in two splits

Usage:
    python -m codegen.data.splits --download   # first time: fetch APPS + TACO-verified
    python -m codegen.data.splits              # rebuild; refuses to change eval.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from codegen.data import apps, dedup, taco

FIXED_SPLITS = ("dev", "gepa_val", "gepa_train")  # filled in this order
ALL_SPLITS = ("eval",) + FIXED_SPLITS + ("grpo_train",)


@dataclass
class SplitConfig:
    seed: int = 0
    eval_counts: dict[str, int] = field(
        default_factory=lambda: {"introductory": 150, "interview": 300, "competition": 50}
    )
    # Kept small: after the min_tests and leakage rules the pool has ~1.4k problems,
    # and GRPO needs most of them.
    sizes: dict[str, int] = field(
        default_factory=lambda: {"dev": 200, "gepa_val": 100, "gepa_train": 100}
    )
    apps_difficulties: tuple[str, ...] = ("introductory", "interview")
    taco_difficulties: tuple[str, ...] = ("EASY", "MEDIUM", "MEDIUM_HARD")
    kinds: tuple[str, ...] = ("stdio",)
    min_tests: int = 5
    require_solution: bool = True
    # Measured on the real pool vs APPS-test: copies (formatted differently) score
    # 0.45-1.0, unrelated problems <= 0.22, and nothing falls in between.
    dedup_threshold: float = 0.35


@dataclass
class SplitResult:
    ids: dict[str, list[str]]
    # reason -> [(dropped uid, uid it duplicates, jaccard)]
    dropped: dict[str, list[tuple[str, str, float]]]


def stratum(p: apps.Problem) -> tuple[str, str, str]:
    return (p.source_split, p.difficulty, p.test_kind)


def sample_eval(candidates: list[apps.Problem], cfg: SplitConfig) -> list[apps.Problem]:
    rng = random.Random(f"{cfg.seed}-eval")  # independent of train-side config
    by_diff = defaultdict(list)
    for p in sorted(candidates, key=lambda p: p.uid):
        if p.n_solutions > 0 or not cfg.require_solution:
            by_diff[p.difficulty].append(p)
    chosen = []
    for diff, n in cfg.eval_counts.items():
        pool = by_diff[diff]
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} eval candidates for {diff}, need {n}")
        chosen += rng.sample(pool, n)
    return chosen


def _largest_remainder(total: int, weights: dict) -> dict:
    """Split integer `total` across keys proportionally to `weights`."""
    wsum = sum(weights.values())
    exact = {k: total * w / wsum for k, w in weights.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    by_remainder = sorted(weights, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in by_remainder[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def training_pool(
    apps_train: list[apps.Problem], taco_problems: list[apps.Problem], cfg: SplitConfig
) -> list[apps.Problem]:
    def usable(p: apps.Problem, difficulties) -> bool:
        return (
            p.difficulty in difficulties
            and p.test_kind in cfg.kinds
            and p.n_tests >= cfg.min_tests
            and (p.n_solutions > 0 or not cfg.require_solution)
        )

    return [p for p in apps_train if usable(p, cfg.apps_difficulties)] + [
        p for p in taco_problems if usable(p, cfg.taco_difficulties)
    ]


def build_splits(
    apps_train: list[apps.Problem],
    apps_test: list[apps.Problem],
    taco_problems: list[apps.Problem],
    cfg: SplitConfig,
) -> SplitResult:
    eval_set = sample_eval(apps_test, cfg)
    pool = {p.uid: p for p in training_pool(apps_train, taco_problems, cfg)}
    test_uids = {p.uid for p in apps_test}

    texts = {p.uid: p.question for p in apps_test} | {u: p.question for u, p in pool.items()}
    pairs = dedup.near_duplicate_pairs(texts, threshold=cfg.dedup_threshold)
    score = {(a, b): round(s, 3) for a, b, s in pairs} | {(b, a): round(s, 3) for a, b, s in pairs}

    dropped: dict[str, list] = defaultdict(list)
    leaked = {}
    for a, b, _ in pairs:
        for c, t in ((a, b), (b, a)):
            if c in pool and t in test_uids and c not in leaked:
                leaked[c] = t
    dropped["dup_of_apps_test"] = [(c, t, score[(c, t)]) for c, t in leaked.items()]
    for c in leaked:
        del pool[c]

    # One problem per duplicate cluster: most tests, then uid for determinism.
    reps = []
    for g in dedup.clusters(sorted(pool), pairs):
        keep = min(g, key=lambda u: (-pool[u].n_tests, u))
        reps.append(keep)
        dropped["dup_within_pool"] += [(u, keep, score.get((u, keep))) for u in g if u != keep]

    if sum(cfg.sizes.values()) > len(reps):
        raise ValueError(f"pool has {len(reps)} problems, fixed splits need {sum(cfg.sizes.values())}")

    # Deal problems into the fixed splits, stratified by (source, difficulty, kind).
    rng = random.Random(f"{cfg.seed}-train")
    by_stratum = defaultdict(list)
    for u in reps:
        by_stratum[stratum(pool[u])].append(u)
    weights = {s: len(us) for s, us in by_stratum.items()}
    quotas = {name: _largest_remainder(n, weights) for name, n in cfg.sizes.items()}

    ids: dict[str, list[str]] = {name: [] for name in FIXED_SPLITS + ("grpo_train",)}
    for s in sorted(by_stratum):
        us = sorted(by_stratum[s])
        rng.shuffle(us)
        start = 0
        for name in FIXED_SPLITS:
            ids[name] += us[start : start + quotas[name][s]]
            start += quotas[name][s]
        ids["grpo_train"] += us[start:]

    ids = {"eval": sorted(p.uid for p in eval_set), **{k: sorted(v) for k, v in ids.items()}}
    return SplitResult(ids, {k: sorted(v) for k, v in dropped.items()})


def describe(ids: list[str], by_uid: dict[str, apps.Problem]) -> dict:
    c = Counter(stratum(by_uid[u]) for u in ids)
    tests = sorted(by_uid[u].n_tests for u in ids)
    return {
        "n": len(ids),
        "median_tests": tests[len(tests) // 2] if tests else 0,
        "by_source/difficulty/kind": {"/".join(k): n for k, n in sorted(c.items())},
    }


def write_splits(
    result: SplitResult, cfg: SplitConfig, problems: list[apps.Problem], out_dir: Path,
    force: bool, processed_dir: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval.json"
    if eval_path.exists() and not force:
        if json.loads(eval_path.read_text()) != result.ids["eval"]:
            raise SystemExit(
                f"{eval_path} exists with different ids. The eval set is frozen; "
                "pass --force only if you mean to replace it before any experiment has used it."
            )
    for name in ALL_SPLITS:
        (out_dir / f"{name}.json").write_text(json.dumps(result.ids[name], indent=0) + "\n")

    by_uid = {p.uid: p for p in problems}
    manifest = {
        "sources": {
            "apps": {"hf_repo": apps.HF_REPO, "revision": apps.HF_REVISION, "sha256": apps.SHA256},
            "taco_verified": {"hf_repo": taco.HF_REPO, "revision": taco.HF_REVISION, "sha256": taco.SHA256},
        },
        "config": asdict(cfg),
        "splits": {name: describe(result.ids[name], by_uid) for name in ALL_SPLITS},
        "dropped_counts": {k: len(v) for k, v in result.dropped.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "dropped.json").write_text(json.dumps(result.dropped, indent=1) + "\n")

    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        for name in ALL_SPLITS:
            with open(processed_dir / f"{name}.jsonl", "w") as f:
                for uid in result.ids[name]:
                    f.write(json.dumps(by_uid[uid].to_dict()) + "\n")


def load_split(
    name: str, splits_dir: Path = Path("data/splits"), processed_dir: Path = Path("data/processed")
) -> list[apps.Problem]:
    """Load one split's problems (without solutions), in id order.

    Reads the processed JSONL written by this script and checks it against the
    committed id list, so a stale processed file can't silently be used.
    """
    ids = json.loads((Path(splits_dir) / f"{name}.json").read_text())
    with open(Path(processed_dir) / f"{name}.jsonl") as f:
        problems = [apps.Problem.from_dict(json.loads(line)) for line in f]
    if [p.uid for p in problems] != ids:
        raise RuntimeError(f"{processed_dir}/{name}.jsonl is stale; rerun python -m codegen.data.splits")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps-dir", type=Path, default=Path("data/raw/apps"))
    ap.add_argument("--taco-dir", type=Path, default=Path("data/raw/taco_verified"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", action="store_true", help="fetch the pinned raw files first")
    ap.add_argument("--force", action="store_true", help="allow replacing an existing eval.json")
    args = ap.parse_args()

    if args.download:
        apps.download_apps(args.apps_dir)
        taco.download_taco(args.taco_dir)
    cfg = SplitConfig(seed=args.seed)
    apps_train, s1 = apps.load_problems(args.apps_dir, "train")
    apps_test, s2 = apps.load_problems(args.apps_dir, "test")
    taco_problems, s3 = taco.load_problems(args.taco_dir)
    for label, ps, skipped in (("apps train", apps_train, s1), ("apps test", apps_test, s2), ("taco", taco_problems, s3)):
        print(f"{label:>10}: {len(ps)} usable, skipped {dict(skipped)}")

    result = build_splits(apps_train, apps_test, taco_problems, cfg)
    write_splits(result, cfg, apps_train + apps_test + taco_problems, args.out, args.force, args.processed)
    for name in ALL_SPLITS:
        print(f"{name:>10}: {len(result.ids[name])}")
    for reason, rows in result.dropped.items():
        print(f"dropped {len(rows):>5}  {reason}")
    print(f"wrote {args.out}/ and {args.processed}/")


if __name__ == "__main__":
    main()
