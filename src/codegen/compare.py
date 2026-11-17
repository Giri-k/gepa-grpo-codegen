"""Statistics for the 2x2 (prompt x weights) on the frozen eval split.

    python -m codegen.compare C1=$WORK/gepa-grpo-runs/c1-3538415-forced:strict \
        C2=$WORK/gepa-grpo-runs/c2-1147254 C3=... C4=... [--out results/comparison.json]

Each argument is NAME=RUN_DIR (a run with scores.jsonl). ":strict" on a budget-forced
run scores every forced sample 0, which gives back the strict run it was made from.

Reported:
- pass@1 per condition (mean over problems of the fraction of the 4 samples that pass
  every test), with a 95% bootstrap CI over problems.
- Paired differences between conditions, bootstrapped over problems (the same
  resampled problems for both conditions).
- McNemar's exact test on (problem, sample) pairs: sample k of a problem in one
  condition against sample k in the other (same seed index, different generations).
- The interaction (C4 - C3) - (C2 - C1): does the prompt help as much after GRPO?
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

B = 10_000


def load(spec: str) -> tuple[str, dict[tuple[str, int], float]]:
    name, _, path = spec.partition("=")
    path, _, mode = path.partition(":")
    run = Path(path)
    scores = {(r["uid"], r["sample"]): float(r["passed_all"]) for r in map(json.loads, open(run / "scores.jsonl"))}
    if mode == "strict":
        for r in map(json.loads, open(run / "generations.jsonl")):
            if r.get("forced"):
                scores[(r["uid"], r["sample"])] = 0.0
    return name, scores


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_first": only_a, "only_second": only_b, "p": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR[:strict]")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = dict(load(s) for s in args.runs)
    keys = sorted(set.intersection(*(set(s) for s in runs.values())))
    uids = sorted({u for u, _ in keys})
    n_samples = len(keys) // len(uids)
    assert all(len(s) == len(keys) for s in runs.values()), "runs cover different samples"

    flat = {c: np.array([s[k] for k in keys]) for c, s in runs.items()}           # (problem, sample) order
    per_problem = {c: v.reshape(len(uids), n_samples).mean(1) for c, v in flat.items()}
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(uids), size=(B, len(uids)))

    def summary(x: np.ndarray) -> dict:
        boot = x[idx].mean(1)
        return {"mean": round(float(x.mean()), 4), "ci95": [round(float(q), 4) for q in np.percentile(boot, [2.5, 97.5])],
                "p_le_0": round(float((boot <= 0).mean()), 4)}

    out = {"problems": len(uids), "samples_per_problem": n_samples, "bootstrap": B, "pass@1": {}, "differences": {}}
    for c, x in per_problem.items():
        s = summary(x)
        out["pass@1"][c] = {"mean": s["mean"], "ci95": s["ci95"]}
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["differences"][f"{b} - {a}"] = summary(per_problem[b] - per_problem[a]) | {
                "mcnemar": mcnemar(flat[b], flat[a])}
    if {"C1", "C2", "C3", "C4"} <= set(runs):
        p = per_problem
        out["interaction (C4-C3)-(C2-C1)"] = summary((p["C4"] - p["C3"]) - (p["C2"] - p["C1"]))

    print(json.dumps(out, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
