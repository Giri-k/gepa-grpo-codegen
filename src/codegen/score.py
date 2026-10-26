"""Score generations from `codegen.generate` in the sandbox.

    python -m codegen.score --gen $SCRATCH/.../gen [--workers 32]

Writes `<gen>/scores.jsonl` (one row per sample: status, reward, verdict
counts) and `<gen>/summary.json`, and prints the summary. pass@1 is the mean
over problems of each problem's fraction of samples that pass every test (the
unbiased estimator with n samples), with a 95% bootstrap interval over problems.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from codegen.data.splits import load_split
from codegen.judge import evaluate_many
from codegen.sandbox import Sandbox


def bootstrap_ci(values: list[float], reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def summarize(rows: list[dict], difficulty: dict[str, str]) -> dict:
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["uid"]].append(r)
    pass_rate = {uid: mean(r["passed_all"] for r in rs) for uid, rs in by_problem.items()}
    rates = list(pass_rate.values())
    lo, hi = bootstrap_ci(rates)

    by_difficulty = defaultdict(list)
    for uid, rate in pass_rate.items():
        by_difficulty[difficulty[uid]].append(rate)

    return {
        "problems": len(by_problem),
        "samples": len(rows),
        "pass@1": round(mean(rates), 4),
        "pass@1_ci95": [round(lo, 4), round(hi, 4)],
        "mean_reward": round(mean(r["reward"] for r in rows), 4),
        # GRPO only learns from problems where samples disagree.
        "problems_all_fail": sum(r == 0 for r in rates),
        "problems_mixed": sum(0 < r < 1 for r in rates),
        "problems_all_pass": sum(r == 1 for r in rates),
        "pass@1_by_difficulty": {d: {"problems": len(v), "pass@1": round(mean(v), 4)} for d, v in sorted(by_difficulty.items())},
        "status": dict(Counter(r["status"] for r in rows).most_common()),
        "test_verdicts": dict(sum((Counter(r["verdicts"]) for r in rows), Counter()).most_common()),
        "truncated": round(mean(r["finish_reason"] == "length" for r in rows), 4),
        "mean_completion_tokens": round(mean(r["completion_tokens"] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True, help="directory written by codegen.generate")
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    args = ap.parse_args()

    meta = json.loads((args.gen / "meta.json").read_text())
    problems = {p.uid: p for p in load_split(meta["split"])}
    gens = [json.loads(line) for line in open(args.gen / "generations.jsonl")]

    start = time.time()
    with Sandbox(workers=args.workers) as sb:
        isolation = sb.isolation
        evals = evaluate_many([(problems[g["uid"]], g["text"]) for g in gens], sb)
    seconds = time.time() - start

    rows = []
    with open(args.gen / "scores.jsonl", "w") as f:
        for g, e in zip(gens, evals):
            row = {
                "uid": g["uid"], "sample": g["sample"], "status": e.status, "reward": e.reward,
                "passed_all": e.passed_all, "verdicts": dict(Counter(e.verdicts)), "n_tests": e.n_tests,
                "finish_reason": g["finish_reason"], "completion_tokens": g["completion_tokens"],
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    summary = summarize(rows, {uid: p.difficulty for uid, p in problems.items()})
    summary.update({
        "model": meta["model"], "lora": meta["lora"], "system_prompt": meta["system_prompt_name"],
        "split": meta["split"], "sampling": meta["sampling"],
        "generate_seconds": meta["generate_seconds"], "tokens_per_second": meta["tokens_per_second"],
        "score_seconds": round(seconds, 1), "sandbox_workers": args.workers, "sandbox_isolation": isolation,
    })
    (args.gen / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
