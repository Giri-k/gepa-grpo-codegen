"""Check every split problem's tests by running its official solutions.

A problem whose official solutions all fail our checker has a broken answer
key (often an "any valid answer" problem) and is dropped from the training
splits. Eval keeps all its problems; the labels let us also report pass@1 on
the valid subset. Labels depend only on official solutions, never on model
outputs.

Labels:
    valid               an official solution passes every test
    too_slow            the only failures are timeouts / time budget
    wrong_answer        solutions run but print something the checker rejects
    runtime_error       solutions crash
    no_usable_solution  every solution fails to compile (often Python 2)

Results are cached in data/splits/validation.json (committed); only problems
not already in it are run.

Usage:
    python -m codegen.data.validate              # then: python -m codegen.data.splits
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from codegen.data import apps, splits, taco
from codegen.judge import evaluate_code_many, tokens_match
from codegen.sandbox import Limits, Sandbox

LABELS = ("valid", "too_slow", "wrong_answer", "runtime_error", "no_usable_solution")
CHUNK = 200  # problems per progress line


def _attempt(problem: apps.Problem, e) -> dict:
    """Summary of one solution's run. `case_fix`: would pass if case were ignored."""
    fails = Counter(v for v in e.verdicts if v != "passed")
    case_fix = (
        problem.test_kind == "stdio" and set(fails) == {"wrong_answer"} and all(
            tokens_match(out.lower(), exp.lower())
            for v, out, exp in zip(e.verdicts, e.stdouts, problem.outputs) if v == "wrong_answer"
        )
    )
    return {"status": e.status, "passed": e.passed, "fails": dict(fails), "case_fix": case_fix}


def label(attempts: list[dict]) -> str:
    if any(a["status"] == "ok" and not a["fails"] for a in attempts):
        return "valid"
    ran = [a for a in attempts if a["status"] == "ok"]
    if not ran:
        return "no_usable_solution"
    if any(set(a["fails"]) <= {"timeout", "skipped"} for a in ran):
        return "too_slow"
    if any("wrong_answer" in a["fails"] for a in ran):
        return "wrong_answer"
    return "runtime_error"


def validate(
    problems: list[apps.Problem], sandbox: Sandbox, max_solutions: int = 5,
    limits: Limits | None = None, progress=print,
) -> dict[str, dict]:
    """Try up to `max_solutions` official solutions per problem, stopping at the first that passes."""
    attempts: dict[str, list[dict]] = {p.uid: [] for p in problems}
    pending = list(problems)
    for k in range(max_solutions):
        batch = [p for p in pending if k < len(p.solutions)]
        if not batch:
            break
        start = time.monotonic()
        for i in range(0, len(batch), CHUNK):
            chunk = batch[i : i + CHUNK]
            evals = evaluate_code_many(
                [(p, p.solutions[k]) for p in chunk], sandbox, limits, keep_stdout=True
            )
            for p, e in zip(chunk, evals):
                attempts[p.uid].append(_attempt(p, e))
            progress(f"solution {k + 1}: {min(i + CHUNK, len(batch))}/{len(batch)} problems, "
                     f"{time.monotonic() - start:.0f}s")
        pending = [p for p in batch if label(attempts[p.uid]) != "valid"]

    return {
        p.uid: {
            "label": label(attempts[p.uid]),
            "n_tests": p.n_tests,
            "attempts": attempts[p.uid],
        }
        for p in problems
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps-dir", type=Path, default=Path("data/raw/apps"))
    ap.add_argument("--taco-dir", type=Path, default=Path("data/raw/taco_verified"))
    ap.add_argument("--out", type=Path, default=Path("data/splits/validation.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-solutions", type=int, default=5)
    ap.add_argument("--limit", type=int, help="only validate this many problems (for a quick try)")
    args = ap.parse_args()

    print("loading problems with official solutions...")
    apps_train, _ = apps.load_problems(args.apps_dir, "train", keep_solutions=args.max_solutions)
    apps_test, _ = apps.load_problems(args.apps_dir, "test", keep_solutions=args.max_solutions)
    taco_problems, _ = taco.load_problems(args.taco_dir, keep_solutions=args.max_solutions)
    # Everything that could be placed in a split: eval + the deduplicated pool.
    result = splits.build_splits(apps_train, apps_test, taco_problems, splits.SplitConfig())
    by_uid = {p.uid: p for p in apps_train + apps_test + taco_problems}
    wanted = sorted(u for ids in result.ids.values() for u in ids)

    cache = json.loads(args.out.read_text()) if args.out.exists() else {}
    todo = [by_uid[u] for u in wanted if u not in cache][: args.limit]
    print(f"{len(wanted)} problems, {sum(u in cache for u in wanted)} already validated, running {len(todo)}")
    if todo:
        with Sandbox(workers=args.workers) as sb:
            cache |= validate(todo, sb, args.max_solutions)
        args.out.write_text(json.dumps(dict(sorted(cache.items())), indent=1) + "\n")

    print(f"\n{'split':>10}  " + "  ".join(f"{l:>18}" for l in LABELS) + "   case_fix")
    for name, ids in result.ids.items():
        rows = [cache[u] for u in ids if u in cache]
        counts = Counter(r["label"] for r in rows)
        case_fix = sum(any(a["case_fix"] for a in r["attempts"]) for r in rows if r["label"] != "valid")
        print(f"{name:>10}  " + "  ".join(f"{counts[l]:>18}" for l in LABELS) + f"   {case_fix:>8}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
