"""Summarize a GRPO run's performance logs: where each step's time goes and how
busy the GPUs and CPUs were.

    python -m codegen.perf_report $SCRATCH/gepa-grpo/grpo-<jobid>

Reads <run>/perf.rank*.jsonl (written by codegen.grpo), and <run>/gpu.csv and
<run>/cpu.log (sampled every 5 s by tacc/grpo.sh) when present.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def step_table(run: Path) -> None:
    rows = [json.loads(line) for f in sorted(run.glob("perf.rank*.jsonl")) for line in open(f)]
    if not rows:
        print("no perf.rank*.jsonl rows yet")
        return
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    print("per step (mean over GPUs; generate = vLLM, other = weight sync + log-probs + backward):")
    print(f"{'step':>5} {'step s':>7} {'generate':>9} {'reward':>7} {'other':>7} {'tok/s/GPU':>10} {'tokens':>8} {'peak GB':>8}")
    for step, rs in sorted(by_step.items()):
        tps = [r["generate_tokens_per_second"] for r in rs if r["generate_tokens_per_second"]]
        print(f"{step:>5} {mean(r['step_seconds'] for r in rs):>7.0f} {mean(r['generate_seconds'] for r in rs):>9.0f} "
              f"{mean(r['reward_seconds'] for r in rs):>7.0f} {mean(r['other_seconds'] for r in rs):>7.0f} "
              f"{mean(tps) if tps else 0:>10.0f} {sum(r['completion_tokens'] for r in rs):>8} "
              f"{max(r['peak_allocated_gb'] for r in rs):>8.1f}")
    steady = [r for r in rows if r["step"] > 1] or rows  # step 1 includes compilation and warm-up
    total = mean(r["step_seconds"] for r in steady)
    print(f"\nsteady state (steps > 1): {total:.0f} s/step = "
          + ", ".join(f"{k} {mean(r[k + '_seconds'] for r in steady) / total:.0%}" for k in ("generate", "reward", "other")))
    profile = defaultdict(list)
    for r in steady:
        for k, v in r["profile"].items():
            profile[k].append(v)
    print("profiled blocks, mean s/step: " + ", ".join(f"{k} {mean(v):.1f}" for k, v in sorted(profile.items())))


def gpu_table(run: Path) -> None:
    path = run / "gpu.csv"
    if not path.exists():
        return
    samples = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f, skipinitialspace=True):
            try:
                samples[r["index"]].append((float(r["utilization.gpu [%]"]), float(r["memory.used [MiB]"]) / 1024,
                                            float(r["power.draw [W]"])))
            except (KeyError, ValueError):  # header repeated or a partial last line
                continue
    print("\nGPUs (nvidia-smi every 5 s, whole run):")
    for gpu, s in sorted(samples.items()):
        util = [x[0] for x in s]
        print(f"  GPU {gpu}: util mean {mean(util):.0f}%, idle (<10%) {sum(u < 10 for u in util) / len(util):.0%} of samples, "
              f"memory peak {max(x[1] for x in s):.1f} GB, power mean {mean(x[2] for x in s):.0f} W")


def cpu_table(run: Path) -> None:
    path = run / "cpu.log"
    if not path.exists():
        return
    busy = []
    for line in open(path):
        cols = line.split()
        if len(cols) > 15 and cols[0].isdigit():
            busy.append(100 - int(cols[14]))  # vmstat: us sy id wa st start at column 12; id is 14
    if busy:
        print(f"\nCPU (vmstat every 5 s): busy mean {mean(busy):.0f}%, max {max(busy)}% "
              "(sustained ~100% during reward means the sandbox is CPU-bound)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the GRPO --out directory")
    run = ap.parse_args().run
    step_table(run)
    gpu_table(run)
    cpu_table(run)


if __name__ == "__main__":
    main()
