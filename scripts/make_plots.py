"""Draw the result figures into results/figures/.

    uv run --no-project --with matplotlib --with numpy python scripts/make_plots.py

Reads the run outputs in $WORK/gepa-grpo-runs (per-sample scores and generations)
and the summaries committed under results/. Every number in the figures comes
from those files; nothing is typed in by hand except the run names below and the
node-hours from the compute table in results/RESULTS.md.
"""

from __future__ import annotations

import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("RUNS", Path(os.environ["WORK"]) / "gepa-grpo-runs"))
OUT = REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = {  # name -> (run dir, strict rebuild from a forced run)
    "C1": ("c1-3538415-forced", True),
    "C2": ("c2-1147254", False),
    "C3": ("c3-1145976", False),
    "C4": ("c4-1147270", False),
}
LABEL = {"C1": "C1 base + original", "C2": "C2 base + GEPA", "C3": "C3 GRPO + original", "C4": "C4 GRPO + GEPA"}

# Reference palette (dataviz skill, light mode): categorical slots 1-4 in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
COND_COLOR = {"C1": BLUE, "C2": ORANGE, "C3": AQUA, "C4": YELLOW}
PROMPT_COLOR = {"original": BLUE, "GEPA": ORANGE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def f3(v: float, sign: bool = False) -> str:
    """Round half up to 3 places (as in RESULTS.md: 0.3575 -> 0.358), not to even."""
    d = Decimal(repr(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- per-sample data
def load_condition(run: str, strict: bool) -> list[dict]:
    d = RUNS / run
    gens = {(g["uid"], g["sample"]): g for g in map(json.loads, open(d / "generations.jsonl"))}
    rows = []
    for s in map(json.loads, open(d / "scores.jsonl")):
        g = gens[(s["uid"], s["sample"])]
        if strict and g.get("forced"):
            s = s | {"passed_all": False, "status": "unfinished_think"}
        tokens = g.get("think_tokens", g["completion_tokens"]) if strict else s["completion_tokens"]
        rows.append({"uid": s["uid"], "sample": s["sample"], "passed": bool(s["passed_all"]),
                     "status": s["status"], "tokens": tokens})
    return sorted(rows, key=lambda r: (r["uid"], r["sample"]))


data = {c: load_condition(*v) for c, v in CONDITIONS.items()}
difficulty = {}
for name in ("eval",):
    for line in open(REPO / "data" / "processed" / f"{name}.jsonl"):
        p = json.loads(line)
        difficulty[p.get("uid") or p.get("id")] = p.get("difficulty")
comparison = json.loads((REPO / "results" / "comparison.json").read_text())


def per_problem(rows):
    uids = sorted({r["uid"] for r in rows})
    by = {u: [] for u in uids}
    for r in rows:
        by[r["uid"]].append(r["passed"])
    return uids, np.array([np.mean(by[u]) for u in uids])


def boot_ci(x, b=10_000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(x), size=(b, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


# ---------------------------------------------------------------- fig 1: the 2x2
def grouped_bars(ax, values, cis, groups, ylim=(0, 0.75), labels=True):
    width, gap = 0.36, 0.02
    x = np.arange(len(groups))
    for j, prompt in enumerate(("original", "GEPA")):
        xs = x + (j - 0.5) * (width + gap)
        v = [values[g][prompt] for g in groups]
        lo = [values[g][prompt] - cis[g][prompt][0] for g in groups]
        hi = [cis[g][prompt][1] - values[g][prompt] for g in groups]
        ax.bar(xs, v, width, color=PROMPT_COLOR[prompt], label=f"{prompt} prompt", zorder=2)
        ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
        if labels:
            for xi, vi, h in zip(xs, v, hi):
                ax.text(xi, vi + h + 0.012, f3(vi), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", length=0)


pa = comparison["pass@1"]
cell = {("Base model", "original"): "C1", ("Base model", "GEPA"): "C2",
        ("GRPO model", "original"): "C3", ("GRPO model", "GEPA"): "C4"}
vals = {g: {p: pa[cell[(g, p)]]["mean"] for p in ("original", "GEPA")} for g in ("Base model", "GRPO model")}
cis = {g: {p: pa[cell[(g, p)]]["ci95"] for p in ("original", "GEPA")} for g in ("Base model", "GRPO model")}
fig, ax = plt.subplots(figsize=(6.2, 4.2))
grouped_bars(ax, vals, cis, ["Base model", "GRPO model"])
ax.set_ylabel("pass@1 (strict, 8192 tokens)")
inter = comparison["interaction (C4-C3)-(C2-C1)"]
ax.set_title("pass@1 on 500 APPS-test problems × 4 samples")
fig.text(0.01, -0.03, f"Whiskers: 95% bootstrap CI over problems. Interaction (C4−C3)−(C2−C1) = {f3(inter['mean'], True)} "
         f"[{f3(inter['ci95'][0], True)}, {f3(inter['ci95'][1], True)}].", fontsize=8.5, color=INK2)
ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 0.98))
save(fig, "fig1_main_2x2.png")

# ---------------------------------------------------------------- fig 2: effects (forest plot)
effects = [
    ("Prompt on base (C2 − C1)", comparison["differences"]["C2 - C1"]),
    ("Prompt on GRPO (C4 − C3)", comparison["differences"]["C4 - C3"]),
    ("GRPO, original prompt (C3 − C1)", comparison["differences"]["C3 - C1"]),
    ("GRPO, GEPA prompt (C4 − C2)", comparison["differences"]["C4 - C2"]),
    ("Both (C4 − C1)", comparison["differences"]["C4 - C1"]),
    ("Interaction (C4−C3)−(C2−C1)", inter),
]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for i, (name, e) in enumerate(effects[::-1]):
    ax.plot(e["ci95"], [i, i], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(e["mean"], i, "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    ax.text(e["ci95"][1] + 0.008, i, f3(e['mean'], True), va="center", fontsize=9, color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(effects)), [n for n, _ in effects[::-1]])
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlabel("Δ pass@1 (paired, 95% bootstrap CI over problems)")
ax.set_xlim(-0.15, 0.3)
ax.set_title("Effects of the prompt and of GRPO")
save(fig, "fig2_effects.png")

# ---------------------------------------------------------------- fig 3: by difficulty
levels = ["introductory", "interview", "competition"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
for ax, level in zip(axes, levels):
    v, c = {}, {}
    for g in ("Base model", "GRPO model"):
        v[g], c[g] = {}, {}
        for p in ("original", "GEPA"):
            uids, x = per_problem(data[cell[(g, p)]])
            x = np.array([xi for u, xi in zip(uids, x) if difficulty.get(u) == level])
            v[g][p], c[g][p] = float(x.mean()), boot_ci(x)
    n = sum(1 for u in uids if difficulty.get(u) == level)
    grouped_bars(ax, v, c, ["Base model", "GRPO model"], ylim=(0, 1.0))
    ax.set_title(f"{level.capitalize()} ({n} problems)")
axes[0].set_ylabel("pass@1")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.07))
save(fig, "fig3_by_difficulty.png")

# ---------------------------------------------------------------- fig 4: outcomes
outcomes = [
    ("Passed every test", BLUE, lambda r: r["passed"]),
    ("Code ran, failed a test", ORANGE, lambda r: not r["passed"] and r["status"] == "ok"),
    ("Out of tokens while thinking", AQUA, lambda r: r["status"] == "unfinished_think"),
    ("No code / syntax error", YELLOW, lambda r: r["status"] in ("no_code", "syntax_error")),
]
fig, ax = plt.subplots(figsize=(8.4, 3.2))
order = ["C4", "C3", "C2", "C1"]
for i, c in enumerate(order):
    rows, left = data[c], 0.0
    for name, color, test in outcomes:
        share = sum(map(test, rows)) / len(rows)
        ax.barh(i, share, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2,
                label=name if i == 0 else None)
        if share > 0.06:
            ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center", fontsize=9, color=INK)
        left += share
ax.set_yticks(range(len(order)), [LABEL[c] for c in order])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_title("What happened to each of the 2,000 samples")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncols=4, fontsize=8.5, handlelength=1.2)
save(fig, "fig4_outcomes.png")

# ---------------------------------------------------------------- fig 5: token use
fig, ax = plt.subplots(figsize=(6.8, 4.0))
for c in ("C1", "C2", "C3", "C4"):
    t = np.sort([r["tokens"] for r in data[c]])
    y = np.arange(1, len(t) + 1) / len(t)
    keep = t < 8192
    ax.plot(np.append(t[keep], 8192), np.append(y[keep], y[keep][-1]), color=COND_COLOR[c], label=LABEL[c])
    trunc = 1 - y[keep][-1]
    ax.text(8250, y[keep][-1], f"{LABEL[c].split()[0]}: {trunc:.0%} hit the limit", va="center", fontsize=8.5, color=INK)
ax.axvline(8192, color=INK2, lw=1, ls="--")
ax.set_xlim(0, 8192)
ax.set_ylim(0, 1)
ax.set_xlabel("Completion tokens (reasoning + answer)")
ax.set_ylabel("Share of samples finished")
ax.grid(axis="x", visible=True)
ax.set_title("Samples that finish within the token budget")
ax.legend(loc="upper left")
save(fig, "fig5_token_use.png")

# ---------------------------------------------------------------- fig 6: GRPO training
state = json.loads((REPO / "results" / "grpo-main" / "trainer_state.json").read_text())
train = [r for r in state["log_history"] if "completions/mean_length" in r and r["step"] <= 120]
evals = {}
for r in state["log_history"]:
    if "eval_rewards/passed_all/mean" in r and r["step"] not in (1,):
        evals[r["step"]] = r["eval_rewards/passed_all/mean"]
steps = np.array([r["step"] for r in train])


def rolling(x, w=10):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1): i + 1].mean() for i in range(len(x))])


def raw_and_smooth(ax, y, color, label):
    ax.plot(steps, y, color=color, lw=1, alpha=0.3)
    ax.plot(steps, rolling(y), color=color, label=label)


fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
ax = axes[0, 0]
raw_and_smooth(ax, [r["rewards/tests_passed/mean"] for r in train], BLUE, "reward (fraction of tests passed)")
raw_and_smooth(ax, [r["rewards/passed_all/mean"] for r in train], ORANGE, "passed every test")
ax.set_title("Training reward (16 × 8 per step)")
ax.legend(loc="lower right", fontsize=8.5)
ax.set_ylim(0, 1)

ax = axes[0, 1]
es = sorted(evals)
ax.plot(es, [evals[s] for s in es], color=BLUE, marker="o", ms=7, mec=SURFACE, mew=2)
for s in (es[0], es[-1]):
    ax.text(s, evals[s] + 0.02, f3(evals[s]), ha="center", fontsize=9, color=INK)
ax.set_ylim(0.3, 0.7)
ax.set_title("Dev pass@1 (64 × 4)")

ax = axes[0, 2]
raw_and_smooth(ax, [r["completions/mean_length"] for r in train], BLUE, "mean completion tokens")
ax.axhline(8192, color=INK2, lw=1, ls="--")
ax.text(steps[0], 8050, "limit 8192", fontsize=8.5, color=INK2, va="top")
ax.set_ylim(0, 8600)
ax.set_title("Mean completion length (tokens)")

ax = axes[1, 0]
raw_and_smooth(ax, [r["completions/clipped_ratio"] for r in train], BLUE, "truncated")
ax.set_ylim(0, 1)
ax.set_title("Share of samples cut off at 8192 tokens")

ax = axes[1, 1]
raw_and_smooth(ax, [r["frac_reward_zero_std"] for r in train], BLUE, "zero-variance groups")
ax.set_ylim(0, 1)
ax.set_title("Groups with all 8 rewards equal")

ax = axes[1, 2]
raw_and_smooth(ax, [r["entropy"] for r in train], BLUE, "entropy")
ax.set_ylim(bottom=0)
ax.set_title("Token entropy")
for ax in axes.flat:
    ax.set_xlabel("GRPO step")
    ax.set_xlim(0, 121)
fig.text(0.01, -0.01, "Faint lines: per step. Solid: 10-step trailing mean.", fontsize=8.5, color=INK2)
fig.tight_layout()
save(fig, "fig6_grpo_training.png")

# ---------------------------------------------------------------- fig 7: GEPA search
summary = json.loads((REPO / "results" / "gepa-main" / "gepa_summary.json").read_text())
cands = summary["candidates"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for c in cands:
    for p in c["parents"] or []:
        if p is not None:
            q = cands[p]
            ax.plot([q["found_after_metric_calls"], c["found_after_metric_calls"]], [q["val_score"], c["val_score"]],
                    color=GRID, lw=1.2, zorder=1)
calls = [c["found_after_metric_calls"] for c in cands]
scores = [c["val_score"] for c in cands]
best, xs, ys = 0, [], []
for x, s in sorted(zip(calls, scores)):
    best = max(best, s)
    xs.append(x)
    ys.append(best)
xs.append(summary["total_metric_calls"])
ys.append(best)
ax.step(xs, ys, where="post", color=ORANGE, lw=2, label="best so far", zorder=2)
ax.plot(calls, scores, "o", color=BLUE, ms=8, mec=SURFACE, mew=2, label="candidate prompt", zorder=3)
for c in cands:
    if c["idx"] in (0, 1, 3):
        name = "original" if c["idx"] == 0 else f"#{c['idx']}" + (" → gepa_v1" if c["idx"] == 1 else "")
        ax.annotate(name, (c["found_after_metric_calls"], c["val_score"]), xytext=(6, -14 if c["idx"] == 3 else 6),
                    textcoords="offset points", fontsize=9, color=INK)
ax.axvline(1060, color=INK2, lw=1, ls=":")
ax.text(1075, 0.31, "job 2 resumes", fontsize=8.5, color=INK2)
ax.set_xlabel("GEPA metric calls (problem rollouts) when the prompt was found")
ax.set_ylabel("pass@1 on gepa_val (100 problems)")
ax.set_ylim(0.3, 0.6)
ax.set_xlim(-40, summary["total_metric_calls"] + 40)
ax.grid(axis="x", visible=True)
ax.set_title("GEPA prompt search: 63 proposals, 14 kept")
ax.legend(loc="lower right")
save(fig, "fig7_gepa_search.png")

# ---------------------------------------------------------------- fig 8: GEPA prompt choice on dev
dev = {}
for d in sorted((REPO / "results").glob("dev-*")):
    s = json.loads((d / "summary.json").read_text())
    dev[s["system_prompt"]] = s
val = {"original": cands[0]["val_score"], "gepa_cand1": cands[1]["val_score"], "gepa_cand2": cands[2]["val_score"]}
names = {"original": "original", "gepa_cand1": "#1 (gepa_v1)", "gepa_cand2": "#2 (no reasoning)"}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
keys = ["original", "gepa_cand1", "gepa_cand2"]
x = np.arange(len(keys))
for j, (label, color, get) in enumerate([("gepa_val (100 × 1)", BLUE, lambda k: val[k]),
                                         ("dev (200 × 4)", ORANGE, lambda k: dev[k]["pass@1"])]):
    xs = x + (j - 0.5) * 0.38
    v = [get(k) for k in keys]
    ax.bar(xs, v, 0.36, color=color, label=label, zorder=2)
    for xi, vi in zip(xs, v):
        ax.text(xi, vi + 0.01, f3(vi), ha="center", fontsize=9, color=INK)
ax.set_xticks(x, [names[k] for k in keys])
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 0.65)
ax.set_ylabel("pass@1")
ax.set_title("Choosing the GEPA prompt: validation vs held-out dev")
ax.legend(loc="upper right")
save(fig, "fig8_gepa_dev_choice.png")

# ---------------------------------------------------------------- fig 9: gain vs compute
NODE_HOURS = {"GEPA": 6.1, "GRPO": 24.1, "GEPA + GRPO": 30.2}  # results/RESULTS.md, compute table
gain = {"GEPA": comparison["differences"]["C2 - C1"], "GRPO": comparison["differences"]["C3 - C1"],
        "GEPA + GRPO": comparison["differences"]["C4 - C1"]}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for k in NODE_HOURS:
    g = gain[k]
    ax.plot([NODE_HOURS[k]] * 2, g["ci95"], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(NODE_HOURS[k], g["mean"], "o", color=BLUE, ms=9, mec=SURFACE, mew=2)
    ax.annotate(f"{k}\n{f3(g['mean'], True)}", (NODE_HOURS[k], g["mean"]), xytext=(9, -4),
                textcoords="offset points", fontsize=9, color=INK, va="center")
ax.set_xlim(0, 36)
ax.set_ylim(0, 0.28)
ax.set_xlabel("GH200 node-hours to obtain the method (search / training)")
ax.set_ylabel("Δ pass@1 over C1")
ax.grid(axis="x", visible=True)
ax.set_title("Gain over the baseline vs compute spent")
save(fig, "fig9_gain_vs_compute.png")
