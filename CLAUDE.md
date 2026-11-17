# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This is a UT "Post-Training of LLMs" course project by Girishkaushik Karthikeyan and Sergio Sánchez Manso; the proposal PDF lives one directory up and is not in the repo. `plan.md` holds the original dataset survey and design options; some choices changed since (below).

**All four conditions are done.** See `results/RESULTS.md` for the 2×2 results, statistics, GEPA/GRPO details, compute table and caveats, and `results/C1.md` for the baseline and budget forcing. Remaining: the course write-up.

Decisions that differ from `plan.md`:
- **Model:** `Qwen/Qwen3-4B-Thinking-2507` (thinking on), strict 8192-token limit shared by reasoning and answer. Sampling: T 0.6, top-p 0.95, top-k 20, seed 0, 4 samples per problem.
- **GRPO:** TRL `dr_grpo`, β 0, LoRA r 32, lr 1e-5, 16 problems × 8 samples per step. Reward is the fraction of tests passed, and a truncated sample scores 0. The final adapter is checkpoint-120 at `$WORK/gepa-grpo-runs/grpo-main/checkpoint-120`.
- **GEPA:** the `gepa` 0.1.4 package with `gepa_adapter.py`. The reflection model is Qwen3-30B-A3B-Thinking-2507 on a second node. The final prompt is `src/codegen/prompts/gepa_v1.txt`, which is candidate #1, chosen on `dev`.
- **Budget forcing** (`force_answer.py`) is reported for C1 only. All conditions are compared on the strict metric.

Large runs happen on **TACC Vista** (GH200, 1 GPU per node; account CDA26008; 1 SU per node-hour). TACC has no Docker, so the sandbox uses Apptainer, and Apptainer does not run on login nodes. Large outputs go in `$SCRATCH`, which is purged, and are copied to `$WORK/gepa-grpo-runs/<run>`. Each run's summary also goes in `results/<run>`. TACC refuses `sbatch` from compute nodes, so a job stopped by its time limit is continued by submitting the same command again from a login node.

```bash
cd $WORK/post-training/gepa-grpo-codegen
NAME=c2 PROMPT=gepa_v1 sbatch -p gh-dev -t 02:00:00 tacc/eval.sh                  # one condition (SPLIT=dev, LORA=... also)
OUT=$SCRATCH/gepa-grpo/grpo-<id> sbatch tacc/grpo.sh --max-steps 200 --resume      # GRPO (continue)
OUT=$SCRATCH/gepa-grpo/gepa-main sbatch tacc/gepa.sh --max-metric-calls 2000       # GEPA, 2 nodes (resumes from OUT)
PYTHONPATH=src .venv-gpu/bin/python -m codegen.compare C1=$WORK/gepa-grpo-runs/c1-3538415-forced:strict \
  C2=... C3=... C4=... --out results/comparison.json                               # 2x2 statistics
```

The rest of this file describes the data pipeline and evaluator from the planning phase; the Docker details apply to local development only.

## Commands

```bash
uv sync                                        # create .venv (Python >=3.10) with dev deps
uv run pytest                                  # all tests (sandbox tests skip if Docker isn't running)
uv run pytest tests/test_splits.py::test_write_freeze_guard_and_load_roundtrip   # one test
uv run python -m codegen.data.splits --download  # fetch pinned APPS + TACO-verified (~3.3 GB) and build splits
uv run python -m codegen.data.validate         # run official solutions in the sandbox (Docker), cached in data/splits/validation.json
uv run python -m codegen.data.splits           # rebuild splits, ~5 min (refuses to change eval.json without --force)
uv run python scripts/sandbox_demo.py          # see the sandbox handle a few small programs
```

## Data pipeline (`src/codegen/data/`)

**Dataset decision:** eval = APPS-test (as in the proposal). All other splits = **stdio problems with ≥5 tests and a reference solution**, drawn from APPS-train + TACO-verified (EASY/MEDIUM/MEDIUM_HARD). Why:
- APPS-test is ~99% stdio, while APPS-train introductory problems are ~86% call-based.
- Every stdio APPS-train intro/interview problem has exactly **one** test, usually the example in the statement, which a model can pass by echoing it. So in practice almost all training data comes from TACO.
- Dropping copies of APPS-test removes more than half of TACO's multi-test problems, which leaves ~1.4k problems. That's why the fixed splits are small (dev 200, gepa_val 100, gepa_train 100).

- `apps.py` / `taco.py` download raw files from HF at **pinned revisions with SHA-256 checks**. Loading fails on a checksum mismatch, which has already caught a hand-edited raw file once. Raw files live in gitignored `data/raw/`. Both datasets share the APPS record format and `apps.parse_problem`. Stdio tests are normalized to strings. Call-based outputs are kept raw (APPS often wraps the return value in a one-element list, so the evaluator must accept both forms). Ids are `train-0001`, `test-0042`, `taco-01234`.
- `splits.py` writes id lists `data/splits/{eval,dev,gepa_val,gepa_train,grpo_train}.json` plus `manifest.json` and `dropped.json` (committed), and full problems to `data/processed/*.jsonl` (gitignored). The fixed splits are stratified by (source, difficulty, kind), and `grpo_train` gets everything left.
- Leakage rules (`dedup.py`: 5-word-shingle Jaccard ≥ 0.35 after stripping LaTeX, `$` and `<image>`, a cutoff chosen from a measured gap between copies and distinct problems): drop pool problems that near-duplicate **any** APPS-test problem (TACO contains APPS-test verbatim). Each duplicate cluster in the pool keeps one problem, the one with the most tests.
- Validity rules: training problems with "print any valid answer" wording are excluded (expected outputs came from one solution, so a different correct answer fails, and validation can't detect it). `validate.py` runs up to 5 official solutions per problem, and training splits keep only `valid` ones. Eval is never filtered; its labels let us report pass@1 on all 500 and on the valid subset. Adding pool problems means re-running `validate` (only uncached problems run) before `splits`.
- `eval` uses its own RNG stream, so train-side config changes never alter it. **`eval.json` is frozen once committed.** Use `load_split(name)`, which reads `data/processed/` and checks it against the committed ids.

## Sandbox (`src/codegen/sandbox/`)

- `Sandbox(workers=N)` keeps N long-running Docker containers, with no network, no host mounts, a read-only filesystem except a small `/tmp`, memory and process caps, no capabilities, and an unprivileged user. Each container runs `worker.py` (stdlib only, copied into the image), which takes one JSON job per line.
- Each test is a fresh `python solution.py` with limits (4 s per test, 30 s per program, then `skipped`; 4 MB stdout cap, above the largest expected output of 1.4 MB). After every job the worker kills stray processes and empties `/tmp`. **Expected outputs never enter the container.** The sandbox returns raw stdout plus an outcome, and comparison happens on the host.
- `Sandbox()` rebuilds the image on start (Docker caches it), so edits to `worker.py` take effect. On TACC (no Docker), the same image must be converted for Apptainer. Whether unprivileged network isolation works there is still unverified.

## Judge (`src/codegen/judge.py`)

`evaluate_many(items, sandbox)` takes (problem, model text) pairs and runs extract → sandbox → compare. `evaluate_code_many` does the same for plain code (official solutions). Stdio comparison works on whitespace tokens, allows 1e-6 tolerance only where the expected token is a decimal, and is otherwise exact and case-sensitive (validation found 0 official solutions failing on case). Call-based answers are wrapped by a harness and their return value compared as JSON, accepting APPS's `[x]` wrapping. `Evaluation.reward` is the fraction of tests passed (the GRPO reward), `passed_all` is pass@1, and `feedback` is text for GEPA.

## Research question

Can prompt optimization (GEPA), weight post-training (GRPO + LoRA), or both together improve **first-attempt (pass@1) Python code generation** for **Qwen3-4B**, evaluated on **APPS-test**?

## Experimental design (constraints that code must respect)

- **Four conditions** (a 2×2 design with a single GRPO run):
  1. base model + original prompt
  2. base model + GEPA-optimized prompt
  3. GRPO-trained model + original prompt
  4. GRPO-trained model + GEPA prompt
- **Data splits:** APPS *train* is divided into disjoint subsets for GRPO training, GEPA prompt optimization, and development decisions. A fixed, prespecified sample of APPS *test* is used **only** for final evaluation; never use it for tuning. Start with introductory and interview-level problems that have usable tests, and audit a sample before training.
- **Baseline prompt:** problem statement + any starter code + a fixed instruction to return a Python solution. Prompt format, generation settings, output-token limit, and execution procedure are recorded and held constant across conditions wherever possible.
- **Metrics:** primary is pass@1 (the first generated solution passes every available test). Also report the fraction of tests passed, invalid/timed-out solutions, the number of model generations, and GPU-hours per method. Break failures down by difficulty and error type, and report uncertainty across problems.

## Planned architecture

- **One shared evaluator** used by GEPA, GRPO, and final eval, so feedback is consistent. It extracts code from the model output, runs it against APPS input/output tests, and returns:
  - a scalar reward (proportion of tests passed; full reward only when all tests pass) for GRPO
  - structured execution feedback (wrong output / runtime error / timeout) for GEPA's reflective prompt updates
- **Sandboxing:** generated code runs in isolated containers with network disabled, restricted filesystem access, memory and CPU limits, and a wall-clock timeout. Expected outputs stay **outside** the container.
- **GRPO:** a LoRA adapter on Qwen3-4B. Several candidate solutions per problem, each with a verifiable reward from the evaluator.
- **GEPA:** optimizes only the system prompt, against the frozen base model.

## First milestone

An end-to-end pilot on 50 problems: code extraction → test execution → reward logging → baseline pass@1.

## Known risks to keep in mind

- APPS input/output formats are inconsistent, so the evaluator must handle the variants.
- APPS tests have weak coverage and can produce false positives. Spot-check passing solutions.
- GRPO gets no signal when every candidate for a problem has the same score. Measure reward variance before scaling up the training set.
- Generating and executing code repeatedly is expensive. Measure runtime on a small subset before expanding.
