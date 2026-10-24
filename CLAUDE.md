# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This is a UT "Post-Training of LLMs" course project by Girishkaushik Karthikeyan and Sergio Sánchez Manso; the proposal PDF lives one directory up and is not in the repo. `plan.md` holds the dataset survey, design options (with recommendations), the proposed repo layout, and the phased task list, so follow it when you scaffold code. Only the data pipeline (Phase 1) exists so far.

Training and large-scale generation run on **TACC** (Slurm, no Docker, so use Apptainer; models and data are pre-downloaded on login nodes; large outputs go on `$SCRATCH`, which is purged). Development and small pilots run locally.

## Commands

```bash
uv sync                                        # create .venv (Python >=3.10) with dev deps
uv run pytest                                  # all tests
uv run pytest tests/test_splits.py::test_write_freeze_guard_and_load_roundtrip   # one test
uv run python -m codegen.data.splits --download  # fetch pinned APPS + TACO-verified (~3.3 GB) and build splits
uv run python -m codegen.data.splits           # rebuild splits, ~5 min (refuses to change eval.json without --force)
```

## Data pipeline (`src/codegen/data/`)

**Dataset decision:** eval = APPS-test (as in the proposal). All other splits = **stdio problems with ≥5 tests and a reference solution**, drawn from APPS-train + TACO-verified (EASY/MEDIUM/MEDIUM_HARD). Why:
- APPS-test is ~99% stdio, while APPS-train introductory problems are ~86% call-based.
- Every stdio APPS-train intro/interview problem has exactly **one** test, usually the example in the statement, which a model can pass by echoing it. So in practice almost all training data comes from TACO.
- Dropping copies of APPS-test removes more than half of TACO's multi-test problems, which leaves ~1.4k problems. That's why the fixed splits are small (dev 200, gepa_val 100, gepa_train 100).

- `apps.py` / `taco.py` download raw files from HF at **pinned revisions with SHA-256 checks**. Loading fails on a checksum mismatch, which has already caught a hand-edited raw file once. Raw files live in gitignored `data/raw/`. Both datasets share the APPS record format and `apps.parse_problem`. Stdio tests are normalized to strings. Call-based outputs are kept raw (APPS often wraps the return value in a one-element list, so the evaluator must accept both forms). Ids are `train-0001`, `test-0042`, `taco-01234`.
- `splits.py` writes id lists `data/splits/{eval,dev,gepa_val,gepa_train,grpo_train}.json` plus `manifest.json` and `dropped.json` (committed), and full problems to `data/processed/*.jsonl` (gitignored). The fixed splits are stratified by (source, difficulty, kind), and `grpo_train` gets everything left.
- Leakage rules (`dedup.py`: 5-word-shingle Jaccard ≥ 0.35 after stripping LaTeX, `$` and `<image>`, a cutoff chosen from a measured gap between copies and distinct problems): drop pool problems that near-duplicate **any** APPS-test problem (TACO contains APPS-test verbatim). Each duplicate cluster in the pool keeps one problem, the one with the most tests.
- `eval` uses its own RNG stream, so train-side config changes never alter it. **`eval.json` is frozen once committed.** Use `load_split(name)`, which reads `data/processed/` and checks it against the committed ids.

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
