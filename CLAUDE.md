# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

No code exists yet. This is a UT "Post-Training of LLMs" course project by Girishkaushik Karthikeyan and Sergio Sánchez Manso; the proposal PDF lives one directory up and is not in the repo. `plan.md` holds the dataset survey, design options (with recommendations), the proposed repo layout, and the phased task list, so follow it when you scaffold code. When build, test, or run commands are added, document them here.

Training and large-scale generation run on **TACC** (Slurm, no Docker, so use Apptainer; models and data are pre-downloaded on login nodes; large outputs go on `$SCRATCH`, which is purged). Development and small pilots run locally.

## Research question

Can prompt optimization (GEPA), weight post-training (GRPO + LoRA), or both together improve **first-attempt (pass@1) Python code generation** for **Qwen3-4B** on **APPS**?

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
