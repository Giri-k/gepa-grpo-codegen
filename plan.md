# Project Plan: GEPA × GRPO for First-Attempt Code Generation

**Question:** Can prompt optimization (GEPA), weight post-training (GRPO + LoRA), or their combination best improve first-attempt (pass@1) correctness of Qwen3-4B on Python programming problems?

**Compute:** TACC (training + large-scale generation). Laptop for development and small pilots.

---

## 1. Datasets

### 1.1 Candidate datasets

| Dataset | HF id | Size | Test format | Role it could play | Notes / caveats |
|---|---|---|---|---|---|
| **APPS** | `codeparrot/apps` | 10k (5k train / 5k test). Test split: 1k intro / 3k interview / 1k competition | stdin/stdout **and** call-based (`fn_name`) in `input_output` JSON | **Primary** (as proposed): train subsets + final eval | Some train problems have no or broken tests. The HF repo uses a loading script, which recent `datasets` versions no longer run, so load the raw JSONL/parquet instead. Weak test coverage leads to false positives. Likely seen in Qwen3 pretraining. |
| **TACO** | `BAAI/TACO` | ~25.4k train / 1k test | Same schema as APPS, many more tests per problem | Larger or cleaner **training pool** | **Superset of APPS, including APPS test problems.** You must deduplicate it against your APPS eval sample before training on it. |
| **TACO-verified** | `likaixin/TACO-verified` | ~14k | Same as TACO | Cleaner training pool | Problems whose tests were checked against reference solutions. Same dedup caveat. |
| **CodeContests** | `deepmind/code_contests` | ~13.3k train / 117 valid / 165 test | stdin/stdout. Public, private, and generated tests | Training pool, or OOD eval (its 165 test problems) | Its generated tests are much stronger than APPS tests. Problems are hard for a 4B model, so expect low reward variance. |
| **DeepCoder-Preview** | `agentica-org/DeepCoder-Preview-Dataset` | ~24k | Filtered to problems with ≥5 tests | RL training pool (built for GRPO on code) | Drawn from TACO-verified, PrimeIntellect, and LiveCodeBench, so dedup against APPS test. |
| **PrimeIntellect verifiable coding** | `PrimeIntellect/verifiable-coding-problems` (SYNTHETIC-1) | ~144k | stdin/stdout + call-based | Large RL pool | Mixed sources and quality. |
| **KodCode-V1** | `KodCode/KodCode-V1` | ~447k synthetic | pytest unit tests | RL pool for function-style coding | Synthetic, with a different style from APPS. Useful mainly if you also evaluate on HumanEval/MBPP. |
| **open-r1 Codeforces** | `open-r1/codeforces` | ~10k problems | Verified tests + checkers | Training or eval | Some problems need custom checkers. |
| **LiveCodeBench** | `livecodebench/code_generation_lite` | Grows over time, problems are date-stamped | stdin/stdout + LeetCode-style functional | **OOD / contamination-controlled eval** | Use problems released after the Qwen3 release date to rule out contamination. |
| **HumanEval+ / MBPP+** | `evalplus/humanevalplus`, `evalplus/mbppplus` | 164 / 378 | Function-level, strong tests | Cheap OOD sanity eval | Near-saturated for modern models. Different format from APPS. |
| **BigCodeBench** | `bigcode/bigcodebench` | 1,140 | Library-heavy unittest | OOD generalization | Needs many pip packages in the sandbox. |

### 1.2 Recommended data setup

- **Training/opt pool:** APPS-train intro + interview problems that have ≥1 working test. Optionally add **TACO-verified**, deduplicated against the eval sample, if APPS-train is too small after filtering.
- **Disjoint splits from that pool:**
  - `grpo_train`: ~2–4k problems (after reward-variance filtering, see §4.3)
  - `gepa_train` (feedback minibatches): ~150–300 problems
  - `gepa_val` (Pareto selection set): ~100–200 problems
  - `dev`: ~300 problems, for all hyperparameter, checkpoint, and decoding decisions
- **Final eval (frozen, prespecified, touched once per condition):** a stratified APPS-test sample, e.g. 500 problems (e.g. 150 intro / 300 interview / 50 competition). Save the IDs to `data/splits/eval_ids.json` and commit them before running any experiment.
- **Secondary eval (optional, strengthens conclusions):** post-cutoff LiveCodeBench problems + HumanEval+/MBPP+. These show whether gains transfer beyond APPS.
- **Dedup:** hash normalized problem statements (lowercase, collapse whitespace) and drop near-duplicates (e.g. MinHash, Jaccard > 0.8) of any eval problem from every training/opt split.

---

## 2. Design Options (pros / cons)

### 2.1 Qwen3-4B variant and reasoning mode

| Option | Pros | Cons |
|---|---|---|
| **A. Qwen3-4B, `enable_thinking=False`** (recommended) | Short outputs (~300–1500 tokens), so GRPO and GEPA are cheap. Matches the "first-attempt answer" framing. | Lower absolute pass@1 than thinking mode. |
| B. Qwen3-4B, thinking mode on, with a token budget (e.g. 4–8k) | Higher ceiling. RL on reasoning traces is a popular research direction. | 5–10× generation cost, and truncation at the budget produces many zero rewards. Hard to fit in a course budget. |
| C. `Qwen3-4B-Instruct-2507` (non-thinking checkpoint) | Stronger non-thinking baseline. No chat-template flag issues. | Deviates from the proposal's "Qwen3-4B" wording, and it was released later, so contamination profile differs. |

### 2.2 GRPO training framework

| Option | Pros | Cons |
|---|---|---|
| **A. TRL `GRPOTrainer` + vLLM (colocate mode) + PEFT LoRA** (recommended) | Simplest. A custom reward function is just a Python callable. Fits one GPU for 4B + LoRA. Good docs. | Slower than verl at scale. Async rewards need care to avoid blocking on the sandbox. |
| B. verl | Fastest, and used by many code-RL papers (DeepCoder, Skywork). Supports DAPO/Dr.GRPO variants out of the box. | Steeper setup (Ray, configs). LoRA support is less mature. |
| C. Unsloth GRPO | Very memory-efficient, quick to start. | Single-GPU focus. Less control over the vLLM/rollout plumbing. ARM (Vista) support uncertain. |
| D. OpenRLHF | Mature, Ray-based. | Heavier than needed for 4B + LoRA. |

### 2.3 GEPA implementation

| Option | Pros | Cons |
|---|---|---|
| **A. `gepa` package (`gepa.optimize`) with a custom adapter** (recommended) | Direct control over the candidate (just the system prompt), the evaluator, and the textual feedback. No DSPy program abstractions. | You write the adapter (evaluate + make reflective dataset). |
| B. `dspy.GEPA` with a DSPy module | Less glue code. Built-in metric-with-feedback API. | DSPy wraps prompts in its own format (signatures, field markers), so the "original prompt" baseline is no longer a plain chat prompt. Harder to apply the GEPA prompt verbatim to the GRPO model. |

**Reflection LM choice:**

| Option | Pros | Cons |
|---|---|---|
| **Qwen3-32B (or similar open model) served on TACC via vLLM** | Reproducible, no API cost, keeps everything on-cluster. | Uses GPU hours. Weaker reflection than frontier models. |
| Frontier API model (e.g. Claude / GPT) | Best prompt proposals, cheap in tokens (GEPA needs few reflection calls). | External dependency. Compute accounting is not apples-to-apples with GRPO. TACC compute nodes may lack outbound internet, so reflection may need to run from a login node or laptop. |
| Qwen3-4B itself | Self-contained and "fair". | Likely poor reflection quality. |

Whatever you pick, report reflection-LM calls separately in the compute table.

### 2.4 Code execution sandbox (TACC has no Docker)

| Option | Pros | Cons |
|---|---|---|
| **A. One long-lived Apptainer container hosting a worker pool. Each program runs in a subprocess with `setrlimit` (CPU, AS/memory, NPROC, FSIZE), a wall-clock timeout, a fresh tmpdir, and no network** (recommended) | Fast (no per-run container startup). Meets "isolated container + limits". Apptainer is supported on TACC. | Network isolation inside Apptainer (`--net --network none`) may require admin-enabled config, so verify. Otherwise rely on compute-node firewalling + `unshare -n` if user namespaces are allowed. |
| B. New Apptainer instance per program | Strongest isolation. | ~0.3–1 s overhead × (G samples × tests × steps) gets too slow for GRPO. |
| C. Plain subprocess + rlimits, no container | Simplest, fastest. | Weaker isolation than the proposal promises. OK for local dev only. |
| D. Remote sandbox service (e.g. SandboxFusion / E2B) | Battle-tested. | Network dependency from compute nodes. External cost. |

Expected outputs never enter the sandbox. The worker sends stdin in, captures stdout, and compares **outside**.

### 2.5 Reward function for GRPO

| Option | Pros | Cons |
|---|---|---|
| **A. Fraction of tests passed, with a bonus/override to 1.0 only if all pass** (as proposed) | Denser signal than binary, which helps a 4B model. | Can reward partial hacks (e.g. hardcoding sample outputs). |
| B. Binary all-pass | Directly optimizes pass@1. Hard to hack. | Sparse, so many groups have zero variance. |
| C. A + small format penalty (no code block / syntax error → −0.1 or 0) | Teaches extraction-friendly output quickly. | Extra hyperparameter. |

Recommendation: **A, with partial credit scaled down** (e.g. `0.5 * frac` if not all pass, `1.0` if all pass), plus C's format gate. Run B as an ablation only if time allows.

### 2.6 GRPO loss / algorithm variant

| Option | Pros | Cons |
|---|---|---|
| Vanilla GRPO (sequence-mean, std-normalized advantages, KL β>0) | Matches the proposal citation. | Known length bias. KL adds cost. |
| **Dr. GRPO / DAPO-style (token-level loss, no std norm, β=0, clip-higher, dynamic sampling)** (recommended; TRL supports via `loss_type`, `beta=0`, `epsilon_high`) | More stable. Reduces length bias. Dynamic sampling fixes the zero-variance group problem. | Deviates slightly from "GRPO" as cited. Describe it as a GRPO variant in the report. |

### 2.7 How to measure pass@1

| Option | Pros | Cons |
|---|---|---|
| Greedy decoding, 1 sample | Literal "first attempt". Cheapest. | High variance per problem. Greedy can behave oddly for Qwen3 (repetition). |
| **Sample n=5–10 at the recommended temperature (T=0.7, top-p 0.8 for non-thinking), report unbiased pass@1 = mean success rate** (recommended) | Lower variance, better CIs. Standard practice. | n× generation cost (cheap at eval scale). |

Report both greedy and sampled if budget allows.

### 2.8 Statistics

- Paired comparisons on identical problems: **McNemar's test** (greedy) or a **paired bootstrap** over problems (sampled pass@1). Report 95% CIs.
- Main effects and interaction in the 2×2 (prompt × weights), e.g. a logistic mixed model with a random intercept per problem, or simply the bootstrap CI of `(C4 − C3) − (C2 − C1)`.

---

## 3. Repository Layout (proposed)

```
gepa-grpo-codegen/
├── configs/            # YAML: model, generation, grpo, gepa, eval
├── data/splits/        # committed ID lists (eval_ids.json, grpo_train_ids.json, ...)
├── src/codegen/
│   ├── data/           # loaders for APPS/TACO, normalization, dedup, splitting
│   ├── prompts/        # baseline prompt, GEPA output prompts (versioned)
│   ├── extract.py      # pull Python code from model output
│   ├── sandbox/        # executor worker pool, rlimits, Apptainer def file
│   ├── evaluator.py    # SHARED: run tests → reward + structured feedback
│   ├── generate.py     # vLLM batch generation for any (model, prompt) pair
│   ├── grpo/           # TRL trainer script + reward fn wrapper
│   ├── gepa/           # gepa adapter + run script
│   └── analysis/       # metrics, stats, plots, failure taxonomy
├── slurm/              # TACC sbatch scripts
├── tests/              # unit tests for extractor, evaluator, sandbox
└── results/            # small JSON summaries only (big outputs live on $SCRATCH)
```

---

## 4. Step-by-Step Task List

### Phase 0: Environment and TACC setup (week 1)

- [ ] Confirm which TACC system and allocation you have (**Vista**: GH200 96 GB, **aarch64/ARM**; **Lonestar6**: 3× A100 40 GB per node, x86). This determines wheel availability.
- [ ] On Vista, verify `vllm`, `flash-attn`, and `torch` aarch64 builds. Prefer TACC-provided modules or NGC containers via Apptainer.
- [ ] Build an Apptainer image (`.sif`) with Python, torch, vllm, trl, peft, gepa. Store it on `$WORK`. Put HF cache (`HF_HOME`) and outputs on `$SCRATCH` (purged periodically, so sync important results back).
- [ ] Pre-download Qwen3-4B and datasets on a login node (compute nodes may lack internet).
- [ ] Write `slurm/` templates: interactive dev (`idev`), generation job, GRPO job, GEPA job.
- [ ] Local env (`uv`) for laptop dev with a tiny model (e.g. Qwen3-0.6B) for fast iteration.
- [ ] Add Sergio as a collaborator on the GitHub repo.

### Phase 1: Data (week 1–2)

- [ ] Loader for APPS raw files. Parse `input_output` (stdin/stdout vs `fn_name` call-based) and `starter_code`.
- [ ] Validate tests by running the **reference solutions** in the sandbox. Keep only problems where ≥1 reference solution passes all tests (this flags broken tests).
- [ ] Filter to intro + interview for training. Cap the number of tests per problem (e.g. ≤20) for RL speed, and keep all tests for eval.
- [ ] Build and **commit** split ID files: `eval_ids`, `dev_ids`, `gepa_train_ids`, `gepa_val_ids`, `grpo_train_ids`. Fix the random seed.
- [ ] Dedup any extra training data (TACO etc.) against `eval_ids`.
- [ ] Manual audit: read ~30 problems + tests and note format issues (trailing whitespace, multiple valid answers, float tolerance).

### Phase 2: Shared evaluator + sandbox (week 2), *the critical path*

- [ ] `extract.py`: take the last fenced ```python block, fall back to the whole output, strip `<think>` blocks. Unit tests on messy outputs.
- [ ] Sandbox executor: subprocess per test (or per problem with a loop harness), rlimits, timeout (e.g. 4 s/test, 10–30 s/problem), no network, tmpdir. Concurrency via process pool sized to CPU cores.
- [ ] Call-based harness: wrap the solution, call `fn_name` with JSON args, and compare return values.
- [ ] Output comparison: exact after whitespace normalization, plus a float-tolerance fallback (match APPS's official eval semantics, and document deviations).
- [ ] `evaluator.py` returns, per problem: `passed`, `n_pass/n_total`, `reward`, `status ∈ {pass, wrong_answer, runtime_error, timeout, syntax_error, no_code}`, and a short textual feedback (first failing input, expected vs. got truncated, stderr tail). GRPO uses `reward` and GEPA uses `feedback`.
- [ ] Throughput benchmark: problems/sec at N workers. This determines GRPO step time.
- [ ] Unit tests: known-correct reference solutions → 1.0; infinite loop → timeout; `import os; os.system(...)`/network attempt → blocked or harmless; fork bomb → contained.

### Phase 3: Baseline pilot, *milestone from the proposal* (week 2–3)

- [ ] Write the **original prompt** (system + user template with problem + starter code + "return a single Python code block reading stdin / implementing `fn_name`"). Freeze it in `prompts/original.txt`.
- [ ] Freeze generation config: thinking mode, temperature, top-p, max_new_tokens (e.g. 2048), stop sequences, chat template.
- [ ] `generate.py` with vLLM. Run on **50 dev problems**, evaluate, and log rewards.
- [ ] Check: extraction failure rate, timeout rate, reward distribution, wall-clock per problem.
- [ ] Run baseline on full `dev` (n samples) to get **C1-dev**. Measure the per-problem pass rate distribution (needed for §4.3).

### Phase 4: GEPA (week 3–4), can run in parallel with Phase 5

- [ ] Write the GEPA adapter: candidate = `{"system_prompt": str}`. `evaluate()` calls vLLM-served Qwen3-4B + the shared evaluator. `make_reflective_dataset()` emits (problem summary, model output, status, feedback).
- [ ] Serve Qwen3-4B with vLLM (OpenAI-compatible) on a TACC node. Serve or connect the reflection LM.
- [ ] Set a metric-call budget (e.g. 2–5k rollouts) and record it.
- [ ] Run GEPA from the original prompt on `gepa_train` / `gepa_val`. Save the full candidate history and the final prompt to `prompts/gepa_v1.txt`.
- [ ] Evaluate the GEPA prompt on `dev` (C2-dev). Sanity-check that the prompt doesn't leak problem-specific content.

### Phase 5: GRPO (week 3–5)

- [ ] **Reward-variance filter (§4.3):** using the base model's dev-style pass rates on `grpo_train` (sample G=8 each), drop problems with pass rate 0 or 1 (no learning signal). Or rely on dynamic sampling.
- [ ] TRL config to start with: LoRA r=32–64, α=2r, all linear layers; lr ~1e-5–5e-5 (LoRA); G=8 generations/prompt; ~64–128 prompts/step; max_completion_length 2048; temp 1.0; β=0; `loss_type` Dr.GRPO/DAPO; vLLM colocate.
- [ ] Reward fn = shared evaluator (async/process pool so the GPU doesn't idle).
- [ ] Short smoke run (~20 steps) and check reward curves, completion length, KL/entropy, and step time.
- [ ] Full run with checkpoints every N steps. Choose the checkpoint on `dev` **with the original prompt** (GRPO must not see the GEPA prompt).
- [ ] Log GPU-hours and the number of generations.

### Phase 6: Final evaluation, the 2×2 (week 5–6)

- [ ] Freeze: original prompt, GEPA prompt, base weights, chosen LoRA checkpoint, generation config.
- [ ] Run all four conditions on `eval_ids` (+ optional LiveCodeBench / HumanEval+):
  - C1 base + original · C2 base + GEPA · C3 GRPO + original · C4 GRPO + GEPA
- [ ] Compute pass@1 (greedy and/or sampled), test-fraction passed, invalid/timeout rates, by difficulty.
- [ ] Stats: paired bootstrap CIs, McNemar, interaction term.
- [ ] Compute table: generations, GPU-hours, reflection-LM calls per method.

### Phase 7: Analysis and write-up (week 6–7)

- [ ] Failure taxonomy per condition (status × difficulty), plus manual labels on ~50 failures (misread format, wrong algorithm, edge case, TLE).
- [ ] **False-positive audit:** manually inspect ~30 passing solutions per condition for test-gaming (hardcoded samples, special-casing).
- [ ] Qualitative: diff the original vs. the GEPA prompt, and compare example outputs before and after GRPO (e.g. length, structure).
- [ ] Answer the proposal's decision rule explicitly: does GEPA alone match GRPO? Do they stack?
- [ ] Report + slides. Clean the repo, document reproduction commands.

### 4.3 Note on GRPO signal

If a group of G samples all score the same, the advantage is 0 and there's no gradient. For a 4B model on APPS, many interview problems will be all-0 and many intro problems all-1. Mitigations, in order of cost: (1) pre-filter on base-model pass rate ∈ (0, 1); (2) partial-credit reward (§2.5); (3) dynamic sampling (resample until the batch has variance); (4) larger G.

---

## 5. Compute Budget (rough, to be validated in the pilot)

| Item | Estimate |
|---|---|
| Baseline eval, 500 problems × n=8, ~1k tokens each | < 1 GPU-hour on one A100/GH200 with vLLM |
| GEPA, ~3k rollouts + reflection | ~1–3 GPU-hours (4B) + reflection LM |
| GRPO, ~300 steps × 128 prompts × G=8 ≈ 300k generations | ~20–60 GPU-hours, often **bound by sandbox CPU throughput** rather than the GPU |
| Final 2×2 eval | ~2–4 GPU-hours |

Request CPU-heavy nodes or co-locate many CPU workers. Measure sandbox throughput in Phase 2 before committing to the GRPO scale.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Sandbox too slow for GRPO | Long-lived worker pool, cap tests per problem during training, early-exit on first failure for binary signal, async rewards |
| ARM (Vista) wheel issues | Use NGC/TACC containers. Fall back to Lonestar6 (x86 A100) |
| No internet on compute nodes | Pre-download models/data. Run the API-based reflection LM from a login node or laptop, or use a local reflection LM |
| Zero-variance groups | §4.3 |
| Reward hacking (hardcoded outputs) | All-pass bonus, hidden tests (use a subset of tests for training reward if many exist), false-positive audit |
| Contamination (APPS in Qwen3 pretraining, TACO overlap) | Dedup. Add post-cutoff LiveCodeBench as a secondary eval. State it as a limitation |
| `$SCRATCH` purge | Sync checkpoints and results to `$WORK` / GitHub (small files) regularly |
| GEPA overfits `gepa_val` | Pick the final prompt on `dev`, which GEPA never sees |

---

## 7. Open Decisions to Make Before Phase 3

1. TACC system (Vista vs. Lonestar6) and allocation size.
2. Thinking mode on or off (recommend off).
3. Reflection LM for GEPA (local open model vs. API).
4. Eval sample size and difficulty mix, and whether to add LiveCodeBench.
5. Greedy vs. sampled pass@1 as the *primary* metric (prespecify it).
