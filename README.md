# gepa-grpo-codegen

Can prompt optimization (GEPA) and weight post-training (GRPO + LoRA) improve first-attempt Python code generation together?

Course project for *Post-Training of Large Language Models* (UT Austin, Fall 2026). Target model: Qwen3-4B. Data: APPS. Training runs on TACC.

The four conditions:

| | Original prompt | GEPA prompt |
|---|---|---|
| **Base Qwen3-4B** | C1 | C2 |
| **GRPO (LoRA) Qwen3-4B** | C3 | C4 |

Primary metric: pass@1 on a fixed, prespecified APPS test sample.

## Results

Model: Qwen3-4B-Thinking-2507. 500 APPS-test problems × 4 samples, strict 8192-token budget. Brackets are 95% bootstrap intervals.

| | Original prompt | GEPA prompt |
|---|---|---|
| **Base** | C1: 0.358 [0.320, 0.397] | C2: 0.468 [0.429, 0.508] |
| **GRPO** | C3: 0.558 [0.518, 0.598] | C4: 0.577 [0.538, 0.616] |

Both methods help, but their gains overlap. The interaction (C4 − C3) − (C2 − C1) is −0.092 [−0.118, −0.066], mostly because both cut the reasoning that would otherwise run past the token budget. Details, the compute table and caveats are in [results/RESULTS.md](results/RESULTS.md).

See [plan.md](plan.md) for datasets, design options, and the task list.
