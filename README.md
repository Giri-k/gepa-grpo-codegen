# gepa-grpo-codegen

Can prompt optimization (GEPA) and weight post-training (GRPO + LoRA) improve first-attempt Python code generation together?

Course project for *Post-Training of Large Language Models* (UT Austin, Fall 2026). Target model: Qwen3-4B. Data: APPS. Training runs on TACC.

The four conditions:

| | Original prompt | GEPA prompt |
|---|---|---|
| **Base Qwen3-4B** | C1 | C2 |
| **GRPO (LoRA) Qwen3-4B** | C3 | C4 |

Primary metric: pass@1 on a fixed, prespecified APPS test sample.

See [plan.md](plan.md) for datasets, design options, and the task list.
