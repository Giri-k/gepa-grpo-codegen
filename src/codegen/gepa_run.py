"""Run GEPA on the system prompt of Qwen3-4B-Thinking-2507 (condition C2).

    python -m codegen.gepa_run --out $SCRATCH/gepa-grpo/gepa-<id> --reflect-url http://<node>:8000/v1 \
        [--max-metric-calls 2000] [--deadline <unix time>]

Qwen3-4B runs in this process (vLLM) and is scored in the sandbox; the reflection
model is any OpenAI-compatible server (tacc/gepa.sh starts Qwen3-30B-A3B-Thinking-2507
on a second node). GEPA keeps its state in --out and continues from it when rerun
with the same --out. Exits with status 3 if stopped by --deadline before the budget
was used, so the job script can say how to continue.

Outputs in --out: run_log.txt and gepa_state (GEPA's own), evals.jsonl (every
generation batch), reflections.jsonl (every reflection call), gepa_summary.json and
best_prompt.txt (written when the run ends or stops).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from codegen import prompts
from codegen.data.splits import load_split
from codegen.generate import MODEL, STOPPED, select
from codegen.gepa_adapter import COMPONENT, CodegenAdapter, reflection_template


class DeadlineStopper:
    """Stop before an iteration would run past the deadline.

    GEPA calls the stopper once per iteration, so the gap between calls is one
    iteration (two minibatch passes, a reflection, and a full validation pass when
    the new prompt is accepted)."""

    def __init__(self, deadline: float | None, first_guess: float = 900):
        self.deadline, self.longest, self.last = deadline, first_guess, None
        self.triggered = False

    def __call__(self, state) -> bool:
        now = time.time()
        if self.last is not None:
            self.longest = max(self.longest, now - self.last)
        self.last = now
        if self.deadline is not None and now + 1.2 * self.longest > self.deadline:
            self.triggered = True
            return True
        return False


class ReflectionLM:
    def __init__(self, url: str, model: str, max_tokens: int, log_path: Path):
        from openai import OpenAI

        self.client = OpenAI(base_url=url, api_key="none", timeout=1800)
        self.model, self.max_tokens, self.log_path = model, max_tokens, log_path

    def _call(self, messages: list[dict], seed: int):
        for attempt in range(5):
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=self.max_tokens,
                    temperature=0.6, top_p=0.95, seed=seed, extra_body={"top_k": 20},
                )
            except Exception as err:  # server restarting or overloaded: wait and retry
                print(f"reflection call failed ({err}); retry {attempt + 1}", flush=True)
                time.sleep(30)
        raise RuntimeError("reflection server unreachable")

    def __call__(self, prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        # The server runs without a reasoning parser: the content is the thinking, "</think>",
        # then the answer. A reply that never closes its thinking has no prompt in it; ask again.
        for attempt in range(3):
            start = time.time()
            r = self._call(messages, seed=attempt)
            content = r.choices[0].message.content or ""
            reasoning, _, answer = content.rpartition("</think>")
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps({
                    "time": round(time.time()), "seconds": round(time.time() - start, 1), "attempt": attempt,
                    "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                    "finish": r.choices[0].finish_reason, "prompt": messages[-1]["content"],
                    "reasoning": reasoning, "response": answer,
                }) + "\n")
            if reasoning and "```" in answer:
                return answer.strip()
            print(f"reflection reply has no finished answer (finish={r.choices[0].finish_reason}); asking again",
                  flush=True)
        raise RuntimeError("reflection model gave no usable answer in 3 tries")


def write_summary(result, out: Path, args, stopped: bool) -> None:
    rows = [
        {"idx": i, "parents": result.parents[i], "val_score": result.val_aggregate_scores[i],
         "found_after_metric_calls": result.discovery_eval_counts[i], "prompt": c[COMPONENT]}
        for i, c in enumerate(result.candidates)
    ]
    front = sorted({i for s in result.per_val_instance_best_candidates.values() for i in s})
    summary = {
        "best_idx": result.best_idx, "best_val_score": result.val_aggregate_scores[result.best_idx],
        "seed_val_score": result.val_aggregate_scores[0], "pareto_front": front,
        "total_metric_calls": result.total_metric_calls, "num_full_val_evals": result.num_full_val_evals,
        "stopped_by_deadline": stopped, "settings": vars(args) | {"out": str(args.out)},
        "candidates": rows,
    }
    (out / "gepa_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "best_prompt.txt").write_text(result.best_candidate[COMPONENT].strip() + "\n")
    print(f"candidates: {len(rows)}; best #{result.best_idx} val {summary['best_val_score']:.3f} "
          f"(seed {summary['seed_val_score']:.3f}); Pareto front {front}; metric calls {result.total_metric_calls}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-split", default="gepa_train")
    ap.add_argument("--val-split", default="gepa_val")
    ap.add_argument("--train-limit", type=int, help="random subset (smoke tests)")
    ap.add_argument("--val-limit", type=int, help="random subset (smoke tests)")
    ap.add_argument("--seed-prompt", default="original")
    ap.add_argument("--max-metric-calls", type=int, default=2000)
    ap.add_argument("--minibatch", type=int, default=5)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--reflect-url", required=True)
    ap.add_argument("--reflect-model", default="reflect")
    ap.add_argument("--reflect-max-tokens", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 4))
    ap.add_argument("--deadline", type=float, help="unix time; don't start an iteration expected to end later")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import gepa
    from vllm import LLM

    from codegen.sandbox import Sandbox

    args.out.mkdir(parents=True, exist_ok=True)
    train = select(load_split(args.train_split), args.train_limit, args.seed)
    val = select(load_split(args.val_split), args.val_limit, args.seed)
    print(f"train {len(train)} ({args.train_split}), val {len(val)} ({args.val_split}), "
          f"budget {args.max_metric_calls} metric calls", flush=True)

    llm = LLM(model=MODEL, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              seed=args.seed)
    stopper = DeadlineStopper(args.deadline)
    reflection = ReflectionLM(args.reflect_url, args.reflect_model, args.reflect_max_tokens,
                              args.out / "reflections.jsonl")
    with Sandbox(workers=args.workers) as sandbox:
        adapter = CodegenAdapter(llm, sandbox, args.max_tokens, args.seed, args.out / "evals.jsonl")
        result = gepa.optimize(
            seed_candidate={COMPONENT: prompts.load_system_prompt(args.seed_prompt)},
            trainset=train, valset=val, adapter=adapter, reflection_lm=reflection,
            reflection_prompt_template=reflection_template(args.max_tokens),
            candidate_selection_strategy="pareto", reflection_minibatch_size=args.minibatch,
            use_merge=not args.no_merge, max_metric_calls=args.max_metric_calls, stop_callbacks=stopper,
            run_dir=str(args.out), seed=args.seed, track_best_outputs=False, display_progress_bar=False,
        )
    write_summary(result, args.out, args, stopper.triggered)
    if stopper.triggered and (result.total_metric_calls or 0) < args.max_metric_calls:
        print("stopped before the deadline; rerun with the same --out to continue", flush=True)
        sys.exit(STOPPED)


if __name__ == "__main__":
    main()
