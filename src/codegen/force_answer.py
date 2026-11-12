"""Budget forcing: give samples that ran out of tokens while reasoning a final answer.

    python -m codegen.force_answer --gen $WORK/gepa-grpo-runs/c1-<id> --out $SCRATCH/.../c1-<id>-forced

A sample cut off inside <think> scores 0 under the normal rules (the extractor
reports `unfinished_think`, since code drafted while thinking is not the answer).
This script continues each such sample the way Qwen recommends for budgeting
thinking models: append an early-stop sentence and close the reasoning block, then
let the model write its answer with up to --answer-tokens more tokens. Samples
that finished reasoning are copied unchanged, so the result is a complete run
that `codegen.score` scores like any other.

The forced run uses up to max_tokens + --answer-tokens tokens per sample, so it
is reported next to the original run, not in place of it. Each forced row keeps
the original text, followed by EARLY_STOP and the new answer, and records
`forced: true`, `think_tokens` (the original completion) and `answer_tokens`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from codegen.data.splits import load_split
from codegen.extract import extract_code
from codegen.generate import TEMPERATURE, TOP_K, TOP_P
from codegen import prompts

# Qwen3's suggested text for stopping reasoning at a budget (Qwen3 model card,
# "thinking budget"), followed by the end of the reasoning block.
EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the "
              "thinking directly now.\n</think>\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True, help="a finished codegen.generate run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--answer-tokens", type=int, default=2048, help="budget for the forced answer")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.gen / "meta.json").read_text())
    rows = [json.loads(line) for line in open(args.gen / "generations.jsonl")]
    todo = [i for i, r in enumerate(rows) if extract_code(r["text"]).method == "unfinished_think"]
    print(f"{len(todo)}/{len(rows)} samples stopped while reasoning; forcing an answer", flush=True)

    problems = {p.uid: p for p in load_split(meta["split"])}
    from vllm import LLM, SamplingParams  # imported here so --help works without a GPU

    max_model_len = meta["max_model_len"] + args.answer_tokens + 64
    llm = LLM(
        model=meta["model"], max_model_len=max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed, enable_lora=bool(meta["lora"]), max_lora_rank=64, enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()
    inputs, sampling = [], []
    for i in todo:
        r = rows[i]
        chat = tok.apply_chat_template(prompts.messages(problems[r["uid"]], meta["system_prompt"]),
                                       tokenize=False, add_generation_prompt=True)
        # The row's text starts with the "<think>\n" that ends the chat template.
        if not (chat.endswith("<think>\n") and r["text"].startswith("<think>\n")):
            raise SystemExit(f"{r['uid']}: expected a prompt ending in <think> and a text starting with it")
        inputs.append(chat + r["text"][len("<think>\n"):] + EARLY_STOP)
        # One seed per sample, so a rerun reproduces it.
        sampling.append(SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                                       max_tokens=args.answer_tokens, seed=args.seed * 100_000 + i))
    lora = None
    if meta["lora"]:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("grpo", 1, meta["lora"])

    start = time.time()
    outputs = llm.generate(inputs, sampling, lora_request=lora)
    seconds = time.time() - start

    answer_tokens = 0
    for i, out in zip(todo, outputs):
        o = out.outputs[0]
        r = rows[i]
        answer_tokens += len(o.token_ids)
        rows[i] = {
            **r, "text": r["text"] + EARLY_STOP + o.text, "finish_reason": o.finish_reason,
            "completion_tokens": r["completion_tokens"] + len(o.token_ids),
            "forced": True, "think_tokens": r["completion_tokens"], "answer_tokens": len(o.token_ids),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "generations.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    meta.update({
        "forced_from": str(args.gen), "forced_samples": len(todo), "forced_answer_tokens": args.answer_tokens,
        "forced_early_stop": EARLY_STOP, "forced_seed": args.seed, "forced_seconds": round(seconds, 1),
        "forced_tokens": answer_tokens, "forced_max_model_len": max_model_len,
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
    })
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(args.gen / "meta.json", args.out / "meta.original.json")
    print(f"forced {len(todo)} answers, {answer_tokens} tokens in {seconds:.0f}s "
          f"({answer_tokens / max(seconds, 0.1):.0f} tok/s), "
          f"{sum(rows[i]['finish_reason'] == 'length' for i in todo)} hit --answer-tokens", flush=True)


if __name__ == "__main__":
    main()
