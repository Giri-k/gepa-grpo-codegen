"""Generate answers for a split with vLLM.

    python -m codegen.generate --split dev --limit 50 --n 8 --out $SCRATCH/.../gen

Writes `<out>/generations.jsonl` (one row per sample) and `<out>/meta.json`
(model, prompt, sampling settings, timing). Each row's `text` is the full
assistant turn: when the chat template opens the reasoning block itself (Qwen3
thinking models end the prompt with "<think>\\n"), that prefix is put back, so
an answer cut off mid-reasoning has an unclosed <think> and the extractor
reports `unfinished_think` instead of taking code drafted while thinking.

With --gpus N (N > 1) the problems are split into N shards, one vLLM process
per GPU (a 4B model fits on one A100, so data parallelism beats tensor
parallelism), and the shard outputs are merged into the same two files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from codegen import prompts
from codegen.data.splits import load_split

MODEL = "Qwen/Qwen3-4B-Thinking-2507"
# Qwen's recommended settings for the thinking model (generation_config.json).
TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20


def select(problems: list, limit: int | None, seed: int) -> list:
    """A seeded random subset, returned in id order (the splits are stratified,
    so the first `limit` by id would over-represent one source)."""
    if limit is None or limit >= len(problems):
        return problems
    keep = set(random.Random(seed).sample(range(len(problems)), limit))
    return [p for i, p in enumerate(problems) if i in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, help="random subset of this many problems")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=8, help="samples per problem")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora", help="LoRA adapter directory (GRPO model)")
    ap.add_argument("--system-prompt", default="original", help="name of a prompts/*.txt file")
    ap.add_argument("--max-tokens", type=int, default=8192, help="output budget, reasoning included")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpus", type=int, default=1, help="one vLLM process per GPU, each on a shard of the problems")
    ap.add_argument("--shard", help=argparse.SUPPRESS)  # "i/N", set by the --gpus launcher
    args = ap.parse_args()

    if args.gpus > 1 and not args.shard:
        launch_shards(args)
        return

    from vllm import LLM, SamplingParams  # imported here so --help works without a GPU

    problems = select(load_split(args.split), args.limit, args.subset_seed)
    out_dir, suffix = args.out, ""
    if args.shard:
        i, k = map(int, args.shard.split("/"))
        problems, suffix = problems[i::k], f".shard{i}"
    system_prompt = prompts.load_system_prompt(args.system_prompt)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed, enable_lora=bool(args.lora), max_lora_rank=64,
    )
    tok = llm.get_tokenizer()
    chats = [
        tok.apply_chat_template(prompts.messages(p, system_prompt), tokenize=False, add_generation_prompt=True)
        for p in problems
    ]
    prompt_tokens = [len(tok(c).input_ids) for c in chats]
    too_long = [p.uid for p, n in zip(problems, prompt_tokens) if n + args.max_tokens > args.max_model_len]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts exceed --max-model-len with this --max-tokens: {too_long[:5]}")

    sampling = SamplingParams(
        n=args.n, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=args.max_tokens, seed=args.seed,
    )
    lora = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("grpo", 1, args.lora)

    start = time.time()
    outputs = llm.generate(chats, sampling, lora_request=lora)
    seconds = time.time() - start

    rows, completion_tokens = 0, 0
    with open(out_dir / f"generations{suffix}.jsonl", "w") as f:
        for problem, chat, out in zip(problems, chats, outputs):
            prefix = "<think>\n" if chat.endswith("<think>\n") else ""
            for i, c in enumerate(out.outputs):
                completion_tokens += len(c.token_ids)
                f.write(json.dumps({
                    "uid": problem.uid, "sample": i, "text": prefix + c.text,
                    "finish_reason": c.finish_reason, "completion_tokens": len(c.token_ids),
                }) + "\n")
                rows += 1

    meta = {
        "split": args.split, "limit": args.limit, "subset_seed": args.subset_seed,
        "uids": [p.uid for p in problems], "n": args.n, "model": args.model, "lora": args.lora,
        "system_prompt_name": args.system_prompt, "system_prompt": system_prompt,
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "max_tokens": args.max_tokens,
                     "seed": args.seed},
        "max_model_len": args.max_model_len,
        "generations": rows, "prompt_tokens_max": max(prompt_tokens), "completion_tokens": completion_tokens,
        "generate_seconds": round(seconds, 1), "tokens_per_second": round(completion_tokens / max(seconds, 0.1), 1),
    }
    (out_dir / f"meta{suffix}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("uids", "system_prompt")}, indent=2))


def launch_shards(args) -> None:
    """Run one shard per GPU in parallel, then merge them in problem order."""
    args.out.mkdir(parents=True, exist_ok=True)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.gpus)))).split(",")
    if len(gpus) < args.gpus:
        raise SystemExit(f"--gpus {args.gpus} but CUDA_VISIBLE_DEVICES={','.join(gpus)}")
    procs = []
    for i in range(args.gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i]}
        log = open(args.out / f"shard{i}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "codegen.generate", *sys.argv[1:], "--shard", f"{i}/{args.gpus}"],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        ))
    failed = [i for i, p in enumerate(procs) if p.wait() != 0]
    if failed:
        raise SystemExit(f"shards {failed} failed; see {args.out}/shard<i>.log")

    metas = [json.loads((args.out / f"meta.shard{i}.json").read_text()) for i in range(args.gpus)]
    order = select(load_split(args.split), args.limit, args.subset_seed)
    if sorted(u for m in metas for u in m["uids"]) != sorted(p.uid for p in order):
        raise SystemExit("shards do not cover the selected problems")
    rows = {}
    for i in range(args.gpus):
        for line in open(args.out / f"generations.shard{i}.jsonl"):
            rows.setdefault(json.loads(line)["uid"], []).append(line)
    with open(args.out / "generations.jsonl", "w") as f:
        for p in order:
            f.writelines(rows[p.uid])

    meta = {k: v for k, v in metas[0].items() if k not in ("uids", "generations", "completion_tokens", "prompt_tokens_max",
                                                           "generate_seconds", "tokens_per_second")}
    completion_tokens = sum(m["completion_tokens"] for m in metas)
    seconds = max(m["generate_seconds"] for m in metas)  # shards run in parallel
    meta.update({
        "uids": [p.uid for p in order], "gpus": args.gpus, "generations": sum(m["generations"] for m in metas),
        "prompt_tokens_max": max(m["prompt_tokens_max"] for m in metas), "completion_tokens": completion_tokens,
        "generate_seconds": seconds, "tokens_per_second": round(completion_tokens / max(seconds, 0.1), 1),
    })
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    for i in range(args.gpus):
        (args.out / f"generations.shard{i}.jsonl").unlink()
        (args.out / f"meta.shard{i}.json").unlink()
    print(json.dumps({k: v for k, v in meta.items() if k not in ("uids", "system_prompt")}, indent=2))


if __name__ == "__main__":
    main()
