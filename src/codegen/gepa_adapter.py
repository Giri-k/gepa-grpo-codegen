"""GEPA adapter: one candidate is one system prompt for Qwen3-4B-Thinking-2507.

    candidate {"system_prompt": str}
      -> vLLM generation (same sampling and 8192-token budget as the evaluations)
      -> judge.evaluate_many in the sandbox
      -> score 1.0 if every test passes, else 0.0 (pass@1)

The reflection model never sees a whole trace (6-8k tokens of reasoning each).
Each example shows the problem, the start and the end of the reasoning, the final
answer if there is one, and the judge's feedback.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from gepa import EvaluationBatch

from codegen import prompts
from codegen.data.apps import Problem
from codegen.judge import evaluate_many

COMPONENT = "system_prompt"
TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20  # as in codegen.generate

PROBLEM_CHARS = 2500
HEAD_TOKENS, TAIL_TOKENS, ANSWER_TOKENS = 1000, 1500, 1500

REFLECTION_TEMPLATE = """You are improving the system prompt of a small reasoning model (Qwen3-4B-Thinking) that solves competitive programming problems in Python.

How the model is used:
- It receives the system prompt below, then a user message holding the problem statement and a fixed instruction on the answer format (a complete Python 3 program in a single ```python block, or a completed starter function). Only the system prompt can change.
- It thinks inside <think>...</think> before answering. Thinking and answer share a hard budget of {max_tokens} tokens. If the budget runs out while the model is still thinking, it gives no code and scores 0.
- An answer counts only if it passes every hidden test. Tests also check speed and memory.

The current system prompt:
```
<curr_param>
```

Below are problems the model attempted with this system prompt. Each shows the problem, an excerpt of the model's reasoning (start and end; the middle is omitted), its final answer if it gave one, and the test results:
```
<side_info>
```

Your task: write an improved system prompt.

- Study what went wrong and right: running out of budget, slow or wrong algorithms, misread input formats, edge cases, output formatting, and how the model spends its reasoning.
- Give guidance that will help on new, unseen problems of the same kind. Do not include facts, algorithms or solutions specific to these particular problems, and do not repeat the answer-format instruction the user message already gives.
- Keep the prompt focused; a long prompt is not better than a clear one.

Put the new system prompt, and nothing else, inside a single ``` block."""


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


class CodegenAdapter:
    propose_new_texts = None  # GEPA's default proposer, with our reflection template

    def __init__(self, llm, sandbox, max_tokens: int = 8192, seed: int = 0, log_path: Path | None = None):
        from vllm import SamplingParams

        self.llm, self.sandbox, self.max_tokens = llm, sandbox, max_tokens
        self.tok = llm.get_tokenizer()
        self.sampling = SamplingParams(
            n=1, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=max_tokens, seed=seed,
        )
        self.log_path = log_path

    # GEPA calls this for minibatches (capture_traces=True) and for full validation passes.
    def evaluate(self, batch: list[Problem], candidate: dict[str, str], capture_traces: bool = False):
        system_prompt = candidate[COMPONENT]
        chats = [
            self.tok.apply_chat_template(prompts.messages(p, system_prompt), tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        start = time.time()
        outputs = self.llm.generate(chats, self.sampling, use_tqdm=False)
        gen_seconds = time.time() - start
        texts, tokens, finish = [], [], []
        for chat, out in zip(chats, outputs):
            o = out.outputs[0]
            texts.append(("<think>\n" if chat.endswith("<think>\n") else "") + o.text)
            tokens.append(len(o.token_ids))
            finish.append(o.finish_reason)
        evals = evaluate_many(list(zip(batch, texts)), self.sandbox)
        scores = [1.0 if e.passed_all else 0.0 for e in evals]
        results = [
            {"uid": p.uid, "status": e.status, "passed": e.passed, "n_tests": e.n_tests, "tokens": n, "finish": f}
            for p, e, n, f in zip(batch, evals, tokens, finish)
        ]
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps({
                    "time": round(time.time()), "prompt": _key(system_prompt), "traces": capture_traces,
                    "n": len(batch), "pass": sum(scores), "gen_seconds": round(gen_seconds, 1),
                    "tokens": sum(tokens), "results": results,
                }) + "\n")
        trajectories = None
        if capture_traces:
            trajectories = [
                {"problem": p, "text": t, "eval": e, "tokens": n, "finish": f}
                for p, t, e, n, f in zip(batch, texts, evals, tokens, finish)
            ]
        return EvaluationBatch(outputs=results, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [self._record(t) for t in eval_batch.trajectories]
        return {c: records for c in components_to_update}

    def _clip(self, text: str, head: int, tail: int) -> str:
        ids = self.tok(text, add_special_tokens=False).input_ids
        if len(ids) <= head + tail:
            return text
        return (self.tok.decode(ids[:head]) + f"\n\n[... {len(ids) - head - tail} tokens omitted ...]\n\n"
                + self.tok.decode(ids[-tail:]))

    def _record(self, t: dict) -> dict:
        problem, text, e = t["problem"], t["text"], t["eval"]
        statement = prompts.user_message(problem)
        if len(statement) > PROBLEM_CHARS:
            statement = statement[:PROBLEM_CHARS] + "\n[... problem statement truncated ...]"
        body = text.removeprefix("<think>\n")
        if "</think>" in body:
            reasoning, answer = body.split("</think>", 1)
        else:
            reasoning, answer = body, None
        output = "<think>\n" + self._clip(reasoning.strip(), HEAD_TOKENS, TAIL_TOKENS) + "\n"
        output += ("</think>\n\n" + self._clip(answer.strip(), ANSWER_TOKENS, 0)) if answer is not None else \
            "[no </think>: the token budget ran out during reasoning]"

        used = f"Used {t['tokens']} of {self.max_tokens} tokens."
        if e.passed_all:
            feedback = f"PASSED all {e.n_tests} tests. {used}"
        elif e.status == "unfinished_think":
            feedback = (f"FAILED: ran out of the {self.max_tokens}-token budget while still thinking, so no code "
                        f"was given. {used}")
        else:
            feedback = f"FAILED. {used}\n{e.feedback}"
        return {"Problem": statement, "Model output (excerpt)": output, "Test results": feedback}


def reflection_template(max_tokens: int) -> str:
    return REFLECTION_TEMPLATE.replace("{max_tokens}", str(max_tokens))
