"""Standalone eval: run model on AIME2025 via sglang, report pass@1.

Usage:
    python -m sao.standalone.eval --model-path /path/to/model --data /path/to/aime2025.jsonl

No slime, no ray, no torch_memory_saver.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .sglang_server import start_server, stop_server, env_setup
from .rollout import generate_batch
from .reward import math_reward


def load_aime_data(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
    return samples


def build_prompt(sample: dict) -> str:
    """Build chat prompt from AIME sample."""
    messages = [{"role": "user", "content": sample["input"]}]
    # Apply Qwen chat template
    from transformers import AutoTokenizer
    tok_path = os.environ.get("MODEL_PATH", "")
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_eval(args):
    data = load_aime_data(args.data)
    print(f"Loaded {len(data)} problems from {args.data}")

    # Start sglang server
    print(f"\nStarting sglang server (TP={args.tp}, disable_cuda_graph={args.disable_cuda_graph})...")
    proc = start_server(
        model_path=args.model_path,
        port=args.port,
        tp=args.tp,
        mem_fraction_static=args.mem_fraction,
        max_total_tokens=args.max_total_tokens,
        disable_cuda_graph=args.disable_cuda_graph,
    )

    try:
        tok = None
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        except Exception:
            pass

        # Prepare result output dir
        results_dir = os.path.join(os.path.dirname(args.data), "..", "eval_results", args.tag)
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "results.jsonl")
        wrong_file = os.path.join(results_dir, "wrong.jsonl")
        rf = open(results_file, "w")
        wf = open(wrong_file, "w")

        correct = 0
        total = 0
        t0 = time.time()

        for i, sample in enumerate(data):
            gt = sample.get("label") or sample.get("answer") or ""
            prompt_text = sample["input"]

            # Use chat template if tokenizer available
            if tok:
                messages = [{"role": "user", "content": sample["input"]}]
                prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Generate
            from .rollout import _post
            payload = {
                "model": args.model_path,
                "messages": [{"role": "user", "content": sample["input"]}],
                "n": args.n_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_new_tokens,
            }
            try:
                resp = _post(args.port, "/v1/chat/completions", payload, timeout=3600)
            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")
                continue

            choices = resp.get("choices", [])
            sample_rewards = []
            sample_responses = []
            for ci, choice in enumerate(choices):
                msg = choice.get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                if not content:
                    content = reasoning
                if not content:
                    print(f"  [{i+1}] WARNING: empty response content")
                r = math_reward(content, gt)
                sample_rewards.append(r)
                sample_responses.append({
                    "choice_idx": ci,
                    "content": content,
                    "reasoning_content": reasoning,
                    "reward": r,
                })

            if sample_rewards:
                pass_at_1 = sum(sample_rewards) / len(sample_rewards)
            else:
                pass_at_1 = 0.0

            correct += pass_at_1
            total += 1
            elapsed = time.time() - t0

            # Save full result
            result = {
                "idx": i,
                "input": sample["input"],
                "ground_truth": gt,
                "pass_at_1": pass_at_1,
                "n_samples": len(sample_rewards),
                "rewards": sample_rewards,
                "responses": sample_responses,
            }
            rf.write(json.dumps(result, ensure_ascii=False) + "\n")
            rf.flush()

            # Save wrong problems separately
            if pass_at_1 < 1.0:
                wf.write(json.dumps(result, ensure_ascii=False) + "\n")
                wf.flush()

            status = "OK" if pass_at_1 >= 1.0 else "WRONG" if pass_at_1 == 0.0 else "PARTIAL"
            print(f"  [{i+1}/{len(data)}] {status} reward={pass_at_1:.2f} ({len(sample_rewards)} samples) "
                  f"running_acc={correct/total:.1%} elapsed={elapsed:.0f}s")

        rf.close()
        wf.close()

        acc = correct / max(total, 1)
        elapsed = time.time() - t0
        n_wrong = sum(1 for line in open(wrong_file))
        print(f"\n{'='*60}")
        print(f"AIME2025 pass@1: {acc:.1%} ({total} problems, {args.n_samples} samples each)")
        print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Wrong/partial problems: {n_wrong}")
        print(f"Results: {results_file}")
        print(f"Wrong:   {wrong_file}")
        print(f"{'='*60}")
        return acc

    finally:
        stop_server(proc)


def main():
    parser = argparse.ArgumentParser(description="Standalone AIME eval via sglang")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True, help="Path to aime2025 jsonl")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-total-tokens", type=int, default=40960)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--disable-cuda-graph", action="store_true", default=True)
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    run_eval(args)


if __name__ == "__main__":
    main()
