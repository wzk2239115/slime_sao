"""Async rollout worker: continuously generates trajectories and writes to queue.

Uses sglang /generate API for exact token IDs + log-probs alignment.
Supports concurrent generation (multiple trajectories in parallel).
Supports TIR (Tool-Integrated Reasoning) with Python code execution.

Paper §3.2: "a sample is immediately fed into training upon generation"
Paper §3.1: "we directly use π_rollout log-probabilities"

Usage:
    python -m sao.standalone.rollout_worker \
        --sglang-host 127.0.0.1 --sglang-port 30000 \
        --data /path/to/train.jsonl \
        --queue-dir /shared/queue \
        --checkpoint-dir /shared/checkpoints \
        --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import threading
import time

from .rollout import _post
from .reward import math_reward


def load_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("step_")])
    if not ckpts:
        return None
    return os.path.join(ckpt_dir, ckpts[-1])


def generate_via_sglang(
    port: int,
    prompt_ids: list[int],
    host: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout: int = 3600,
) -> dict | None:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": len(prompt_ids),
        "return_text_in_logprobs": False,
    }
    try:
        resp = _post(port, "/generate", payload, host=host, timeout=timeout)
    except Exception:
        try:
            resp = _post(port, "/v1/chat/completions", {
                "model": "default",
                "messages": [{"role": "user", "content": ""}],
                "n": 1,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
                "logprobs": True,
                "top_logprobs": 1,
            }, host=host, timeout=timeout)
            choice = resp.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content") or \
                      choice.get("message", {}).get("reasoning_content") or ""
            lp_data = choice.get("logprobs", {})
            tokens_text = []
            logprobs = []
            if lp_data and lp_data.get("content"):
                for lp in lp_data["content"]:
                    tokens_text.append(lp.get("token", ""))
                    logprobs.append(lp.get("logprob", 0.0))
            return {
                "text": content or "".join(tokens_text),
                "output_ids": [],
                "output_logprobs": logprobs,
            }
        except Exception as e:
            return None

    meta = resp.get("meta_info", {})
    logprob_data = meta.get("output_token_logprobs", [])

    if logprob_data and isinstance(logprob_data[0], list):
        output_ids = [entry[1] for entry in logprob_data]
        output_logprobs = [
            entry[0] if entry[0] is not None else 0.0
            for entry in logprob_data
        ]
        text = resp.get("text", "")
        return {
            "output_ids": output_ids,
            "output_logprobs": output_logprobs,
            "text": text,
        }

    output_ids = resp.get("output_ids", [])
    text = resp.get("text", "")

    if not output_ids and "samples" in resp:
        sample = resp["samples"][0] if resp["samples"] else {}
        output_ids = sample.get("output_ids", [])
        text = sample.get("text", "")

    return {
        "output_ids": output_ids,
        "output_logprobs": [],
        "text": text,
    }


def run_rollout_worker(args):
    from transformers import AutoTokenizer

    data = load_data(args.data)
    print(f"Loaded {len(data)} prompts from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.queue_dir, exist_ok=True)
    pending_dir = os.path.join(args.queue_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    concurrency = getattr(args, "concurrency", 1)
    hostname = socket.gethostname()
    print(f"Concurrency: {concurrency} (sglang will batch concurrent requests)")

    state = {"traj_id": 0, "rewards": [], "t0": time.time(), "total": 0}
    lock = threading.Lock()
    worker_id = f"{int(time.time())}_{random.randint(1000,9999)}"
    current_ckpt = get_latest_checkpoint(args.checkpoint_dir)

    def generate_one():
        local_ckpt = current_ckpt
        while True:
            with lock:
                tid = state["traj_id"]
                if tid >= args.max_trajectories:
                    return
                state["traj_id"] += 1

            latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
            if latest_ckpt != local_ckpt and latest_ckpt is not None:
                signal_file = os.path.join(args.checkpoint_dir, f".reload_signal_{hostname}")
                reload_done = os.path.join(args.checkpoint_dir, f".reload_done_{hostname}")
                if os.path.exists(reload_done):
                    os.remove(reload_done)
                with open(signal_file, "w") as f:
                    f.write(str(time.time()))
                for _ in range(600):
                    if os.path.exists(reload_done):
                        break
                    time.sleep(3)
                local_ckpt = latest_ckpt

            sample = random.choice(data)
            gt = sample.get("label") or sample.get("answer") or ""
            prompt_text = sample["input"]

            messages = [{"role": "user", "content": prompt_text}]
            if getattr(args, "enable_tir", False):
                from .tir_rollout import TIR_SYSTEM_PROMPT
                messages = [
                    {"role": "system", "content": TIR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]

            if getattr(args, "enable_tir", False):
                from .tir_rollout import generate_tir_trajectory
                result = generate_tir_trajectory(
                    port=args.sglang_port,
                    prompt_ids=prompt_ids,
                    tokenizer=tokenizer,
                    host=args.sglang_host,
                    max_turns=args.tir_max_turns,
                    max_new_tokens=args.tir_max_tokens_per_turn,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    code_timeout=args.tir_code_timeout,
                    context_limit=args.max_seq_len,
                )
                resp_ids = result["resp_ids"]
                output_logprobs = result["token_logprobs"]
                text = result["text"]
                action_mask = result["action_mask"]
                n_code = result["n_code_exec"]
            else:
                result = generate_via_sglang(
                    port=args.sglang_port,
                    prompt_ids=prompt_ids,
                    host=args.sglang_host,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                if result is None:
                    time.sleep(5)
                    continue
                resp_ids = result.get("output_ids", [])
                output_logprobs = result.get("output_logprobs", [])
                text = result.get("text", "")
                action_mask = None
                n_code = 0
                if not resp_ids:
                    resp_text = text or ""
                    resp_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]
                    output_logprobs = [0.0] * len(resp_ids)

            reward = math_reward(text, gt)

            total_len = len(prompt_ids) + len(resp_ids)
            if total_len > args.max_seq_len:
                excess = total_len - args.max_seq_len
                if excess < len(prompt_ids):
                    prompt_ids = prompt_ids[excess:]
                else:
                    resp_ids = resp_ids[:args.max_seq_len - len(prompt_ids)]
                    output_logprobs = output_logprobs[:len(resp_ids)]

            traj = {
                "id": tid,
                "prompt_ids": prompt_ids,
                "resp_ids": resp_ids,
                "logprobs": output_logprobs,
                "response_text": text,
                "ground_truth": gt,
                "reward": reward,
                "timestamp": time.time(),
                "resp_len": len(resp_ids),
                "prompt_len": len(prompt_ids),
                "action_mask": action_mask,
                "n_code_exec": n_code,
            }
            traj_file = os.path.join(pending_dir, f"traj_{worker_id}_{tid:08d}.json")
            tmp_file = traj_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(traj, f)
            os.rename(tmp_file, traj_file)

            with lock:
                state["rewards"].append(reward)
                if len(state["rewards"]) > 100:
                    state["rewards"].pop(0)
                state["total"] += 1
                total = state["total"]
                avg_r = sum(state["rewards"]) / len(state["rewards"])
                elapsed = time.time() - state["t0"]

            if total % 10 == 0:
                rate = total / max(elapsed, 1) * 60
                print(f"  [{hostname}:{tid}] total={total} avg100={avg_r:.2f} "
                      f"rate={rate:.1f}/min elapsed={elapsed/60:.1f}min")

    threads = [threading.Thread(target=generate_one, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - state["t0"]
    avg_r = sum(state["rewards"]) / max(len(state["rewards"]), 1)
    print(f"\n[worker:{hostname}] Done. {state['total']} trajectories "
          f"in {elapsed/60:.1f} min, avg reward={avg_r:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Async rollout worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sglang-host", default="127.0.0.1")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-seq-len", type=int, default=131072)
    parser.add_argument("--max-trajectories", type=int, default=100000)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent trajectory generation threads")
    parser.add_argument("--enable-tir", action="store_true",
                        help="Enable Tool-Integrated Reasoning (Python code execution)")
    parser.add_argument("--tir-max-turns", type=int, default=20,
                        help="Max code execution turns for TIR")
    parser.add_argument("--tir-max-tokens-per-turn", type=int, default=32768)
    parser.add_argument("--tir-code-timeout", type=int, default=10)
    args = parser.parse_args()
    run_rollout_worker(args)


if __name__ == "__main__":
    main()
