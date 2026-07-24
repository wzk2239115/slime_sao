"""Async rollout worker: continuously generates trajectories and writes to queue.

Uses sglang /generate API for exact token IDs + log-probs alignment.
This is CRITICAL for DIS: ratio = exp(log_pi_theta - log_pi_rollout)
If token IDs don't match, the ratio is meaningless.

Paper §3.2: "a sample is immediately fed into training upon generation"
Paper §3.1: "we directly use π_rollout log-probabilities"

Usage:
    python -m sao.standalone.rollout_worker \
        --sglang-host 127.0.0.1 --sglang-port 30000 \
        --data /path/to/train.jsonl \
        --queue-dir /shared/queue \
        --checkpoint-dir /shared/checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
    """Generate via sglang /generate API with return_logprob=True.

    Returns dict with keys:
      - output_ids: list[int]     generated token IDs
      - output_logprobs: list[float]  log π_rollout for each output token
      - text: str                 decoded text
    """
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
            print(f"  [worker] Generation failed: {e}")
            return None

    # Primary: extract from meta_info.output_token_logprobs
    # Format: [[logprob, token_id, top_logprobs], ...]
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

    # Fallback: top-level output_ids (no logprobs)
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

    traj_id = 0
    current_ckpt = None
    rewards_recent = []
    t0 = time.time()
    worker_id = f"{int(time.time())}_{random.randint(1000,9999)}"

    while traj_id < args.max_trajectories:
        latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
        if latest_ckpt != current_ckpt:
            if latest_ckpt is not None:
                print(f"\n[worker] New checkpoint detected: {latest_ckpt}")
                print(f"[worker] Writing reload signal for sglang daemon...")
                signal_file = os.path.join(args.checkpoint_dir, ".reload_signal")
                reload_done = os.path.join(args.checkpoint_dir, ".reload_done")
                if os.path.exists(reload_done):
                    os.remove(reload_done)
                with open(signal_file, "w") as f:
                    f.write(str(time.time()))
                print(f"[worker] Waiting for sglang to reload (up to 30 min)...")
                for _ in range(600):
                    if os.path.exists(reload_done):
                        current_ckpt = latest_ckpt
                        print(f"[worker] sglang reloaded ✓")
                        break
                    time.sleep(3)
                else:
                    print(f"[worker] WARNING: reload timeout, continuing with old model")
                    current_ckpt = latest_ckpt

        sample = random.choice(data)
        gt = sample.get("label") or sample.get("answer") or ""
        prompt_text = sample["input"]

        messages = [{"role": "user", "content": prompt_text}]
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]

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

        output_ids = result.get("output_ids", [])
        output_logprobs = result.get("output_logprobs", [])
        text = result.get("text", "")

        if not output_ids:
            resp_text = text or ""
            resp_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]
            output_logprobs = [0.0] * len(resp_ids)
        else:
            resp_ids = output_ids
            if not text:
                text = tokenizer.decode(resp_ids, skip_special_tokens=False)
            if len(output_logprobs) != len(resp_ids):
                print(f"  [worker] WARN: logprob len {len(output_logprobs)} != "
                      f"token len {len(resp_ids)}, padding with 0.0")
                if len(output_logprobs) < len(resp_ids):
                    output_logprobs.extend([0.0] * (len(resp_ids) - len(output_logprobs)))
                else:
                    output_logprobs = output_logprobs[:len(resp_ids)]

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
            "id": traj_id,
            "prompt_ids": prompt_ids,
            "resp_ids": resp_ids,
            "logprobs": output_logprobs,
            "response_text": text,
            "ground_truth": gt,
            "reward": reward,
            "timestamp": time.time(),
            "resp_len": len(resp_ids),
            "prompt_len": len(prompt_ids),
        }
        traj_file = os.path.join(pending_dir, f"traj_{worker_id}_{traj_id:08d}.json")
        tmp_file = traj_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(traj, f)
        os.rename(tmp_file, traj_file)

        rewards_recent.append(reward)
        if len(rewards_recent) > 100:
            rewards_recent.pop(0)

        avg_r = sum(rewards_recent) / len(rewards_recent)
        elapsed = time.time() - t0
        rate = (traj_id + 1) / max(elapsed, 1) * 60

        if traj_id % 10 == 0 or reward > 0:
            print(f"  [{traj_id}] r={reward} avg100={avg_r:.2f} "
                  f"len={len(resp_ids)} rate={rate:.1f}/min "
                  f"elapsed={elapsed/60:.1f}min")

        traj_id += 1

    print(f"\n[worker] Done. Generated {traj_id} trajectories "
          f"in {(time.time()-t0)/60:.1f} min, avg reward={avg_r:.3f}")


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
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--max-trajectories", type=int, default=100000)
    args = parser.parse_args()
    run_rollout_worker(args)


if __name__ == "__main__":
    main()
