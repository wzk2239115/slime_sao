"""Async rollout worker: continuously generates trajectories and writes to queue.

Runs on inference machine. Each trajectory is generated with n=1 (single rollout),
rewarded, and immediately queued for training.

Paper §3.2: "a sample is immediately fed into training upon generation"

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
    """Find latest checkpoint directory."""
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("step_")])
    if not ckpts:
        return None
    return os.path.join(ckpt_dir, ckpts[-1])


def run_rollout_worker(args):
    from transformers import AutoTokenizer

    data = load_data(args.data)
    print(f"Loaded {len(data)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    os.makedirs(args.queue_dir, exist_ok=True)
    pending_dir = os.path.join(args.queue_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    traj_id = 0
    current_ckpt = None
    rewards_recent = []
    t0 = time.time()

    while traj_id < args.max_trajectories:
        # Check for new checkpoint (model updated by trainer)
        latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
        if latest_ckpt != current_ckpt:
            if latest_ckpt is not None:
                print(f"\n[worker] New checkpoint detected: {latest_ckpt}")
                print(f"[worker] Writing reload signal for sglang daemon...")
                signal_file = os.path.join(args.checkpoint_dir, ".reload_signal")
                with open(signal_file, "w") as f:
                    f.write(str(time.time()))
                # Wait for sglang to reload
                print(f"[worker] Waiting for sglang to reload (up to 15 min)...")
                reload_done = os.path.join(args.checkpoint_dir, ".reload_done")
                if os.path.exists(reload_done):
                    os.remove(reload_done)
                for _ in range(300):  # 300 * 3s = 15 min
                    if os.path.exists(reload_done):
                        current_ckpt = latest_ckpt
                        print(f"[worker] sglang reloaded.")
                        break
                    time.sleep(3)
                else:
                    print(f"[worker] WARNING: sglang reload timeout, continuing with old model")
                    current_ckpt = latest_ckpt

        # Sample a prompt
        sample = random.choice(data)
        gt = sample.get("label") or sample.get("answer") or ""

        # Generate via sglang (n=1, single rollout)
        payload = {
            "model": args.model_path,
            "messages": [{"role": "user", "content": sample["input"]}],
            "n": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_new_tokens,
            "logprobs": True,
            "top_logprobs": 1,
        }

        try:
            resp = _post(args.sglang_port, "/v1/chat/completions", payload,
                        host=args.sglang_host, timeout=3600)
        except Exception as e:
            print(f"  [worker] Generation error: {e}")
            time.sleep(5)
            continue

        choices = resp.get("choices", [])
        if not choices:
            print(f"  [worker] No choices in response")
            continue

        choice = choices[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not content:
            content = reasoning

        reward = math_reward(content, gt)

        # Extract token-level log-probs
        token_ids = []
        token_logprobs = []
        lp_data = choice.get("logprobs")
        if lp_data and lp_data.get("content"):
            for lp in lp_data["content"]:
                token_ids.append(lp.get("token", 0))
                token_logprobs.append(lp.get("logprob", 0.0))

        # Tokenize for training
        messages = [{"role": "user", "content": sample["input"]}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        resp_ids = tokenizer(content, add_special_tokens=False)["input_ids"] if content else []

        # Align logprobs with response tokens
        if len(token_logprobs) != len(resp_ids):
            token_logprobs = [0.0] * len(resp_ids)
            token_ids = resp_ids[:]

        # Write trajectory to queue
        traj = {
            "id": traj_id,
            "prompt_text": full_prompt,
            "prompt_ids": prompt_ids,
            "response_text": content,
            "resp_ids": resp_ids,
            "logprobs": token_logprobs,
            "ground_truth": gt,
            "reward": reward,
            "timestamp": time.time(),
        }
        traj_file = os.path.join(pending_dir, f"traj_{traj_id:08d}.json")
        with open(traj_file, "w") as f:
            json.dump(traj, f)

        rewards_recent.append(reward)
        if len(rewards_recent) > 100:
            rewards_recent.pop(0)

        avg_r = sum(rewards_recent) / len(rewards_recent)
        elapsed = time.time() - t0
        rate = (traj_id + 1) / max(elapsed, 1) * 60

        if traj_id % 10 == 0:
            print(f"  [{traj_id}] reward={reward} avg100={avg_r:.2f} "
                  f"rate={rate:.1f}/min elapsed={elapsed/60:.1f}min")

        traj_id += 1

    print(f"\n[worker] Done. Generated {traj_id} trajectories in {(time.time()-t0)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Async rollout worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sglang-host", default="127.0.0.1")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--queue-dir", required=True, help="Shared queue directory")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory to watch")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-trajectories", type=int, default=100000)
    args = parser.parse_args()

    run_rollout_worker(args)


if __name__ == "__main__":
    main()
