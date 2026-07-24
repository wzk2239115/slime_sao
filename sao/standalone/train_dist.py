"""Distributed GRPO/SAO training: remote sglang + local HF training.

Architecture:
  - Inference machine (ctm-05): runs sglang daemon, generates responses
  - Training machine (ctm-06): runs this script, computes gradients + updates model
  - Shared storage: checkpoints visible to both machines
  - Weight sync: training saves checkpoint → writes .reload_signal → daemon restarts sglang

Usage:
  python -m sao.standalone.train_dist \
      --model-path /path/to/model \
      --sglang-host 11.131.211.65 \
      --sglang-port 30000 \
      --data /path/to/train.jsonl \
      --save-dir /shared/checkpoints \
      --num-steps 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

from .rollout import _post, RolloutSample
from .reward import math_reward
from .grpo_step import (
    compute_grpo_advantages,
    compute_sao_advantages,
    compute_log_probs,
    dis_policy_loss,
    grpo_policy_loss,
)


def train_step(
    model,
    optimizer,
    samples,
    tokenizer,
    device,
    algo="sao",
    clip_low=0.7,
    clip_high=6.0,
    eps_clip=0.2,
    eps_clip_high=0.28,
    running_mean=0.0,
    group_size=8,
    max_seq_len=32768,
    gradient_checkpointing=True,
):
    model.train()
    rewards = [s.reward for s in samples]
    if algo == "sao":
        advantages = compute_sao_advantages(rewards, running_mean)
    else:
        advantages = compute_grpo_advantages(rewards, group_size)

    input_ids_list = []
    response_lens = []
    rollout_log_probs_list = []

    for s in samples:
        prompt_ids = s.prompt_token_ids
        resp_ids = s.response_token_ids
        if not prompt_ids:
            prompt_ids = tokenizer(s.prompt_text, add_special_tokens=False)["input_ids"]
        if not resp_ids:
            resp_ids = tokenizer(s.response_text, add_special_tokens=False)["input_ids"]
        total_ids = prompt_ids + resp_ids
        if len(total_ids) > max_seq_len:
            excess = len(total_ids) - max_seq_len
            prompt_ids = prompt_ids[excess:]
            total_ids = prompt_ids + resp_ids
        input_ids_list.append(torch.tensor(total_ids, dtype=torch.long))
        response_lens.append(len(resp_ids))
        rlp = s.response_logprobs if len(s.response_logprobs) == len(resp_ids) else [0.0] * len(resp_ids)
        rollout_log_probs_list.append(torch.tensor(rlp, dtype=torch.float32))

    train_log_probs = compute_log_probs(model, input_ids_list, response_lens, device, gradient_checkpointing)

    if algo == "sao":
        loss, metrics = dis_policy_loss(train_log_probs, rollout_log_probs_list, advantages, clip_low=clip_low, clip_high=clip_high)
    else:
        loss, metrics = grpo_policy_loss(train_log_probs, rollout_log_probs_list, advantages, eps_clip=eps_clip, eps_clip_high=eps_clip_high)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), metrics


def load_train_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} training prompts from {path}")
    return data


def wait_for_sglang(host: str, port: int, timeout: int = 900):
    """Wait for remote sglang to be ready."""
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(f"http://{host}:{port}/health")
            opener.open(req, timeout=5)
            print(f"[sglang] Ready at {host}:{port} ({time.time()-t0:.0f}s)")
            return True
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"sglang at {host}:{port} not ready within {timeout}s")


def reload_sglang(save_dir: str):
    """Signal the sglang daemon to reload weights."""
    signal_file = os.path.join(save_dir, ".reload_signal")
    with open(signal_file, "w") as f:
        f.write(str(time.time()))
    print(f"  [reload] Signal written, waiting for sglang to restart...")


def generate_via_remote_sglang(
    host: str,
    port: int,
    prompts: list[str],
    ground_truths: list[str],
    tokenizer,
    n: int = 8,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 32768,
    model_name: str = "default",
) -> list[RolloutSample]:
    """Generate responses from remote sglang server."""
    all_samples = []

    for prompt_text, gt in zip(prompts, ground_truths):
        messages = [{"role": "user", "content": prompt_text}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        try:
            resp = _post(port, "/v1/chat/completions", payload, host=host, timeout=3600)
        except Exception as e:
            print(f"  Generation error: {e}")
            continue

        for choice in resp.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            if not content:
                content = reasoning
            resp_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
            reward = math_reward(content, gt)

            lp_data = choice.get("logprobs")
            logprobs = []
            if lp_data and lp_data.get("content"):
                for lp in lp_data["content"]:
                    logprobs.append(lp.get("logprob", 0.0))
            if len(logprobs) != len(resp_ids):
                logprobs = [0.0] * len(resp_ids)

            all_samples.append(RolloutSample(
                prompt_text=full_prompt,
                response_text=content,
                prompt_token_ids=prompt_ids,
                response_token_ids=resp_ids,
                response_logprobs=logprobs,
                reward=reward,
            ))

    return all_samples


def run_training(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = load_train_data(args.data)
    device = torch.device("cuda")

    # ============ Tokenizer ============
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============ Load model ONCE (stays on GPU for all steps) ============
    print(f"\nLoading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    # ============ Wait for remote sglang ============
    print(f"\nWaiting for sglang at {args.sglang_host}:{args.sglang_port}...")
    wait_for_sglang(args.sglang_host, args.sglang_port)

    # ============ Training loop ============
    running_mean = 0.0
    current_model_path = args.model_path
    train_t0 = time.time()
    reward_history = []

    for step in range(args.num_steps):
        pct = (step + 1) / args.num_steps
        bar_len = 20
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"\n{'='*60}")
        print(f"[{bar}] {step+1}/{args.num_steps} ({pct:.0%})")
        print(f"{'='*60}")

        # ---- Phase 1: Rollout (remote sglang) ----
        batch_prompts = random.sample(data, min(args.batch_size, len(data)))
        prompt_texts = [p["input"] for p in batch_prompts]
        ground_truths = [p.get("label", p.get("answer", "")) for p in batch_prompts]

        t0 = time.time()
        all_samples = generate_via_remote_sglang(
            host=args.sglang_host,
            port=args.sglang_port,
            prompts=prompt_texts,
            ground_truths=ground_truths,
            tokenizer=tokenizer,
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        gen_time = time.time() - t0

        mean_reward = sum(s.reward for s in all_samples) / max(len(all_samples), 1)
        for s in all_samples:
            running_mean = 0.95 * running_mean + 0.05 * s.reward
        reward_history.append(mean_reward)

        if len(all_samples) < 2:
            print("  Too few samples, skipping")
            continue

        # ---- Phase 2: Training (local HF model) ----
        t0 = time.time()
        loss, metrics = train_step(
            model, optimizer, all_samples, tokenizer, device,
            algo=args.algo,
            clip_low=args.clip_low, clip_high=args.clip_high,
            eps_clip=args.eps_clip, eps_clip_high=args.eps_clip_high,
            running_mean=running_mean,
            group_size=args.n_samples,
            max_seq_len=args.max_seq_len,
        )
        train_time = time.time() - t0

        # ---- Progress line ----
        elapsed = time.time() - train_t0
        avg_step = elapsed / (step + 1)
        eta = avg_step * (args.num_steps - step - 1)
        recent_r = sum(reward_history[-10:]) / max(len(reward_history[-10:]), 1)

        print(f"  reward={mean_reward:.2f} (avg10={recent_r:.2f}) | loss={loss:.4f} | "
              f"clip={metrics['clip_ratio']:.1%} | ratio={metrics['mean_ratio']:.2f}")
        print(f"  gen={gen_time:.0f}s train={train_time:.0f}s | "
              f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")

        # ---- Phase 3: Save checkpoint + signal sglang reload ----
        if (step + 1) % args.save_interval == 0 or step == args.num_steps - 1:
            ckpt_dir = os.path.join(args.save_dir, f"step_{step+1}")
            print(f"  Saving checkpoint to {ckpt_dir}...")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir, safe_serialization=True)
            tokenizer.save_pretrained(ckpt_dir)
            current_model_path = ckpt_dir

            # Signal sglang daemon to reload
            reload_sglang(args.save_dir)

            # Wait for sglang to come back up
            time.sleep(10)
            wait_for_sglang(args.sglang_host, args.sglang_port, timeout=900)
            print(f"  sglang reloaded with new weights.")

        print(f"  Step {step+1} done.")

    print(f"\n{'='*60}")
    print(f"Training complete. Final checkpoint: {current_model_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Distributed GRPO/SAO RL training")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--sglang-host", required=True, help="Inference machine IP")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--algo", choices=["sao", "grpo"], default="sao")
    parser.add_argument("--clip-low", type=float, default=0.7)
    parser.add_argument("--clip-high", type=float, default=6.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--eps-clip-high", type=float, default=0.28)
    parser.add_argument("--save-interval", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    run_training(args)


if __name__ == "__main__":
    main()
