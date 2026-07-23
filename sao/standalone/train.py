"""Standalone GRPO/SAO RL training loop.

Architecture:
  1. sglang server generates responses (rollout phase)
  2. Kill sglang, free GPU memory
  3. HF model computes log-probs + gradient update (training phase)
  4. Save checkpoint, restart sglang with new weights
  5. Repeat

No slime, no megatron, no torch_memory_saver.
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

from .sglang_server import start_server, stop_server
from .rollout import generate_batch, RolloutSample
from .reward import math_reward
from .grpo_step import (
    compute_grpo_advantages,
    compute_sao_advantages,
    compute_log_probs,
    dis_policy_loss,
    grpo_policy_loss,
)


def load_train_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} training prompts from {path}")
    return data


def build_token_ids(prompt: str, response: str, tokenizer) -> tuple[list[int], list[int]]:
    """Tokenize prompt and response separately."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # Response: add the leading space or special token as appropriate
    response_text = response
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    return prompt_ids, response_ids


def train_step(
    model,
    optimizer,
    samples: list[RolloutSample],
    tokenizer,
    device: torch.device,
    algo: str = "sao",  # "sao" or "grpo"
    clip_low: float = 0.7,
    clip_high: float = 6.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    running_mean: float = 0.0,
    group_size: int = 8,
    max_seq_len: int = 32768,
    gradient_checkpointing: bool = True,
) -> tuple[float, dict]:
    """One training step over a batch of samples."""
    model.train()

    # Compute advantages
    rewards = [s.reward for s in samples]
    if algo == "sao":
        advantages = compute_sao_advantages(rewards, running_mean)
    else:
        advantages = compute_grpo_advantages(rewards, group_size)

    # Build input_ids for each sample
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
            # Truncate from the left of prompt to keep response
            excess = len(total_ids) - max_seq_len
            prompt_ids = prompt_ids[excess:]
            total_ids = prompt_ids + resp_ids

        input_ids_list.append(torch.tensor(total_ids, dtype=torch.long))
        response_lens.append(len(resp_ids))

        # Rollout log-probs as tensor
        rlp = s.response_logprobs if len(s.response_logprobs) == len(resp_ids) else [0.0] * len(resp_ids)
        rollout_log_probs_list.append(torch.tensor(rlp, dtype=torch.float32))

    # Compute train log-probs (with gradient)
    train_log_probs = compute_log_probs(
        model, input_ids_list, response_lens, device, gradient_checkpointing
    )

    # Compute loss
    if algo == "sao":
        loss, metrics = dis_policy_loss(
            train_log_probs, rollout_log_probs_list, advantages,
            clip_low=clip_low, clip_high=clip_high,
        )
    else:
        loss, metrics = grpo_policy_loss(
            train_log_probs, rollout_log_probs_list, advantages,
            eps_clip=eps_clip, eps_clip_high=eps_clip_high,
        )

    # Backward + step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), metrics


def run_training(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = load_train_data(args.data)
    device = torch.device("cuda")

    # ============ Tokenizer ============
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============ Model + Optimizer ============
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

    # ============ Training loop ============
    running_mean = 0.0
    running_count = 0

    for step in range(args.num_steps):
        # ---- Phase 1: Rollout (sglang) ----
        print(f"\n{'='*60}")
        print(f"Step {step+1}/{args.num_steps}: Starting rollout phase")
        print(f"{'='*60}")

        # Free model memory for sglang
        del model
        torch.cuda.empty_cache()

        proc = start_server(
            model_path=args.model_path,
            port=args.port,
            tp=args.tp,
            mem_fraction_static=args.mem_fraction,
            max_total_tokens=args.max_total_tokens,
            disable_cuda_graph=args.disable_cuda_graph,
        )

        # Sample prompts
        batch_prompts = random.sample(data, min(args.batch_size, len(data)))
        prompt_texts = [p["input"] for p in batch_prompts]
        ground_truths = [p.get("label", p.get("answer", "")) for p in batch_prompts]

        # Generate with chat template
        t0 = time.time()
        all_samples = []
        for prompt_text, gt in zip(prompt_texts, ground_truths):
            messages = [{"role": "user", "content": prompt_text}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]

            from .rollout import _post
            payload = {
                "model": args.model_path,
                "messages": [{"role": "user", "content": prompt_text}],
                "n": args.n_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_new_tokens,
            }
            try:
                resp = _post(args.port, "/v1/chat/completions", payload, timeout=600)
            except Exception as e:
                print(f"  Generation error: {e}")
                continue

            for choice in resp.get("choices", []):
                content = choice["message"]["content"]
                resp_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
                reward = math_reward(content, gt)

                # Get logprobs from response if available
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
                running_mean = 0.95 * running_mean + 0.05 * reward
                running_count += 1

        gen_time = time.time() - t0
        mean_reward = sum(s.reward for s in all_samples) / max(len(all_samples), 1)
        print(f"  Generated {len(all_samples)} samples, mean_reward={mean_reward:.3f}, time={gen_time:.0f}s")

        # Kill sglang
        stop_server(proc)
        torch.cuda.empty_cache()
        time.sleep(3)

        if len(all_samples) < 2:
            print("  Too few samples, skipping training step")
            # Reload model
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, attn_implementation="flash_attention_2",
            )
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            continue

        # ---- Phase 2: Training (HF model) ----
        print(f"  Reloading model for training...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
        )

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
        print(f"  Training: loss={loss:.4f}, clip_ratio={metrics['clip_ratio']:.3f}, "
              f"mean_ratio={metrics['mean_ratio']:.3f}, time={train_time:.0f}s")

        # ---- Phase 3: Save checkpoint ----
        if (step + 1) % args.save_interval == 0 or step == args.num_steps - 1:
            ckpt_dir = os.path.join(args.save_dir, f"step_{step+1}")
            print(f"  Saving checkpoint to {ckpt_dir}...")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir, safe_serialization=True)
            tokenizer.save_pretrained(ckpt_dir)
            args.model_path = ckpt_dir  # Use updated model for next rollout
            print(f"  Checkpoint saved.")

        print(f"  Step {step+1} done: mean_reward={mean_reward:.3f}, running_mean={running_mean:.3f}, "
              f"loss={loss:.4f}")

    print(f"\n{'='*60}")
    print(f"Training complete. Final checkpoint: {args.model_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Standalone GRPO/SAO RL training")
    parser.add_argument("--model-path", required=True, help="HF model checkpoint path")
    parser.add_argument("--data", required=True, help="Training data jsonl")
    parser.add_argument("--save-dir", required=True, help="Checkpoint save dir")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per step")
    parser.add_argument("--n-samples", type=int, default=8, help="Samples per prompt (GRPO); 1 for SAO")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-total-tokens", type=int, default=40960)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--disable-cuda-graph", action="store_true", default=True)
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
