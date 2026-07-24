"""Value model pretraining: train critic before RL to solve cold-start problem.

Paper §3.2: "the 'cold start' problem in value estimation is a major bottleneck.
By significantly increasing the scale of the value pretraining corpus, we provide
a robust initialization point."

Uses TIR/SFT data: (prompt, response, reward) triples.
The critic learns to predict the trajectory reward from any position.

Usage:
    python -m sao.standalone.value_pretrain \
        --model-path /path/to/base_model \
        --data /path/to/tir_data.jsonl \
        --save-dir /path/to/critic_pretrained \
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn

from .critic import ValueModel


def load_pretrain_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} pretraining samples from {path}")
    return data


def compute_reward(text: str, gt: str) -> float:
    """Binary reward for pretraining data."""
    from .reward import math_reward
    return math_reward(text, gt)


def run_pretrain(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    data = load_pretrain_data(args.data)

    # ============ Tokenizer ============
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============ Model ============
    print(f"\nLoading base model from {args.model_path} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    critic = ValueModel(base, hidden_size=base.config.hidden_size)
    if args.freeze_attention:
        critic.freeze_attention()
    critic = critic.to(device)

    params = [p for p in critic.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )

    # ============ Training ============
    print(f"\nPretraining critic for {args.epochs} epochs, {len(data)} samples")
    print(f"Trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(data)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(data), args.batch_size):
            batch = data[i:i + args.batch_size]
            if len(batch) < 2:
                continue

            # Prepare batch
            input_ids_list = []
            response_lens = []
            rewards = []

            for sample in batch:
                messages = [{"role": "user", "content": sample["input"]}]
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]

                response = sample.get("output") or sample.get("response") or ""
                resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

                gt = sample.get("label") or sample.get("answer") or ""
                reward = sample.get("reward", None)
                if reward is None:
                    reward = compute_reward(response, gt)

                total = prompt_ids + resp_ids
                if len(total) > args.max_seq_len:
                    excess = len(total) - args.max_seq_len
                    prompt_ids = prompt_ids[excess:]
                    total = prompt_ids + resp_ids

                input_ids_list.append(torch.tensor(total, dtype=torch.long))
                response_lens.append(len(resp_ids))
                rewards.append(reward)

            # Forward critic → values for response tokens
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            total_tokens = 0

            for input_ids, resp_len, reward in zip(input_ids_list, response_lens, rewards):
                ids = input_ids.unsqueeze(0).to(device)
                values = critic(ids)[0]  # [total_len]
                resp_values = values[-resp_len:]  # [resp_len]

                # Target: reward for all response tokens (λ_critic=1, γ=1 → MC return)
                target = torch.full_like(resp_values, reward)
                loss = (resp_values - target).pow(2).sum()
                total_loss = total_loss + loss
                total_tokens += resp_len

            loss = total_loss / max(total_tokens, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % args.log_interval == 0:
                print(f"  [epoch {epoch+1} step {step}] loss={loss.item():.4f} "
                      f"avg_epoch_loss={epoch_loss/n_batches:.4f}")

        avg = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}: avg_loss={avg:.4f}")

    # ============ Save ============
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\nSaving pretrained critic to {args.save_dir} ...")
    torch.save(critic.state_dict(), os.path.join(args.save_dir, "critic.pt"))
    # Also save base model config for easy loading
    base.config.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Done. Use --critic-path {args.save_dir} for RL training.")


def main():
    parser = argparse.ArgumentParser(description="Value model pretraining")
    parser.add_argument("--model-path", required=True, help="Base model checkpoint")
    parser.add_argument("--data", required=True, help="TIR/SFT training data jsonl")
    parser.add_argument("--save-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Paper: 3 epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6, help="Critic lr (paper: 5e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--freeze-attention", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    run_pretrain(args)


if __name__ == "__main__":
    main()
