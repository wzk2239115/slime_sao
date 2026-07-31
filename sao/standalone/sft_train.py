#!/usr/bin/env python3
"""SFT: 用正确的 TIR 轨迹微调 Qwen3。

输入: rollouts.jsonl (collect_rollouts.py 产出)
Loss: next-token cross-entropy, 只在 action tokens 上计算
Epochs: 3 (论文 §4.1)

用法:
    BASH_ENV= python3 sao/standalone/sft_train.py \
        --data datasets/self_distill/rollouts.jsonl \
        --model models/Qwen3-30B-A3B-Thinking-2507 \
        --output checkpoints/sft \
        --epochs 3 --lr 1e-5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_data(path: str, correct_only: bool = True):
    data = []
    for line in open(path):
        d = json.loads(line.strip())
        if correct_only and d.get("reward", 0) <= 0:
            continue
        total_len = len(d["prompt_ids"]) + len(d["resp_ids"])
        if total_len > 30000:
            continue
        data.append(d)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 数据
    print("Loading data...")
    data = load_data(args.data, correct_only=True)
    print(f"  {len(data)} correct trajectories")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # model
    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}")
    max_memory = {i: "78GB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GB"

    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    first_device = next(model.parameters()).device

    # 训练
    step = 0
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(args.epochs):
        import random
        random.seed(epoch)
        random.shuffle(data)

        epoch_loss = 0.0
        epoch_n = 0

        for i, sample in enumerate(data):
            prompt_ids = sample["prompt_ids"]
            resp_ids = sample["resp_ids"]
            action_mask = sample["action_mask"]

            input_ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device=first_device)

            # labels: action tokens 保留, 其余 -100
            labels = torch.full_like(input_ids, -100)
            offset = len(prompt_ids)
            for j in range(len(resp_ids)):
                if j < len(action_mask) and action_mask[j] == 1:
                    labels[0, offset + j] = input_ids[0, offset + j]

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.accum_steps
            loss.backward()
            epoch_loss += outputs.loss.item()
            epoch_n += 1

            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0:
                    avg_loss = epoch_loss / max(epoch_n, 1)
                    print(f"  epoch {epoch+1}/{args.epochs} step {step} "
                          f"[{i+1}/{len(data)}] loss={avg_loss:.4f}")

                if step % args.save_every == 0:
                    ckpt = f"{args.output}/step_{step}"
                    print(f"  Saving {ckpt}...")
                    model.save_pretrained(ckpt, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt)

        # epoch end save
        ckpt = f"{args.output}/epoch_{epoch+1}"
        print(f"\nEpoch {epoch+1} done. avg_loss={epoch_loss/max(epoch_n,1):.4f}. Saving {ckpt}...")
        model.save_pretrained(ckpt, safe_serialization=True)
        tokenizer.save_pretrained(ckpt)

    print(f"\nSFT complete. Checkpoints in {args.output}")


if __name__ == "__main__":
    main()
