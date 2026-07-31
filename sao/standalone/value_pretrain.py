#!/usr/bin/env python3
"""Value pretrain: 用 TIR 轨迹训练 critic value head。

输入: rollouts.jsonl (collect_rollouts.py 产出, 正确+错误)
Loss: MSE(V(s_t), reward) 对所有 response tokens
模型: SFT checkpoint + 随机初始化 value head

用法:
    BASH_ENV= python3 sao/standalone/value_pretrain.py \
        --data datasets/self_distill/rollouts.jsonl \
        --model checkpoints/sft/epoch_3 \
        --output checkpoints/value_pretrain \
        --epochs 5 --lr 1e-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_data(path: str):
    data = []
    for line in open(path):
        d = json.loads(line.strip())
        total_len = len(d["prompt_ids"]) + len(d["resp_ids"])
        if total_len > 30000:
            continue
        data.append(d)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="SFT checkpoint 或 base model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--freeze-base", action="store_true", help="冻结 base model, 只训 value head")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from critic import ValueModel

    # 数据
    print("Loading data...")
    data = load_data(args.data)
    correct = sum(1 for d in data if d.get("reward", 0) > 0)
    print(f"  {len(data)} trajectories ({correct} correct, {len(data)-correct} wrong)")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # model
    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}")
    max_memory = {i: "78GB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GB"

    print(f"Loading base model from {args.model}...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if not args.freeze_base:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    base.config.use_cache = False

    # ValueModel = base + value_head
    critic = ValueModel(base, hidden_size=base.config.hidden_size)

    if args.freeze_base:
        for p in critic.model.parameters():
            p.requires_grad = False
        print("  Base model frozen, only training value head")

    # optimizer
    trainable = [p for p in critic.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_trainable / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    first_device = next(critic.parameters()).device

    # 计算 explained variance
    def explained_variance(values, targets):
        var_values = values.var()
        var_targets = targets.var()
        if var_targets < 1e-8:
            return torch.tensor(0.0)
        return 1.0 - var_values / var_targets

    # 训练
    step = 0
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(args.epochs):
        import random
        random.seed(epoch)
        random.shuffle(data)

        epoch_loss = 0.0
        epoch_ev = 0.0
        epoch_n = 0

        for i, sample in enumerate(data):
            prompt_ids = sample["prompt_ids"]
            resp_ids = sample["resp_ids"]
            reward = float(sample.get("reward", 0))

            input_ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device=first_device)
            prompt_len = len(prompt_ids)
            resp_len = len(resp_ids)

            # forward: V(s_t) for all positions
            values = critic(input_ids)  # [1, seq_len]
            resp_values = values[0, prompt_len:prompt_len + resp_len]  # [resp_len]

            # target: reward for all response tokens
            target = torch.full_like(resp_values, reward)

            # MSE loss
            loss = nn.functional.mse_loss(resp_values, target)
            loss_scaled = loss / args.accum_steps
            loss_scaled.backward()

            epoch_loss += loss.item()

            # EV (batch size 1, 所以只看单个轨迹的预测方差)
            with torch.no_grad():
                ev = explained_variance(resp_values.detach(), target)
                epoch_ev += ev.item()

            epoch_n += 1

            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0:
                    avg_loss = epoch_loss / max(epoch_n, 1)
                    avg_ev = epoch_ev / max(epoch_n, 1)
                    print(f"  epoch {epoch+1}/{args.epochs} step {step} "
                          f"[{i+1}/{len(data)}] mse={avg_loss:.4f} ev={avg_ev:.4f}")

                if step % args.save_every == 0:
                    ckpt = f"{args.output}/step_{step}"
                    print(f"  Saving {ckpt}...")
                    critic.model.save_pretrained(ckpt, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt)
                    torch.save(critic.value_head.state_dict(), f"{ckpt}/value_head.pt")
                    print(f"  value_head → {ckpt}/value_head.pt")

        # epoch end
        ckpt = f"{args.output}/epoch_{epoch+1}"
        avg_loss = epoch_loss / max(epoch_n, 1)
        avg_ev = epoch_ev / max(epoch_n, 1)
        print(f"\nEpoch {epoch+1} done. mse={avg_loss:.4f} ev={avg_ev:.4f}. Saving {ckpt}...")
        critic.model.save_pretrained(ckpt, safe_serialization=True)
        tokenizer.save_pretrained(ckpt)
        torch.save(critic.value_head.state_dict(), f"{ckpt}/value_head.pt")

    print(f"\nValue pretrain complete. Checkpoints in {args.output}")


if __name__ == "__main__":
    main()
