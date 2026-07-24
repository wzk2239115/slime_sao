"""Async SAO trainer: consumes trajectories from queue, trains actor + critic.

Runs on training machine. Continuously polls the queue directory for new
trajectories and immediately trains on them (single-rollout async).

Paper SAO components:
- DIS (§3.1): token-level double-sided importance sampling with hard masking
- GAE with length-adaptive λ (§4.1): λ_policy = 1-1/(α·L), α=1.5
- λ_critic = 1 for value targets (pure Monte-Carlo return)
- TTUR K=2 (§3.2): critic updated twice per actor step
- Frozen attention (§3.2): critic attention params frozen, only MoE+head trained
- Critic warmup (§4.1): 10 steps before critic training starts

Memory optimization for 2×30B on 8×80GB GPUs:
- 8-bit AdamW (bitsandbytes) → 4× optimizer memory reduction
- Gradient checkpointing → minimal activation memory
- Sequential sample processing → no batch padding overhead

Usage:
    python -m sao.standalone.trainer \
        --model-path /path/to/model \
        --critic-path /path/to/critic \
        --queue-dir /shared/queue \
        --save-dir /shared/checkpoints \
        --num-steps 1000 --batch-size 32
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import time

import torch
import torch.nn as nn

from .grpo_step import compute_log_probs, dis_policy_loss
from .critic import (
    ValueModel, compute_values, compute_gae_batch, train_critic_step,
)


def poll_queue(queue_dir: str, max_items: int = 128) -> list[dict]:
    """Atomically get pending trajectories from queue."""
    pending_dir = os.path.join(queue_dir, "pending")
    done_dir = os.path.join(queue_dir, "done")
    os.makedirs(done_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(pending_dir, "traj_*.json")))[:max_items]
    trajs = []
    for f in files:
        try:
            with open(f) as fh:
                trajs.append(json.load(fh))
            os.rename(f, os.path.join(done_dir, os.path.basename(f)))
        except Exception:
            pass
    return trajs


def queue_size(queue_dir: str) -> int:
    pending_dir = os.path.join(queue_dir, "pending")
    return len(glob.glob(os.path.join(pending_dir, "traj_*.json")))


def create_optimizer(params, lr: float, weight_decay: float, use_8bit: bool = True):
    """Create optimizer: 8-bit AdamW if bitsandbytes available, else standard AdamW."""
    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98),
            )
            print(f"  Using 8-bit AdamW (bitsandbytes) — 4× optimizer memory saved")
            return opt
        except ImportError:
            print(f"  bitsandbytes not available, using standard AdamW")
    return torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98),
    )


def run_trainer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    print(f"GPUs visible: {n_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============ Actor ============
    print(f"\nLoading actor from {args.model_path} ...")
    max_memory = {i: "78GB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GB"

    actor = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    actor.gradient_checkpointing_enable()
    actor.config.use_cache = False

    actor_optimizer = create_optimizer(
        actor.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        use_8bit=args.use_8bit_adam,
    )

    # ============ Critic ============
    print(f"\nLoading critic from {args.critic_path} ...")
    base_critic = AutoModelForCausalLM.from_pretrained(
        args.critic_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_critic.gradient_checkpointing_enable()
    base_critic.config.use_cache = False

    critic = ValueModel(base_critic, hidden_size=actor.config.hidden_size)
    critic.freeze_attention()

    critic_params = [p for p in critic.parameters() if p.requires_grad]
    critic_optimizer = create_optimizer(
        critic_params, lr=args.critic_lr, weight_decay=args.weight_decay,
        use_8bit=args.use_8bit_adam,
    )

    critic_warmup_steps = args.critic_warmup
    n_trainable = sum(p.numel() for p in critic_params)
    n_total = sum(p.numel() for p in critic.parameters())
    print(f"  Critic: {n_trainable/1e6:.0f}M trainable / {n_total/1e6:.0f}M total")

    # ============ Training loop ============
    step = 0
    reward_history = []
    train_t0 = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    while step < args.num_steps:
        queue_n = queue_size(args.queue_dir)
        needed = args.batch_size
        if queue_n < needed:
            if step == 0 or step % 10 == 0:
                print(f"  Waiting for trajectories... (queue: {queue_n}/{needed})")
            time.sleep(5)
            continue

        trajs = poll_queue(args.queue_dir, max_items=needed)
        if len(trajs) < 2:
            time.sleep(5)
            continue

        # ---- Prepare batch ----
        input_ids_list = []
        response_lens = []
        rollout_log_probs_list = []
        rewards_float = []

        for t in trajs:
            prompt_ids = t.get("prompt_ids", [])
            resp_ids = t.get("resp_ids", [])
            if not prompt_ids or not resp_ids:
                continue
            total = prompt_ids + resp_ids
            if len(total) > args.max_seq_len:
                excess = len(total) - args.max_seq_len
                prompt_ids = prompt_ids[excess:]
                total = prompt_ids + resp_ids
            input_ids_list.append(torch.tensor(total, dtype=torch.long))
            response_lens.append(len(resp_ids))
            rlp = t.get("logprobs", [])
            if len(rlp) != len(resp_ids):
                rlp = [0.0] * len(resp_ids)
            rollout_log_probs_list.append(torch.tensor(rlp, dtype=torch.float32))
            rewards_float.append(t.get("reward", 0.0))

        if len(input_ids_list) < 2:
            time.sleep(5)
            continue

        mean_reward = sum(rewards_float) / len(rewards_float)
        reward_history.append(mean_reward)

        # ---- Critic forward → GAE (no grad) ----
        with torch.no_grad():
            values_list = compute_values(critic, input_ids_list, response_lens, device)

        advantages_list, returns_list = compute_gae_batch(
            values_list, rewards_float, response_lens,
            gamma=args.gamma, alpha=args.gae_alpha,
        )

        # ---- Actor step (DIS) ----
        actor.train()
        train_log_probs = compute_log_probs(
            actor, input_ids_list, response_lens, device,
            gradient_checkpointing=True,
        )
        actor_loss, actor_metrics = dis_policy_loss(
            train_log_probs, rollout_log_probs_list, advantages_list,
            clip_low=args.clip_low, clip_high=args.clip_high,
        )
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optimizer.step()

        # Free actor activations
        del train_log_probs, actor_loss
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Critic step (TTUR K=2, skip during warmup) ----
        critic_loss_val = 0.0
        if step >= critic_warmup_steps:
            critic.train()
            critic_loss_val, _ = train_critic_step(
                critic, critic_optimizer, input_ids_list, response_lens,
                returns_list, device,
                value_clip=args.value_clip, k_epochs=args.critic_k,
            )
            del _
            gc.collect()
            torch.cuda.empty_cache()

        step += 1

        # ---- Progress ----
        elapsed = time.time() - train_t0
        avg_step = elapsed / step
        eta = avg_step * (args.num_steps - step)
        recent_r = sum(reward_history[-20:]) / max(len(reward_history[-20:]), 1)
        pct = step / args.num_steps
        bar_len = 20
        filled = int(bar_len * pct)
        bar = "=" * filled + "-" * (bar_len - filled)

        warmup_str = "" if step >= critic_warmup_steps else " [critic warmup]"
        gpu_mem = torch.cuda.memory_allocated() / 1e9

        print(f"[{bar}] {step}/{args.num_steps} ({pct:.0%}) | "
              f"r={mean_reward:.2f}(avg20={recent_r:.2f}) "
              f"al={actor_metrics.get('loss',0):.4f} cl={critic_loss_val:.4f} "
              f"clip={actor_metrics.get('clip_ratio',0):.0%} "
              f"ratio={actor_metrics.get('mean_ratio',0):.3f} | "
              f"{avg_step:.0f}s/step ETA={eta/3600:.1f}h "
              f"GPU={gpu_mem:.0f}GB{warmup_str}")

        # ---- Save checkpoint ----
        if step % args.save_interval == 0 or step == args.num_steps:
            ckpt_dir = os.path.join(args.save_dir, f"step_{step}")
            print(f"  Saving checkpoint to {ckpt_dir}...")
            os.makedirs(ckpt_dir, exist_ok=True)
            actor.save_pretrained(ckpt_dir, safe_serialization=True)
            tokenizer.save_pretrained(ckpt_dir)
            critic_dir = os.path.join(ckpt_dir, "critic")
            os.makedirs(critic_dir, exist_ok=True)
            torch.save(critic.state_dict(), os.path.join(critic_dir, "critic.pt"))
            print(f"  Checkpoint saved. Rollout worker will auto-reload.")

    print(f"\n{'='*60}")
    print(f"Training complete. {step} steps, final reward={recent_r:.3f}")
    print(f"Total time: {(time.time()-train_t0)/3600:.1f}h")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Async SAO trainer")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--critic-path", required=True)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Paper: 128 (reduce if memory constrained)")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Actor lr (paper: 1e-6)")
    parser.add_argument("--critic-lr", type=float, default=5e-6,
                        help="Critic lr (paper: 5e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-low", type=float, default=0.7,
                        help="1-epsilon_l (paper: 1-0.3=0.7)")
    parser.add_argument("--clip-high", type=float, default=6.0,
                        help="1+epsilon_h (paper: 1+5.0=6.0)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="GAE gamma (paper: 1.0)")
    parser.add_argument("--gae-alpha", type=float, default=1.5,
                        help="lambda=1-1/(alpha*L) (paper: 1.5)")
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--critic-k", type=int, default=2,
                        help="TTUR K (paper: 2)")
    parser.add_argument("--critic-warmup", type=int, default=10,
                        help="Critic warmup steps (paper: 10)")
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--use-8bit-adam", action="store_true", default=True,
                        help="Use 8-bit AdamW for memory efficiency")
    args = parser.parse_args()
    run_trainer(args)


if __name__ == "__main__":
    main()
