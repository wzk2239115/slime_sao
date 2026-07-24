"""Async SAO trainer: consumes trajectories from queue, trains actor + critic.

Runs on training machine. Continuously polls the queue directory for new
trajectories and immediately trains on them (single-rollout async).

Paper SAO components:
- DIS (§3.1): token-level double-sided importance sampling
- GAE with length-adaptive λ (§4.1): λ_policy = 1-1/(α·L)
- λ_critic = 1 for value targets
- TTUR K=2 (§3.2): critic updated twice per actor step
- Frozen attention (§3.2): critic attention params frozen

Usage:
    python -m sao.standalone.trainer \
        --model-path /path/to/model \
        --critic-path /path/to/critic_pretrained \
        --queue-dir /shared/queue \
        --save-dir /shared/checkpoints \
        --num-steps 1000 --batch-size 128
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch

from .grpo_step import compute_log_probs, dis_policy_loss
from .critic import (
    ValueModel, compute_values, compute_gae_batch, train_critic_step,
)


def poll_queue(queue_dir: str, max_items: int = 128) -> list[dict]:
    """Get pending trajectories from queue."""
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


def run_trainer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")

    # ============ Tokenizer ============
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============ Actor ============
    print(f"Loading actor from {args.model_path} ...")
    actor = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    actor.gradient_checkpointing_enable()
    actor.config.use_cache = False
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )

    # ============ Critic ============
    print(f"Loading critic from {args.critic_path} ...")
    base_critic = AutoModelForCausalLM.from_pretrained(
        args.critic_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_critic.gradient_checkpointing_enable()
    base_critic.config.use_cache = False
    critic = ValueModel(base_critic, hidden_size=actor.config.hidden_size)
    critic.freeze_attention()
    critic = critic.to(device)
    critic_params = [p for p in critic.parameters() if p.requires_grad]
    critic_optimizer = torch.optim.AdamW(
        critic_params, lr=args.critic_lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )

    # Warmup counter (paper §4.1: 10-step critic warmup)
    critic_warmup_steps = args.critic_warmup

    # ============ Training loop ============
    step = 0
    reward_history = []
    train_t0 = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    while step < args.num_steps:
        # ---- Collect trajectories from queue ----
        queue_n = queue_size(args.queue_dir)
        needed = args.batch_size
        if queue_n < needed:
            if step == 0:
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
            if not prompt_ids:
                prompt_ids = tokenizer(t["prompt_text"], add_special_tokens=False)["input_ids"]
            if not resp_ids:
                resp_ids = tokenizer(t.get("response_text", ""), add_special_tokens=False)["input_ids"]
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

        mean_reward = sum(rewards_float) / len(rewards_float)
        reward_history.append(mean_reward)

        # ---- Critic forward → GAE ----
        with torch.no_grad():
            values_list = compute_values(critic, input_ids_list, response_lens, device)

        advantages_list, returns_list = compute_gae_batch(
            values_list, rewards_float, response_lens,
            gamma=args.gamma, alpha=args.gae_alpha,
        )

        # ---- Actor step (DIS) ----
        actor.train()
        train_log_probs = compute_log_probs(
            actor, input_ids_list, response_lens, device, True
        )
        actor_loss, actor_metrics = dis_policy_loss(
            train_log_probs, rollout_log_probs_list, advantages_list,
            clip_low=args.clip_low, clip_high=args.clip_high,
        )
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optimizer.step()

        # ---- Critic step (TTUR K=2, skip during warmup) ----
        critic_loss_val = 0.0
        if step >= critic_warmup_steps:
            critic_loss_val, _ = train_critic_step(
                critic, critic_optimizer, input_ids_list, response_lens,
                returns_list, device,
                value_clip=args.value_clip, k_epochs=args.critic_k,
            )

        step += 1

        # ---- Progress ----
        elapsed = time.time() - train_t0
        avg_step = elapsed / step
        eta = avg_step * (args.num_steps - step)
        recent_r = sum(reward_history[-20:]) / max(len(reward_history[-20:]), 1)
        pct = step / args.num_steps
        bar_len = 20
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)

        warmup_str = "" if step >= critic_warmup_steps else " [critic warmup]"
        print(f"[{bar}] {step}/{args.num_steps} ({pct:.0%}) | "
              f"r={mean_reward:.2f}(avg20={recent_r:.2f}) "
              f"al={actor_loss:.4f} cl={critic_loss_val:.4f} "
              f"clip={actor_metrics['clip_ratio']:.0%} | "
              f"{avg_step:.0f}s/step ETA {eta/60:.0f}min{warmup_str}")

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
    print(f"Total time: {(time.time()-train_t0)/60:.0f} min")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Async SAO trainer")
    parser.add_argument("--model-path", required=True, help="Actor initial checkpoint")
    parser.add_argument("--critic-path", required=True, help="Critic initial checkpoint (value pretrained)")
    parser.add_argument("--queue-dir", required=True, help="Shared queue directory")
    parser.add_argument("--save-dir", required=True, help="Checkpoint save directory")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128, help="Paper §4.1: 128")
    parser.add_argument("--lr", type=float, default=1e-6, help="Actor lr (paper: 1e-6)")
    parser.add_argument("--critic-lr", type=float, default=5e-6, help="Critic lr (paper: 5e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-low", type=float, default=0.7, help="1-ε_l (paper: 1-0.3=0.7)")
    parser.add_argument("--clip-high", type=float, default=6.0, help="1+ε_h (paper: 1+5.0=6.0)")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE γ (paper: 1.0)")
    parser.add_argument("--gae-alpha", type=float, default=1.5, help="λ=1-1/(α·L) (paper: 1.5)")
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--critic-k", type=int, default=2, help="TTUR K (paper: 2)")
    parser.add_argument("--critic-warmup", type=int, default=10, help="Critic warmup steps (paper: 10)")
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    run_trainer(args)


if __name__ == "__main__":
    main()
