"""GRPO/SAO training step: compute advantages, DIS mask, policy gradient loss.

Pure PyTorch. Works with any HF CausalLM model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_grpo_advantages(
    rewards: list[float],
    group_size: int,
    eps: float = 1e-8,
) -> list[float]:
    """GRPO advantage: (r - mean) / (std + eps) within each group.

    For SAO (group_size=1), this returns [reward - baseline] where
    baseline is a running mean (passed in separately).
    """
    advantages = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i:i + group_size]
        mean = sum(group) / len(group)
        std = (sum((r - mean) ** 2 for r in group) / len(group)) ** 0.5
        for r in group:
            advantages.append((r - mean) / (std + eps))
    return advantages


def compute_sao_advantages(
    rewards: list[float],
    running_mean: float,
) -> list[float]:
    """SAO single-rollout advantage: r - running_mean."""
    return [r - running_mean for r in rewards]


def compute_log_probs(
    model,
    input_ids_list: list[torch.Tensor],
    response_lens: list[int],
    device: torch.device,
    gradient_checkpointing: bool = True,
) -> list[torch.Tensor]:
    """Compute log π_θ for response tokens of each sample.

    Returns list of tensors, each [response_len_i], with gradient.
    Handles device_map="auto" (model split across GPUs).
    """
    log_probs_list = []
    first_device = next(model.parameters()).device

    for input_ids, resp_len in zip(input_ids_list, response_lens):
        input_ids = input_ids.unsqueeze(0).to(first_device)

        if gradient_checkpointing and model.training:
            outputs = model(input_ids, use_cache=False)
        else:
            outputs = model(input_ids, use_cache=False)

        logits = outputs.logits[0]  # [total_len, vocab]
        shift_logits = logits[:-1]
        shift_labels = input_ids[0, 1:]

        resp_logits = shift_logits[-resp_len:]
        resp_labels = shift_labels[-resp_len:]

        log_probs = F.log_softmax(resp_logits, dim=-1)
        token_log_probs = log_probs.gather(1, resp_labels.unsqueeze(1)).squeeze(1)

        log_probs_list.append(token_log_probs)

    return log_probs_list


def dis_policy_loss(
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    advantages: list[float],
    clip_low: float = 0.7,   # 1 - ε_l  (SAO TIR: ε_l=0.3)
    clip_high: float = 6.0,  # 1 + ε_h  (SAO TIR: ε_h=5.0)
) -> tuple[torch.Tensor, dict]:
    """SAO DIS policy gradient loss (Eq. 1-3).

    L = -mean_t[ f(r_t; ε_l, ε_h) · Â_t · 1 ]  (log π already in train_log_probs via ratio)

    Actually, following SAO Eq.1:
      L(θ) = E[ f(r_t, ε_l, ε_h) · Â_t · log π_θ(a_t|s_t) ]
    
    But the standard PPO formulation uses ratio * advantage:
      pg_loss = -ratio * advantage
    
    With DIS mask applied. We follow the ratio formulation since it's
    equivalent and more numerically stable.
    """
    total_loss = torch.tensor(0.0, device=train_log_probs[0].device)
    total_tokens = 0
    total_clipped = 0

    for tlp, rlp, adv in zip(train_log_probs, rollout_log_probs, advantages):
        rlp = rlp.to(tlp.device)
        # ratio = π_θ / π_rollout = exp(log π_θ - log π_rollout)
        ratio = torch.exp(tlp - rlp)
        adv_t = torch.tensor(adv, device=tlp.device, dtype=tlp.dtype)

        # DIS mask: zero out tokens outside [clip_low, clip_high]
        in_region = (ratio >= clip_low) & (ratio <= clip_high)
        mask = in_region.to(tlp.dtype)

        # Policy loss: -ratio * advantage * mask (negative for gradient descent)
        sample_loss = -(ratio * adv_t * mask).sum()
        total_loss = total_loss + sample_loss
        total_tokens += len(tlp)
        total_clipped += (1 - mask).sum().item()

    loss = total_loss / max(total_tokens, 1)
    metrics = {
        "loss": loss.item(),
        "clip_ratio": total_clipped / max(total_tokens, 1),
        "mean_ratio": torch.cat([torch.exp(t - r) for t, r in zip(train_log_probs, rollout_log_probs)]).mean().item(),
    }
    return loss, metrics


def grpo_policy_loss(
    train_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    advantages: list[float],
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
) -> tuple[torch.Tensor, dict]:
    """Standard GRPO policy loss (clip-higher variant).

    L = -min(ratio * A, clip(ratio, 1-ε, 1+ε_h) * A)
    """
    total_loss = torch.tensor(0.0, device=train_log_probs[0].device)
    total_tokens = 0
    total_clipped = 0

    for tlp, olp, adv in zip(train_log_probs, old_log_probs, advantages):
        olp = olp.to(tlp.device)
        ratio = torch.exp(tlp - olp)
        adv_t = torch.tensor(adv, device=tlp.device, dtype=tlp.dtype)

        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high) * adv_t
        sample_loss = -torch.min(surr1, surr2).sum()
        total_loss = total_loss + sample_loss
        total_tokens += len(tlp)
        total_clipped += (ratio.detach() > 1 + eps_clip_high).sum().item()
        total_clipped += (ratio.detach() < 1 - eps_clip).sum().item()

    loss = total_loss / max(total_tokens, 1)
    metrics = {
        "loss": loss.item(),
        "clip_ratio": total_clipped / max(total_tokens, 1),
        "mean_ratio": torch.cat([torch.exp(t - o) for t, o in zip(train_log_probs, old_log_probs)]).mean().item(),
    }
    return loss, metrics
