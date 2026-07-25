"""Value model (critic) + GAE + critic training for SAO.

Paper §3.2 components:
- ValueModel: base LM + value head → per-token V(s_t)
- compute_gae: token-level GAE with length-adaptive λ
- train_critic_step: value loss with clipping, frozen attention, K=2 (TTUR)
"""
from __future__ import annotations

import re
import torch
import torch.nn as nn


# ============================================================
# Value Model
# ============================================================
class ValueModel(nn.Module):
    """Base LM + linear value head. Same architecture as actor, outputs V(s_t).

    Handles device_map="auto" (model split across GPUs).
    """

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.model = base_model
        self.value_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

        # Find output device (device of last decoder layer params)
        output_device = torch.device("cpu")
        for param in self.model.parameters():
            output_device = param.device
        self.value_head = self.value_head.to(output_device)
        print(f"  [ValueModel] value_head on {output_device}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Move input to first layer's device (embedding layer)
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)

        outputs = self.model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]

        # Move hidden to value_head's device + dtype
        hidden = hidden.to(device=self.value_head.weight.device,
                           dtype=self.value_head.weight.dtype)
        values = self.value_head(hidden).squeeze(-1)  # [batch, seq]
        return values

    def freeze_attention(self):
        """SAO §3.2: freeze attention params, only train MoE + value head.
        
        Qwen3-MoE attention module (self_attn) contains:
          q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
        All must be frozen per the paper.
        """
        for name, param in self.named_parameters():
            # "self_attn" catches ALL attention params (q/k/v/o_proj + q/k_norm)
            # "post_attention_layernorm" is tied to attention output
            if any(pat in name for pat in ["self_attn", "post_attention_layernorm"]):
                param.requires_grad = False
        n_frozen = sum(1 for _, p in self.named_parameters() if not p.requires_grad)
        n_total = sum(1 for _ in self.parameters())
        print(f"  [critic] Frozen {n_frozen}/{n_total} params (attention frozen)")


# ============================================================
# GAE (Generalized Advantage Estimation)
# ============================================================
def compute_gae_single(
    values: torch.Tensor,
    reward: float,
    gamma: float = 1.0,
    lambd: float = 1.0,
    action_mask: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE for a single trajectory.

    Paper §3.2 Skip-Obs GAE (Eq.4-5): when action_mask is provided,
    observation tokens (mask=0) are skipped. Advantage bridges from
    end of one action segment to start of next action segment.

    Without action_mask (all None): standard GAE (pure reasoning mode).

    Returns:
        advantages: [resp_len]
        returns: [resp_len] = advantages + values
    """
    T = len(values)
    advantages = torch.zeros(T, device=values.device, dtype=values.dtype)

    rewards = torch.zeros(T, device=values.device, dtype=values.dtype)
    rewards[-1] = reward

    if action_mask is None:
        action_mask = [1] * T

    # Find action segments: [(start, end), ...] where mask=1
    segments = []
    in_action = False
    seg_start = 0
    for t in range(T):
        if action_mask[t] == 1 and not in_action:
            seg_start = t
            in_action = True
        elif action_mask[t] == 0 and in_action:
            segments.append((seg_start, t - 1))
            in_action = False
    if in_action:
        segments.append((seg_start, T - 1))

    if not segments:
        return advantages, values.clone()

    # Backward GAE across action segments (Skip-Obs)
    lastgae = torch.tensor(0.0, device=values.device, dtype=values.dtype)

    for seg_idx in reversed(range(len(segments))):
        seg_start, seg_end = segments[seg_idx]

        # Next action's first token value (for cross-segment TD bridge)
        if seg_idx < len(segments) - 1:
            next_action_start = segments[seg_idx + 1][0]
            next_val = values[next_action_start]
        else:
            next_val = torch.tensor(0.0, device=values.device, dtype=values.dtype)

        for t in range(seg_end, seg_start - 1, -1):
            if t == seg_end:
                # Last token of action: bridge to next action (Eq.5)
                delta = rewards[t] + gamma * next_val - values[t]
            else:
                # Normal TD within action segment
                next_v = values[t + 1]
                delta = rewards[t] + gamma * next_v - values[t]
            lastgae = delta + gamma * lambd * lastgae
            advantages[t] = lastgae

    returns = advantages + values
    return advantages, returns


def length_adaptive_lambda(resp_len: int, alpha: float = 1.5) -> float:
    """λ = clamp(1 - 1/(α·L), 0, 1). Paper §4.1."""
    if resp_len <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - 1.0 / (alpha * resp_len)))


def compute_gae_batch(
    values_list: list[torch.Tensor],
    rewards: list[float],
    response_lens: list[int],
    gamma: float = 1.0,
    alpha: float = 1.5,
    use_length_adaptive: bool = True,
    critic_lambd: float = 1.0,
    action_masks: list[list[int]] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """GAE for a batch of samples. Supports Skip-Obs GAE (TIR).

    Args:
        action_masks: per-token [1=action, 0=observation] for each sample.
                      None for pure reasoning (all action tokens).
    """
    adv_list = []
    ret_list = []

    for i, (values, reward, resp_len) in enumerate(zip(values_list, rewards, response_lens)):
        vals = values[:resp_len].detach()
        mask = action_masks[i] if action_masks else None

        if use_length_adaptive:
            lam_policy = length_adaptive_lambda(resp_len, alpha)
        else:
            lam_policy = critic_lambd

        adv, _ = compute_gae_single(vals, reward, gamma=gamma, lambd=lam_policy, action_mask=mask)
        _, ret = compute_gae_single(vals, reward, gamma=gamma, lambd=1.0, action_mask=mask)

        adv_list.append(adv)
        ret_list.append(ret)

    return adv_list, ret_list


# ============================================================
# Critic Training Step (TTUR K=2)
# ============================================================
def compute_values(
    critic: ValueModel,
    input_ids_list: list[torch.Tensor],
    response_lens: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """Forward critic to get V(s_t) for response tokens. No gradient."""
    values_list = []
    with torch.no_grad():
        for input_ids, resp_len in zip(input_ids_list, response_lens):
            ids = input_ids.unsqueeze(0)
            vals = critic(ids)[0]  # ValueModel handles device internally
            values_list.append(vals[-resp_len:].clone())
    return values_list


def train_critic_step(
    critic: ValueModel,
    optimizer,
    input_ids_list: list[torch.Tensor],
    response_lens: list[int],
    returns_list: list[torch.Tensor],
    device: torch.device,
    value_clip: float = 0.2,
    k_epochs: int = 2,
    action_masks: list[list[int]] | None = None,
) -> tuple[float, dict]:
    """SAO critic training with TTUR (K=2), value clipping, and action masking.

    With action_masks (TIR): only compute value loss on action tokens.
    """
    # Step 1: Get V_old (clip reference)
    old_values_list = []
    with torch.no_grad():
        for input_ids, resp_len in zip(input_ids_list, response_lens):
            ids = input_ids.unsqueeze(0)
            vals = critic(ids)[0]
            old_values_list.append(vals[-resp_len:].clone())

    # Step 2: K iterations of value loss (gradient accumulation per sample)
    output_device = critic.value_head.weight.device
    total_loss = 0.0
    for epoch in range(k_epochs):
        epoch_loss_val = 0.0
        total_tokens = 0

        optimizer.zero_grad()

        for input_ids, resp_len, old_v, ret in zip(
            input_ids_list, response_lens, old_values_list, returns_list
        ):
            ids = input_ids.unsqueeze(0)
            vals = critic(ids)[0]
            resp_vals = vals[-resp_len:]

            old_v = old_v.to(resp_vals.device)
            ret = ret.to(resp_vals.device)

            vals_clipped = old_v + (resp_vals - old_v).clamp(-value_clip, value_clip)
            loss_unclipped = (resp_vals - ret).pow(2)
            loss_clipped = (vals_clipped - ret).pow(2)
            loss = torch.max(loss_unclipped, loss_clipped)

            # Mask out observation tokens (TIR): only train on action tokens
            if action_masks is not None:
                am = torch.tensor(action_masks[i], device=resp_vals.device, dtype=resp_vals.dtype)
                loss = loss * am

            loss = loss.sum()

            # Per-sample backward (free graph immediately)
            loss.backward()
            epoch_loss_val += loss.item()
            total_tokens += resp_len

            del vals, resp_vals, loss, vals_clipped, loss_unclipped, loss_clipped

        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += epoch_loss_val / max(total_tokens, 1)

    avg_loss = total_loss / k_epochs
    return avg_loss, {"critic_loss": avg_loss}
