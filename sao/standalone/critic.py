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
    """Base LM + linear value head. Same architecture as actor, outputs V(s_t)."""

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.model = base_model  # AutoModelForCausalLM
        self.value_head = nn.Linear(hidden_size, 1, bias=True)
        # Init value head
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass, return per-token values [batch, seq_len]."""
        outputs = self.model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        values = self.value_head(hidden).squeeze(-1)  # [batch, seq]
        return values

    def freeze_attention(self):
        """SAO §3.2: freeze attention params, only train MoE + value head."""
        for name, param in self.named_parameters():
            if any(pat in name for pat in ["self_attention", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "qkv"]):
                param.requires_grad = False
        n_frozen = sum(1 for _, p in self.named_parameters() if not p.requires_grad)
        n_total = sum(1 for _ in self.parameters())
        print(f"  [critic] Frozen {n_frozen}/{n_total} params (attention frozen)")


# ============================================================
# GAE (Generalized Advantage Estimation)
# ============================================================
def compute_gae_single(
    values: torch.Tensor,  # [resp_len] V(s_t) for response tokens
    reward: float,         # scalar reward for the trajectory
    gamma: float = 1.0,
    lambd: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE for a single trajectory.

    For math reasoning, reward is sparse at the end (last token gets reward).

    Returns:
        advantages: [resp_len]
        returns: [resp_len] = advantages + values (target for critic)
    """
    T = len(values)
    advantages = torch.zeros(T, device=values.device, dtype=values.dtype)

    # Build reward vector: all zeros except last token
    rewards = torch.zeros(T, device=values.device, dtype=values.dtype)
    rewards[-1] = reward

    # Backward GAE accumulation
    lastgae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
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
    values_list: list[torch.Tensor],  # per-sample [resp_len_i]
    rewards: list[float],
    response_lens: list[int],
    gamma: float = 1.0,
    alpha: float = 1.5,
    use_length_adaptive: bool = True,
    critic_lambd: float = 1.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """GAE for a batch of samples.

    Returns:
        advantages_list: per-sample advantages
        returns_list: per-sample returns (for critic training)
    """
    adv_list = []
    ret_list = []

    for values, reward, resp_len in zip(values_list, rewards, response_lens):
        # Truncate values to response length
        vals = values[:resp_len].detach()

        if use_length_adaptive:
            lam = length_adaptive_lambda(resp_len, alpha)
        else:
            lam = critic_lambd

        adv, ret = compute_gae_single(vals, reward, gamma=gamma, lambd=lam)
        adv_list.append(adv)
        ret_list.append(ret)

    return adv_list, ret_list


# ============================================================
# Critic Training Step (TTUR K=2)
# ============================================================
def compute_values(
    critic: ValueModel,
    input_ids_list: list[torch.Tensor],  # full prompt+response
    response_lens: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """Forward critic to get V(s_t) for response tokens. No gradient."""
    values_list = []
    with torch.no_grad():
        for input_ids, resp_len in zip(input_ids_list, response_lens):
            ids = input_ids.unsqueeze(0).to(device)
            vals = critic(ids)[0]  # [total_len]
            # Response tokens are the last resp_len positions
            values_list.append(vals[-resp_len:].clone())
    return values_list


def train_critic_step(
    critic: ValueModel,
    optimizer: torch.optim.Optimizer,
    input_ids_list: list[torch.Tensor],
    response_lens: list[int],
    returns_list: list[torch.Tensor],  # targets from GAE
    device: torch.device,
    value_clip: float = 0.2,
    k_epochs: int = 2,
) -> tuple[float, dict]:
    """SAO critic training with TTUR (K=2) and value clipping.

    Steps:
      1. Forward once (no grad) → V_old (clip reference)
      2. Repeat K times:
         - Forward (with grad) → V_new
         - V_clipped = V_old + clip(V_new - V_old, -ε_v, ε_v)
         - Loss = max((V_new - R)², (V_clipped - R)²)
         - Backward + step
    """
    # Step 1: Get V_old (clip reference)
    old_values_list = []
    with torch.no_grad():
        for input_ids, resp_len in zip(input_ids_list, response_lens):
            ids = input_ids.unsqueeze(0).to(device)
            vals = critic(ids)[0]
            old_values_list.append(vals[-resp_len:].clone())

    # Step 2: K iterations of value loss
    total_loss = 0.0
    for epoch in range(k_epochs):
        epoch_loss = torch.tensor(0.0, device=device)
        total_tokens = 0

        optimizer.zero_grad()

        for input_ids, resp_len, old_v, ret in zip(
            input_ids_list, response_lens, old_values_list, returns_list
        ):
            ids = input_ids.unsqueeze(0).to(device)
            vals = critic(ids)[0]  # [total_len]
            resp_vals = vals[-resp_len:]  # [resp_len]

            old_v = old_v.to(device)
            ret = ret.to(device)

            # Value clipping
            vals_clipped = old_v + (resp_vals - old_v).clamp(-value_clip, value_clip)
            loss_unclipped = (resp_vals - ret).pow(2)
            loss_clipped = (vals_clipped - ret).pow(2)
            loss = torch.max(loss_unclipped, loss_clipped).sum()

            epoch_loss = epoch_loss + loss
            total_tokens += resp_len

        epoch_loss = epoch_loss / max(total_tokens, 1)
        epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += epoch_loss.item()

    avg_loss = total_loss / k_epochs
    return avg_loss, {"critic_loss": avg_loss}
