"""组件 4: Skip-Observation Token-level GAE — 论文 §3.2

------------------------------------------------------------------------
为什么需要 Skip-Obs GAE
------------------------------------------------------------------------
agentic / multi-turn 轨迹结构:  T = [a_0, o_0, a_1, o_1, ..., a_n]
  - a_i: 模型生成的 action token (loss_mask=1)
  - o_i: 环境反馈 observation token (loss_mask=0, 模型没生成)

标准 GAE 在相邻 token 之间算 TD 残差 δ_t = r_t + γ·V(s_{t+1}) − V(s_t).
但在 action → observation 的边界处, 模型并没有「生成」obs token,
V(o_{i,start}) 试图预测环境状态, 噪声极大.

SAO 的修正 (论文 Eq.4-5): 把 action i 末尾的 value 直接连到 action i+1 起点的 value,
跨过中间所有 obs token:

    δ(a_{i,N}) = r + γ · V(a_{i+1,0}) − V(a_{i,N})           (Eq.5)
    Â(a_{i,N}) = δ + γλ · Â(a_{i+1,0})                      (Eq.4)

观测 token 本身不参与 advantage 传播.

------------------------------------------------------------------------
slime 现状
------------------------------------------------------------------------
slime 的 ``vanilla_gae`` / ``chunked_gae`` (slime/utils/ppo_utils.py:471+)
把整个 response 当作连续序列算 GAE, 不区分 action/obs.

slime multi-turn 路径 (``slime/utils/mask_utils.py``) 已经按 token 生成了
``loss_mask``:
  - loss_mask[t] = 1 → action token (模型生成)
  - loss_mask[t] = 0 → observation token (环境返回)

这正好给我们提供了 action/obs 边界信息, 不需要额外标注.

------------------------------------------------------------------------
本文件实现
------------------------------------------------------------------------
``skip_obs_gae``: 输入 rewards/values/loss_masks, 输出 advantages/returns.
逻辑:
  1. 从 response 末尾反向扫描;
  2. 对每个 action token t:
       - 找它的 next-action token (跨过中间的 obs), 作为 s_{t+1};
       - δ_t = r_t + γ · V(next_action_token) − V(t);
       - Â_t = δ_t + γλ · Â(next_action_token);
  3. obs token: advantage = 0 (不参与梯度), returns = values.

支持 batch (B, T), 支持每个 sample 独立的 loss_mask.
"""

from __future__ import annotations

import torch


# =========================================================================
# 核心实现: Skip-Observation GAE
# =========================================================================
def skip_obs_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_masks: torch.Tensor,
    gamma: float = 1.0,
    lambd: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SAO Skip-Observation Token-level GAE.

    Args:
        rewards:    [B, T]  每 token 奖励 (通常末位才有 reward, 其余 0).
        values:     [B, T]  critic 对每个 token 的 V(s_t) 预测.
        loss_masks: [B, T]  binary, 1=action token, 0=observation token.
        gamma:      discount factor γ.
        lambd:      GAE λ.

    Returns:
        advantages: [B, T]  obs token 位置为 0.
        returns:    [B, T]  advantages + values.
    """
    assert rewards.ndim == 2 and values.ndim == 2 and loss_masks.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T) and loss_masks.shape == (B, T)
    device = rewards.device
    dtype = values.dtype

    advantages = torch.zeros(B, T, device=device, dtype=dtype)

    # 反向扫描: 从 T-1 到 0
    # next_adv[t]  = Â(s_{t+1}) (如果 t 是 action, s_{t+1} 是下一个 action 的起点)
    next_adv = torch.zeros(B, device=device, dtype=dtype)
    # next_val[t] = V(s_{t+1})  (跨过 obs)
    # 边界: T-1 位置的 next_val 默认 0 (标准 PPO 假设 episode 末尾 V=0)
    next_val = torch.zeros(B, device=device, dtype=dtype)

    for t in reversed(range(T)):
        is_action = loss_masks[:, t].to(torch.bool)  # [B]

        # δ_t = r_t + γ · V(next_action) − V(t)
        # 对 action token 才有意义; 对 obs token 我们直接令 δ=0
        delta = torch.where(
            is_action,
            rewards[:, t] + gamma * next_val - values[:, t],
            torch.zeros_like(rewards[:, t]),
        )
        # Â_t = δ_t + γλ · Â(next_action)
        new_adv = delta + gamma * lambd * next_adv

        # 写回 advantages (obs token 位置保持 0)
        advantages[:, t] = torch.where(is_action, new_adv, torch.zeros_like(new_adv))

        # 更新 next_adv / next_val 给前一个 token 用
        # 对 action token: 下一个 token 的 V/Â 就是当前的 new_adv / values[t]
        # 对 obs token: 「穿透」——保持上一个 action 的 next_adv/next_val 不变
        next_adv = torch.where(is_action, new_adv, next_adv)
        next_val = torch.where(is_action, values[:, t], next_val)

    returns = advantages + values
    return advantages, returns


# =========================================================================
# 对照基线: 标准 GAE (不分 action/obs), 用来证明 skip-obs 的差异
# =========================================================================
def vanilla_gae_ref(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lambd: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """slime ``vanilla_gae`` 的等价实现, 用于对照.

    把所有 token 当作连续序列算 GAE, 不跳过 obs.
    """
    B, T = rewards.shape
    advantages = torch.zeros(B, T, device=rewards.device, dtype=values.dtype)
    lastgaelam = torch.zeros(B, device=rewards.device, dtype=values.dtype)
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=rewards.device, dtype=values.dtype)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages[:, t] = lastgaelam
    return advantages, advantages + values


# =========================================================================
# 单测 / 演示
# =========================================================================
def _demo():
    print("=" * 70)
    print("Skip-Observation GAE: action 边界跨过 obs token")
    print("=" * 70)

    # 构造 1 条轨迹, 形如 [a, a, o, o, a, a, o, a]:
    #   - action token (loss_mask=1): 0, 1, 4, 5, 7
    #   - obs token    (loss_mask=0): 2, 3, 6
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
    values = torch.tensor([[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]])

    # 只在最后一个 action token 上给 reward = 1.0
    rewards = torch.zeros(1, 8)
    rewards[0, 7] = 1.0

    gamma, lambd = 1.0, 1.0

    adv_skip, ret_skip = skip_obs_gae(rewards, values, loss_mask, gamma, lambd)
    adv_van, ret_van = vanilla_gae_ref(rewards, values, gamma, lambd)

    print(f"\nloss_mask : {loss_mask[0].tolist()}")
    print(f"values    : {values[0].tolist()}")
    print(f"rewards   : {rewards[0].tolist()}\n")
    print(f"vanilla GAE adv  : {[round(x, 3) for x in adv_van[0].tolist()]}")
    print(f"skip-obs   adv  : {[round(x, 3) for x in adv_skip[0].tolist()]}")
    print()
    print("差异解释:")
    print(" - vanilla: obs 位置 2,3,6 也参与 advantage 传播, V(obs) 噪声被引入")
    print(" - skip-obs: obs 位置 adv=0, 仅在 action 边界算 δ, 信号更纯净")

    # 断言: obs 位置的 advantage 必须为 0
    obs_positions = (loss_mask[0] == 0).nonzero(as_tuple=True)[0]
    for pos in obs_positions:
        assert adv_skip[0, pos] == 0.0, f"obs token at {pos.item()} 的 advantage 必须为 0"
    # 断言: action 位置的 advantage 与 vanilla 不同 (因为跳过 obs 改变了 next_value)
    action_positions = (loss_mask[0] == 1).nonzero(as_tuple=True)[0]
    # 至少有一些 action token 的 adv 与 vanilla 不同
    diff_count = (adv_skip[0, action_positions] != adv_van[0, action_positions]).sum().item()
    assert diff_count > 0, "skip-obs 应该改变 action token 的 advantage (除非 V 恰好相同)"
    print(f"\n✅ Skip-Obs GAE 单测通过 (obs 位置 adv=0, action 位置与 vanilla 不同)")


if __name__ == "__main__":
    _demo()
