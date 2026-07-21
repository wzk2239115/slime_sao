"""组件 5: Length-Adaptive GAE (per-sample λ) — 论文 §4.1

------------------------------------------------------------------------
为什么需要 length-adaptive λ
------------------------------------------------------------------------
论文 §4.1 引用 VAPO (Yue et al., 2025):

> We adopt a length-adaptive GAE with λ_policy = 1 − 1/(α·l),
> where l is the response length and α = 1.5.

直觉:
  - 长序列 (复杂推理):  把 λ 调接近 1, 让 GAE 多依赖 value function
                        (low variance, 长 horizon 需要);
  - 短序列:             λ 偏小, 多依赖 immediate reward (low bias).

固定 λ 在长短序列混合的 batch 里会顾此失彼.

注意论文同时给:
  - λ_policy = 1 − 1/(α·l)  → actor advantage 用
  - λ_critic = 1            → value target 用 (即 pure Monte-Carlo return)

------------------------------------------------------------------------
slime 现状
------------------------------------------------------------------------
slime ``--lambd`` 是全局标量, 传入 ``get_advantages_and_returns_batch``
(ppo_utils.py:471) 再传给 ``vanilla_gae`` / ``chunked_gae``.

我们要做的:
  1. 新增 ``--gae-alpha`` 参数;
  2. 在算 GAE 前, 按 batch 内每个 sample 的 response_length 算 λ_i;
  3. ``vanilla_gae`` / ``chunked_gae`` 支持 per-row λ (广播成 [B,1]).

------------------------------------------------------------------------
本文件实现
------------------------------------------------------------------------
1. ``compute_per_sample_lambd``: 根据 response_lengths 算 λ list.
2. ``length_adaptive_gae``: per-sample λ 版本的 vanilla GAE (toy 实现).
3. 接入 slime 的说明 (改 ``vanilla_gae`` 一行即可支持 per-row λ).
"""

from __future__ import annotations

import torch


# =========================================================================
# 1. 计算每 sample 的 λ
# =========================================================================
def compute_per_sample_lambd(
    response_lengths: list[int] | torch.Tensor,
    alpha: float = 1.5,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    """SAO length-adaptive λ: λ_i = clamp(1 − 1/(α·L_i), 0, 1).

    Args:
        response_lengths: 每 sample 的 response token 数.
        alpha:            论文取 1.5.
        lambda_min/max:   防止 L 过小时出现负数 / L 过大无意义.

    Returns:
        [B] 的 λ tensor.
    """
    if not isinstance(response_lengths, torch.Tensor):
        response_lengths = torch.tensor(response_lengths, dtype=torch.float32)
    lengths = response_lengths.clamp_min(1).float()  # 避免 L=0 时除零
    lambd = 1.0 - 1.0 / (alpha * lengths)
    return lambd.clamp(lambda_min, lambda_max)


# =========================================================================
# 2. per-sample λ 版本的 GAE (toy 实现, 直观展示行为)
# =========================================================================
def length_adaptive_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_lengths: list[int] | torch.Tensor,
    gamma: float = 1.0,
    alpha: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """length-adaptive GAE: 每 sample 用自己的 λ_i.

    与 vanilla GAE 的唯一差别: ``w = γ · λ_i`` 而不是 ``γ · λ``.

    Args:
        rewards:    [B, T]
        values:     [B, T]
        response_lengths: list of len B, 每 sample 的真实 response 长度.
        gamma:      discount factor.
        alpha:      length-adaptive 系数.

    Returns:
        advantages: [B, T]
        returns:    [B, T]
        lambds:     [B] 每 sample 实际用的 λ (供 logging)
    """
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape

    # ---- 1. 算 per-sample λ ------------------------------------------------
    lambds = compute_per_sample_lambd(response_lengths, alpha=alpha)  # [B]
    lambd_b = lambds.view(B, 1).to(values.dtype)  # [B, 1] 用于广播

    # ---- 2. 反向扫描, 每行用各自的 λ --------------------------------------
    advantages = torch.zeros_like(values)
    lastgaelam = torch.zeros(B, device=values.device, dtype=values.dtype)
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=values.device, dtype=values.dtype)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        # 关键: 用 per-row λ
        lastgaelam = delta + gamma * lambd_b[:, 0] * lastgaelam
        advantages[:, t] = lastgaelam

    returns = advantages + values
    return advantages, returns, lambds


# =========================================================================
# 3. slime 接入指南 (改 vanilla_gae 一行)
# =========================================================================
def _slime_integration_notes():
    """slime ``vanilla_gae`` 改成 per-sample λ 的最小 diff.

    slime/utils/ppo_utils.py:579 vanilla_gae 原版:

        def vanilla_gae(rewards, values, gamma, lambd):
            B, T = rewards.shape
            ...
            lastgaelam = torch.zeros(B, ...)
            for t in reversed(range(T)):
                next_value = values[:, t + 1] if t < T - 1 else 0.0
                delta = rewards[:, t] + gamma * next_value - values[:, t]
                lastgaelam = delta + gamma * lambd * lastgaelam  # ← lambd 是标量
                ...

    改为 (接受 lambd 既可以是 float 也可以是 [B] tensor):

        def vanilla_gae(rewards, values, gamma, lambd):
            B, T = rewards.shape
            # 兼容 per-sample λ
            if not torch.is_tensor(lambd):
                lambd = torch.tensor(lambd, device=values.device, dtype=values.dtype)
            lambd_b = lambd.view(B, 1) if lambd.numel() == B else lambd  # [B,1] 广播

            lastgaelam = torch.zeros(B, ...)
            for t in reversed(range(T)):
                next_value = values[:, t + 1] if t < T - 1 else 0.0
                delta = rewards[:, t] + gamma * next_value - values[:, t]
                lastgaelam = delta + gamma * lambd_b[:, 0] * lastgaelam  # ← 每 row 独立
                ...

    上游调用 ``get_advantages_and_returns_batch`` (ppo_utils.py:471):
    在 ``args.gae_alpha is not None`` 时, 用
        lambd = compute_per_sample_lambd(response_lengths, args.gae_alpha)
    替代标量 args.lambd.

    新增 CLI 参数:

        parser.add_argument(
            "--gae-alpha", type=float, default=None,
            help="开启 length-adaptive GAE. λ_i = 1 - 1/(alpha * L_i). 论文取 1.5.",
        )

    chunked_gae 的修改略复杂 (parallel scan kernel M 用了 w = γλ 作 exponent),
    但只需把 w 改成 [B] 即可, 因为 w ** diff[mask] 会自然广播.
    """
    return None


# =========================================================================
# 单测 / 演示
# =========================================================================
def _demo():
    print("=" * 70)
    print("Length-Adaptive GAE: 不同长度 sample 用不同 λ")
    print("=" * 70)

    # 3 个 sample, 长度分别 10 / 100 / 1000
    response_lengths = [10, 100, 1000]
    lambds = compute_per_sample_lambd(response_lengths, alpha=1.5)
    print(f"\nresponse_lengths = {response_lengths}, alpha = 1.5")
    print(f"per-sample λ    = {[round(x, 4) for x in lambds.tolist()]}")
    print("直觉: L 越大 → λ 越接近 1 (更依赖 value, lower variance)")

    # 断言公式
    expected = [1 - 1 / (1.5 * L) for L in response_lengths]
    for got, exp in zip(lambds.tolist(), expected):
        assert abs(got - exp) < 1e-6, f"λ 公式算错: {got} != {exp}"
    print("✅ λ 公式正确")

    # 完整 GAE 对照
    print("\n--- GAE 行为对比 (固定 λ=0.95 vs length-adaptive) ---")
    B, T = 2, 8
    rewards = torch.zeros(B, T)
    rewards[:, -1] = 1.0  # 只在末尾给 reward
    values = torch.linspace(0.1, 0.8, T).unsqueeze(0).repeat(B, 1)
    response_lengths = [T, T]  # 两个 sample 等长 (这里只为演示)

    # sample 0 用短长度概念 (alpha 把它的 λ 调小), sample 1 用长长度
    # 我们手动改 response_lengths 让两个 sample 有不同 λ
    response_lengths_diff = [4, 8]
    lambds_diff = compute_per_sample_lambd(response_lengths_diff, alpha=1.5)
    print(f"  sample 0 (L=4):  λ = {lambds_diff[0].item():.4f}")
    print(f"  sample 1 (L=8):  λ = {lambds_diff[1].item():.4f}")

    adv, ret, used_lambds = length_adaptive_gae(
        rewards, values, response_lengths=response_lengths_diff, gamma=1.0, alpha=1.5
    )
    print(f"\n  advantage (sample 0, 短): {[round(x, 3) for x in adv[0].tolist()]}")
    print(f"  advantage (sample 1, 长): {[round(x, 3) for x in adv[1].tolist()]}")

    # 断言: λ 不同的 sample 产生不同的 advantage
    assert not torch.allclose(adv[0], adv[1]), "不同 λ 应该产生不同 advantage"
    print("\n✅ Length-Adaptive GAE 单测通过")


if __name__ == "__main__":
    _demo()
