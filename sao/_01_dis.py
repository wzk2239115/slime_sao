"""组件 1: DIS (Direct double-sided Importance Sampling) — 论文 §3.1

------------------------------------------------------------------------
为什么需要 DIS
------------------------------------------------------------------------
异步 RL 训练时, rollout 引擎和训练模型是解耦的:
  - rollout 用 π_rollout 采样
  - 训练时策略已经更新到 π_θ
  - 经典 PPO 用 π_{θ_old} / π_θ 做重要性采样修正

但在异步场景下, π_{θ_old} 难以精确跟踪 (rollout 期间训练侧可能更新多次),
维护 checkpoint 历史又不可行. SAO 的做法非常激进:

  1. **直接用 π_rollout 当 behavior proxy**, 即
        r_t(θ) = π_θ(y_t|...) / π_rollout(y_t|...)                       (Eq.2)

     这一步在 slime 里通过 ``--use-rollout-logprobs`` 已经默认开启:
     rollout 阶段会把 log π_rollout 写到 sample.rollout_log_probs,
     训练时用 rollout log-probs 当作 ``old_log_probs``.

  2. **双向 mask 而不是 clip**: 落在 [1-ε_l, 1+ε_h] 之外的 token,
     梯度直接置 0 (而不是 PPO 风格的 clip-and-pass-through):

        f(r; ε_l, ε_h) = r   if 1-ε_l < r < 1+ε_h
                       = 0   otherwise                                (Eq.3)

        L(θ) = E[ f(r_t; ε_l, ε_h) · Â_t · log π_θ(a_t|s_t) ]         (Eq.1)

------------------------------------------------------------------------
slime 现状 & 改动点
------------------------------------------------------------------------
slime 的 ``--use-tis`` + ``--custom-tis-function-path`` 是为 Icepop / TIS
设计的扩展点, 它接受一个签名为::

    fn(args, *, pg_loss, train_log_probs, rollout_log_probs, loss_masks, **kw)
        -> (pg_loss, modified_response_masks, metrics)

的函数. 我们只要把 DIS 写成同样签名的函数即可直接挂上去:

    --use-rollout-logprobs
    --use-tis
    --custom-tis-function-path SAO.sao._01_dis.dis_tis_function
    --tis-clip-low 0.7      # = 1 - ε_l  (ε_l=0.3 for TIR math)
    --tis-clip     6.0      # = 1 + ε_h  (ε_h=5.0 for TIR math)

注意: slime 现有 ``icepop_function`` 已经实现了「区间外置 0」,
**但它把 pg_loss 又乘以 ratio**, 得到 ``-r²·A`` 的梯度, 与 SAO 不一致.
本文件实现的 DIS 严格匹配论文 Eq.(1).

------------------------------------------------------------------------
⚠️  注意: slime 的 ``arguments.py:1802`` 当前断言
    ``use_rollout_logprobs`` 和 ``use_tis`` 不能同时为 True.
    要启用 DIS, 必须先放开这个断言 (或新增 ``--use-dis`` 旁路).
    详见 TODO 文件 §2.2.
"""

from __future__ import annotations

from typing import Any

import torch


# =========================================================================
# 核心实现: DIS mask 函数
# =========================================================================
def dis_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """SAO DIS: token-level 双向硬 mask.

    等价于把论文 Eq.(3) 的 ``f(r; ε_l, ε_h)`` 直接乘到 slime 已经算好的
    ``pg_loss = -r · A`` 上:
        - 区间内: ``pg_loss = -r · A``           梯度 = ``r · A · ∇log π_θ``
        - 区间外: ``pg_loss = 0``                梯度 = 0

    Args:
        args:           必须包含 ``tis_clip_low`` (下界, 如 0.7) 和 ``tis_clip`` (上界, 如 6.0).
        pg_loss:        slime 算好的 ``-ratio · advantage`` tensor, shape=[total_tokens].
        train_log_probs: 当前策略 π_θ 的 log-prob, 每个 sample 一个 tensor.
        rollout_log_probs: π_rollout 的 log-prob (rollout 阶段写入 sample).
        loss_masks:     每 sample 的 response-only mask (这里不修改, 原样返回).

    Returns:
        ``(new_pg_loss, loss_masks, metrics)`` 三元组, 与 slime 现有 TIS
        函数签名一致, metrics 含 ``tis`` (ratio), ``tis_clipfrac`` (被 mask 比例),
        ``tis_abs`` (|ratio-1|).
    """
    # ---- 1. 拼成一维 tensor -------------------------------------------------
    # slime 的 train_log_probs 是 per-sample list[ Tensor[resp_len_i] ].
    # 注意每个 sample 内部已经做了 CP slice, 直接 cat 即可.
    train_lp = torch.cat(train_log_probs, dim=0)
    rollout_lp = torch.cat(rollout_log_probs, dim=0)

    # ---- 2. 计算 ratio = π_θ / π_rollout -----------------------------------
    # 论文 Eq.(2): r_t(θ) = exp(log π_θ - log π_rollout)
    ratio = torch.exp(train_lp - rollout_lp)

    # ---- 3. 区间 mask: 落在 [tis_clip_low, tis_clip] 内的位置才贡献梯度 -----
    #   tis_clip_low = 1 - ε_l  (例如 ε_l=0.3 → 0.7)
    #   tis_clip     = 1 + ε_h  (例如 ε_h=5.0 → 6.0)
    in_trust_region = (ratio >= args.tis_clip_low) & (ratio <= args.tis_clip)
    mask = in_trust_region.to(pg_loss.dtype)

    # ---- 4. 应用 mask -------------------------------------------------------
    # pg_loss 已经是 -ratio · A, 再乘 mask 就实现了:
    #   区间内: -ratio · A  →  梯度 = ratio · A · ∇log π_θ   (与 SAO Eq.1 一致)
    #   区间外: 0           →  无梯度
    pg_loss = pg_loss * mask

    # ---- 5. 打点指标 (会自动进 wandb / tensorboard) -------------------------
    #   tis_clipfrac: 被 mask 掉的比例 (1 = 全部被裁)
    #   tis_abs:      |ratio - 1|, 反映 off-policy 程度
    metrics = {
        "tis": ratio.detach(),
        "tis_clipfrac": (1.0 - mask).detach(),
        "tis_abs": (ratio - 1.0).abs().detach(),
    }
    return pg_loss, loss_masks, metrics


# =========================================================================
# 单测 / 演示: 直接 ``python SAO/sao/_01_dis.py`` 运行
# =========================================================================
def _demo():
    """手造 6 个 token, 验证 DIS 在/不在信任域的行为."""
    # 模拟 args: ε_l=0.3, ε_h=5.0  → ratio 信任域 [0.7, 6.0]
    args = type("Args", (), {"tis_clip_low": 0.7, "tis_clip": 6.0})()

    # 6 个 token 的 log π_θ 和 log π_rollout
    # 故意构造 ratio 跨越区间:
    #   ratio = exp(train - rollout)
    # token 0: 1.00   (在区间)
    # token 1: 4.48   (在区间)
    # token 2: 0.50   (低于 0.7, 应 mask)
    # token 3: 7.39   (高于 6.0, 应 mask)
    # token 4: 1.00   (在区间)
    # token 5: 1.00   (在区间)
    train_lp = [torch.tensor([-0.5, -1.0, -1.0, -1.0, -0.5, -0.5])]
    rollout_lp = [torch.tensor([-0.5, -2.5, -0.31, 0.00, -0.5, -0.5])]

    ratio = torch.exp(train_lp[0] - rollout_lp[0])
    print(f"ratio:        {ratio.tolist()}")

    # 假装 pg_loss = -ratio * A (slime compute_policy_loss 的输出), A=1
    pg_loss = -ratio * 1.0

    out_pg, _, metrics = dis_tis_function(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_lp,
        rollout_log_probs=rollout_lp,
        loss_masks=[torch.ones(6)],
    )

    expected_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    expected_clipfrac = 1.0 - expected_mask

    print(f"DIS mask:     {expected_mask.tolist()}")
    print(f"DIS clipfrac: {metrics['tis_clipfrac'].tolist()}")
    print(f"DIS pg_loss:  {out_pg.tolist()}  (被 mask 的 token 应为 0)")

    # 断言
    assert torch.allclose(metrics["tis_clipfrac"], expected_clipfrac), "DIS mask 计算错误"
    assert out_pg[2] == 0.0 and out_pg[3] == 0.0, "区间外 token 必须 pg_loss=0"
    assert out_pg[0] != 0.0 and out_pg[1] != 0.0, "区间内 token 应保留 pg_loss"
    print("\n✅ DIS 单测通过")


if __name__ == "__main__":
    _demo()
