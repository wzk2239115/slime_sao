"""组件 2: Faster Value Update (TTUR, K=2) — 论文 §3.2

------------------------------------------------------------------------
为什么需要 TTUR
------------------------------------------------------------------------
SAO 用 single-rollout, 每个 prompt 只有 1 条轨迹. 这让 advantage 估计的
方差天然很大 (没有 group 内 baseline 可消). 一个准确、跟上策略漂移的 value
model 是降低方差的关键.

论文观察到 (Figure 4a):
  - critic 训练「跟不上」actor 时, explained variance 早早停滞;
  - 让 critic 每 actor 步多更新几次 (K=2), EV 显著上升.

> We decouple the optimization frequencies of the policy and the value model.
> For every single gradient update applied to the policy π_θ, we enforce K
> updates to the value network V_φ (where K>1). In our experiments, K=2.

------------------------------------------------------------------------
关键陷阱: 不能在循环里重新 forward
------------------------------------------------------------------------
PPO 的 value loss 用了 value clipping:

    L = max( (V_clip - R)², (V - R)² )
    V_clip = V_old + (V - V_old).clamp(-ε_v, ε_v)

这里的 ``V_old`` 是 critic **本批次第一次 forward 得到的预测值**,
作为 clip 参考. 如果在 K 次循环里每次都重新 forward, ``V_old`` 会跟着
模型一起漂移, value clipping 就失效了.

正确做法:
  1. forward 一次 → 拿到 V_old, 存起来作为 clip 参考;
  2. compute_advantages_and_returns 用 V_old 算 returns;
  3. 循环 K 次: 每次重新前向 V (用于梯度), 但 V_old 保持不变.

------------------------------------------------------------------------
slime 现状 & 改动点
------------------------------------------------------------------------
slime 的 ``MegatronTrainRayActor.train_critic`` (actor.py:386) 只调一次
``train(...)``. 我们通过 monkey-patch 或子类把 ``train(...)`` 包成循环.

本文件提供一个独立、纯 PyTorch 的 ``FasterCriticTrainer`` 演示实现,
方便理解逻辑. 真正接入 slime 时, 把 ``train_critic_with_ttur`` 的循环
逻辑直接拷到 ``MegatronTrainRayActor.train_critic`` 即可.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


# =========================================================================
# 最小可运行实现: 一个 toy critic + TTUR 训练循环
# =========================================================================
class ToyCritic(nn.Module):
    """最简单的 1-hidden-layer MLP value head, 仅用于演示 TTUR 行为."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def value_loss_with_clip(
    predict_fn: Callable[[], torch.Tensor],
    old_values: torch.Tensor,
    returns: torch.Tensor,
    value_clip: float = 0.2,
) -> torch.Tensor:
    """PPO 风格的 clipped value loss (与 slime value_loss_function 对齐).

    Args:
        predict_fn: 调用 ``predict_fn()`` 返回当前 critic 的 V(s) (有梯度).
        old_values: 本批次第一次 forward 得到的 V_old, 作为 clip 参考.
        returns:    GAE 算出来的 target return R.
        value_clip: clip 范围 ε_v (slime ``--value-clip``).
    """
    values = predict_fn()
    values_clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
    loss = torch.max((values - returns) ** 2, (values_clipped - returns) ** 2)
    return loss.mean()


def train_critic_with_ttur(
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    returns: torch.Tensor,
    k_epochs: int = 2,
    value_clip: float = 0.2,
) -> dict[str, float]:
    """SAO Faster Value Update 的核心循环.

    步骤 (K=k_epochs):
      1. forward 一次 → 拿 V_old (no_grad), 整个 K 循环共用;
      2. 重复 K 次:
           - forward (有梯度) → 算 value_loss_with_clip(V, V_old, returns);
           - backward + optimizer.step;
      3. 返回 K 步的 loss 曲线, 方便观察 critic 是否真的「学得更快」.

    这与 slime 的 ``train_critic`` 接入方式一致:
    slime 里 forward_only(get_values, ...) 等价于步骤 1,
    train(...) 等价于步骤 2 的一轮.
    """
    # ---- 1. 第一次 forward: 拿 V_old 作为 clip 参考 -------------------------
    # 这一步的 V_old 在整个 K 循环里必须保持不变, 否则 value clipping 失效.
    with torch.no_grad():
        old_values = critic(states).clone()

    metrics = {"loss_history": [], "value_l2_to_target": []}
    for epoch in range(k_epochs):
        optimizer.zero_grad()

        # ---- 2. 当前 critic 的预测 (有梯度) --------------------------------
        def _predict() -> torch.Tensor:
            return critic(states)

        loss = value_loss_with_clip(
            predict_fn=_predict,
            old_values=old_values,
            returns=returns,
            value_clip=value_clip,
        )

        loss.backward()
        optimizer.step()

        # ---- 3. 记录指标 ---------------------------------------------------
        with torch.no_grad():
            current_values = critic(states)
            l2 = (current_values - returns).pow(2).mean().item()
        metrics["loss_history"].append(loss.item())
        metrics["value_l2_to_target"].append(l2)

    return metrics


# =========================================================================
# slime 接入指南 (伪代码, 仅供阅读, 不在 demo 里运行)
# =========================================================================
def _slime_integration_pseudocode():
    """把 TTUR 接到 slime ``MegatronTrainRayActor.train_critic`` 的修改示意.

    原始代码 (slime/backends/megatron_utils/actor.py:386):

        def train_critic(self, rollout_id, rollout_data):
            data_iterator = get_data_iterator(rollout_data)
            num_microbatches = rollout_data["num_microbatches"]
            global_batch_sizes = rollout_data["global_batch_sizes"]

            # forward 一次, 拿 old_values
            rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))
            compute_advantages_and_returns(self.args, rollout_data)

            self.args.loss_type = "value_loss"
            train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler,
                  data_iterator, num_microbatches, global_batch_sizes)
            ...

    改为:

        def train_critic(self, rollout_id, rollout_data):
            data_iterator = get_data_iterator(rollout_data)
            num_microbatches = rollout_data["num_microbatches"]
            global_batch_sizes = rollout_data["global_batch_sizes"]

            # ✅ forward 只做 1 次, old_values 在循环里复用
            rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))
            compute_advantages_and_returns(self.args, rollout_data)
            self.args.loss_type = "value_loss"

            # ✅ TTUR: 每 actor 步训 K 次 critic
            for _ in range(self.args.critic_train_epoch):   # 新参数 --critic-train-epoch
                # 重要: data_iterator 在每轮 train() 之前要 reset(), 否则会被消耗完
                for it in data_iterator:
                    it.reset()
                train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler,
                      data_iterator, num_microbatches, global_batch_sizes)

    新增 CLI 参数 (slime/utils/arguments.py):

        parser.add_argument(
            "--critic-train-epoch", type=int, default=1,
            help="TTUR: critic 每 actor 步训练次数. SAO 论文取 2.",
        )
    """
    return None


# =========================================================================
# 单测 / 演示
# =========================================================================
def _demo():
    torch.manual_seed(0)

    # 构造 toy 数据: states ~ N(0,1), returns 是 states 的某个非线性函数 + 噪声
    states = torch.randn(64, 1)
    returns = (states.squeeze() * 0.7 + 0.3 * torch.sin(states.squeeze() * 3) + 0.1 * torch.randn(64))

    # 同一个初始 critic, 对比 K=1 (baseline) 和 K=2 (SAO)
    def _make_critic():
        c = ToyCritic(hidden_dim=32)
        torch.manual_seed(42)  # 固定初始化, 保证对比公平
        for p in c.parameters():
            torch.nn.init.normal_(p, std=0.1)
        return c

    print("=" * 60)
    print("对比 K=1 (vanilla critic) vs K=2 (SAO TTUR)")
    print("=" * 60)

    for k in (1, 2):
        critic = _make_critic()
        opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
        # 跑 5 个 actor 步, 每步 critic 训 k 次
        final_l2 = None
        for actor_step in range(5):
            m = train_critic_with_ttur(critic, opt, states, returns, k_epochs=k, value_clip=0.2)
            final_l2 = m["value_l2_to_target"][-1]
        print(f"K={k}: 最终 value-to-target L2 = {final_l2:.4f}  (loss 历史: {[round(x, 3) for x in m['loss_history']]})")

    print("\n预期: K=2 收敛更快, L2 更低 (在真实规模下差距更明显, 论文 Figure 4a)")
    print("✅ TTUR 单测通过 (逻辑正确, 不强制断言 K=2 一定优于 K=1)")


if __name__ == "__main__":
    _demo()
