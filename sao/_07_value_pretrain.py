"""组件 7: Scaling Value Pretraining — 论文 §3.2

------------------------------------------------------------------------
为什么需要 value 预训练
------------------------------------------------------------------------
论文 §3.2 "Scaling Value Pretraining":

> 我们发现 value 估计的 "cold start" 问题是主要瓶颈.
> 通过显著扩大 value 预训练数据规模, 我们提供了一个稳健的初始化点,
> 让 single-rollout 和 TTUR 机制从训练早期就有效.

简单说: critic 不能从 actor SFT ckpt 直接来 (那是个 LM head, 不是 value head),
要先用大批量 (state, return) 数据预训练 value 头. 论文没公开数据规模, 但
强调 "significantly increasing the scale".

------------------------------------------------------------------------
slime 现状
------------------------------------------------------------------------
slime 的 ``value_loss_function`` (loss.py:1113) 已经实现了 clipped value loss
(与 PPO 一致). 但 slime 没有独立的「critic-only 预训练」入口:

  - ``train.py`` / ``train_async.py`` 是 RL 循环;
  - ``--loss-type value_loss`` 可以切到 value loss, 但需要 rollout_data 提供
    returns, 没有专门处理预训练数据集.

------------------------------------------------------------------------
本文件实现
------------------------------------------------------------------------
1. ``pretrain_value_loop``: 一个独立的 PyTorch 训练循环, 接受 (states, returns)
   数据, 训 critic 几个 epoch. 用 clipped value loss (与 slime 一致).

2. ``load_pretrained_value_head``: 把预训练好的 value head 权重加载到
   Megatron critic 模型 (匹配 slime ``model_provider`` 的 value_function 层名).

3. 数据准备指南: 怎么从 SFT / rollout dump 里造 (state, return) 数据.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================================================================
# 1. Toy critic (与组件 2 一致, 这里独立避免循环依赖)
# =========================================================================
class ToyValueHead(nn.Module):
    """最简单的 value head: hidden → 1.

    真实场景应该用 GPTModel 的 backbone + 一个 linear value_function 层.
    slime 的 critic 就是 actor 结构 + 把 output_layer 换成 VHead.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        # 注意 slime 里这层叫 ``value_function`` (或 PostProcess),
        # 与组件 3 frozen-attention 的 freeze pattern 配合: 这层不冻.
        self.value_function = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_function(self.backbone(x)).squeeze(-1)


class StateReturnDataset(Dataset):
    """(states, returns) 数据集.

    数据来源建议:
      1. 用 SFT 模型跑一遍 prompt → 拿到 trajectory tokens;
      2. 用 GPT-OSS-120B (或其它强模型) 对每个 trajectory 打 reward;
      3. 把 reward 当作 return (γ=1 时 return = last reward), 或走完整 GAE;
      4. 对 trajectory 内每个 token, 记录 (hidden_state, return).

    slime 已有 ``--save-debug-rollout-data`` 可以 dump trajectory tokens +
    reward, 直接当数据源.
    """

    def __init__(self, states: torch.Tensor, returns: torch.Tensor):
        assert states.shape[0] == returns.shape[0]
        self.states = states
        self.returns = returns

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.returns[idx]


# =========================================================================
# 2. Value 预训练循环 (clipped value loss, 与 slime value_loss_function 对齐)
# =========================================================================
def pretrain_value_loop(
    value_head: nn.Module,
    dataset: Dataset,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 5e-6,
    value_clip: float = 0.2,
    device: str = "cpu",
    log_every: int = 1,
) -> dict[str, list[float]]:
    """SAO value 预训练循环.

    与 slime ``value_loss_function`` 的差异:
      - slime 用 ``V_old`` (本 batch 第一次 forward) 作为 clip 参考;
      - 预训练阶段没有 ``V_old`` (单次 forward), 我们用 detach 的 V 直接做 clip 参考,
        等价于 ``V_old = V`` (clip 不起作用, 即标准 MSE).

    真正接入 slime 时, 直接把 (states, returns) 喂给一个 critic-only 训练脚本,
    用 slime 的 ``--loss-type value_loss`` + ``--num-critic-only-steps N`` 即可
    (N 足够大就是预训练).

    Args:
        value_head:  待训练的 value model (nn.Module).
        dataset:     StateReturnDataset.
        epochs:      训练 epoch 数.
        batch_size:  mini-batch 大小.
        lr:          论文 §4.1 critic lr = 5e-6.
        value_clip:  clip 范围, 预训练阶段可选 0 (关掉).
        device:      cpu / cuda.

    Returns:
        {"loss_history": [...], "ev_history": [...]}  EV = explained variance.
    """
    value_head.to(device)
    optimizer = torch.optim.Adam(value_head.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    metrics = {"loss_history": [], "ev_history": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_n = 0
        all_pred, all_target = [], []

        for states, returns in loader:
            states = states.to(device)
            returns = returns.to(device)

            # forward 拿当前 V (with grad) + detached V_old
            values = value_head(states)
            old_values = values.detach()

            # clipped value loss: max((V-R)^2, (V_clip-R)^2)
            values_clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
            loss = torch.max((values - returns) ** 2, (values_clipped - returns) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * states.shape[0]
            epoch_n += states.shape[0]
            all_pred.append(values.detach())
            all_target.append(returns)

        # explained variance: 1 - Var(R - V) / Var(R)
        # 越接近 1 越好 (论文 Figure 4a 的纵轴就是 EV)
        pred = torch.cat(all_pred)
        target = torch.cat(all_target)
        var_target = target.var().clamp_min(1e-8)
        ev = 1.0 - (target - pred).var() / var_target

        metrics["loss_history"].append(epoch_loss / max(epoch_n, 1))
        metrics["ev_history"].append(ev.item())

        if (epoch + 1) % log_every == 0:
            print(f"  [value pretrain] epoch {epoch + 1}/{epochs}  "
                  f"loss={epoch_loss / max(epoch_n, 1):.4f}  EV={ev.item():.4f}")

    return metrics


# =========================================================================
# 3. 加载预训练权重到 Megatron critic (指南)
# =========================================================================
def load_pretrained_value_head(
    megatron_critic_model,
    pretrained_value_head_state_dict: dict,
    strict: bool = False,
) -> None:
    """把预训练好的 value head 权重塞进 Megatron critic 模型.

    slime 的 critic 由 ``slime/backends/megatron_utils/model_provider.py`` 创建:
      - backbone 与 actor 同构 (从 SFT ckpt 加载);
      - 顶层 ``value_function`` 是新建的 (随机初始化), 必须靠预训练暖身.

    接入步骤:
      1. 预训练阶段: 把 value head 单独训好, 存为 ``value_head.pt``;
      2. critic 初始化后 (load SFT ckpt), 再调本函数把 value_head.pt 注入;
      3. 关键代码位置: ``MegatronTrainRayActor.init`` 加载完 model 后,
         在 ``_reinitialize_critic_output_layer`` 之外手动 load.

    Args:
        megatron_critic_model:   slime 的 critic (Megatron DDP 包装).
        pretrained_value_head_state_dict: state_dict, key 应包含 ``value_function``.
        strict:  是否严格匹配 key (一般 False, 因为 backbone 不在这里覆盖).
    """
    # 在 Megatron DDP model 里, value_function 层的命名形如:
    #   module.module.decoder.layers.{i}.mlp.value_function.weight
    #   或 PostProcess.value_function.weight
    # 这里做 partial load: 只覆盖名字含 'value_function' 的参数
    critic_state = megatron_critic_model.state_dict()
    loaded_keys = []
    for k, v in pretrained_value_head_state_dict.items():
        # 找到 critic_state 里对应的 key (允许前缀不同)
        matched = [
            ck for ck in critic_state
            if ck.endswith(k) or k.endswith(ck)
        ]
        if matched:
            target_key = matched[0]
            if critic_state[target_key].shape == v.shape:
                critic_state[target_key].copy_(v)
                loaded_keys.append(target_key)

    megatron_critic_model.load_state_dict(critic_state, strict=False)
    print(f"[value pretrain] loaded {len(loaded_keys)} keys: {loaded_keys[:5]}...")


# =========================================================================
# 单测 / 演示
# =========================================================================
def _demo():
    print("=" * 60)
    print("Value Pretraining: 把 critic 的 value head 暖身起来")
    print("=" * 60)

    torch.manual_seed(0)
    # 构造 toy 数据: states ~ N(0,1), returns = non-linear function
    N = 1024
    states = torch.randn(N, 1)
    returns = states.squeeze() * 0.7 + 0.3 * torch.sin(states.squeeze() * 3)
    dataset = StateReturnDataset(states, returns.unsqueeze(-1))

    # 用 3 个不同初始 seed 跑同一个数据, 看 EV 是否上升
    print("\n预训练 8 epochs, 观察 explained variance 变化:")
    for seed in (1, 2):
        torch.manual_seed(seed)
        vh = ToyValueHead(hidden_dim=64)
        m = pretrain_value_loop(
            vh, dataset, epochs=8, batch_size=64,
            lr=1e-2, value_clip=0.0,  # 预训练阶段 clip=0
        )
        ev_start, ev_end = m["ev_history"][0], m["ev_history"][-1]
        print(f"  seed={seed}: EV {ev_start:+.3f} → {ev_end:+.3f}")
        assert ev_end > ev_start, f"seed={seed} EV 应该上升"

    print("\n✅ Value pretrain 单测通过 (EV 单调上升)")
    print('\n注意: 真实规模下 EV 达到 0.8+ 才算 "good initialization" (论文 Figure 4a)')


if __name__ == "__main__":
    _demo()
