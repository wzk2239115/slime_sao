"""组件 3: Frozen-Attention Critic — 论文 §3.2

------------------------------------------------------------------------
为什么冻结 attention
------------------------------------------------------------------------
论文 §3.2 "Stabilizing Value Model Training via Parameter Freezing":

> 我们发现 value model 训练不稳定, 梯度范数显著大于策略模型.
> 进一步分解发现: 不稳定主要来自 Full Attention 层, MoE 层相对稳定.
> 因此对 V_φ 采用 "Frozen-Attention" 策略: 冻结 attention 参数, 只优化
> MoE 投影.

Figure 4(b) 对比了 frozen-attention 和 full-parameter 两种配置的 critic
gradient norm, frozen 版明显更平滑.

------------------------------------------------------------------------
slime 现状
------------------------------------------------------------------------
slime 已经有 ``--freeze-params-name-list`` (regex) 和
``--only-train-params-name-list``, 实现在
``slime/backends/megatron_utils/model_provider.py:272 freeze_model_params``.

但有两个坑:

  1. ``--freeze-params-name-list`` 是**全局参数**, actor 也会被冻.
     SAO 只想冻 critic, 不动 actor.

  2. slime 通过 ``--megatron-config-path`` 支持 role-tagged YAML
     (``slime/utils/arguments.py:parse_megatron_role_args``), 可以给
     critic 单独传 freeze list. 这是推荐路径.

------------------------------------------------------------------------
本文件提供
------------------------------------------------------------------------
1. ``freeze_critic_attention``: 通用的「按 regex 冻结参数」工具函数,
   对任何 nn.Module 都能用, 方便在 toy 模型上观察效果.

2. ``verify_frozen_params``: 打印各层 requires_grad 统计, 用于调试.

3. ``CRITIC_MEGATRON_CONFIG_YAML``: slime megatron-config-path 的示例
   配置文本, 直接 ``--megatron-config-path <file>`` 即可启用.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn


# =========================================================================
# 通用工具: 按 regex 冻结参数
# =========================================================================
def freeze_critic_attention(
    model: nn.Module,
    attention_patterns: Iterable[str] = ("self_attention", "attention"),
) -> dict[str, int]:
    """冻结名字匹配任一 regex 的参数 (典型: attention 层).

    匹配规则与 slime ``freeze_model_params`` 一致, 使用 ``re.search``
    (子串匹配即可, 不需要 full match).

    Args:
        model:               critic 模型 (nn.Module 或 Megatron DDP 包装的模型)
        attention_patterns:  参数名 regex 列表, 命中任一即冻结.
                             默认覆盖 Megatron 的 ``self_attention`` 子模块名.

    Returns:
        统计 dict: ``{"frozen": N_frozen, "trainable": N_trainable,
                     "frozen_params": total_frozen_param_count}``.
    """
    compiled = [re.compile(p) for p in attention_patterns]
    frozen, trainable, frozen_params = 0, 0, 0
    for name, param in model.named_parameters():
        if any(p.search(name) for p in compiled):
            param.requires_grad = False
            frozen += 1
            frozen_params += param.numel()
        else:
            # 显式置 True, 避免继承之前的 requires_grad 状态
            param.requires_grad = True
            trainable += 1
    return {"frozen": frozen, "trainable": trainable, "frozen_params": frozen_params}


def verify_frozen_params(model: nn.Module, max_print: int = 20) -> None:
    """打印模型各层 requires_grad 状态, 用于调试 freeze 是否生效.

    每个 parameter 显示一行: ``[FROZEN|TRAIN] name (shape, numels)``.
    只打印前 ``max_print`` 行, 避免巨型模型刷屏.
    """
    n_shown = 0
    for name, param in model.named_parameters():
        tag = "FROZEN" if not param.requires_grad else "TRAIN"
        if n_shown < max_print:
            print(f"  [{tag}] {name}  shape={tuple(param.shape)}  numels={param.numel()}")
            n_shown += 1
    if n_shown == max_print:
        rest = sum(1 for _ in model.named_parameters()) - max_print
        print(f"  ... ({rest} more params not shown)")

    n_frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    n_total = sum(1 for _ in model.named_parameters())
    print(f"  Summary: {n_frozen}/{n_total} params frozen ({n_frozen / max(n_total, 1):.1%})")


# =========================================================================
# slime 接入: megatron-config-path YAML 示例
# =========================================================================
CRITIC_MEGATRON_CONFIG_YAML = """\
# SAO Frozen-Attention critic 的 megatron-config-path YAML 示例
#
# 用法:
#   --megatron-config-path /path/to/this_file.yaml
#
# slime 会根据 role=actor / role=critic 选取对应配置覆盖 args.
# 关键是给 critic 单独配 freeze_params_name_list, 不影响 actor.

megatron:
  - role: actor
    # actor 全参数训练, 不冻结
    freeze_params_name_list: null

  - role: critic
    # 只训 MoE 专家 + value head, 冻结 attention + embedding + output_layer
    # 正则按子串匹配 (re.search), 与 slime freeze_model_params 一致
    freeze_params_name_list:
      - self_attention          # Megatron 通用 attention 子模块
      - attention               # 万一 model 用了别的命名
      - embedding               # word embedding
      - output_layer            # 注意: critic 自己的 value head 不在这里
                               # value head 在 PostProcess (VHead), 不冻结
    # 可选: 用 only-train 更精确地白名单
    # only_train_params_name_list:
    #   - experts
    #   - value_function        # critic 专属 output layer 名 (视具体模型)

    # SAO 论文: critic lr = 5e-6, actor lr = 1e-6
    lr: 5.0e-6
    # critic warmup 10 步 (论文 §4.1)
    lr_warmup_iters: 10

    # TTUR: 每 actor 步训 2 次 critic (需要在 train_critic 里循环)
    critic_train_epoch: 2
"""


# =========================================================================
# 单测 / 演示
# =========================================================================
class _ToyMoEBlock(nn.Module):
    """模拟一个 attention + moe expert 的 transformer 层."""

    def __init__(self, hidden_dim: int = 16, n_experts: int = 4):
        super().__init__()
        # attention 部分 (论文要求冻结)
        self.self_attention = nn.ModuleDict({
            "q_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
            "k_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
            "v_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
            "output": nn.Linear(hidden_dim, hidden_dim, bias=False),
        })
        # MoE 专家 (论文要求保留训练)
        self.moe_experts = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False) for _ in range(n_experts)
        ])
        # value head (critic 专属, 不能冻)
        self.value_function = nn.Linear(hidden_dim, 1, bias=True)


def _demo():
    print("=" * 60)
    print("Frozen-Attention: 冻结 attention, 保留 MoE + value head")
    print("=" * 60)

    torch.manual_seed(0)
    model = _ToyMoEBlock(hidden_dim=16, n_experts=4)

    print("\n[冻结前] 所有参数 requires_grad 状态:")
    verify_frozen_params(model, max_print=8)

    stats = freeze_critic_attention(model, attention_patterns=("self_attention",))
    print(f"\n冻结统计: {stats}")

    print("\n[冻结后] 参数状态 (应只看到 self_attention.* 被 FROZEN):")
    verify_frozen_params(model, max_print=10)

    # 断言
    sa_frozen = all(
        not p.requires_grad
        for n, p in model.named_parameters() if "self_attention" in n
    )
    moe_trainable = all(
        p.requires_grad
        for n, p in model.named_parameters() if "moe_experts" in n
    )
    vhead_trainable = all(
        p.requires_grad
        for n, p in model.named_parameters() if "value_function" in n
    )
    assert sa_frozen, "self_attention 参数应该全部被冻结"
    assert moe_trainable, "moe_experts 参数应该保持可训练"
    assert vhead_trainable, "value_function (value head) 不能被冻"
    print("\n✅ Frozen-Attention 单测通过: attention 冻结, MoE/value head 保留")

    # 顺便打印 YAML 示例
    print("\n" + "=" * 60)
    print("slime megatron-config-path YAML 示例 (保存为 .yaml 即可用):")
    print("=" * 60)
    print(CRITIC_MEGATRON_CONFIG_YAML)


if __name__ == "__main__":
    _demo()
