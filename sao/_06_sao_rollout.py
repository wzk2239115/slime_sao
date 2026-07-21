"""组件 6: 端到端 SAO single-rollout async rollout 配置示例

------------------------------------------------------------------------
本文件做什么
------------------------------------------------------------------------
不是新算法, 而是把前面 5 个组件串起来, 给出 slime 命令行参数 + 一个
custom_rollout_function 的范例, 让用户能直接拿来跑 (替换模型路径后).

SAO 的 single-rollout 体现在两点:
  1. ``--n-samples-per-prompt 1``       (slime 默认就是 1)
  2. ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``
     这条路径会持续生成轨迹、来一条训一条, 不等 group 凑齐.

------------------------------------------------------------------------
文件内容
------------------------------------------------------------------------
1. ``SAO_ROLLOUT_ARGS``        : slime CLI 参数清单 (核心)
2. ``build_sao_rollout_args``  : 把参数打包成 dict, 方便脚本化
3. ``run_sao_tir_math_sh``     : TIR 数学任务启动脚本模板
4. ``run_sao_coding_sh``       : SWE-Bench coding 任务启动脚本模板
"""

from __future__ import annotations


# =========================================================================
# slime CLI 参数清单 (核心)
# =========================================================================
SAO_ROLLOUT_ARGS = {
    # ---------- 数据 & 采样 ----------
    "n_samples_per_prompt": 1,            # ✅ SAO 核心: 单 rollout
    "rollout_batch_size": 128,            # 论文 §4.1 batch=128
    "global_batch_size": 128,             # 每 rollout 训 1 个 actor step
    "num_steps_per_rollout": 1,
    "rollout_global_dataset": True,       # 全局 dataset, 配合 fully_async
    "disable_grpo_std_normalization": True,  # n=1 时 GRPO 标准化无意义

    # ---------- 异步管线 ----------
    "rollout_function_path": "slime.rollout.fully_async_rollout.generate_rollout_fully_async",

    # ---------- 算法核心: PPO + critic (SAO 是 value-based) ----------
    "advantage_estimator": "ppo",         # 触发 use_critic=True
    "use_critic": True,                   # 等价于 advantage_estimator=ppo

    # ---------- DIS: 直接双向重要性采样 (组件 1) ----------
    # 注意: slime 当前 arguments.py:1802 断言 use_rollout_logprobs 和 use_tis
    # 互斥, 接入 DIS 前需先放开此断言 (详见 TODO §2.2)
    "use_rollout_logprobs": True,
    "use_tis": True,
    "custom_tis_function_path": "SAO.sao._01_dis.dis_tis_function",
    # TIR 数学任务: ε_l=0.3, ε_h=5.0 → ratio 信任域 [0.7, 6.0]
    # coding 任务: ε_l=0.8, ε_h=3.0 → ratio 信任域 [0.2, 4.0]
    "tis_clip_low": 0.7,
    "tis_clip":     6.0,

    # ---------- Optimizer ----------
    # actor: lr=1e-6;  critic: lr=5e-6 (通过 megatron-config-path 单独配)
    "lr": 1.0e-6,
    "lr_decay_style": "constant",
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,

    # ---------- GAE ----------
    "gamma": 1.0,
    # length-adaptive λ: 接入组件 5 后启用, 否则用标量 lambd
    "lambd": 1.0,             # λ_critic=1 (论文)
    "gae_alpha": 1.5,         # length-adaptive: λ_policy = 1 - 1/(α·L); 接入组件 5 后启用

    # ---------- Critic (组件 2/3) ----------
    "num_critic_only_steps": 10,    # 论文 §4.1 critic warmup 10 步
    "critic_train_epoch": 2,        # 组件 2: TTUR K=2 (需要改 train_critic)
    "value_clip": 0.2,
    "megatron_config_path": "/path/to/sao_critic_config.yaml",  # 组件 3: frozen-attn

    # ---------- 上下文长度 (agentic 必备) ----------
    "rollout_max_context_len": 131072,  # 128k
    "rollout_max_response_len": 32768,
    "rollout_temperature": 1.0,

    # ---------- KL 关掉 ----------
    # SAO 用 DIS 做硬 mask, 不需要 KL 软惩罚
    "kl_coef": 0.0,
    "use_kl_loss": False,
}


def build_sao_rollout_args(task: str = "tir_math") -> dict:
    """根据任务类型返回 SAO 推荐参数.

    Args:
        task: "tir_math"  → TIR 数学 (论文 Table 1)
             "coding"     → SWE-Bench coding agent (论文 Table 2)

    Returns:
        dict, 可直接展开成 slime CLI flags (key --kebab-case).
    """
    base = dict(SAO_ROLLOUT_ARGS)

    if task == "tir_math":
        # 论文 §4.1: ε_low=0.3, ε_high=5.0
        base.update({
            "tis_clip_low": 0.7,    # 1 - 0.3
            "tis_clip":     6.0,    # 1 + 5.0
        })
    elif task == "coding":
        # 论文 §4.1 coding agent: ε_low=0.8, ε_high=3.0
        base.update({
            "tis_clip_low": 0.2,    # 1 - 0.8
            "tis_clip":     4.0,    # 1 + 3.0
        })
    else:
        raise ValueError(f"unknown task: {task}")
    return base


# =========================================================================
# 启动脚本模板
# =========================================================================
RUN_SAO_TIR_MATH_SH = """\
#!/bin/bash
# SAO TIR 数学任务启动脚本 (Qwen3-30B-A3B-Thinking)
# 参考 examples/fully_async/run-qwen2.5-0.5B-fully_async.sh

set -ex
export PYTHONUNBUFFERED=1

# 路径 — 按实际环境改
HF_CHECKPOINT=${HF_CHECKPOINT:-/path/to/Qwen3-30B-A3B-Thinking}
REF_MODEL_PATH=${REF_MODEL_PATH:-/path/to/Qwen3-30B-A3B-Thinking_torch_dist}
CRITIC_LOAD=${CRITIC_LOAD:-/path/to/value_model_pretrained}    # 组件 7
PROMPT_DATA=${PROMPT_DATA:-/path/to/tir_math_train.jsonl}

# critic frozen-attention YAML (组件 3 的 CRITIC_MEGATRON_CONFIG_YAML)
# 写到本地后用 --megatron-config-path 引用
MEGATRON_CONFIG=/path/to/sao_critic_config.yaml

# Ray 启动
NUM_GPUS=${NUM_GPUS:-64}
ray start --head --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \\
    --num-gpus "${NUM_GPUS}" --disable-usage-stats

# SAO 训练入口: 用 train_async.py (异步) 而不是 train.py
ray job submit --address="http://127.0.0.1:8265" \\
    -- python3 train_async.py \\
    --actor-num-nodes ${ACTOR_NUM_NODES:-8} \\
    --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE:-8} \\
    --rollout-num-gpus ${ROLLOUT_GPUS:-32} \\
    --hf-checkpoint "${HF_CHECKPOINT}" \\
    --ref-load "${REF_MODEL_PATH}" \\
    --megatron-config-path "${MEGATRON_CONFIG}" \\
    --prompt-data "${PROMPT_DATA}" \\
    --input-key prompt --label-key label \\
    --apply-chat-template \\
    --rollout-shuffle \\
    --rm-type deepscaler \\
    --num-rollout 1000 \\
    --rollout-batch-size 128 \\
    --n-samples-per-prompt 1 \\
    --global-batch-size 128 \\
    --num-steps-per-rollout 1 \\
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \\
    --rollout-global-dataset \\
    --rollout-max-context-len 131072 \\
    --rollout-max-response-len 32768 \\
    --rollout-temperature 1.0 \\
    --advantage-estimator ppo \\
    --use-rollout-logprobs \\
    --use-tis \\
    --custom-tis-function-path SAO.sao._01_dis.dis_tis_function \\
    --tis-clip-low 0.7 --tis-clip 6.0 \\
    --eps-clip 0.3 --eps-clip-high 5.0 \\
    --gamma 1.0 --lambd 1.0 \\
    --num-critic-only-steps 10 \\
    --critic-train-epoch 2 \\
    --value-clip 0.2 \\
    --kl-coef 0.0 \\
    --lr 1e-6 \\
    --lr-decay-style constant \\
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \\
    --save /path/to/sao_ckpt/ --save-interval 50
"""

RUN_SAO_CODING_SH = """\
#!/bin/bash
# SAO coding agent 启动脚本 (Qwen3-30B-A3B + OpenHands scaffold)
# 参考 examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh

set -ex

# 与 TIR 脚本的差异:
#   1. custom_generate_function 走 OpenHands (需写 harness, slime 现有 claude_code/codex)
#   2. ε_low=0.8, ε_high=3.0 (论文 §4.1 coding)
#   3. context 128k, interaction 300 turn
#   4. reward 走 SWE-bench test pass (examples/coding_agent_rl/swe.py)

# ε_low=0.8 → tis_clip_low=0.2 ;  ε_high=3.0 → tis_clip=4.0
# --tis-clip-low 0.2 --tis-clip 4.0
# --eps-clip 0.8 --eps-clip-high 3.0

# 其余参数与 TIR 一致, 把 PROMPT_DATA 改成 SWE-bench 训练集,
# custom-generate-function-path 指向 OpenHands adapter
echo "详见 RUN_SAO_TIR_MATH_SH, 仅修改上述差异项即可"
"""


# =========================================================================
# 演示
# =========================================================================
def _demo():
    print("=" * 60)
    print("SAO 端到端 single-rollout async 参数清单")
    print("=" * 60)

    for task in ("tir_math", "coding"):
        args = build_sao_rollout_args(task)
        print(f"\n--- task = {task} ---")
        for k, v in args.items():
            print(f"  --{k.replace('_', '-')}  {v}")

    print("\n" + "=" * 60)
    print("启动脚本 (RUN_SAO_TIR_MATH_SH 节选)")
    print("=" * 60)
    # 只打印前 25 行避免刷屏
    lines = RUN_SAO_TIR_MATH_SH.split("\n")
    print("\n".join(lines[:25]))

    print("\n✅ 配置示例生成完毕. 详见 RUN_SAO_TIR_MATH_SH / RUN_SAO_CODING_SH")


if __name__ == "__main__":
    _demo()
