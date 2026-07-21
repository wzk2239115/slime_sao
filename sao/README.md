# SAO 复现组件 (slime)

本目录是论文 **SAO: Single-rollout Asynchronous Optimization for Agentic RL**
(`../SAO.md`) 在 slime 上的分组件实现, 每个组件独立可运行、带单测、附详细注释.

## 快速开始

```bash
cd /path/to/slime

# 逐个组件运行 (按顺序学习推荐)
python -m SAO.sao._01_dis                  # DIS 直接双向重要性采样
python -m SAO.sao._02_faster_value_update  # Faster Value Update (TTUR)
python -m SAO.sao._03_frozen_attention     # Frozen-Attention critic
python -m SAO.sao._04_skip_obs_gae         # Skip-Observation GAE
python -m SAO.sao._05_length_adaptive_gae  # Length-Adaptive GAE
python -m SAO.sao._06_sao_rollout          # 端到端配置示例
python -m SAO.sao._07_value_pretrain       # 价值模型预训练
```

每个脚本不需要 GPU、不需要 slime 的 Megatron 后端, 纯 PyTorch + torch 即可跑.

## 组件 ↔ 论文对照表

| 文件                       | 论文章节   | SAO 设计点                            | slime 接入方式                                                                |
| -------------------------- | ---------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| `_01_dis.py`               | §3.1       | DIS 直接双向重要性采样                | `--custom-tis-function-path SAO.sao._01_dis.dis_tis_function`                 |
| `_02_faster_value_update.py` | §3.2     | Faster Value Update (TTUR, K=2)       | 新增 `--critic-train-epoch`, 改 `MegatronTrainRayActor.train_critic`          |
| `_03_frozen_attention.py`  | §3.2       | Frozen-Attention critic               | `--megatron-config-path` 给 critic 单独配 `freeze_params_name_list`           |
| `_04_skip_obs_gae.py`      | §3.2       | Skip-Observation Token-level GAE      | 新增 `--use-skip-obs-gae`, 改 `compute_advantages_and_returns` 分支           |
| `_05_length_adaptive_gae.py` | §4.1     | Length-Adaptive GAE (per-sample λ)    | 新增 `--gae-alpha`, 改 `vanilla_gae` 支持 per-row λ                           |
| `_06_sao_rollout.py`       | §3 + §4.1  | 端到端 single-rollout async 配置      | `train_async.py` + `--rollout-function-path ...fully_async_rollout...`       |
| `_07_value_pretrain.py`    | §3.2       | Scaling Value Pretraining             | 独立训练脚本 + `load_pretrained_value_head` 注入 critic                       |

## slime 接入 checklist (按优先级)

1. **解除断言** (1 行改动): `slime/utils/arguments.py:1802` 当前禁止
   `use_rollout_logprobs` 与 `use_tis` 同时为 True, DIS 需要两者都开.
   建议新增独立 `--use-dis` 旁路.

2. **DIS**: 直接 `--custom-tis-function-path SAO.sao._01_dis.dis_tis_function`,
   无需改 slime 代码.

3. **Faster Value Update**: 改 `MegatronTrainRayActor.train_critic` 包一层循环
   (详见 `_02_faster_value_update.py` 的 `_slime_integration_pseudocode`).
   新增 `--critic-train-epoch` 参数.

4. **Frozen-Attention**: 把 `_03_frozen_attention.py` 末尾的 YAML 存为
   `sao_critic_config.yaml`, 用 `--megatron-config-path` 引用.

5. **Skip-Obs GAE**: 把 `_04_skip_obs_gae.py:skip_obs_gae` 接到
   `compute_advantages_and_returns` 的 PPO 分支 (根据 `args.use_skip_obs_gae` 选择).

6. **Length-Adaptive GAE**: 修改 `vanilla_gae` / `chunked_gae` 支持接收
   `lambd: Tensor[B]` (per-row λ), 在 `get_advantages_and_returns_batch`
   里按 `args.gae_alpha` 算 λ.

7. **Value Pretrain**: 用组件 7 的 `pretrain_value_loop` 在 RL 训练前独立训 critic,
   再用 `load_pretrained_value_head` 注入 Megatron critic.

## 端到端启动

参考 `_06_sao_rollout.py` 里 `RUN_SAO_TIR_MATH_SH` / `RUN_SAO_CODING_SH` 的模板,
替换模型路径即可.

## 与论文超参对照 (TIR 数学任务)

| 超参              | 论文取值   | slime CLI                                                                 |
| ----------------- | ---------- | ------------------------------------------------------------------------- |
| batch size        | 128        | `--rollout-batch-size 128 --global-batch-size 128`                        |
| group size        | 1          | `--n-samples-per-prompt 1`                                                |
| max length        | 128k       | `--rollout-max-context-len 131072`                                        |
| actor lr          | 1e-6       | `--lr 1e-6`                                                               |
| critic lr         | 5e-6       | megatron-config critic.role.lr: 5e-6                                      |
| ε_low / ε_high    | 0.3 / 5.0  | `--tis-clip-low 0.7 --tis-clip 6.0` (等价于 ε_l/ε_h)                      |
| α (length GAE)    | 1.5        | `--gae-alpha 1.5`                                                         |
| λ_critic          | 1          | `--lambd 1.0`                                                             |
| critic warmup     | 10 step    | `--num-critic-only-steps 10`                                              |
| K (TTUR)          | 2          | `--critic-train-epoch 2`                                                  |

coding agent 任务只改 ε: `--tis-clip-low 0.2 --tis-clip 4.0` (即 ε_l=0.8, ε_h=3.0).
