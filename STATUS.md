# SAO 复现: 完整状态文档

## 项目概述
复现论文 SAO (Single-Rollout Asynchronous Optimization, arXiv:2607.07508) 在 AIME2025 上的实验结果。目标: Qwen3-30B-A3B SAO = 97.3% (论文 Table 1)。

## 算力资源
- **5 台机器 × 8 GPU = 40 张 A800-80GB**
- ctm-03: `11.131.248.23` (正在跑 baseline eval)
- ctm-05: `11.131.211.65` (推理机)
- ctm-06: `11.131.215.38` (训练机)
- 共享存储: `/home/jovyan/h800fast/wangzekai/` (所有机器可访问)
- **注意**: 所有机器 glibc 已升级到 2.39, GLIBCXX 3.4.33

## 代码仓库
- **slime_sao**: `https://github.com/wzk2239115/slime_sao.git`
- SAO 算法代码: `/home/wzk/projects/slime/SAO/sao/`
- 复现脚本: `/home/wzk/projects/slime/SAO/sao/standalone/`
- 本地路径: `/home/wzk/projects/slime/SAO/` (git repo, push 到 slime_sao)
- 算力机路径: `/home/jovyan/h800fast/wangzekai/slime_sao/`

## 环境配置 (已完成)
1. **glibc 升级**: 2.35 → 2.39 (从 rootfs 复制, 原子替换)
   - 脚本: `repro/upgrade_glibc.sh`
   - 备份: `/opt/glibc235-backup/`
2. **rootfs**: `slime_latest.tar` 已解压到 `/home/jovyan/h800fast/wangzekai/slime_rootfs/`
   - 包含: torch 2.9.1+cu129, TE 2.10.0, megatron 0.16, sglang, ray 2.53
   - python3 symlink 修复: `rootfs/usr/bin/python3 → python3.12`
3. **新机器初始化**: 只需跑 `repro/upgrade_glibc.sh` (rootfs 共享, 无需重新解压)

## SAO 算法实现 (standalone, 不依赖 slime/megatron)

### 架构: 推理训练分离 + 异步队列
```
推理机 (sglang)          训练机 (HF transformers)
    │                          │
    ├─ sglang_daemon.sh        ├─ trainer.py
    │   (管理 sglang 生命周期)  │   (actor DIS + critic GAE+TTUR)
    │                          │
    ├─ rollout_worker.py       │
    │   (持续生成 n=1 轨迹)     │
    │        │                 │
    │        ▼                 ▼
    │   queue/pending/    ←→  poll_queue()
    │   traj_XXXXXX.json       │
    │                          │
    │   .reload_signal    ←→  save checkpoint
    │   .reload_done            │
```

### 文件清单
| 文件 | 功能 | 论文对应 |
|---|---|---|
| `standalone/sglang_server.py` | sglang 启停 + HTTP 工具 | - |
| `standalone/rollout.py` | sglang HTTP API (生成 + log-probs) | - |
| `standalone/reward.py` | math reward (extract \boxed{}) | - |
| `standalone/grpo_step.py` | DIS policy loss + GRPO loss | §3.1 Eq.1-3 |
| `standalone/critic.py` | ValueModel + GAE + TTUR K=2 | §3.2 |
| `standalone/rollout_worker.py` | 异步 rollout 生成器 (持续写 queue) | §3.2 single rollout |
| `standalone/trainer.py` | 异步 trainer (actor+critic) | 完整 SAO |
| `standalone/value_pretrain.py` | critic 预训练 (cold start) | §3.2 value pretraining |
| `standalone/eval.py` | 独立 eval (sglang HTTP) | - |
| `standalone/sglang_daemon.sh` | sglang 生命周期管理 + auto-reload | - |
| `standalone/run_sao.sh` | 统一启动器 | - |
| `standalone/run_eval.sh` | eval 启动脚本 | - |

### 论文参数对齐 (trainer.py 默认值)
| 参数 | 代码默认值 | 论文值 | 论文出处 |
|---|---|---|---|
| batch_size | 128 | 128 | §4.1 |
| n_samples (single rollout) | 1 | 1 | §3.2 |
| actor lr | 1e-6 | 1×10⁻⁶ | §4.1 |
| critic lr | 5e-6 | 5×10⁻⁶ | §4.1 |
| DIS clip_low | 0.7 | 1-ε_l=1-0.3 | §3.1 |
| DIS clip_high | 6.0 | 1+ε_h=1+5.0 | §3.1 |
| GAE γ | 1.0 | 1.0 | §4.1 |
| λ_policy | 1-1/(1.5·L) | α=1.5 | §4.1 |
| λ_critic | 1.0 | 1.0 | §4.1 |
| TTUR K | 2 | 2 | §3.2 |
| critic warmup | 10 steps | 10 | §4.1 |
| value_clip | 0.2 | - | §4.1 |
| frozen attention | ✅ | ✅ | §3.2 |
| num_steps | 1000 | ~1000 | §4.2 |

### SAO 7 个算法组件 (sao/ 目录, 早期 slime 插件版本)
| 文件 | 组件 | 论文 | 状态 |
|---|---|---|---|
| `_01_dis.py` | DIS 双向裁剪 | §3.1 Eq.1-3 | ✅ 已实现 + 单测 |
| `_02_faster_value_update.py` | TTUR K=2 | §3.2 | ✅ 已实现 + 单测 |
| `_03_frozen_attention.py` | 冻结 critic attention | §3.2 | ✅ 已实现 + 单测 |
| `_04_skip_obs_gae.py` | Skip-Observation GAE | §3.2 Eq.4-5 | ✅ 已实现 |
| `_05_length_adaptive_gae.py` | Length-Adaptive λ | §4.1 | ✅ 已实现 + 单测 |
| `_06_sao_rollout.py` | 端到端参数配置 | §4.1 | ✅ 配置模板 |
| `_07_value_pretrain.py` | Value 预训练 | §3.2 | ✅ 已实现 |

## 启动方式

### Step 0 (可选): Value 预训练
```bash
# 训练机上
bash sao/standalone/run_sao.sh pretrain sao_v1
```

### Step 1: 推理机 (sglang + rollout worker)
```bash
bash sao/standalone/run_sao.sh all sao_v1
```

### Step 2: 训练机 (async trainer)
```bash
bash sao/standalone/run_sao.sh train sao_v1
```

### Eval
```bash
bash sao/standalone/run_eval.sh baseline_full
```

## 当前状态

### ✅ 已完成
- 环境搭建 (glibc 2.39, rootfs, python, proxy bypass)
- sglang 独立推理 (--disable-cuda-graph 绕过 torch_memory_saver)
- Baseline eval 跑通 (ctm-03 正在跑 32k token 版, 之前 16k 版 = 46.7%, 确认是截断导致)
- 完整 SAO 算法实现 (DIS + GAE + TTUR + frozen attention + async pipeline)
- Value pretraining 脚本
- 多机推理训练分离架构 (ctm-05 推理 + ctm-06 训练)
- GPU 自动检测 (TP 自动适配 4/8 卡)

### ⏳ 进行中
- ctm-03: baseline eval (32k token, 预计 15-20h)
- ctm-05/ctm-06: 可立即启动 SAO 训练

### ❌ 待完成
- SFT/TIR 数据准备 (论文要求先用 GPT-OSS-120B 生成的 TIR 数据做 3 epochs SFT)
- 多机 FSDP 训练 (当前单机 device_map="auto", 需扩展到 torchrun + DDP)
- 128k context 支持 (当前 32k, 受显存限制)
- Scale batch_size 到 128 (需要多机训练)

## 重要注意事项
1. **Squid Proxy**: 算力机有公司 HTTP 代理, localhost 请求会被拦截。所有脚本已设 `no_proxy="*""`
2. **CUDA Graph**: sglang 必须用 `--disable-cuda-graph` (否则触发 torch_memory_saver 的 hook_mode 断言)
3. **推理速度**: 禁用 CUDA graph 后约 20-40 token/s (正常应 100+), 不影响正确性
4. **content=None**: Qwen3 Thinking 模型可能把回答放在 `reasoning_content` 字段, 代码已做 fallback
5. **rootfs python3**: symlink `→ /etc/alternatives/python3` 曾断链, 已修复为 `→ python3.12` (相对路径)
6. **checkpoint reload**: trainer 存 checkpoint → 写 `.reload_signal` → daemon kill+restart sglang → 写 `.reload_done` → rollout worker 继续

## 论文结果对照
| 方法 | AIME2025 | BeyondAIME | 说明 |
|---|---|---|---|
| Qwen3-30B-A3B w/o python | 85.0 | 63.0 | baseline (eval 目标) |
| GRPO (w/ DIS) | 93.5 | 70.8 | GRPO + DIS clip |
| Running mean baseline | 79.8 | 55.3 | 无 critic 的 SAO |
| **SAO (ours)** | **97.3** | **74.8** | **完整复现目标** |
| SAO (w/ DIS only) | 94.2 | 71.5 | DIS 但无 value model |

## 数据集
- **AIME2025 eval**: `/home/jovyan/h800fast/wangzekai/slime_sao/datasets/AIME2025/slime/aime2025-all.jsonl` (30 题)
- **AIME2025 train**: 同上 (当前用作 RL 训练 prompt)
- **TIR distillation**: 之前验证过 23/30 正确的 TIR 轨迹 (可用于 value pretraining + SFT)
- **eval 结果**: `datasets/eval_results/{tag}/results.jsonl` + `wrong.jsonl`

## Git 工作流
```bash
# 本地开发 (开发机)
cd /home/wzk/projects/slime/SAO
git add -A && git commit -m "msg" && git push

# 算力机同步
cd /home/jovyan/h800fast/wangzekai/slime_sao && git pull
```
