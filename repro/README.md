# SAO 论文复现 — 基于现有资源的实操路径

## 现状清单

| 资源 | 路径 | 状态 |
|---|---|---|
| AIME2025 评测集 | `/home/wzk/datasets/AIME2025/` | ✅ 30 题 (I + II 各 15 题) |
| Qwen3-30B-A3B 基础模型 | `/home/wzk/models/Qwen3-30B-A3B/` | ✅ 60GB, MoE 48 层 128 expert |
| GPU | 1 × NVIDIA GB10 (workstation) | ⚠️ 单卡 |

## 与论文 SAO 实验的差距

1. **模型版本**: 论文用 `Qwen3-30B-A3B-Thinking-2507` + GPT-OSS-120B 蒸馏 SFT 起步;
   你只有 `Qwen3-30B-A3B` 基础版 (无 thinking 训练, 未 SFT).
2. **训练数据**: 论文 TIR 数据由 GPT-OSS-120B 蒸馏, **未公开**.
3. **硬件**: 论文 batch=128, 128k context, 1000+ RL steps, 需 64+ GPU.
   单卡 GB10 跑不动 30B-A3B 的 RL.

## 可行的 3 个复现 Tier

### Tier 0 — Baseline eval (立刻能跑, 本目录已实现)

**目标**: 复现论文 Table 1 的 `Qwen3-30B-A3B w/o python = 85.0` 行.
直接评测基础模型在 AIME2025 上的 pass@1, 不训练.

**步骤**:
```bash
cd /home/wzk/projects/slime
# 1. 数据转换 (question/answer → input/label)
python SAO/repro/01_convert_aime2025.py

# 2. 启动 eval-only job
bash SAO/repro/run_eval_baseline.sh
```

**输出对照**:
| 指标 | 论文 | 你的 baseline (预期) |
|---|---|---|
| Qwen3-30B-A3B w/o python | 85.0 | 50–80 (取决于是否带 thinking) |

> ⚠️ 基础模型不输出 `</think>` 标签, deepscaler RM 会判 0.
> eval_aime2025.yaml 已用 `rm_type: math` (更宽松, 不强制 `</think>`).

---

### Tier 1 — 算法正确性验证 (需小模型 + 开源数据, 单卡可跑)

**目标**: 验证 `SAO/sao/` 5 个算法组件 (DIS / TTUR / frozen-attn / skip-obs GAE /
length-adaptive λ) 在 slime pipeline 上能端到端跑通.

**替代资源**:
| 论文原配 | Tier 1 替代 | 来源 |
|---|---|---|
| Qwen3-30B-A3B-Thinking | Qwen3-1.7B / Qwen3-4B | HF 开源 |
| GPT-OSS 蒸馏 TIR 数据 | dapo-math-17k / numina-math | HF 开源 |
| 128k context | 8k–16k | 单卡显存限制 |
| batch=128, 1000 steps | batch=16, 50 steps | 验证用 |

**步骤** (大致):
1. 下载 `Qwen3-1.7B` + `dapo-math-17k` 到本地
2. 跑 slime 自带的 `examples/fully_async/run-qwen2.5-0.5B-fully_async.sh` 验证 async pipeline
3. 按 `SAO/sao/_06_sao_rollout.py` 的参数清单换上 SAO 配置 (开 DIS, critic_train_epoch=2 等)
4. 跑 50 步看 reward 曲线 + clip_ratio 是否合理
5. 在 AIME2025 上 eval

**预期**: 不复现论文 SOTA, 只验证 SAO 训练能稳定跑完不崩.

---

### Tier 2 — 完整论文复现 (本机不可行)

需要:
- 多节点集群 (≥ 8 × H100 80GB 或等价)
- Qwen3-30B-A3B-Thinking-2507 ckpt
- 自蒸馏 TIR 数据 (用 GPT-OSS-120B / DeepSeek-V3.2 等强模型)
- 128k context 支持

按 `SAO/sao/_06_sao_rollout.py` 的 `RUN_SAO_TIR_MATH_SH` 启动.
本机硬件无法承载, 跳过.

---

## 推荐路径

1. **先跑 Tier 0** (今天就能出结果): 验证 slime pipeline + 数据格式 + reward 计算
   都对齐论文, 拿到一个 baseline 数字.

2. **如果 Tier 0 数字合理** (50–85 区间), 说明评测 pipeline 没问题,
   再考虑是否值得投入 Tier 1 (需要找训练数据 + 下载小模型).

3. **Tier 2** 需要团队级资源, 不在单机复现范围.

## 文件清单

```
SAO/repro/
├── 01_convert_aime2025.py   # AIME2025 数据 → slime eval schema (question→input, answer→label)
├── eval_aime2025.yaml       # eval-config: top-p=1, T=1, max_resp=32k, n=4
└── run_eval_baseline.sh     # eval-only 启动脚本 (num_rollout=0)
```
