# 用 360 API (GLM-5.2) 蒸馏 SAO 冷启动数据

## 这是什么

用 360 API 提供的 **GLM-5.2** 当教师模型，配合 slime 自带的 `PythonSandbox` 执行
python 代码，蒸馏 **TIR (Tool-Integrated Reasoning) 多轮数学轨迹**，作为 SAO 论文
里 GPT-OSS-120B 蒸馏数据的开源替代。

## 能力验证 (已跑通)

| 项 | 结果 |
|---|---|
| API 连通 | ✅ `https://api.360.cn/v1`, model `z-ai/glm-5.2` |
| 数学推理质量 | ✅ AIME2025 第 1 题答对 (70) |
| Thinking 输出 | ✅ `reasoning_content` + `content` 分离 |
| Tool call | ✅ 标准 OpenAI tools 协议, 支持 `python` 函数 |
| PythonSandbox | ✅ 复用 `examples/retool/tool_sandbox.py`, 本地执行 + 安全过滤 |
| 端到端 1 条 | ✅ **53 秒**，完整 trajectory，答案正确 |

## 单条耗时分解

实测 1 条 AIME 题（2 个 assistant turn + 1 个 tool call）= **~53 秒**.
并发 5 会 timeout, **建议并发 2-3**.

## 数据规模估算 (基于 53s/条, 并发=2)

| 目标 | 题数 | 预计耗时 | 适用场景 |
|---|---|---|---|
| Sanity check | 30 (AIME2025) | ~13 min | 验证 pipeline + 看数据格式 |
| 小规模 SFT | 500 | ~3.5 hr | Tier 1 复现起步 |
| 中规模 SFT | 2 000 | ~14 hr | 论文级 SFT 起步 |
| 大规模 SFT | 10 000+ | ~70 hr+ | 完整论文复现 |

> 论文 §4.1 没公开蒸馏数据规模，只说「3 epoch on TIR data」.
> batch=128, 几千条 trajectory 是合理估计.

## 用法

### 1. 设置 API key

```bash
# 从 ~/.config/opencode/opencode.json 拷贝
export API_360_KEY="<your-360-api-key>"
```

如果不设环境变量, 脚本会自动从 `~/.config/opencode/opencode.json` 读.

### 2. 准备输入数据

slime 格式 jsonl, 每行 `{"input": "题目", "label": "标准答案"}`.
AIME2025 已经被 `SAO/repro/01_convert_aime2025.py` 转好了:

```
/home/wzk/datasets/AIME2025/slime/aime2025-all.jsonl   # 30 题
```

要扩大规模, 可以下载开源数学题集:
- `dapo-math-17k` (slime examples 已经在用)
- `numina-math` / `MATH` / `Big-Math-RL-Verified`

### 3. 跑蒸馏

```bash
cd /home/wzk/projects/slime

# 跑 AIME2025 30 题, sanity check
python SAO/repro/distill/distill_tir.py \
    --src /home/wzk/datasets/AIME2025/slime/aime2025-all.jsonl \
    --dst /home/wzk/datasets/sao_sft/aime2025_distilled.jsonl \
    --concurrency 2 --max-turns 8

# 跑 dapo-math-17k, 前 500 条
python SAO/repro/distill/distill_tir.py \
    --src /path/to/dapo-math-17k.jsonl \
    --dst /home/wzk/datasets/sao_sft/dapo_distilled.jsonl \
    --concurrency 2 --max-turns 8 --max-samples 500
```

### 4. 用产出做 SFT

蒸馏输出已经是 slime multi-turn SFT 格式 (`messages` 字段).
直接用 slime 的 `sft_rollout` 路径:

```bash
python3 train.py \
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
    --prompt-data /home/wzk/datasets/sao_sft/dapo_distilled.jsonl \
    --input-key messages \
    --loss-mask-type qwen3 \
    --loss-type sft_loss \
    ...
```

## 文件

```
SAO/repro/distill/
├── distill_tir.py    # 主脚本: 360 API + slime PythonSandbox
└── README.md         # 本文档
```

## 注意事项

1. **API 配额**: 没显式 rate limit header, 但 5 并发会 timeout. 已用 `concurrency=2`
   + 指数退避重试 (3 次) 缓解.

2. **sandbox 安全限制**: slime 的 `PythonSandbox` 禁用 `os/sys/subprocess` 等,
   只允许 `math/random/datetime` 等数学相关模块. 复杂场景 (numpy/sympy) 需扩展
   `examples/retool/tool_sandbox.py:PythonSandbox.allowed_modules`.

3. **正确性过滤**: 蒸馏出的 trajectory 不一定全对. 建议跑完后用
   `slime/rollout/rm_hub/deepscaler.py:get_deepscaler_rule_based_reward` 过一遍,
   只保留 answer 正确的样本.

4. **Ckpt 残留**: 中途失败的题目会跳过, 已写出的会保留 (流式 flush).

5. **价格**: 360 API 计费规则查 https://api.360.cn/. 单条 trajectory 平均
   2-4k token output, 蒸馏 1k 条大约 2-4M token, 看计费档位.
