#!/bin/bash
# SAO 论文复现 — Tier 0: Qwen3-30B-A3B baseline eval on AIME2025
#
# 目标: 复现论文 Table 1 的 "Qwen3-30B-A3B w/o python" = 85.0 (基础模型 baseline)
#       SAO 论文 Table 1 行: Qwen3-30B-A3B SAO (ours) = 97.3
#                            - SAO (w/ DIS only)         = 94.2
#                            - GRPO (+ DIS)              = 93.5
#
# 这个脚本不训练, 只做 eval-only:
#   - 加载 Qwen3-30B-A3B 基础模型到 sglang
#   - 在 AIME2025 (30 题) 上跑 n_samples_per_eval_prompt 次推理
#   - 用 deepscaler/math rule-based reward 评分, 报 pass@1
#
# slime 的 eval-only 入口 (train.py:36):
#   if args.num_rollout == 0 and args.eval_interval is not None:
#       ray.get(rollout_manager.eval.remote(rollout_id=0))
#
# --------------------------------------------------------------------
# 硬件要求
# --------------------------------------------------------------------
# Qwen3-30B-A3B 权重 60GB (16 个 safetensors, 每个 ~4GB).
# bf16 全量加载需 ~60GB 显存. 1 张 GB10 (128GB unified memory) 应该能放下.
# 如果显存不够, 把 ROLLOUT_TP_SIZE 调大 (多卡 TP), 或用 FP8 量化版.
#
# --------------------------------------------------------------------
# 运行前必须做的事
# --------------------------------------------------------------------
# 1. 确认模型路径:
#      /home/wzk/models/Qwen3-30B-A3B/  (config.json + *.safetensors + tokenizer)
#
# 2. 转换数据 (已自动跑过, 这里再确认):
#      python SAO/repro/01_convert_aime2025.py
#    产出: /home/wzk/datasets/AIME2025/slime/aime2025-{I,II,all}.jsonl
#
# 3. 确认 slime 的 sglang 后端能跑 Qwen3-30B-A3B (MoE 128 expert).
#    slime 已内置 qwen3-30B-A3B 配置: scripts/models/qwen3-30B-A3B.sh

set -ex
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ============ 路径 (按需修改) ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/home/wzk/models/Qwen3-30B-A3B}"
# slime 需要一个 megatron torch_dist 格式的权重用于训练侧;
# eval-only 模式 (num_rollout=0) 实际不训练, 可以指向同一个 HF ckpt 作占位.
REF_LOAD="${REF_LOAD:-${HF_CHECKPOINT}}"
EVAL_CONFIG="${SCRIPT_DIR}/eval_aime2025.yaml"

# ============ 模型架构 (来自 scripts/models/qwen3-30B-A3B.sh) ============
source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

# ============ 并行度 (按 GPU 数量调) ============
# 1 张 GB10: TP=1 试试看; 显存不够就改 TP=2 多卡
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   # eval-only 模式不需要 save, 但 slime args 校验需要这两个
   --save /tmp/sao_eval_dummy
   --save-interval 999999
)

ROLLOUT_ARGS=(
   # eval-only 时 prompt-data 可以是任意 jsonl, 不会真正训练;
   # 但 slime 要求有合法路径, 这里用 AIME 全集占位
   --prompt-data /home/wzk/datasets/AIME2025/slime/aime2025-all.jsonl
   --input-key input
   --label-key label
   --apply-chat-template

   --rm-type math                    # 与 eval_config 一致
   --num-rollout 0                   # ✅ 关键: 不训练, 只 eval
   --rollout-batch-size 1
   --n-samples-per-prompt 1
   --rollout-max-response-len 32768
   --rollout-temperature 1.0
   --rollout-top-p 1.0
)

EVAL_ARGS=(
   --eval-interval 1                 # 触发 eval (train.py:36 的判断需要这个)
   --eval-config "${EVAL_CONFIG}"
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size ${ETP_SIZE}
)

GRPO_ARGS=(
   # eval-only 不训练, advantage-estimator 任意; 用 grpo 默认即可
   --advantage-estimator grpo
   --eps-clip 0.2 --eps-clip-high 0.28
   --kl-coef 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant
   --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP_SIZE}
   --sglang-mem-fraction-static 0.85
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============ Ray 启动 ============
NUM_GPUS="${NUM_GPUS:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# 清理残留进程
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

ray start --head --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_GPUS}" --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

# ============ 提交 eval-only job ============
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${SCRIPT_DIR}/eval_baseline.log"

echo ""
echo "==================================================================="
echo "Baseline eval 完成. 关注日志里的:"
echo "  - eval/aime2025_all/reward (整体 pass@1)"
echo "  - eval/aime2025_I/reward"
echo "  - eval/aime2025_II/reward"
echo ""
echo "对照论文 Table 1:"
echo "  Qwen3-30B-A3B w/o python    = 85.0  (论文报的 baseline)"
echo "  Qwen3-30B-A3B SAO (ours)    = 97.3  (完整 SAO 复现目标)"
echo "  - SAO (w/ DIS only)         = 94.2"
echo "  - GRPO (+ DIS)              = 93.5"
echo "==================================================================="
