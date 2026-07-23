#!/bin/bash
# 在 4×A100-80GB 机器上跑 Qwen3-30B-A3B-Thinking-2507 baseline eval on AIME2025
#
# 目标: 复现论文 Table 1 中 "Qwen3-30B-A3B-Thinking-2507 SFT (w/o python) = 14.6"
#       (论文报的 SFT 起点基线; 直接评 Thinking-2507 原版应该更高)
#
# 配置: 4×A100-80GB, 单 sglang 引擎 TP=4 (320GB 总显存)
# 预计耗时: 30 题 × 4 sample × ~60s = ~2 小时
#
# 必须先跑 setup_a100.sh 完成环境准备.

set -ex
export PYTHONUNBUFFERED=1

WORKDIR="${WORKDIR:-/home/jovyan/h800fast/wangzekai}"
# 用 image 里的 slime (跟 image 的 torch/TE/megatron/sglang 版本配套)
# host clone 的 slime 版本可能不一致 (sglang_pipeline_parallel_size 等属性缺失)
SLIME_DIR="${SLIME_ROOTFS:-/home/jovyan/h800fast/wangzekai/slime_rootfs/root/slime}"
cd "${SLIME_DIR}"

# ============ 路径 ============
HF_CHECKPOINT="${WORKDIR}/slime_sao/models/Qwen3-30B-A3B-Thinking-2507"
REF_LOAD="${HF_CHECKPOINT}"   # eval-only 不需要单独的 torch_dist
EVAL_CONFIG="${WORKDIR}/slime_sao/repro/eval_aime2025.yaml"
PROMPT_DATA="${WORKDIR}/slime_sao/datasets/AIME2025/slime/aime2025-all.jsonl"

# ============ 模型架构 ============
source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

# ============ 4×A100 并行度 ============
# 单引擎 TP=4: 每卡 15GB 权重, ~65GB 给 KV cache + 激活
# Qwen3-30B-A3B 总激活 3B, 推理很快
TP_SIZE=4
PP_SIZE=1
CP_SIZE=1
EP_SIZE=4   # 128 experts 分 4 卡, 每卡 32 expert
ETP_SIZE=1

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save /tmp/sao_eval_dummy
   --save-interval 999999
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key input
   --label-key label
   --apply-chat-template

   --rm-type math
   --num-rollout 0                # ✅ eval-only (train.py:36 分支)
   --rollout-batch-size 1
   --n-samples-per-prompt 1
   --rollout-max-response-len 32768
   --rollout-temperature 1.0
   --rollout-top-p 1.0
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-config "${EVAL_CONFIG}"
)

 PERF_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --context-parallel-size ${CP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --expert-tensor-parallel-size ${ETP_SIZE}
    # Qwen3-30B-A3B-Thinking-2507 的 rope_theta=1e7, 但 slime 的 qwen3-30B-A3B.sh
    # 默认 rotary_base=1e6 (基础版). 覆盖成 1e7 跟 HF config 对齐.
    --rotary-base 10000000
 )

GRPO_ARGS=(
   --advantage-estimator grpo    # eval-only 不训练, 任意
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
   --colocate
)

# ============ Ray 启动 ============
NUM_GPUS=4
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# 算力机常走 HTTP proxy, 必须排除 localhost 否则 ray client 连不上本地 dashboard
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

ray start --head --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_GPUS}" --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

# ray worker 进程会继承当前 shell 的环境变量 (PYTHONPATH, LD_LIBRARY_PATH, PATH 等)
# run_env.sh 已经设好这些, 所以 runtime_env 只需要额外设 CUDA 相关变量
RUNTIME_ENV_JSON='{
  "env_vars": {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1"
  }
}'

# ============ 提交 eval-only job ============
LOG_FILE="${WORKDIR}/slime_sao/eval_baseline_$(date +%Y%m%d_%H%M%S).log"
echo "日志: ${LOG_FILE}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo ""
echo "==================================================================="
echo "完成. 关注日志里的:"
echo "  eval/aime2025_all/reward    (整体 pass@1)"
echo "  eval/aime2025_I/reward"
echo "  eval/aime2025_II/reward"
echo ""
echo "对照论文 Table 1:"
echo "  Qwen3-30B-A3B w/o python SFT       = 14.6   (论文 SFT 起点基线)"
echo "  Qwen3-30B-A3B SAO (ours)           = 97.3   (完整 SAO 复现目标)"
echo "==================================================================="
