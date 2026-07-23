#!/bin/bash
# ============================================================
# 一条命令跑 slime (eval 或 RL train)
#
#   bash run.sh eval [tag]       # baseline eval on AIME2025
#   bash run.sh train [tag]      # GRPO RL training
#
# 前提: glibc 已升级到 2.38+, rootfs 已解压
# ============================================================
set -ex

MODE="${1:-eval}"
TAG="${2:-$(date +%m%d_%H%M)}"

# ============================================================
# 环境设置 (自包含，不依赖外部 source)
# ============================================================
WORKDIR="/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS="/home/jovyan/h800fast/wangzekai/slime_rootfs"
SITE_PKG="$ROOTFS/usr/local/lib/python3.12/dist-packages"
SLIME_DIR="$ROOTFS/root/slime"

# Python → rootfs python (统一 /usr/bin /usr/local/bin)
ln -sf "$ROOTFS/usr/bin/python3" /usr/bin/python3
ln -sf "$ROOTFS/usr/bin/python3" /usr/local/bin/python3
ln -sf "$ROOTFS/usr/bin/python3" /usr/local/bin/python

# .pth: 让所有 python 进程找到 rootfs 的包
HOST_SITE="/usr/local/lib/python3.12/dist-packages"
mkdir -p "$HOST_SITE"
echo "$SITE_PKG" > "$HOST_SITE/rootfs_packages.pth"

export LD_LIBRARY_PATH="$ROOTFS/usr/local/cuda/lib64:$ROOTFS/usr/local/nvidia/lib64"
export PYTHONPATH="$WORKDIR/patch:$SITE_PKG:$ROOTFS/root/slime:$ROOTFS/root/Megatron-LM:$ROOTFS/tmp/local_src/python:$WORKDIR"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$ROOTFS/usr/local/cuda/bin"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export NVTE_FRAMEWORK=pytorch
export HOME="${HOME:-/root}"
export no_proxy="127.0.0.1,localhost"

cd "$SLIME_DIR"

# ============================================================
# 路径配置
# ============================================================
HF_CHECKPOINT="$WORKDIR/models/Qwen3-30B-A3B-Thinking-2507"
EVAL_CONFIG="$WORKDIR/repro/eval_aime2025.yaml"
PROMPT_DATA="$WORKDIR/datasets/AIME2025/slime/aime2025-all.jsonl"

# ============================================================
# 模型架构
# ============================================================
source "$SLIME_DIR/scripts/models/qwen3-30B-A3B.sh"

# ============================================================
# 公共参数 (eval + train 共用)
# ============================================================
TP=4; PP=1; CP=1; EP=4; ETP=1; NUM_GPUS=4

COMMON_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$HF_CHECKPOINT"
    --actor-num-nodes 1
    --actor-num-gpus-per-node $NUM_GPUS
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --expert-model-parallel-size $EP
    --expert-tensor-parallel-size $ETP
    --rotary-base 10000000
    --rollout-num-gpus-per-engine $TP
    --sglang-mem-fraction-static 0.85
    --rollout-max-response-len 32768
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rm-type math
    --apply-chat-template
    --input-key input
    --label-key label
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --colocate
    --advantage-estimator grpo
    --eps-clip 0.2 --eps-clip-high 0.28
    --kl-coef 0.0
    --optimizer adam --lr 1e-6 --lr-decay-style constant
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
)

# ============================================================
# Ray 启动
# ============================================================
pkill -9 sglang 2>/dev/null || true
sleep 1
ray stop --force 2>/dev/null || true
sleep 2

ray start --head --node-ip-address 127.0.0.1 \
    --num-gpus $NUM_GPUS --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "等待 dashboard..."
for i in $(seq 1 30); do
    curl -sf http://127.0.0.1:8265/api/version >/dev/null 2>&1 && { echo "Dashboard ready (${i}x2s)"; break; }
    sleep 2
done
curl -sf http://127.0.0.1:8265/api/version >/dev/null 2>&1 || { echo "ERROR: Dashboard 启动失败"; exit 1; }

RUNTIME_ENV='{"env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}}'
LOG_FILE="$WORKDIR/${MODE}_${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "日志: $LOG_FILE"

# ============================================================
# 提交 job
# ============================================================
if [ "$MODE" = "eval" ]; then
    # ---- Baseline Eval ----
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="$RUNTIME_ENV" \
        -- python3 "$SLIME_DIR/train.py" \
        "${COMMON_ARGS[@]}" \
        --prompt-data "$PROMPT_DATA" \
        --num-rollout 0 \
        --rollout-batch-size 1 \
        --n-samples-per-prompt 1 \
        --eval-interval 1 \
        --eval-config "$EVAL_CONFIG" \
        --save /tmp/sao_eval_dummy --save-interval 999999 \
        2>&1 | tee "$LOG_FILE"

elif [ "$MODE" = "train" ]; then
    # ---- GRPO RL Training ----
    SAVE_DIR="${SAVE_DIR:-$WORKDIR/checkpoints/sao_train}"
    mkdir -p "$SAVE_DIR"

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="$RUNTIME_ENV" \
        -- python3 "$SLIME_DIR/train.py" \
        "${COMMON_ARGS[@]}" \
        --prompt-data "$PROMPT_DATA" \
        --num-rollout "${NUM_ROLLOUT:-100}" \
        --rollout-batch-size "${BATCH_SIZE:-4}" \
        --n-samples-per-prompt "${N_SAMPLES:-8}" \
        --save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-10}" \
        --eval-interval "${EVAL_INTERVAL:-10}" \
        --eval-config "$EVAL_CONFIG" \
        2>&1 | tee "$LOG_FILE"
else
    echo "用法: bash run.sh [eval|train] [tag]"
    exit 1
fi

echo ""
echo "==================================================================="
echo "完成. 日志: $LOG_FILE"
echo "  eval 结果: grep 'eval/aime2025_all/reward' $LOG_FILE"
echo "==================================================================="
