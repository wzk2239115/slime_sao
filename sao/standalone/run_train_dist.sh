#!/bin/bash
# 多机分布式训练
#
# 架构:
#   ctm-05 (推理机): bash sao/standalone/sglang_daemon.sh  (先启动)
#   ctm-06 (训练机): bash sao/standalone/run_train_dist.sh  (后启动)
#
# 用法:
#   1. 在 ctm-05 上: bash sao/standalone/sglang_daemon.sh
#   2. 在 ctm-06 上: SGLANG_HOST=11.131.211.65 bash sao/standalone/run_train_dist.sh v1

set -ex
export PYTHONUNBUFFERED=1

TAG="${1:-sao_dist}"
SGLANG_HOST="${SGLANG_HOST:-11.131.211.65}"  # ctm-05 的 IP

WORKDIR="/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS="/home/jovyan/h800fast/wangzekai/slime_rootfs"
SITE_PKG="$ROOTFS/usr/local/lib/python3.12/dist-packages"
HOST_SITE="/usr/local/lib/python3.12/dist-packages"

# Python + 环境
ln -sf "$ROOTFS/usr/bin/python3" /usr/bin/python3 2>/dev/null || true
ln -sf "$ROOTFS/usr/bin/python3" /usr/local/bin/python3 2>/dev/null || true
mkdir -p "$HOST_SITE"
echo "$SITE_PKG" > "$HOST_SITE/rootfs_packages.pth"

export LD_LIBRARY_PATH="$ROOTFS/usr/local/cuda/lib64:$ROOTFS/usr/local/nvidia/lib64"
export PYTHONPATH="$WORKDIR:$SITE_PKG:$ROOTFS/tmp/local_src/python"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$ROOTFS/usr/local/cuda/bin"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONDONTWRITEBYTECODE=1
export no_proxy="$SGLANG_HOST,127.0.0.1,localhost"
export NO_PROXY="$SGLANG_HOST,127.0.0.1,localhost"

MODEL="$WORKDIR/models/Qwen3-30B-A3B-Thinking-2507"
DATA="$WORKDIR/datasets/AIME2025/slime/aime2025-all.jsonl"
SAVE_DIR="$WORKDIR/checkpoints/sao_dist"
LOG_FILE="$WORKDIR/dist_train_${TAG}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$SAVE_DIR"
cd "$WORKDIR"

# 检测 GPU 数量
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs on this machine: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

echo "SGLANG_HOST=$SGLANG_HOST"
echo "SAVE_DIR=$SAVE_DIR"
echo "LOG=$LOG_FILE"

python3 -m sao.standalone.train_dist \
    --model-path "$MODEL" \
    --sglang-host "$SGLANG_HOST" \
    --sglang-port 30000 \
    --data "$DATA" \
    --save-dir "$SAVE_DIR" \
    --num-steps "${NUM_STEPS:-100}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --n-samples "${N_SAMPLES:-8}" \
    --lr 1e-6 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-new-tokens 32768 \
    --max-seq-len 32768 \
    --algo "${ALGO:-sao}" \
    --clip-low 0.7 --clip-high 6.0 \
    --save-interval "${SAVE_INTERVAL:-10}" \
    ${USE_CRITIC:+--use-critic} \
    --critic-lr 5e-6 --critic-k 2 \
    --value-clip 0.2 --gamma 1.0 --gae-alpha 1.5 \
    2>&1 | tee "$LOG_FILE"
