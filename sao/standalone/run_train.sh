#!/bin/bash
# Standalone GRPO/SAO RL training — no slime, no ray, no torch_memory_saver
#
# Usage: bash run_train.sh [tag]
#
# Architecture:
#   1. sglang generates responses (rollout)
#   2. Kill sglang, free GPU
#   3. HF model: compute log-probs + policy gradient (training)
#   4. Save checkpoint, restart sglang with new weights
#   5. Repeat

set -ex
export PYTHONUNBUFFERED=1

TAG="${1:-sao}"

WORKDIR="/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS="/home/jovyan/h800fast/wangzekai/slime_rootfs"
SITE_PKG="$ROOTFS/usr/local/lib/python3.12/dist-packages"

# Python → rootfs python
ln -sf "$ROOTFS/usr/bin/python3" /usr/bin/python3 2>/dev/null || true
ln -sf "$ROOTFS/usr/bin/python3" /usr/local/bin/python3 2>/dev/null || true

# .pth
HOST_SITE="/usr/local/lib/python3.12/dist-packages"
mkdir -p "$HOST_SITE"
echo "$SITE_PKG" > "$HOST_SITE/rootfs_packages.pth"

export LD_LIBRARY_PATH="$ROOTFS/usr/local/cuda/lib64:$ROOTFS/usr/local/nvidia/lib64"
export PYTHONPATH="$WORKDIR:$SITE_PKG:$ROOTFS/tmp/local_src/python:$WORKDIR/patch"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$ROOTFS/usr/local/cuda/bin"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONDONTWRITEBYTECODE=1
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export SAO_WORKDIR="$WORKDIR"
export SAO_ROOTFS="$ROOTFS"

MODEL="$WORKDIR/models/Qwen3-30B-A3B-Thinking-2507"
DATA="$WORKDIR/datasets/AIME2025/slime/aime2025-all.jsonl"
SAVE_DIR="$WORKDIR/checkpoints/sao_standalone"
LOG_FILE="$WORKDIR/standalone_train_${TAG}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$SAVE_DIR"
cd "$WORKDIR"

# Kill any leftover sglang
pkill -9 sglang 2>/dev/null || true
pkill -9 launch_server 2>/dev/null || true
sleep 2

python3 -m sao.standalone.train \
    --model-path "$MODEL" \
    --data "$DATA" \
    --save-dir "$SAVE_DIR" \
    --tp 4 \
    --num-steps "${NUM_STEPS:-100}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --n-samples "${N_SAMPLES:-8}" \
    --lr 1e-6 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-new-tokens 32768 \
    --max-total-tokens 36864 \
    --max-seq-len 32768 \
    --algo "${ALGO:-sao}" \
    --clip-low 0.7 --clip-high 6.0 \
    --save-interval "${SAVE_INTERVAL:-10}" \
    --disable-cuda-graph \
    2>&1 | tee "$LOG_FILE"
