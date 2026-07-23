#!/bin/bash
# Standalone eval — no slime, no ray, no torch_memory_saver
#
# Usage: bash run_eval.sh [tag]
#
# Just sglang server + HTTP API. Period.

set -ex
export PYTHONUNBUFFERED=1

TAG="${1:-baseline}"

WORKDIR="/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS="/home/jovyan/h800fast/wangzekai/slime_rootfs"
SITE_PKG="$ROOTFS/usr/local/lib/python3.12/dist-packages"

# Python → rootfs python
ln -sf "$ROOTFS/usr/bin/python3" /usr/bin/python3 2>/dev/null || true
ln -sf "$ROOTFS/usr/bin/python3" /usr/local/bin/python3 2>/dev/null || true

# .pth: find rootfs packages
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
LOG_FILE="$WORKDIR/standalone_eval_${TAG}_$(date +%Y%m%d_%H%M%S).log"

cd "$WORKDIR"

# Kill any leftover sglang
pkill -9 sglang 2>/dev/null || true
pkill -9 launch_server 2>/dev/null || true
sleep 2

python3 -m sao.standalone.eval \
    --tag "${TAG}" \
    --model-path "$MODEL" \
    --data "$DATA" \
    --tp 4 \
    --n-samples 4 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-new-tokens 16384 \
    --max-total-tokens 20480 \
    --mem-fraction 0.85 \
    --disable-cuda-graph \
    2>&1 | tee "$LOG_FILE"
