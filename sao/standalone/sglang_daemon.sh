#!/bin/bash
# sglang 推理守护进程 — 在推理机上运行 (ctm-05)
#
# 自动加载最新 checkpoint，训练端保存新 checkpoint 后会触发重启。
#
# 用法: bash sglang_daemon.sh
set -ex

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
export NVTE_FRAMEWORK=pytorch
# 彻底禁用 proxy (sglang warmup 访问 0.0.0.0 会被 squid 拦截)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="*"
export NO_PROXY="*"

MODEL="$WORKDIR/models/Qwen3-30B-A3B-Thinking-2507"
CKPT_DIR="$WORKDIR/checkpoints/sao_dist"
PORT=30000
RELOAD_FILE="$CKPT_DIR/.reload_signal"

# 自动检测 GPU 数量
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=${TP:-4}
fi
echo "[daemon] Detected $NUM_GPUS GPUs, using TP=$NUM_GPUS"

mkdir -p "$CKPT_DIR"

get_latest_ckpt() {
    local latest=""
    for d in "$CKPT_DIR"/step_*; do
        [ -d "$d" ] && latest="$d"
    done
    echo "$latest"
}

while true; do
    # 找最新 checkpoint
    CKPT=$(get_latest_ckpt)
    if [ -n "$CKPT" ]; then
        echo "[daemon] Loading checkpoint: $CKPT"
        MODEL_PATH="$CKPT"
    else
        echo "[daemon] No checkpoint found, using base model: $MODEL"
        MODEL_PATH="$MODEL"
    fi

    # 启动 sglang
    pkill -9 sglang 2>/dev/null || true
    pkill -9 launch_server 2>/dev/null || true
    sleep 2

    echo "[daemon] Starting sglang on 0.0.0.0:$PORT with $MODEL_PATH"
    "$ROOTFS/usr/bin/python3" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tp "$NUM_GPUS" \
        --mem-fraction-static 0.85 \
        --context-length 36864 \
        --disable-cuda-graph \
        --reasoning-parser qwen3 &
    SGLANG_PID=$!

    # 等待 sglang 就绪或退出
    while kill -0 $SGLANG_PID 2>/dev/null; do
        sleep 5
        # 检查 reload 信号
        if [ -f "$RELOAD_FILE" ]; then
            echo "[daemon] Reload signal received, restarting sglang..."
            rm -f "$RELOAD_FILE"
            kill -TERM $SGLANG_PID 2>/dev/null || true
            wait $SGLANG_PID 2>/dev/null || true
            break
        fi
    done

    # sglang 退出（非 reload 导致），等一会再重启
    if [ ! -f "$RELOAD_FILE" ]; then
        echo "[daemon] sglang exited unexpectedly, waiting 10s before restart..."
        sleep 10
    fi
done
