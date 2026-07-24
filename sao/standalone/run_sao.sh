#!/bin/bash
# ============================================================
# SAO 完整复现: 异步 RL 训练管线
#
# 论文: Single-Rollout Asynchronous Optimization (arXiv:2607.07508)
#
# 架构 (5×8=40 GPU):
#   Machine 1 (8 GPU): sglang 推理 + rollout worker (持续生成)
#   Machine 2-3 (16 GPU): actor 训练
#   Machine 4-5 (16 GPU): critic 训练 (后续多机版本)
#
# 三个独立进程, 通过共享存储协调:
#   1. sglang_daemon.sh     — 管理 sglang 生命周期 + 自动 reload
#   2. rollout_worker.py     — 持续生成轨迹, 写入 queue/
#   3. trainer.py            — 从 queue 消费轨迹, 异步训练
#
# 启动顺序:
#   Phase 0: value_pretrain.py (可选, 一次性)
#   Phase 1: sglang_daemon.sh (推理机)
#   Phase 2: rollout_worker.py (推理机, 同一台)
#   Phase 3: trainer.py (训练机)
#
# 用法:
#   bash run_sao.sh [phase] [tag]
#   bash run_sao.sh pretrain    # Phase 0: value pretraining
#   bash run_sao.sh daemon      # Phase 1: sglang daemon
#   bash run_sao.sh rollout     # Phase 2: rollout worker
#   bash run_sao.sh train       # Phase 3: async trainer
#   bash run_sao.sh all         # Phase 1+2 (推理机一键启动)
# ============================================================

set -ex
export PYTHONUNBUFFERED=1

PHASE="${1:-all}"
TAG="${2:-sao}"

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
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="*"
export NO_PROXY="*"

MODEL="$WORKDIR/models/Qwen3-30B-A3B-Thinking-2507"
TRAIN_DATA="${TRAIN_DATA:-$WORKDIR/datasets/AIME2025/slime/aime2025-all.jsonl}"
TIR_DATA="${TIR_DATA:-$WORKDIR/datasets/tir_train.jsonl}"  # SFT/TIR 数据
SAVE_DIR="$WORKDIR/checkpoints/sao"
QUEUE_DIR="$WORKDIR/queue"
CRITIC_INIT="$WORKDIR/checkpoints/critic_pretrained"

mkdir -p "$SAVE_DIR" "$QUEUE_DIR" "$QUEUE_DIR/pending" "$QUEUE_DIR/done"
cd "$WORKDIR"

case "$PHASE" in
pretrain)
    # Phase 0: Value model pretraining (在训练机上运行, 一次性)
    LOG="$WORKDIR/value_pretrain_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Phase 0: Value Pretraining ==="
    echo "Log: $LOG"
    python3 -m sao.standalone.value_pretrain \
        --model-path "$MODEL" \
        --data "$TIR_DATA" \
        --save-dir "$CRITIC_INIT" \
        --epochs 3 \
        --batch-size 4 \
        --lr 5e-6 \
        --freeze-attention \
        2>&1 | tee "$LOG"
    ;;

daemon)
    # Phase 1: sglang daemon (推理机)
    exec bash "$WORKDIR/sao/standalone/sglang_daemon.sh"
    ;;

rollout)
    # Phase 2: rollout worker (推理机, 和 daemon 同一台)
    SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
    LOG="$WORKDIR/rollout_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Phase 2: Rollout Worker ==="
    echo "Connecting to sglang at $SGLANG_HOST:30000"
    echo "Log: $LOG"
    python3 -m sao.standalone.rollout_worker \
        --model-path "$MODEL" \
        --data "$TRAIN_DATA" \
        --sglang-host "$SGLANG_HOST" \
        --sglang-port 30000 \
        --queue-dir "$QUEUE_DIR" \
        --checkpoint-dir "$SAVE_DIR" \
        --temperature 1.0 \
        --top-p 1.0 \
        --max-new-tokens 32768 \
        2>&1 | tee "$LOG"
    ;;

train)
    # Phase 3: Async trainer (训练机)
    LOG="$WORKDIR/trainer_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Phase 3: SAO Async Trainer ==="
    echo "Log: $LOG"

    # 如果没有 pretrained critic, 先用 base model
    CRITIC_PATH="$CRITIC_INIT"
    if [ ! -f "$CRITIC_INIT/critic.pt" ]; then
        echo "WARNING: No pretrained critic found, using base model"
        CRITIC_PATH="$MODEL"
    fi

    python3 -m sao.standalone.trainer \
        --model-path "$MODEL" \
        --critic-path "$CRITIC_PATH" \
        --queue-dir "$QUEUE_DIR" \
        --save-dir "$SAVE_DIR" \
        --num-steps "${NUM_STEPS:-1000}" \
        --batch-size "${BATCH_SIZE:-128}" \
        --lr 1e-6 \
        --critic-lr 5e-6 \
        --clip-low 0.7 --clip-high 6.0 \
        --gamma 1.0 --gae-alpha 1.5 \
        --critic-k 2 --critic-warmup 10 \
        --value-clip 0.2 \
        --save-interval "${SAVE_INTERVAL:-50}" \
        2>&1 | tee "$LOG"
    ;;

all)
    # Phase 1+2: daemon + rollout worker (推理机一键启动)
    SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
    LOG_DAEMON="$WORKDIR/daemon_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    LOG_ROLLOUT="$WORKDIR/rollout_${TAG}_$(date +%Y%m%d_%H%M%S).log"

    echo "=== Starting sglang daemon (background) ==="
    bash "$WORKDIR/sao/standalone/sglang_daemon.sh" &
    DAEMON_PID=$!

    # 等 sglang ready
    echo "Waiting for sglang..."
    for i in $(seq 1 300); do
        if curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; then
            echo "sglang ready"
            break
        fi
        sleep 3
    done

    echo "=== Starting rollout worker ==="
    python3 -m sao.standalone.rollout_worker \
        --model-path "$MODEL" \
        --data "$TRAIN_DATA" \
        --sglang-host "$SGLANG_HOST" \
        --sglang-port 30000 \
        --queue-dir "$QUEUE_DIR" \
        --checkpoint-dir "$SAVE_DIR" \
        --temperature 1.0 \
        --top-p 1.0 \
        --max-new-tokens 32768 \
        2>&1 | tee "$LOG_ROLLOUT" &
    ROLLOUT_PID=$!

    # 等待 rollout worker 结束
    wait $ROLLOUT_PID
    kill $DAEMON_PID 2>/dev/null || true
    ;;

*)
    echo "Usage: bash run_sao.sh [pretrain|daemon|rollout|train|all] [tag]"
    echo ""
    echo "Typical workflow (5 machines):"
    echo "  Machine 1 (inference): bash run_sao.sh all"
    echo "  Machine 2+ (training): bash run_sao.sh train"
    echo ""
    echo "Optional: bash run_sao.sh pretrain (on training machine first)"
    exit 1
    ;;
esac
