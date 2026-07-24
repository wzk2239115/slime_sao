#!/bin/bash
# ============================================================
# SAO 5-machine deployment script
#
# Machines:
#   INFERENCE_HOST: sglang + rollout worker (8 GPU)
#   TRAIN_HOST:     actor + critic trainer (8 GPU)
#
# Shared NFS: /home/jovyan/h800fast/wangzekai/slime_sao
#
# Usage:
#   bash run_sao.sh setup          # First-time: upgrade glibc + verify env
#   bash run_sao.sh inference      # On inference machine: sglang + rollout
#   bash run_sao.sh train          # On training machine: async trainer
#   bash run_sao.sh status         # Check queue + checkpoints
# ============================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

PHASE="${1:-help}"
TAG="${2:-sao_v1}"

# ============================================================
# Paths (shared NFS - same on all machines)
# ============================================================
WORKDIR="/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS="/home/jovyan/h800fast/wangzekai/slime_rootfs"
SITE_PKG="$ROOTFS/usr/local/lib/python3.12/dist-packages"
HOST_SITE="/usr/local/lib/python3.12/dist-packages"

MODEL="$WORKDIR/models/Qwen3-30B-A3B-Thinking-2507"
TRAIN_DATA="$WORKDIR/datasets/AIME2025/slime/aime2025-all.jsonl"
SAVE_DIR="$WORKDIR/checkpoints/sao"
QUEUE_DIR="$WORKDIR/queue"
LOG_DIR="$WORKDIR/logs"

# ============================================================
# Environment setup (called on every machine)
# ============================================================
setup_env() {
    echo "=== Setting up environment ==="
    
    # Python symlink
    ln -sf "$ROOTFS/usr/bin/python3" /usr/bin/python3 2>/dev/null || true
    ln -sf "$ROOTFS/usr/bin/python3" /usr/local/bin/python3 2>/dev/null || true
    mkdir -p "$HOST_SITE"
    echo "$SITE_PKG" > "$HOST_SITE/rootfs_packages.pth"
    
    # Verify glibc
    local glibc_ver
    glibc_ver=$(ldd --version 2>&1 | head -1 | awk '{print $NF}')
    echo "  glibc: $glibc_ver"
    
    local glibc_major=$(echo "$glibc_ver" | cut -d. -f1)
    local glibc_minor=$(echo "$glibc_ver" | cut -d. -f2)
    
    if [ "$glibc_major" -lt 2 ] || { [ "$glibc_major" -eq 2 ] && [ "$glibc_minor" -lt 38 ]; }; then
        echo "  glibc < 2.38, upgrading..."
        if [ -f "$WORKDIR/repro/upgrade_glibc.sh" ]; then
            bash "$WORKDIR/repro/upgrade_glibc.sh"
        else
            echo "  ERROR: upgrade_glibc.sh not found!"
            echo "  Run: cd $WORKDIR && git pull"
            exit 1
        fi
    else
        echo "  glibc OK (>= 2.38)"
    fi
    
    # GPU check
    local n_gpu
    n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "  GPUs: $n_gpu"
    
    # Pull latest code
    cd "$WORKDIR"
    git pull --rebase 2>/dev/null || true
    
    # Create directories
    mkdir -p "$SAVE_DIR" "$QUEUE_DIR/pending" "$QUEUE_DIR/done" "$LOG_DIR"
    
    # Verify model exists
    if [ ! -d "$MODEL" ]; then
        echo "  ERROR: Model not found at $MODEL"
        exit 1
    fi
    
    # Verify data exists
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "  ERROR: Training data not found at $TRAIN_DATA"
        exit 1
    fi
    
    echo "=== Environment ready ==="
}

# ============================================================
# Export common env vars
# ============================================================
export_env() {
    export LD_LIBRARY_PATH="$ROOTFS/usr/local/cuda/lib64:$ROOTFS/usr/local/nvidia/lib64"
    export PYTHONPATH="$WORKDIR:$SITE_PKG:$ROOTFS/tmp/local_src/python"
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$ROOTFS/usr/local/cuda/bin"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONDONTWRITEBYTECODE=1
    export NVTE_FRAMEWORK=pytorch
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export no_proxy="*"
    export NO_PROXY="*"
    export TOKENIZERS_PARALLELISM=false
}

# ============================================================
# Phase: setup
# ============================================================
if [ "$PHASE" = "setup" ]; then
    setup_env
    echo ""
    echo "=== Verifying Python + torch ==="
    cd "$WORKDIR"
    python3 -c "
import torch
print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')
" || echo "WARNING: torch import failed"
    
    echo ""
    echo "=== Verifying bitsandbytes (8-bit optimizer) ==="
    python3 -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')" 2>/dev/null \
        && echo "  8-bit AdamW available" \
        || echo "  bitsandbytes not found (will use standard AdamW)"
    
    echo ""
    echo "Setup complete. Next steps:"
    echo "  Inference machine: bash $0 inference"
    echo "  Training machine:  bash $0 train"
    exit 0
fi

# ============================================================
# Phase: inference (sglang daemon + rollout worker)
# ============================================================
if [ "$PHASE" = "inference" ]; then
    setup_env
    export_env
    cd "$WORKDIR"
    
    LOG_DAEMON="$LOG_DIR/daemon_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    LOG_ROLLOUT="$LOG_DIR/rollout_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    
    # Kill any existing sglang
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
    
    # Auto-detect GPU count
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ] && NUM_GPUS=8
    echo "=== Inference machine: $NUM_GPUS GPUs ==="
    
    # ---- Start sglang daemon ----
    echo "=== Starting sglang daemon (log: $LOG_DAEMON) ==="
    
    # Find latest checkpoint or use base model
    LATEST_CKPT=""
    for d in "$SAVE_DIR"/step_*; do
        [ -d "$d" ] && LATEST_CKPT="$d"
    done
    
    if [ -n "$LATEST_CKPT" ]; then
        echo "  Loading checkpoint: $LATEST_CKPT"
        SGLANG_MODEL="$LATEST_CKPT"
    else
        echo "  No checkpoint, using base model"
        SGLANG_MODEL="$MODEL"
    fi
    
    pkill -9 sglang 2>/dev/null || true
    pkill -9 launch_server 2>/dev/null || true
    sleep 2
    
    # CUDA graph: default ON (much faster). Set DISABLE_CUDA_GRAPH=1 to disable.
    CUDA_GRAPH_FLAG=""
    if [ "${DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
        CUDA_GRAPH_FLAG="--disable-cuda-graph"
        echo "  CUDA graph: DISABLED"
    else
        echo "  CUDA graph: ENABLED"
    fi
    
    python3 -m sglang.launch_server \
        --model-path "$SGLANG_MODEL" \
        --host 0.0.0.0 \
        --port 30000 \
        --tp "$NUM_GPUS" \
        --mem-fraction-static 0.85 \
        --context-length 36864 \
        $CUDA_GRAPH_FLAG \
        --reasoning-parser qwen3 \
        > "$LOG_DAEMON" 2>&1 &
    SGLANG_PID=$!
    echo "  sglang PID: $SGLANG_PID"
    
    # Wait for sglang to be ready
    echo "  Waiting for sglang to be ready..."
    READY=0
    for i in $(seq 1 600); do
        if ! kill -0 $SGLANG_PID 2>/dev/null; then
            echo "  ERROR: sglang died! Check $LOG_DAEMON"
            tail -30 "$LOG_DAEMON"
            exit 1
        fi
        if python3 -c "
import urllib.request, json
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
req = urllib.request.Request('http://127.0.0.1:30000/health')
opener.open(req, timeout=5).read()
" 2>/dev/null; then
            READY=1
            echo "  sglang ready! (took ${i}0s)"
            break
        fi
        sleep 10
    done
    
    if [ "$READY" -eq 0 ]; then
        echo "  ERROR: sglang failed to start within 6000s"
        tail -50 "$LOG_DAEMON"
        exit 1
    fi
    
    # ---- Start rollout worker ----
    echo "=== Starting rollout worker (log: $LOG_ROLLOUT) ==="
    
    MAX_TOKENS="${MAX_TOKENS:-32768}"
    echo "  max_new_tokens=$MAX_TOKENS"
    
    python3 -m sao.standalone.rollout_worker \
        --model-path "$MODEL" \
        --data "$TRAIN_DATA" \
        --sglang-host 127.0.0.1 \
        --sglang-port 30000 \
        --queue-dir "$QUEUE_DIR" \
        --checkpoint-dir "$SAVE_DIR" \
        --temperature 1.0 \
        --top-p 1.0 \
        --max-new-tokens "$MAX_TOKENS" \
        --max-seq-len "$MAX_TOKENS" \
        > "$LOG_ROLLOUT" 2>&1 &
    ROLLOUT_PID=$!
    echo "  rollout worker PID: $ROLLOUT_PID"
    
    # ---- Auto-reload watcher (restarts sglang when trainer saves checkpoint) ----
    start_sglang() {
        local model_path="$1"
        pkill -9 sglang 2>/dev/null || true
        pkill -9 launch_server 2>/dev/null || true
        sleep 3
        local cg_flag=""
        [ "${DISABLE_CUDA_GRAPH:-0}" != "1" ] || cg_flag="--disable-cuda-graph"
        python3 -m sglang.launch_server \
            --model-path "$model_path" \
            --host 0.0.0.0 --port 30000 \
            --tp "$NUM_GPUS" \
            --mem-fraction-static 0.85 \
            --context-length 36864 \
            $cg_flag \
            --reasoning-parser qwen3 \
            >> "$LOG_DAEMON" 2>&1 &
        SGLANG_PID=$!
        echo "[watcher] sglang restarted PID=$SGLANG_PID model=$model_path"
        # Wait for health
        for i in $(seq 1 120); do
            kill -0 $SGLANG_PID 2>/dev/null || return 1
            if python3 -c "
import urllib.request
o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
o.open(urllib.request.Request('http://127.0.0.1:30000/health'),timeout=5).read()
" 2>/dev/null; then
                echo "[watcher] sglang healthy after ${i}0s"
                return 0
            fi
            sleep 10
        done
        return 1
    }
    export -f start_sglang
    export SGLANG_PID NUM_GPUS LOG_DAEMON
    
    (
        while true; do
            sleep 10
            if [ -f "$SAVE_DIR/.reload_signal" ]; then
                rm -f "$SAVE_DIR/.reload_signal"
                echo "[watcher] Reload signal received!"
                kill -TERM $SGLANG_PID 2>/dev/null || true
                wait $SGLANG_PID 2>/dev/null || true
                sleep 5
                LATEST=""
                for d in "$SAVE_DIR"/step_*; do [ -d "$d" ] && LATEST="$d"; done
                if [ -n "$LATEST" ]; then
                    start_sglang "$LATEST" && touch "$SAVE_DIR/.reload_done" \
                        || echo "[watcher] ERROR: sglang restart failed!"
                else
                    echo "[watcher] No checkpoint found, skipping reload"
                    touch "$SAVE_DIR/.reload_done"
                fi
            fi
        done
    ) &
    WATCHER_PID=$!
    echo "  watcher PID:  $WATCHER_PID"
    
    echo ""
    echo "=== Inference pipeline running ==="
    echo "  sglang PID:   $SGLANG_PID  (log: $LOG_DAEMON)"
    echo "  rollout PID:  $ROLLOUT_PID (log: $LOG_ROLLOUT)"
    echo "  watcher PID:  $WATCHER_PID"
    echo "  Queue:        $QUEUE_DIR/pending/"
    echo ""
    echo "  Monitor:"
    echo "    tail -f $LOG_ROLLOUT"
    echo "    ls $QUEUE_DIR/pending/ | wc -l"
    echo ""
    echo "  Stop: kill $ROLLOUT_PID $WATCHER_PID $SGLANG_PID"
    
    wait $ROLLOUT_PID
    kill $WATCHER_PID $SGLANG_PID 2>/dev/null || true
    exit 0
fi

# ============================================================
# Phase: train (async SAO trainer)
# ============================================================
if [ "$PHASE" = "train" ]; then
    setup_env
    export_env
    cd "$WORKDIR"
    
    LOG_TRAIN="$LOG_DIR/trainer_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=== Training machine: SAO Async Trainer ==="
    echo "  Log: $LOG_TRAIN"
    
    # Batch size (paper: 128, reduce for single-machine)
    BATCH_SIZE="${BATCH_SIZE:-32}"
    NUM_STEPS="${NUM_STEPS:-1000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
    MAX_TOKENS="${MAX_TOKENS:-32768}"
    
    echo "  batch_size=$BATCH_SIZE  num_steps=$NUM_STEPS"
    echo "  lr=1e-6  critic_lr=5e-6  (paper §4.1)"
    echo "  DIS: clip_low=0.7(ε_l=0.3)  clip_high=6.0(ε_h=5.0)"
    echo "  GAE: gamma=1.0  alpha=1.5  λ_policy=1-1/(1.5·L)"
    echo "  TTUR: K=2  critic_warmup=10  value_clip=0.2"
    echo "  frozen_attention=yes"
    
    # Use base model as critic init if no pretrained critic
    CRITIC_PATH="$MODEL"
    CRITIC_PRETRAINED="$WORKDIR/checkpoints/critic_pretrained"
    if [ -f "$CRITIC_PRETRAINED/critic.pt" ]; then
        CRITIC_PATH="$CRITIC_PRETRAINED"
        echo "  Using pretrained critic: $CRITIC_PATH"
    else
        echo "  No pretrained critic, using base model (cold start)"
    fi
    
    # Clear any stale reload signals
    rm -f "$SAVE_DIR/.reload_signal" "$SAVE_DIR/.reload_done" 2>/dev/null || true
    
    python3 -m sao.standalone.trainer \
        --model-path "$MODEL" \
        --critic-path "$CRITIC_PATH" \
        --queue-dir "$QUEUE_DIR" \
        --save-dir "$SAVE_DIR" \
        --num-steps "$NUM_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --lr 1e-6 \
        --critic-lr 5e-6 \
        --clip-low 0.7 --clip-high 6.0 \
        --gamma 1.0 --gae-alpha 1.5 \
        --critic-k 2 --critic-warmup 10 \
        --value-clip 0.2 \
        --save-interval "$SAVE_INTERVAL" \
        --max-seq-len "$MAX_TOKENS" \
        --use-8bit-adam \
        2>&1 | tee "$LOG_TRAIN"
    exit $?
fi

# ============================================================
# Phase: status
# ============================================================
if [ "$PHASE" = "status" ]; then
    echo "=== SAO Pipeline Status ==="
    
    # Queue
    PENDING=$(ls "$QUEUE_DIR/pending/"traj_*.json 2>/dev/null | wc -l)
    DONE=$(ls "$QUEUE_DIR/done/"traj_*.json 2>/dev/null | wc -l)
    echo "Queue: $PENDING pending, $DONE done"
    
    # Checkpoints
    echo "Checkpoints:"
    for d in "$SAVE_DIR"/step_*; do
        [ -d "$d" ] && echo "  $(basename $d)"
    done
    
    # Latest reward
    LATEST_LOG=$(ls -t "$LOG_DIR"/trainer_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo ""
        echo "Latest trainer output:"
        tail -5 "$LATEST_LOG"
    fi
    
    # GPU status
    echo ""
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
    
    exit 0
fi

# ============================================================
# Help
# ============================================================
echo "Usage: bash run_sao.sh <phase> [tag]"
echo ""
echo "Phases:"
echo "  setup      - First-time setup: glibc upgrade, env check, git pull"
echo "  inference  - Start sglang + rollout worker (inference machine)"
echo "  train      - Start async SAO trainer (training machine)"
echo "  status     - Check pipeline status"
echo ""
echo "Typical workflow:"
echo "  # On EVERY machine:"
echo "  bash run_sao.sh setup"
echo ""
echo "  # On inference machine (e.g. 11.131.211.65):"
echo "  bash run_sao.sh inference sao_v1"
echo ""
echo "  # On training machine (e.g. 11.131.215.38):"
echo "  bash run_sao.sh train sao_v1"
echo ""
echo "Environment overrides:"
echo "  BATCH_SIZE=32   NUM_STEPS=1000   MAX_TOKENS=32768"
echo ""
echo "Paper parameters (hardcoded):"
echo "  lr=1e-6  critic_lr=5e-6  DIS ε_l=0.3 ε_h=5.0"
echo "  γ=1.0  α=1.5  K=2  warmup=10  frozen_attn"
exit 0
