#!/usr/bin/env python3
"""SAO Cluster Manager — one command to rule all 5 machines × 8 GPUs = 40 GPUs.

Runs on ctm-06 (master). SSHes to inference machines to manage processes.

Architecture:
  ctm-05, ctm-01, ctm-02, ctm-04  →  sglang + rollout worker (4 × 8 GPU inference)
  ctm-06                           →  trainer (8 GPU training)

Usage:
    python3 cluster.py setup      # Setup SSH keys + verify all machines
    python3 cluster.py start      # Start inference on 4 machines + trainer locally
    python3 cluster.py stop       # Kill everything on all machines
    python3 cluster.py status     # Check all machines
    python3 cluster.py logs       # Tail trainer log
    python3 cluster.py queue      # Show queue size
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

# ============================================================
# Machine Configuration
# ============================================================
INFER_MACHINES = [
    ("ctm-05", "11.131.211.65"),
    ("ctm-01", "11.131.210.217"),
    ("ctm-02", "11.131.210.2"),
    ("ctm-04", "11.131.210.123"),
]
TRAIN_MACHINE = ("ctm-06", "11.131.215.38")
ALL_MACHINES = INFER_MACHINES + [TRAIN_MACHINE]

WORKDIR = "/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS = "/home/jovyan/h800fast/wangzekai/slime_rootfs"
DATA_FILE = "datasets/MATH_train.jsonl"
MODEL_PATH = "models/Qwen3-30B-A3B-Thinking-2507"
MAX_TOKENS = 32768
BATCH_SIZE = 8
NUM_STEPS = 1000

SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"


# ============================================================
# SSH Helpers
# ============================================================
def ssh_run(ip: str, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run command on remote machine, return (exit_code, stdout, stderr)."""
    full = f"ssh {SSH_OPTS} root@{ip} 'env -u BASH_ENV {cmd}'"
    r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def ssh_bg(ip: str, cmd: str):
    """Start a background command on remote machine (returns immediately)."""
    full = f"ssh {SSH_OPTS} root@{ip} 'env -u BASH_ENV nohup bash -c \"{cmd}\" > /dev/null 2>&1 &'"
    subprocess.run(full, shell=True, timeout=15)


def ssh_check(ip: str) -> bool:
    """Check if SSH works."""
    try:
        code, _, _ = ssh_run(ip, "echo ok", timeout=10)
        return code == 0
    except Exception:
        return False


# ============================================================
# Environment Setup Commands
# ============================================================
ENV_SETUP = f"""export ROOTFS={ROOTFS}
export LD_LIBRARY_PATH=$ROOTFS/usr/local/cuda/lib64:$ROOTFS/usr/local/nvidia/lib64
export PYTHONPATH={WORKDIR}:$ROOTFS/usr/local/lib/python3.12/dist-packages
export PATH=$ROOTFS/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export no_proxy='*' NO_PROXY='*' CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
ln -sf $ROOTFS/sgl-workspace /sgl-workspace 2>/dev/null || true
ln -sf $ROOTFS/usr/bin/python3 /usr/bin/python3 2>/dev/null || true
ln -sf $ROOTFS/usr/bin/python3 /usr/local/bin/python3 2>/dev/null || true
mkdir -p /usr/local/lib/python3.12/dist-packages
echo '$ROOTFS/usr/local/lib/python3.12/dist-packages' > /usr/local/lib/python3.12/dist-packages/rootfs_packages.pth
cd {WORKDIR}
git pull --rebase 2>/dev/null || true
mkdir -p queue/pending queue/done logs checkpoints/sao"""


# ============================================================
# Commands
# ============================================================
def cmd_setup():
    """Setup SSH keys + verify all machines."""
    print("=" * 60)
    print("SAO Cluster Setup")
    print("=" * 60)

    # 1. Check SSH to all machines
    print("\n--- Checking SSH connectivity ---")
    all_ok = True
    for name, ip in ALL_MACHINES:
        ok = ssh_check(ip)
        status = "✓" if ok else "✗"
        print(f"  {status} {name} ({ip})")
        if not ok:
            all_ok = False
            print(f"    Fix: ssh-copy-id root@{ip}")

    if not all_ok:
        print("\n✗ SSH not configured for all machines.")
        print("Run on ctm-06:")
        print("  ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa  # if no key")
        for _, ip in ALL_MACHINES:
            print(f"  ssh-copy-id root@{ip}")
        return

    # 2. Setup environment on each machine
    print("\n--- Setting up environment on all machines ---")
    for name, ip in ALL_MACHINES:
        print(f"  {name} ({ip})...", end=" ", flush=True)
        code, out, err = ssh_run(ip, ENV_SETUP, timeout=60)
        if code == 0:
            # Verify
            code2, out2, _ = ssh_run(ip, f"{ENV_SETUP}; python3 -c \"import torch; print(torch.cuda.device_count())\"", timeout=30)
            print(f"OK ({out2.strip()} GPUs)" if "GPUs" in out2 else "OK")
        else:
            print(f"FAIL: {err[:100]}")

    # 3. Verify shared storage
    print("\n--- Verifying shared storage ---")
    code, out, _ = ssh_run(TRAIN_MACHINE[1], f"ls {WORKDIR}/{DATA_FILE} 2>/dev/null && echo EXISTS || echo MISSING")
    print(f"  Training data ({DATA_FILE}): {out}")
    code, out, _ = ssh_run(TRAIN_MACHINE[1], f"ls {WORKDIR}/{MODEL_PATH}/config.json 2>/dev/null && echo EXISTS || echo MISSING")
    print(f"  Model: {out}")

    print("\n✓ Setup complete. Run: python3 cluster.py start")


def cmd_start():
    """Start all inference machines + trainer."""
    print("=" * 60)
    print("Starting SAO Cluster (5 machines, 40 GPUs)")
    print("=" * 60)

    # Clear queue
    os.system(f"rm -rf {WORKDIR}/queue/pending/* {WORKDIR}/queue/done/* 2>/dev/null")
    os.makedirs(f"{WORKDIR}/queue/pending", exist_ok=True)
    os.makedirs(f"{WORKDIR}/queue/done", exist_ok=True)
    os.makedirs(f"{WORKDIR}/logs", exist_ok=True)
    os.makedirs(f"{WORKDIR}/checkpoints/sao", exist_ok=True)

    # 1. Start inference on all inference machines (parallel)
    print("\n--- Starting inference on 4 machines ---")
    for name, ip in INFER_MACHINES:
        print(f"  {name} ({ip})...", end=" ", flush=True)

        # Kill old processes
        ssh_run(ip, "pkill -9 -f sglang.launch_server 2>/dev/null; pkill -9 -f rollout_worker 2>/dev/null; sleep 2", timeout=15)

        # Start sglang + rollout worker
        start_cmd = f"""{ENV_SETUP}
pkill -9 -f sglang.launch_server 2>/dev/null; sleep 2
python3 -m sglang.launch_server \
    --model-path {MODEL_PATH} \
    --host 0.0.0.0 --port 30000 --tp 8 \
    --mem-fraction-static 0.85 --context-length 36864 \
    --reasoning-parser qwen3 > logs/sglang_{name}.log 2>&1 &
SGLANG_PID=$!
echo "sglang PID=$SGLANG_PID"
# Wait for health (up to 10 min)
HEALTHY=0
for i in $(seq 1 120); do
    kill -0 $SGLANG_PID 2>/dev/null || {{ echo "sglang DIED"; exit 1; }}
    python3 -c "import urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({{}})); o.open(urllib.request.Request('http://127.0.0.1:30000/health'),timeout=5).read()" 2>/dev/null && {{ HEALTHY=1; break; }}
    sleep 5
done
if [ $HEALTHY -eq 0 ]; then echo "sglang TIMEOUT"; exit 1; fi
echo "sglang READY"
# Start rollout worker
python3 -m sao.standalone.rollout_worker \
    --model-path {MODEL_PATH} \
    --data {DATA_FILE} \
    --sglang-host 127.0.0.1 --sglang-port 30000 \
    --queue-dir queue --checkpoint-dir checkpoints/sao \
    --temperature 1.0 --top-p 1.0 \
    --max-new-tokens {MAX_TOKENS} > logs/rollout_{name}.log 2>&1 &
echo "rollout PID=$!"
echo "ALL_OK {name}"
"""
        ssh_bg(ip, start_cmd)
        print("launched (waiting for sglang...)")

    # 2. Wait for all inference machines to be ready
    print("\n--- Waiting for all sglang instances ---")
    for name, ip in INFER_MACHINES:
        print(f"  {name}...", end=" ", flush=True)
        for attempt in range(120):
            code, out, _ = ssh_run(ip, "tail -1 /home/jovyan/h800fast/wangzekai/slime_sao/logs/sglang_" + name + ".log 2>/dev/null", timeout=5)
            if "ALL_OK" in out or "READY" in out:
                print("✓ ready")
                break
            if attempt % 12 == 0:
                print(f"({attempt*5}s)", end=" ", flush=True)
            time.sleep(5)
        else:
            print("✗ TIMEOUT")

    # 3. Check queue has data
    print("\n--- Waiting for trajectories in queue ---")
    for attempt in range(60):
        n = len([f for f in os.listdir(f"{WORKDIR}/queue/pending") if f.endswith(".json")]) if os.path.exists(f"{WORKDIR}/queue/pending") else 0
        if n >= BATCH_SIZE:
            print(f"  Queue: {n} trajectories ✓")
            break
        if attempt % 6 == 0:
            print(f"  Queue: {n}/{BATCH_SIZE}...", flush=True)
        time.sleep(10)
    else:
        print(f"  Queue: {n} (starting trainer anyway)")

    # 4. Start trainer locally
    print("\n--- Starting trainer on ctm-06 ---")
    train_cmd = f"""cd {WORKDIR}
export ROOTFS={ROOTFS}
export LD_LIBRARY_PATH=$ROOTFS/usr/local/cuda/lib64:$ROOTFS/usr/local/nvidia/lib64
export PYTHONPATH={WORKDIR}:$ROOTFS/usr/local/lib/python3.12/dist-packages
export PATH=$ROOTFS/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export no_proxy='*' NO_PROXY='*' CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m sao.standalone.trainer \
    --model-path {MODEL_PATH} \
    --critic-path {MODEL_PATH} \
    --queue-dir queue --save-dir checkpoints/sao \
    --num-steps {NUM_STEPS} --batch-size {BATCH_SIZE} \
    --lr 1e-6 --critic-lr 5e-6 \
    --clip-low 0.7 --clip-high 6.0 \
    --gamma 1.0 --gae-alpha 1.5 \
    --critic-k 2 --critic-warmup 10 \
    --value-clip 0.2 --save-interval 50 \
    --max-seq-len {MAX_TOKENS} \
    --use-8bit-adam 2>&1 | tee logs/trainer_$(date +%Y%m%d_%H%M%S).log
"""
    log_file = f"{WORKDIR}/logs/trainer_cluster.log"
    print(f"  Log: {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            ["bash", "-c", train_cmd],
            stdout=f, stderr=subprocess.STDOUT
        )
    print(f"  Trainer PID: {proc.pid}")
    print(f"  Monitor: tail -f {log_file}")

    print("\n" + "=" * 60)
    print("✓ Cluster running!")
    print(f"  4 inference machines generating trajectories")
    print(f"  1 training machine (PID {proc.pid})")
    print(f"  Monitor: python3 cluster.py status")
    print(f"  Logs: python3 cluster.py logs")
    print(f"  Stop:  python3 cluster.py stop")
    print("=" * 60)


def cmd_stop():
    """Stop everything on all machines."""
    print("=" * 60)
    print("Stopping SAO Cluster")
    print("=" * 60)

    for name, ip in ALL_MACHINES:
        print(f"  {name} ({ip})...", end=" ", flush=True)
        ssh_run(ip, "pkill -9 -f sglang.launch_server 2>/dev/null; pkill -9 -f rollout_worker 2>/dev/null; pkill -9 -f 'sao.standalone.trainer' 2>/dev/null; echo done", timeout=10)
        print("stopped")

    # Also kill local trainer
    os.system("pkill -f 'sao.standalone.trainer' 2>/dev/null")
    print("\n✓ All stopped")


def cmd_status():
    """Check status of all machines."""
    print("=" * 60)
    print("SAO Cluster Status")
    print("=" * 60)

    # Queue
    pending = len([f for f in os.listdir(f"{WORKDIR}/queue/pending") if f.endswith(".json")]) if os.path.exists(f"{WORKDIR}/queue/pending") else 0
    done = len([f for f in os.listdir(f"{WORKDIR}/queue/done") if f.endswith(".json")]) if os.path.exists(f"{WORKDIR}/queue/done") else 0
    print(f"\nQueue: {pending} pending, {done} done")

    # Checkpoints
    ckpts = sorted([d for d in os.listdir(f"{WORKDIR}/checkpoints/sao") if d.startswith("step_")]) if os.path.exists(f"{WORKDIR}/checkpoints/sao") else []
    if ckpts:
        print(f"Checkpoints: {', '.join(ckpts[-3:])}")

    # Machine status
    print(f"\n{'Machine':<12} {'Role':<10} {'sglang':<8} {'rollout':<8} {'GPU mem':<12} {'throughput':<12}")
    print("-" * 70)

    for name, ip in INFER_MACHINES:
        _, sglang, _ = ssh_run(ip, "pgrep -f sglang.launch_server | head -1", timeout=5)
        _, rollout, _ = ssh_run(ip, "pgrep -f rollout_worker | head -1", timeout=5)
        _, gpu, _ = ssh_run(ip, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1", timeout=5)
        _, tput, _ = ssh_run(ip, f"tail -1 {WORKDIR}/logs/sglang_{name}.log 2>/dev/null | grep -oP 'throughput \(token/s\): \K[0-9.]+' | tail -1", timeout=5)
        print(f"  {name:<10} inference  {'✓' if sglang else '✗':<8} {'✓' if rollout else '✗':<8} {gpu+'MB':<12} {tput+' tok/s':<12}")

    name, ip = TRAIN_MACHINE
    _, trainer, _ = ssh_run(ip, "pgrep -f 'sao.standalone.trainer' | head -1", timeout=5)
    _, gpu, _ = ssh_run(ip, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1", timeout=5)
    _, last, _ = ssh_run(ip, f"tail -1 {WORKDIR}/logs/trainer_cluster.log 2>/dev/null", timeout=5)
    print(f"  {name:<10} training   {'✓' if trainer else '✗':<8} {'—':<8} {gpu+'MB':<12} {'—':<12}")

    if last:
        print(f"\nTrainer: {last[:100]}")

    # Latest rollout reward
    for name, ip in INFER_MACHINES:
        _, reward, _ = ssh_run(ip, f"tail -3 {WORKDIR}/logs/rollout_{name}.log 2>/dev/null | grep -oP 'avg100=[0-9.]+' | tail -1", timeout=5)
        if reward:
            print(f"  {name} reward: {reward}")


def cmd_logs():
    """Tail trainer log."""
    os.system(f"tail -50 {WORKDIR}/logs/trainer_cluster.log 2>/dev/null || echo 'No trainer log yet'")


def cmd_queue():
    """Show queue status."""
    pending = len([f for f in os.listdir(f"{WORKDIR}/queue/pending") if f.endswith(".json")]) if os.path.exists(f"{WORKDIR}/queue/pending") else 0
    done = len([f for f in os.listdir(f"{WORKDIR}/queue/done") if f.endswith(".json")]) if os.path.exists(f"{WORKDIR}/queue/done") else 0
    print(f"Queue: {pending} pending, {done} done (total consumed: {done})")
    
    # Recent reward from queue
    import glob, json
    files = sorted(glob.glob(f"{WORKDIR}/queue/pending/traj_*.json"))[-5:]
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
                print(f"  {os.path.basename(f)}: reward={d['reward']}, len={d.get('resp_len', '?')}")
        except Exception:
            pass


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "setup":
        cmd_setup()
    elif cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "status":
        cmd_status()
    elif cmd == "logs":
        cmd_logs()
    elif cmd == "queue":
        cmd_queue()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
