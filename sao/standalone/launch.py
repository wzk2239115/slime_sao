#!/usr/bin/env python3
"""SAO 单机启动器 — 每台机器跑一条命令，通过共享 NFS 协调。

用法 (在对应机器上执行):

  # 推理机 (ctm-05, ctm-01, ctm-02, ctm-04):
  BASH_ENV= python3 sao/standalone/launch.py inference

  # 训练机 (ctm-06):
  BASH_ENV= python3 sao/standalone/launch.py train

  # 停止:
  BASH_ENV= python3 sao/standalone/launch.py stop
"""
from __future__ import annotations

import os, sys, time, subprocess, signal, socket

# ============================================================
# Config
# ============================================================
WORKDIR = "/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS  = "/home/jovyan/h800fast/wangzekai/slime_rootfs"
MODEL   = "models/Qwen3-30B-A3B-Thinking-2507"
DATA    = "datasets/MATH_train.jsonl"
PORT    = 30000
MAX_TOKENS = 32768

def setup_env():
    """Set all environment variables + symlinks."""
    os.environ.update({
        "ROOTFS": ROOTFS,
        "LD_LIBRARY_PATH": f"{ROOTFS}/usr/local/cuda/lib64:{ROOTFS}/usr/local/nvidia/lib64",
        "PYTHONPATH": f"{WORKDIR}:{ROOTFS}/usr/local/lib/python3.12/dist-packages",
        "PATH": f"{ROOTFS}/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "no_proxy": "*", "NO_PROXY": "*",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    # Symlinks (idempotent)
    for src, dst in [
        (f"{ROOTFS}/sgl-workspace", "/sgl-workspace"),
        (f"{ROOTFS}/usr/bin/python3", "/usr/bin/python3"),
        (f"{ROOTFS}/usr/bin/python3", "/usr/local/bin/python3"),
    ]:
        try: os.symlink(src, dst)
        except FileExistsError: pass
    os.makedirs("/usr/local/lib/python3.12/dist-packages", exist_ok=True)
    pth = "/usr/local/lib/python3.12/dist-packages/rootfs_packages.pth"
    with open(pth, "w") as f:
        f.write(f"{ROOTFS}/usr/local/lib/python3.12/dist-packages\n")
    # Dirs
    for d in ["queue/pending", "queue/done", "logs", "checkpoints/sao"]:
        os.makedirs(f"{WORKDIR}/{d}", exist_ok=True)
    os.chdir(WORKDIR)


def check_health(host="127.0.0.1", port=PORT):
    """Check sglang health."""
    import urllib.request
    try:
        o = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        o.open(urllib.request.Request(f"http://{host}:{port}/health"), timeout=5).read()
        return True
    except Exception:
        return False


def wait_health(timeout=600, tag=""):
    """Wait for sglang to be healthy."""
    print(f"[{tag}] Waiting for sglang...", end=" ", flush=True)
    for i in range(timeout // 5):
        if check_health():
            print("ready!")
            return True
        time.sleep(5)
        if i % 12 == 0 and i > 0:
            print(f"{i*5}s", end=" ", flush=True)
    print("TIMEOUT!")
    return False


# ============================================================
# Inference mode
# ============================================================
def start_inference():
    setup_env()
    hostname = socket.gethostname()
    print(f"=== SAO Inference on {hostname} ===")

    # Kill old processes
    os.system("pkill -9 -f sglang.launch_server 2>/dev/null; pkill -9 -f rollout_worker 2>/dev/null; sleep 2")

    num_gpu = subprocess.check_output("nvidia-smi -L 2>/dev/null | wc -l", shell=True).decode().strip()
    print(f"GPUs: {num_gpu}")

    # Start sglang
    log_sglang = f"{WORKDIR}/logs/sglang_{hostname}.log"
    print(f"Starting sglang... (log: {log_sglang})")
    sglang = subprocess.Popen(
        ["python3", "-m", "sglang.launch_server",
         "--model-path", MODEL,
         "--host", "0.0.0.0", "--port", str(PORT),
         "--tp", num_gpu,
         "--mem-fraction-static", "0.85",
         "--context-length", "36864",
         "--reasoning-parser", "qwen3"],
        stdout=open(log_sglang, "w"), stderr=subprocess.STDOUT,
    )
    print(f"sglang PID: {sglang.pid}")

    if not wait_health(tag=hostname):
        print("ERROR: sglang failed to start. Check log:")
        os.system(f"tail -20 {log_sglang}")
        return

    # Start rollout worker
    log_rollout = f"{WORKDIR}/logs/rollout_{hostname}.log"
    print(f"Starting rollout worker... (log: {log_rollout})")
    rollout = subprocess.Popen(
        ["python3", "-m", "sao.standalone.rollout_worker",
         "--model-path", MODEL,
         "--data", DATA,
         "--sglang-host", "127.0.0.1", "--sglang-port", str(PORT),
         "--queue-dir", "queue", "--checkpoint-dir", "checkpoints/sao",
         "--temperature", "1.0", "--top-p", "1.0",
         "--max-new-tokens", str(MAX_TOKENS)],
        stdout=open(log_rollout, "w"), stderr=subprocess.STDOUT,
    )
    print(f"rollout PID: {rollout.pid}")
    print(f"\n✓ Inference running on {hostname}")
    print(f"  Monitor: tail -f {log_rollout}")
    print(f"  Stop:    BASH_ENV= python3 sao/standalone/launch.py stop")

    # Wait for either process to exit
    try:
        while sglang.poll() is None and rollout.poll() is None:
            time.sleep(10)
    except KeyboardInterrupt:
        pass

    print("Process exited, cleaning up...")
    sglang.terminate()
    rollout.terminate()


# ============================================================
# Training mode
# ============================================================
def start_train():
    setup_env()
    hostname = socket.gethostname()
    print(f"=== SAO Trainer on {hostname} ===")

    # Clear stale signals
    for f in [".reload_signal", ".reload_done"]:
        try: os.remove(f"{WORKDIR}/checkpoints/sao/{f}")
        except FileNotFoundError: pass

    # Wait for queue
    print("Waiting for trajectories in queue...", end=" ", flush=True)
    import glob
    while True:
        n = len(glob.glob(f"{WORKDIR}/queue/pending/traj_*.json"))
        if n >= 8:
            print(f"{n} ready!")
            break
        print(f"{n}", end=" ", flush=True)
        time.sleep(10)

    # Start trainer (foreground, so user sees output)
    log_file = f"{WORKDIR}/logs/trainer_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"Starting trainer... (log: {log_file})")
    print(f"  batch_size=8  lr=1e-6  critic_lr=5e-6")
    print(f"  DIS: clip=[0.7, 6.0]  GAE: gamma=1.0 alpha=1.5")
    print(f"  TTUR K=2  warmup=10  frozen_attn  8bit_adam")
    print()

    trainer = subprocess.Popen(
        ["python3", "-m", "sao.standalone.trainer",
         "--model-path", MODEL,
         "--critic-path", MODEL,
         "--queue-dir", "queue",
         "--save-dir", "checkpoints/sao",
         "--num-steps", "1000",
         "--batch-size", "8",
         "--lr", "1e-6", "--critic-lr", "5e-6",
         "--clip-low", "0.7", "--clip-high", "6.0",
         "--gamma", "1.0", "--gae-alpha", "1.5",
         "--critic-k", "2", "--critic-warmup", "10",
         "--value-clip", "0.2", "--save-interval", "50",
         "--max-seq-len", str(MAX_TOKENS),
         "--use-8bit-adam"],
    )
    print(f"Trainer PID: {trainer.pid}")
    print(f"\n Monitor: tail -f {log_file}")
    print(f" Stop:    BASH_ENV= python3 sao/standalone/launch.py stop")
    trainer.wait()


# ============================================================
# Stop mode
# ============================================================
def stop_all():
    print("Stopping SAO processes...")
    os.system("pkill -9 -f sglang.launch_server 2>/dev/null")
    os.system("pkill -9 -f rollout_worker 2>/dev/null")
    os.system("pkill -9 -f 'sao.standalone.trainer' 2>/dev/null")
    print("Done.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"
    if mode == "inference":
        start_inference()
    elif mode == "train":
        start_train()
    elif mode == "stop":
        stop_all()
    elif mode == "health":
        setup_env()
        print("healthy" if check_health() else "not running")
    else:
        print(__doc__)
