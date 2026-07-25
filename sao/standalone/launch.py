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

import os, sys, time, subprocess, signal, socket, threading, glob

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


def get_latest_checkpoint():
    """Find latest checkpoint directory."""
    ckpt_dir = f"{WORKDIR}/checkpoints/sao"
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("step_")])
    if not ckpts:
        return None
    return os.path.join(ckpt_dir, ckpts[-1])


def start_sglang_proc(model_path, num_gpu, log_file):
    """Start sglang server process."""
    disable_cg = os.environ.get("DISABLE_CUDA_GRAPH", "0") == "1"
    cg_flag = ["--disable-cuda-graph"] if disable_cg else []
    return subprocess.Popen(
        ["python3", "-m", "sglang.launch_server",
         "--model-path", model_path,
         "--host", "0.0.0.0", "--port", str(PORT),
         "--tp", str(num_gpu),
         "--mem-fraction-static", "0.85",
         "--context-length", "36864",
         "--disable-custom-all-reduce",
         "--reasoning-parser", "qwen3"] + cg_flag,
        stdout=open(log_file, "a"), stderr=subprocess.STDOUT,
    )


def reload_watcher(state, hostname, num_gpu, log_file):
    """Background thread: watch for .reload_signal, restart sglang with new checkpoint."""
    ckpt_dir = f"{WORKDIR}/checkpoints/sao"
    signal_file = f"{ckpt_dir}/.reload_signal_{hostname}"
    done_file = f"{ckpt_dir}/.reload_done_{hostname}"
    while True:
        time.sleep(10)
        if os.path.exists(signal_file):
            try: os.remove(signal_file)
            except FileNotFoundError: pass
            if os.path.exists(done_file):
                try: os.remove(done_file)
                except FileNotFoundError: pass
            print("[watcher] Reload signal → restarting sglang...", flush=True)
            # Kill old sglang
            if state.get("sglang"):
                state["sglang"].terminate()
                try: state["sglang"].wait(timeout=30)
                except: state["sglang"].kill()
            # Find latest checkpoint
            latest = get_latest_checkpoint()
            model_path = latest if latest else MODEL
            print(f"[watcher] Loading: {model_path}", flush=True)
            state["sglang"] = start_sglang_proc(model_path, num_gpu, log_file)
            # Wait for health
            if wait_health(tag="reload"):
                with open(done_file, "w") as f: f.write(str(time.time()))
                print(f"[watcher] sglang reloaded ✓", flush=True)
            else:
                with open(done_file, "w") as f: f.write(str(time.time()))
                print(f"[watcher] reload FAILED, unblocked anyway", flush=True)


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

    num_gpu = int(subprocess.check_output("nvidia-smi -L 2>/dev/null | wc -l", shell=True).decode().strip())
    print(f"GPUs: {num_gpu}")

    # Load latest checkpoint if exists, else base model
    latest = get_latest_checkpoint()
    model_to_load = latest if latest else MODEL
    if latest:
        print(f"Loading checkpoint: {latest}")
    else:
        print(f"No checkpoint, using base model")

    # Start sglang
    log_sglang = f"{WORKDIR}/logs/sglang_{hostname}.log"
    disable_cg = os.environ.get("DISABLE_CUDA_GRAPH", "0") == "1"
    print(f"Starting sglang (CUDA graph: {'OFF' if disable_cg else 'ON'})...")
    sglang = start_sglang_proc(model_to_load, num_gpu, log_sglang)
    print(f"sglang PID: {sglang.pid}")

    if not wait_health(tag=hostname):
        print("ERROR: sglang failed. Check log:")
        os.system(f"tail -20 {log_sglang}")
        return

    # Start reload watcher (background thread)
    # Clear stale signals for this hostname
    for f in [f".reload_signal_{hostname}", f".reload_done_{hostname}"]:
        try: os.remove(f"{WORKDIR}/checkpoints/sao/{f}")
        except FileNotFoundError: pass
    state = {"sglang": sglang}
    watcher = threading.Thread(
        target=reload_watcher, args=(state, hostname, num_gpu, log_sglang), daemon=True
    )
    watcher.start()
    print("Reload watcher started (auto-restart sglang on new checkpoint)")

    # Start rollout worker
    log_rollout = f"{WORKDIR}/logs/rollout_{hostname}.log"
    enable_tir = os.environ.get("ENABLE_TIR", "0") == "1"
    tir_flag = ["--enable-tir"] if enable_tir else []
    tir_mode = "TIR (Python tools)" if enable_tir else "pure reasoning"
    concurrency = os.environ.get("CONCURRENCY", "1")
    print(f"Starting rollout worker ({tir_mode}, concurrency={concurrency})... (log: {log_rollout})")
    rollout = subprocess.Popen(
        ["python3", "-m", "sao.standalone.rollout_worker",
         "--model-path", MODEL,
         "--data", DATA,
         "--sglang-host", "127.0.0.1", "--sglang-port", str(PORT),
         "--queue-dir", "queue", "--checkpoint-dir", "checkpoints/sao",
         "--temperature", "1.0", "--top-p", "1.0",
         "--max-new-tokens", str(MAX_TOKENS),
         "--concurrency", concurrency] + tir_flag,
        stdout=open(log_rollout, "w"), stderr=subprocess.STDOUT,
    )
    print(f"rollout PID: {rollout.pid}")
    print(f"\n✓ Inference running on {hostname}")
    print(f"  Monitor:  tail -f {log_rollout}")
    print(f"  sglang:   tail -f {log_sglang}")
    print(f"  Stop:     BASH_ENV= python3 sao/standalone/launch.py stop")

    # Wait for either process to exit
    try:
        while state["sglang"].poll() is None and rollout.poll() is None:
            time.sleep(10)
    except KeyboardInterrupt:
        pass

    print("Process exited, cleaning up...")
    state["sglang"].terminate()
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

    import shlex
    trainer_cmd = [
        "python3", "-m", "sao.standalone.trainer",
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
        "--use-8bit-adam",
    ]
    cmd_str = " ".join(shlex.quote(a) for a in trainer_cmd)
    print(f"Trainer PID: starting...")
    print(f"\n Monitor: tail -f {log_file}")
    print(f" Stop:    BASH_ENV= python3 sao/standalone/launch.py stop\n")
    os.system(f"{cmd_str} 2>&1 | tee {log_file}")


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
