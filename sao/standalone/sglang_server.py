"""Launch and manage a standalone sglang server.

No slime, no ray, no torch_memory_saver. Just subprocess + HTTP.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request

WRK = os.environ.get("SAO_WORKDIR", "/home/jovyan/h800fast/wangzekai/slime_sao")
ROOTFS = os.environ.get("SAO_ROOTFS", "/home/jovyan/h800fast/wangzekai/slime_rootfs")


def env_setup():
    """Return dict of env vars needed for sglang on this box."""
    site = f"{ROOTFS}/usr/local/lib/python3.12/dist-packages"
    return {
        "LD_LIBRARY_PATH": f"{ROOTFS}/usr/local/cuda/lib64:{ROOTFS}/usr/local/nvidia/lib64",
        "PYTHONPATH": f"{site}:{ROOTFS}/tmp/local_src/python:{ROOTFS}/root/slime:{ROOTFS}/root/Megatron-LM:{WRK}",
        "PATH": f"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:{ROOTFS}/usr/local/cuda/bin",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NVTE_FRAMEWORK": "pytorch",
        "no_proxy": "127.0.0.1,localhost",
        "NO_PROXY": "127.0.0.1,localhost",
    }


def start_server(
    model_path: str,
    port: int = 30000,
    tp: int = 4,
    mem_fraction_static: float = 0.85,
    max_total_tokens: int = 8192,
    disable_cuda_graph: bool = True,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch sglang server. Returns the Popen handle."""
    py = f"{ROOTFS}/usr/bin/python3"
    cmd = [
        py, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp", str(tp),
        "--mem-fraction-static", str(mem_fraction_static),
        "--context-length", str(max_total_tokens),
        "--reasoning-parser", "qwen3",
    ]
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if extra_args:
        cmd.extend(extra_args)

    full_env = dict(os.environ)
    full_env.update(env_setup())

    proc = subprocess.Popen(cmd, env=full_env, stdout=sys.stderr, stderr=sys.stderr)
    _wait_ready(port, proc, timeout=600)
    return proc


def _wait_ready(port: int, proc: subprocess.Popen, timeout: int = 600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"sglang server exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            print(f"[sglang] Server ready at port {port} ({time.time()-t0:.0f}s)")
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"sglang server not ready within {timeout}s")


def stop_server(proc: subprocess.Popen):
    """Kill sglang server gracefully."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("[sglang] Server stopped")


def update_weights(proc_port_or_server_args, hf_model_path: str):
    """Tell sglang to reload weights from a HF checkpoint path.

    Uses sglang's /update_weights_from_distributed endpoint if available,
    otherwise the caller should stop + restart the server.
    """
    # For now: caller stops and restarts. This is a placeholder.
    raise NotImplementedError("Use stop_server + start_server for weight updates")
