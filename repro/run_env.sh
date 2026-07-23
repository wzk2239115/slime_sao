#!/bin/bash
# ============================================================
# 用 slime image 的 python + 库跑命令 (不需要 chroot/mount)
#
# 原理: 用 image 的 python binary ($ROOTFS/usr/bin/python3),
#       LD_LIBRARY_PATH 指向 image 的 CUDA 库,
#       PYTHONPATH 指向 image 的 python 包.
#       不需要 docker daemon, 不需要 mount 权限.
#
# 用法:
#   bash run_env.sh             # 进交互 shell (用 image 的 bash)
#   bash run_env.sh python ...  # 用 image python 跑命令
#   bash run_env.sh CMD args..  # 在 PATH 内跑任意命令
#
# 配置可通过环境变量覆盖:
#   ROOTFS=/path bash run_env.sh
# ============================================================
set -euo pipefail

# ----------------- 配置 -----------------
ROOTFS="${ROOTFS:-/home/jovyan/h800fast/wangzekai/slime_rootfs}"
# 用户工作目录 (包含 SAO 代码, 模型权重, 数据集)
HOST_WORK="${HOST_WORK:-/home/jovyan/h800fast/wangzekai}"
# ----------------------------------------

[ ! -x "$ROOTFS/bin/bash" ] && { echo "ERROR: rootfs 不存在: $ROOTFS"; echo "先跑: bash setup_env.sh"; exit 1; }

SITE="$ROOTFS/usr/local/lib/python3.12/dist-packages"

# 环境变量 (模仿 NGC image 的默认环境)
export HOME="${HOME:-/root}"
export TERM="${TERM:-xterm-256color}"
# PATH: host 系统命令在前, image 的 cuda/bin 和 /usr/local/bin (ray/sgl-*等) 在后
# (不能用 image 的 /usr/bin, 因为 image 的 ls/cat 等需要 GLIBC_2.38, host 没有)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$ROOTFS/usr/local/cuda/bin:$ROOTFS/usr/local/bin"
export LD_LIBRARY_PATH="$ROOTFS/usr/local/cuda/lib64:$ROOTFS/usr/local/nvidia/lib64"
export PYTHONPATH="$HOST_WORK/slime_sao/patch:$ROOTFS/tmp/local_src/python:$ROOTFS/sgl-workspace/sglang/python:$SITE:$ROOTFS/root/slime:$ROOTFS/root/Megatron-LM:$HOST_WORK/slime_sao"
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1  # 避免 __pycache__ 加载旧版本 (image 的 .py 可能比 .pyc 新)
export NVTE_FRAMEWORK=pytorch

# Python 解释器 (image 的 python, 编译时对齐 torch/TE 的 ABI)
PY="$ROOTFS/usr/bin/python3"

# image 的命令行工具 (ray, sgl-*) 的 shebang 是 #!/usr/bin/python3
# 让 /usr/bin/python3 和 /usr/local/bin/python{,3} 都指向 image 的 python
# (host 容器里 /usr/bin/python3 没有装 torch/TE, 必须覆盖)
ln -sf "$PY" /usr/local/bin/python3
ln -sf "$PY" /usr/local/bin/python
# 只在 /usr/bin/python3 存在但不是 image python 时覆盖
if [ "$(readlink -f /usr/bin/python3 2>/dev/null)" != "$PY" ]; then
    [ ! -f /usr/bin/python3.host.bak ] && cp -a /usr/bin/python3 /usr/bin/python3.host.bak 2>/dev/null || true
    ln -sf "$PY" /usr/bin/python3
fi

if [ $# -gt 0 ]; then
    # 非交互: 在配置好的环境下跑命令
    # 第一个参数是命令名 (python/python3/nvidia-smi/...), 后面是参数
    exec "$@"
else
    # 交互: 进 bash (PATH/PYTHONPATH/LD_LIBRARY_PATH 都已设好)
    echo "=== slime 环境 (ROOTFS=$ROOTFS) ==="
    echo "Python: $PY"
    echo "PYTHONPATH 首项: $(echo $PYTHONPATH | cut -d: -f1)"
    echo ""
    echo "快捷命令:"
    echo "  python                # image 的 python 3.12 (torch/TE/megatron 都能 import)"
    echo "  nvidia-smi            # GPU 状态"
    echo "  python train.py ...   # 跑 slime 训练"
    echo ""
    exec /bin/bash --noprofile --norc
fi
