#!/bin/bash
# ============================================================
# 每次进入 slime chroot 环境 (bind mount + chroot)
#
# 用法:
#   bash run_env.sh             # 进交互 shell
#   bash run_env.sh CMD args..  # 在 chroot 内跑命令
#
# 例子:
#   bash run_env.sh python -c "import torch; print(torch.__version__)"
#   bash run_env.sh nvidia-smi
#   bash run_env.sh cd /root/slime \&\& python train.py --help
#
# 配置可通过环境变量覆盖:
#   ROOTFS=... bash run_env.sh
#   HOST_BINDS="/home/jovyan/h800fast /data" bash run_env.sh
# ============================================================
set -euo pipefail

# ----------------- 配置 -----------------
ROOTFS="${ROOTFS:-/home/jovyan/h800fast/wangzekai/slime_rootfs}"
# 要 bind mount 进 chroot 的 host 目录 (让 chroot 能访问模型/数据/代码)
HOST_BINDS="${HOST_BINDS:-/home/jovyan/h800fast}"
# ----------------------------------------

[ "$(id -u)" != "0" ] && { echo "ERROR: 必须 root 运行"; exit 1; }
[ ! -x "$ROOTFS/bin/bash" ] && { echo "ERROR: rootfs 不存在: $ROOTFS"; echo "先跑: bash setup_env.sh"; exit 1; }

# 清理可能残留的 mount (避免 "already mounted" 错误)
echo "=== 清理旧 mount ==="
for sub in proc sys dev; do
    if mountpoint -q "$ROOTFS/$sub" 2>/dev/null; then
        umount -R "$ROOTFS/$sub" 2>/dev/null || umount -f "$ROOTFS/$sub" 2>/dev/null || true
    fi
done
for host_dir in $HOST_BINDS; do
    mountpoint -q "$ROOTFS$host_dir" 2>/dev/null && umount "$ROOTFS$host_dir" 2>/dev/null || true
done

# bind mount
echo "=== bind mount ==="
mount --bind /proc "$ROOTFS/proc" && echo "✅ /proc"
mount --bind /sys  "$ROOTFS/sys"  && echo "✅ /sys"
mount --bind /dev  "$ROOTFS/dev"  && echo "✅ /dev"
mkdir -p "$ROOTFS/dev/pts"
mount -t devpts devpts "$ROOTFS/dev/pts" 2>/dev/null && echo "✅ /dev/pts" || echo "⚠️  /dev/pts (非致命)"

# host 工作目录
for host_dir in $HOST_BINDS; do
    [ -d "$host_dir" ] || { echo "⚠️  跳过 (host 不存在): $host_dir"; continue; }
    mkdir -p "$ROOTFS$host_dir"
    mount --bind "$host_dir" "$ROOTFS$host_dir" && echo "✅ $host_dir" || echo "⚠️  $host_dir mount failed"
done

# DNS (chroot 内能解析域名)
cp /etc/resolv.conf "$ROOTFS/etc/resolv.conf" 2>/dev/null || true
cp /etc/hosts       "$ROOTFS/etc/hosts"       2>/dev/null || true

# 进 chroot
ENV_VARS=(
    HOME=/root
    TERM="${TERM:-xterm-256color}"
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/nvidia/bin
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    NVIDIA_VISIBLE_DEVICES=all
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
    PYTHONPATH=/root/Megatron-LM:/root/slime
    CUDA_DEVICE_MAX_CONNECTIONS=1
    PYTHONUNBUFFERED=1
)

if [ $# -gt 0 ]; then
    # 非交互: 在 chroot 内执行命令
    exec chroot "$ROOTFS" /usr/bin/env -i "${ENV_VARS[@]}" /bin/bash -c "$*"
else
    echo ""
    echo "=== 进入 chroot ($ROOTFS) ==="
    echo "退出: exit 或 Ctrl+D"
    echo ""
    exec chroot "$ROOTFS" /usr/bin/env -i "${ENV_VARS[@]}" /bin/bash --login
fi
