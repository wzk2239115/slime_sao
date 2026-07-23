#!/bin/bash
# 一键升级 host 的 glibc 2.35 → 2.38 + libstdc++ → 支持 GLIBCXX_3.4.32
# 从 rootfs 复制，原子替换。备份旧文件可回滚。
#
# 用法: bash upgrade_glibc.sh
set -euo pipefail

ROOTFS="${ROOTFS:-/home/jovyan/h800fast/wangzekai/slime_rootfs}"
RFS_LIB="$ROOTFS/lib/x86_64-linux-gnu"
RFS_USR_LIB="$ROOTFS/usr/lib/x86_64-linux-gnu"

HOST_LIB="/lib/x86_64-linux-gnu"
HOST_USR_LIB="/usr/lib/x86_64-linux-gnu"
BACKUP_DIR="/opt/glibc235-backup"

echo "=== 检查 rootfs ==="
[ ! -f "$RFS_LIB/libc.so.6" ] && { echo "ERROR: rootfs libc 不存在"; exit 1; }

echo "rootfs glibc 版本:"
strings "$RFS_LIB/libc.so.6" | grep "^GLIBC_2\." | sort -V | tail -3
echo "rootfs libstdc++ 版本:"
strings "$RFS_USR_LIB/libstdc++.so.6" | grep GLIBCXX | sort -V | tail -3
echo ""
echo "host glibc 版本:"
strings "$HOST_LIB/libc.so.6" | grep "^GLIBC_2\." | sort -V | tail -3
echo "host libstdc++ 版本:"
strings "$HOST_USR_LIB/libstdc++.so.6" | grep GLIBCXX | sort -V | tail -3
echo ""

# 1. 备份
echo "=== 备份旧文件到 $BACKUP_DIR ==="
mkdir -p "$BACKUP_DIR"
# glibc 组件
for f in libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 \
         libresolv.so.2 libutil.so.1 ld-linux-x86-64.so.2; do
    [ -f "$HOST_LIB/$f" ] && cp -a "$HOST_LIB/$f" "$BACKUP_DIR/" && echo "  备份 $f"
done
# libstdc++
for f in $(ls "$HOST_USR_LIB"/libstdc++.so.6* 2>/dev/null); do
    cp -a "$f" "$BACKUP_DIR/" && echo "  备份 $(basename $f)"
done

# 2. 替换 glibc（先 cp 为 .new，再原子 mv）
echo ""
echo "=== 替换 glibc ==="
for f in libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 \
         libresolv.so.2 libutil.so.1; do
    if [ -f "$RFS_LIB/$f" ]; then
        cp "$RFS_LIB/$f" "$HOST_LIB/$f.new"
        mv "$HOST_LIB/$f.new" "$HOST_LIB/$f"
        echo "  替换 $f"
    fi
done

# ld-linux-x86-64.so.2
if [ -f "$RFS_LIB/ld-linux-x86-64.so.2" ]; then
    cp "$RFS_LIB/ld-linux-x86-64.so.2" "$HOST_LIB/ld-linux-x86-64.so.2.new"
    mv "$HOST_LIB/ld-linux-x86-64.so.2.new" "$HOST_LIB/ld-linux-x86-64.so.2"
    echo "  替换 ld-linux-x86-64.so.2"
fi

# 3. 替换 libstdc++
echo ""
echo "=== 替换 libstdc++ ==="
# 先找 rootfs 里 libstdc++.so.6 的实际文件 (可能是 symlink → libstdc++.so.6.0.x)
RFS_STDCPP_REAL=$(readlink -f "$RFS_USR_LIB/libstdc++.so.6")
echo "  rootfs libstdc++ 实际文件: $RFS_STDCPP_REAL"
cp "$RFS_STDCPP_REAL" "$HOST_USR_LIB/"
# 更新 symlink
ln -sf "$(basename "$RFS_STDCPP_REAL")" "$HOST_USR_LIB/libstdc++.so.6"
echo "  替换 libstdc++.so.6 → $(basename "$RFS_STDCPP_REAL")"

# 4. 验证
echo ""
echo "=== 验证 ==="
echo "新 glibc 版本:"
strings "$HOST_LIB/libc.so.6" | grep "^GLIBC_2\." | sort -V | tail -3
echo "新 libstdc++ 版本:"
strings "$HOST_USR_LIB/libstdc++.so.6" | grep GLIBCXX | sort -V | tail -3

echo ""
echo "测试基本命令:"
ls /tmp >/dev/null && echo "  ls OK"
python3 --version 2>&1 && echo "  python3 OK"
bash -c 'echo test' && echo "  bash OK"

echo ""
echo "=========================================="
echo "glibc 升级完成!"
echo "备份在: $BACKUP_DIR"
echo ""
echo "如需回滚:"
echo "  cd $BACKUP_DIR"
echo "  cp libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 ld-linux-x86-64.so.2 $HOST_LIB/"
echo "  cp libstdc++.so.6* $HOST_USR_LIB/"
echo "=========================================="
