#!/bin/bash
# ============================================================
# 一次性: 把 slimerl/slime:latest docker image 解压成 rootfs
#
# 之后日常使用只需要: bash run_env.sh
# 换机器时: rsync 整个 $ROOTFS 到新机器, 再跑 bash run_env.sh
#
# 用法: bash setup_env.sh
# ============================================================
set -euo pipefail

# ----------------- 配置 -----------------
TAR="${TAR:-/home/jovyan/h800fast/wangzekai/slime_sao/slime_latest.tar}"
ROOTFS="${ROOTFS:-/home/jovyan/h800fast/wangzekai/slime_rootfs}"
TMP_ROOTFS="/tmp/slime_rootfs"  # 之前可能解压过的临时位置
# ----------------------------------------

[ "$(id -u)" != "0" ] && { echo "ERROR: 必须 root 运行"; exit 1; }
[ ! -f "$TAR" ] && { echo "ERROR: TAR 不存在: $TAR"; exit 1; }

# 如果 /tmp 里之前解压过, 直接 mv 过去 (省 30 分钟)
if [ -d "$TMP_ROOTFS" ] && [ -x "$TMP_ROOTFS/bin/bash" ] && [ ! -x "$ROOTFS/bin/bash" ]; then
    echo "=== 发现 $TMP_ROOTFS 已解压, 移动到 $ROOTFS ==="
    mkdir -p "$(dirname "$ROOTFS")"
    mv "$TMP_ROOTFS" "$ROOTFS"
fi

if [ -x "$ROOTFS/bin/bash" ] && [ -d "$ROOTFS/usr/local/lib" ]; then
    echo "$ROOTFS 已存在, 跳过解压"
    echo "重新解压: rm -rf $ROOTFS && bash $0"
else
    echo "=== 解压 $TAR 到 $ROOTFS (20-40 分钟) ==="
    mkdir -p "$ROOTFS"
    TAR="$TAR" ROOTFS="$ROOTFS" python3 << 'PYEOF'
import json, subprocess, os
rootfs = os.environ['ROOTFS']
tar_path = os.environ['TAR']

manifest = subprocess.check_output(['tar', 'xf', tar_path, '-O', 'manifest.json'])
layers = json.loads(manifest)[0]['Layers']
print(f'共 {len(layers)} 层, 开始解压...', flush=True)

for i, layer in enumerate(layers):
    print(f'[{i+1}/{len(layers)}] {layer}', flush=True)
    p1 = subprocess.Popen(['tar', 'xf', tar_path, '-O', layer], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['tar', 'xf', '-', '-C', rootfs],
                          stdin=p1.stdout, stderr=subprocess.DEVNULL)
    p1.stdout.close()
    p2.communicate()
    if p2.returncode != 0:
        print(f'  WARN: exit {p2.returncode}', flush=True)

print('=== 解压完成 ===', flush=True)
PYEOF
fi

# 清理 docker image 残留的 /dev /proc /sys (避免 bind mount 冲突)
echo "=== 清理 mount 点 ==="
rm -rf "$ROOTFS/dev" "$ROOTFS/proc" "$ROOTFS/sys"
mkdir -p "$ROOTFS/dev" "$ROOTFS/proc" "$ROOTFS/sys"

# 验证关键路径
echo ""
echo "=== 验证 ==="
echo "ROOTFS: $ROOTFS"
ls "$ROOTFS/root/slime" >/dev/null 2>&1 && echo "✅ /root/slime 存在" || echo "⚠️  /root/slime 不存在"
ls "$ROOTFS"/usr/local/lib/python3.*/dist-packages/ 2>/dev/null | grep -E "^(torch|transformer_engine|megatron|sglang)" | head -5

echo ""
echo "✅ 完成. 进入环境:"
echo "  bash run_env.sh"
