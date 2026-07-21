#!/bin/bash
# 在 4×A100 机器上 setup SAO 复现环境
#
# 用法 (在 slime_sao/ 的父目录执行):
#     bash slime_sao/SAO/repro/setup_a100.sh
#
# 前提: 当前机器已经有以下目录
#   - slime_sao/                          # 本仓库
#   - slime_sao/models/Qwen3-30B-A3B-Thinking-2507/
#   - slime_sao/datasets/AIME2025/

set -e

WORKDIR="${WORKDIR:-/home/jovyan/h800fast/wangzekai}"
cd "${WORKDIR}"

echo "=== Step 1: 检查已有目录 ==="
for d in slime_sao slime_sao/models/Qwen3-30B-A3B-Thinking-2507 slime_sao/datasets/AIME2025; do
    if [[ ! -d "$d" ]]; then
        echo "❌ 缺少目录: ${WORKDIR}/${d}"
        exit 1
    fi
done
echo "✅ 目录齐全"
ls slime_sao/datasets/AIME2025/

echo ""
echo "=== Step 2: clone slime 主仓库 ==="
if [[ ! -d slime ]]; then
    git clone https://github.com/THUDM/slime.git
fi
cd slime

echo ""
echo "=== Step 3: 软链 SAO 复现代码到 slime/SAO ==="
if [[ ! -L SAO && ! -d SAO ]]; then
    ln -s "${WORKDIR}/slime_sao/SAO" SAO
elif [[ -L SAO ]]; then
    echo "SAO 软链已存在: $(readlink SAO)"
fi
ls -la SAO

echo ""
echo "=== Step 4: 数据格式转换 ==="
# AIME2025 原始格式 (question/answer) → slime 格式 (input/label)
python SAO/repro/01_convert_aime2025.py \
    --src "${WORKDIR}/slime_sao/datasets/AIME2025" \
    --dst "${WORKDIR}/slime_sao/datasets/AIME2025/slime"

echo ""
echo "=== Step 5: 检查 slime 运行依赖 ==="
# 检查 PYTHONPATH 里的 Megatron-LM, sglang 等关键依赖
echo "PYTHONPATH=${PYTHONPATH}"
python -c "import sglang; print('sglang:', sglang.__version__)" 2>&1 || echo "⚠️ sglang 未安装"
python -c "import megatron; print('megatron ok')" 2>&1 || echo "⚠️ Megatron-LM 未在 PYTHONPATH"

echo ""
echo "==================================================================="
echo "✅ Setup 完成. 下一步:"
echo "  bash SAO/repro/run_eval_a100.sh"
echo "==================================================================="
