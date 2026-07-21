#!/bin/bash
# 在 4×A100 算力机上安装 slime + Megatron-LM + sglang 全套依赖
#
# 用法:
#     bash slime_sao/SAO/repro/install_deps.sh
#
# 安装位置:
#   - Megatron-LM  → /root/Megatron-LM  (slime 默认 PYTHONPATH)
#   - slime        → 已经 clone 在 slime_sao 旁边 (setup_a100.sh 做的)
#   - Python deps  → 当前 conda/venv 环境

set -e

WORKDIR="${WORKDIR:-/home/jovyan/h800fast/wangzekai}"
SLIME_DIR="${WORKDIR}/slime"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"   # slime Dockerfile 锁定的版本

# ====================================================================
# Step 0: 确保 slime 主仓库已 clone (后面要用它的 patch)
# ====================================================================
echo "=== Step 0: 检查 slime 主仓库 ==="
if [[ ! -d "${SLIME_DIR}" ]]; then
    echo "slime 主仓库不存在, 自动跑 setup_a100.sh..."
    bash "$(dirname "$0")/setup_a100.sh"
fi
ls "${SLIME_DIR}/docker/patch/latest/megatron.patch" || {
    echo "❌ slime 主仓库 patch 文件不存在"
    exit 1
}

# ====================================================================
# Step 1: clone + checkout Megatron-LM (slime 适配的 commit)
# ====================================================================
echo "=== Step 1: clone Megatron-LM @ ${MEGATRON_COMMIT} ==="
if [[ ! -d "${MEGATRON_DIR}" ]]; then
    cd "$(dirname ${MEGATRON_DIR})"
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
    cd Megatron-LM
    git checkout "${MEGATRON_COMMIT}"
    git submodule update --init --recursive
else
    echo "✅ 已存在: ${MEGATRON_DIR}"
fi

# ====================================================================
# Step 2: apply slime 的 megatron patch
# ====================================================================
echo ""
echo "=== Step 2: apply slime megatron.patch ==="
PATCH_SRC="${SLIME_DIR}/docker/patch/latest/megatron.patch"
if [[ ! -f "${PATCH_SRC}" ]]; then
    echo "❌ patch 文件不存在: ${PATCH_SRC}"
    echo "   检查 slime 是否完整 clone, 或重新跑 setup_a100.sh"
    exit 1
fi
cp "${PATCH_SRC}" "${MEGATRON_DIR}/megatron.patch"
cd "${MEGATRON_DIR}"
if git apply --check megatron.patch 2>/dev/null; then
    git apply megatron.patch --3way
    echo "✅ patch 已应用"
else
    echo "⚠️ patch 可能已应用过或冲突, 跳过 (如果之前装过可以忽略)"
fi
rm -f megatron.patch

# ====================================================================
# Step 3: install Megatron-Bridge (slime 用的桥接层)
# ====================================================================
echo ""
echo "=== Step 3: install Megatron-Bridge ==="
pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation

# ====================================================================
# Step 4: install slime Python 依赖
# ====================================================================
echo ""
echo "=== Step 4: install slime Python deps ==="
pip install -r "${SLIME_DIR}/requirements.txt"

# ====================================================================
# Step 5: install sglang (slime Dockerfile 里走 sglang.patch, 这里简化为 pip)
# ====================================================================
echo ""
echo "=== Step 5: install sglang ==="
# slime 用的是定制 sglang, 完整对齐见 slime/docker/Dockerfile.
# 这里先装开源版本, 跑起来报错再换.
pip install "sglang[all]" sglang-router

# ====================================================================
# Step 6: 验证
# ====================================================================
echo ""
echo "=== Step 6: 验证安装 ==="
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PYTHONPATH}"
echo "PYTHONPATH=${PYTHONPATH}"

echo ""
echo "--- 关键包 ---"
python -c "import megatron; print('✅ megatron:', megatron.__file__)"
python -c "import sglang; print('✅ sglang:', sglang.__version__)"
python -c "import slime; print('✅ slime:', slime.__file__)"
python -c "import torch; print('✅ torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"

echo ""
echo "==================================================================="
echo "✅ 安装完成. 永久环境变量设置:"
echo ""
echo '    echo "export PYTHONPATH=${MEGATRON_DIR}:\${SLIME_DIR}:\${PYTHONPATH}" >> ~/.bashrc'
echo '    source ~/.bashrc'
echo ""
echo "然后跑:"
echo "    bash ${SLIME_DIR}/SAO/repro/run_eval_a100.sh"
echo "==================================================================="
