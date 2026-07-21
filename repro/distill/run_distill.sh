#!/bin/bash
# SAO 冷启动数据蒸馏 - 启动脚本
#
# 用 360 API (GLM-5.2) + slime PythonSandbox 蒸馏 TIR 数学轨迹.
# 详细文档见 README.md.

set -e

cd "$(dirname "$0")/../../.."   # 切到 slime 项目根 (distill/ → repro/ → SAO/ → slime/)

# ============ 配置 (按需修改) ============
export API_360_KEY="${API_360_KEY:-<your-360-api-key>}"

# 源数据集: 三选一 (取消注释)
# (a) AIME2025 (30 题, sanity check, ~13 分钟)
SRC="${SRC:-/home/wzk/datasets/AIME2025/slime/aime2025-all.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"

# (b) dapo-math-17k 子集 (500 条, ~3.5 小时)
# SRC="${SRC:-/path/to/dapo-math-17k.jsonl}"
# MAX_SAMPLES="${MAX_SAMPLES:-500}"

# (c) 全量 dapo-math-17k (17000 条, ~120 小时, 不推荐单机)
# SRC="${SRC:-/path/to/dapo-math-17k.jsonl}"
# MAX_SAMPLES="${MAX_SAMPLES:-}"

# 输出路径
DST_DIR="${DST_DIR:-/home/wzk/datasets/sao_sft}"
mkdir -p "${DST_DIR}"
DST="${DST:-${DST_DIR}/distilled_$(date +%Y%m%d_%H%M%S).jsonl}"

# 蒸馏参数
CONCURRENCY="${CONCURRENCY:-2}"     # 360 API 5 并发会 timeout, 建议 2-3
MAX_TURNS="${MAX_TURNS:-8}"         # 单条最多 8 轮 tool call
MAX_TOKENS="${MAX_TOKENS:-16384}"   # 单轮 API 输出上限 (GLM-5.2 thinking 长, 建议 16k+)

# ============ 启动 ============
ARGS=(
    --src "${SRC}"
    --dst "${DST}"
    --concurrency "${CONCURRENCY}"
    --max-turns "${MAX_TURNS}"
    --max-tokens "${MAX_TOKENS}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
    ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "======================================================================="
echo "SAO 冷启动数据蒸馏"
echo "  源数据 : ${SRC}"
echo "  输出   : ${DST}"
echo "  并发   : ${CONCURRENCY}, 单条最大 ${MAX_TURNS} 轮"
echo "======================================================================="
echo ""

python SAO/repro/distill/distill_tir.py "${ARGS[@]}"

echo ""
echo "======================================================================="
echo "完成. 下一步:"
echo "  1. 检查数据质量:"
echo "     python3 -c \"import json; lines=open('${DST}').readlines(); print(f'{len(lines)} 条'); print(json.loads(lines[0])['messages'][1]['content'][:200])\""
echo "  2. 用 slime sft_rollout 做 SFT 起步"
echo "======================================================================="
