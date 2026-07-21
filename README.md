# SAO: Single-rollout Asynchronous Optimization for Agentic RL — slime 复现

本仓库是对论文 [**Single-rollout Asynchronous Optimization for Agentic Reinforcement Learning**](https://arxiv.org/html/2607.07508v1) (Hou et al., 2026)
在 [slime](https://github.com/THUDM/slime) RL 框架上的开源复现工作。

## 目录结构

```
SAO/
├── SAO.md                  # 论文笔记 (方法 / 公式 / 实验设置)
├── TODO                    # 复现清单: 5 大算法组件 + 实验路径 + 风险点
│
├── sao/                    # 算法组件实现 (每文件独立可运行, 带单测)
│   ├── _01_dis.py                    # DIS 直接双向重要性采样        (论文 §3.1)
│   ├── _02_faster_value_update.py    # Faster Value Update (TTUR)   (论文 §3.2)
│   ├── _03_frozen_attention.py       # Frozen-Attention critic       (论文 §3.2)
│   ├── _04_skip_obs_gae.py           # Skip-Observation GAE          (论文 §3.2)
│   ├── _05_length_adaptive_gae.py    # Length-Adaptive GAE           (论文 §4.1)
│   ├── _06_sao_rollout.py            # 端到端 single-rollout async 参数
│   ├── _07_value_pretrain.py         # 价值模型冷启动预训练          (论文 §3.2)
│   └── README.md                     # 组件导航 + slime 接入 checklist
│
└── repro/                  # 端到端复现脚本 (基于本机资源)
    ├── README.md                     # Tier 0/1/2 复现路径
    ├── 01_convert_aime2025.py        # AIME2025 数据 → slime eval schema
    ├── eval_aime2025.yaml            # AIME2025 eval 配置
    ├── run_eval_baseline.sh          # baseline eval 启动脚本
    └── distill/                      # 用 360 API (GLM-5.2) 蒸馏 TIR 冷启动数据
        ├── distill_tir.py            # 蒸馏主脚本
        ├── run_distill.sh            # 启动脚本
        ├── check_quality.py          # 蒸馏数据正确率检查
        ├── filter_correct.py         # 过滤错误样本
        └── README.md
```

## 快速上手

### 1. 学习算法组件 (按编号顺序, 每个文件可独立运行)

```bash
git clone https://github.com/wzk2239115/slime_sao.git
cd slime_sao/sao

python _01_dis.py                  # DIS 双向 mask
python _02_faster_value_update.py  # TTUR K=2
python _03_frozen_attention.py     # 冻结 attention
python _04_skip_obs_gae.py         # Skip-obs GAE
python _05_length_adaptive_gae.py  # per-sample λ
python _07_value_pretrain.py       # value 预训练
```

### 2. 接入 slime 主仓库

算法组件需要 slime 作为运行时依赖:

```bash
# 克隆 slime 到 SAO 同级目录
git clone https://github.com/THUDM/slime.git ../slime
cd ../slime

# 把本仓库软链或拷贝到 slime/SAO/
ln -s /path/to/slime_sao SAO

# 跑组件 (slime 主仓库为 PYTHONPATH)
PYTHONPATH=/path/to/slime python -m SAO.sao._01_dis
```

### 3. 蒸馏 TIR 冷启动数据

替代论文里 GPT-OSS-120B 蒸馏的角色, 用 360 API 的 GLM-5.2 + slime PythonSandbox:

```bash
export API_360_KEY="<your-key>"
cd /path/to/slime
bash SAO/repro/distill/run_distill.sh
```

详见 `repro/distill/README.md`。

## 已验证

- ✅ 7 个算法组件单测全过 (`python -m SAO.sao._XX`)
- ✅ 蒸馏 pipeline 端到端跑通 (AIME2025 30 题, 36 分钟, 80% 正确率)
- ✅ 产出 23 条高质量 slime multi-turn SFT 数据

## 复现进度

参考 `TODO` 文件里的 4 个 sprint:

- [x] **Sprint 1**: 算法组件实现 (DIS / TTUR / frozen-attn / skip-obs GAE / length-adaptive)
- [x] **Sprint 2 (部分)**: 蒸馏 pipeline 跑通, 产出 23 条 SFT 数据
- [ ] **Sprint 2 (完整)**: 接 slime SFT 路径, 小模型 (Qwen3-0.6B) 验证数据格式
- [ ] **Sprint 3**: 主实验 (需多节点 GPU, 单机不可行)

## 与论文 SAO 的现状差距

| 项 | 论文 SAO | 本仓库 |
|---|---|---|
| 模型 | Qwen3-30B-A3B-Thinking-2507 + SFT 起步 | 本地有 ckpt, 但单机 GPU 跑不动 30B 训练 |
| 训练数据 | GPT-OSS-120B 蒸馏 (未公开) | 用 360 API GLM-5.2 自蒸馏 (TIR) |
| 硬件 | 64+ GPU | 单机 |
| 评测 | AIME/BeyondAIME/HMMT/IMOAnswerBench | AIME2025 (30 题) |

完整复现需多节点 + 自备蒸馏数据, 详见 `TODO` §6 风险点。

## 引用

```bibtex
@article{hou2026sao,
  title={Single-rollout Asynchronous Optimization for Agentic Reinforcement Learning},
  author={Hou, Zhenyu and Li, Yujiang and Tang, Jie and Dong, Yuxiao},
  journal={arXiv preprint arXiv:2607.07508},
  year={2026}
}
```
