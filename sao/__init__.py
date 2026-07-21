"""SAO (Single-rollout Asynchronous Optimization) 复现组件包.

每个子模块对应论文 (SAO/SAO.md) 的一个独立设计点, 可单独 ``python -m SAO.sao.<name>``
运行调试, 也可以被 slime 主流程通过 ``--custom-...-path`` 引用.

组件清单 (建议按编号顺序阅读):
    01_dis                    : Direct double-sided Importance Sampling (DIS)        论文 §3.1
    02_faster_value_update    : Faster Value Update (TTUR, K=2)                     论文 §3.2
    03_frozen_attention       : Frozen-Attention critic 配置工具                   论文 §3.2
    04_skip_obs_gae           : Skip-Observation Token-level GAE                   论文 §3.2
    05_length_adaptive_gae    : Length-Adaptive GAE (per-sample λ)                 论文 §4.1
    06_sao_rollout            : 端到端 single-rollout async rollout 配置示例
    07_value_pretrain         : 价值模型冷启动预训练循环

注意: 这里不在 __init__ 里预导入子模块, 避免 ``python -m SAO.sao._xx`` 触发
"found in sys.modules after import of package" 的 RuntimeWarning.
按需 ``from SAO.sao._01_dis import dis_tis_function`` 即可.
"""
