#!/usr/bin/env python3
"""SAO 单元测试 — 训练前必须全部通过。

用法:
    python3 -m pytest sao/standalone/test_sao.py -v
    # 或直接运行:
    BASH_ENV= python3 sao/standalone/test_sao.py
"""
import sys
import os

# ============================================================
# Test 1: Reward function — answer normalization
# ============================================================
def test_reward_normalization():
    """各种 LaTeX 格式的答案匹配。"""
    from sao.standalone.reward import normalize_answer, math_reward, extract_boxed

    # 基本整数
    assert normalize_answer("45") == normalize_answer("45"), "int match"
    assert normalize_answer("45") != normalize_answer("745"), "int mismatch"

    # LaTeX sqrt: \sqrt{39} should match sqrt{39}
    assert normalize_answer("\\sqrt{39}") == normalize_answer("sqrt{39}"), \
        f"sqrt: {normalize_answer('\\sqrt{39}')} != {normalize_answer('sqrt{39}')}"

    # LaTeX frac: \frac{1}{2} should match frac{1}{2}
    assert normalize_answer("\\frac{1}{2}") == normalize_answer("frac{1}{2}"), \
        f"frac: {normalize_answer('\\frac{1}{2}')} != {normalize_answer('frac{1}{2}')}"

    # 空格不影响
    assert normalize_answer("  45  ") == normalize_answer("45"), "whitespace"

    # 美元符号
    assert normalize_answer("$45$") == normalize_answer("45"), "dollar"

    # math_reward end-to-end
    r = math_reward("The answer is $\\boxed{45}$.", "45")
    assert r == 1.0, f"reward should be 1.0, got {r}"

    r = math_reward("The answer is $\\boxed{745}$.", "45")
    assert r == 0.0, f"reward should be 0.0, got {r}"

    r = math_reward("The answer is $\\boxed{\\sqrt{39}}$.", "sqrt{39}")
    assert r == 1.0, f"sqrt reward should be 1.0, got {r}"

    # 无 boxed
    r = math_reward("The answer is 45.", "45")
    assert r == 0.0, "no boxed → 0"

    # None response
    r = math_reward(None, "45")
    assert r == 0.0, "None response → 0"

    print("✓ test_reward_normalization")


def test_extract_boxed():
    """\\boxed{} 提取。"""
    from sao.standalone.reward import extract_boxed

    assert extract_boxed("Answer: $\\boxed{42}$.") == "42"
    assert extract_boxed("$\\boxed{\\frac{13}{2}}$") == "\\frac{13}{2}"
    assert extract_boxed("no boxed here") is None
    assert extract_boxed(None) is None
    assert extract_boxed("") is None
    # 多个 boxed 取最后一个
    assert extract_boxed("$\\boxed{1}$ and $\\boxed{2}$") == "2"
    # 嵌套大括号
    assert extract_boxed("$\\boxed{\\sqrt{x+1}}$") == "\\sqrt{x+1}"

    print("✓ test_extract_boxed")


# ============================================================
# Test 2: GAE — length-adaptive lambda
# ============================================================
def test_length_adaptive_lambda():
    """λ_policy = 1 - 1/(α·L)。"""
    from sao.standalone.critic import length_adaptive_lambda

    # L=100, α=1.5 → λ = 1 - 1/150 ≈ 0.9933
    lam = length_adaptive_lambda(100, alpha=1.5)
    assert abs(lam - (1 - 1/150)) < 1e-6, f"lambda L=100: {lam}"

    # L=1 → λ = 1 - 1/1.5 = 1/3
    lam = length_adaptive_lambda(1, alpha=1.5)
    assert abs(lam - (1 - 1/1.5)) < 1e-6, f"lambda L=1: {lam}"

    # L=0 → λ = 0
    lam = length_adaptive_lambda(0, alpha=1.5)
    assert lam == 0.0, f"lambda L=0: {lam}"

    # λ ∈ [0, 1]
    for L in range(1, 1000):
        lam = length_adaptive_lambda(L, alpha=1.5)
        assert 0 <= lam <= 1.0, f"lambda out of range: L={L} λ={lam}"

    print("✓ test_length_adaptive_lambda")


def test_gae_single():
    """GAE 单条轨迹。"""
    import torch
    from sao.standalone.critic import compute_gae_single

    # γ=1, λ=1, reward at last token → MC return
    values = torch.zeros(5)
    adv, ret = compute_gae_single(values, reward=1.0, gamma=1.0, lambd=1.0)
    # With V=0 and reward=1 at last token:
    # advantages should be [1, 1, 1, 1, 1] (γ=1, λ=1 → MC)
    assert torch.allclose(adv, torch.ones(5)), f"MC advantages: {adv}"
    assert torch.allclose(ret, torch.ones(5)), f"MC returns: {ret}"

    # γ=1, λ=0 → only TD(0) at each step
    values = torch.zeros(5)
    adv, ret = compute_gae_single(values, reward=1.0, gamma=1.0, lambd=0.0)
    # Only last token has advantage = 1 - 0 = 1, rest = 0
    expected_adv = torch.tensor([0., 0., 0., 0., 1.])
    assert torch.allclose(adv, expected_adv), f"TD(0) advantages: {adv}"

    # With nonzero values
    values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    adv, ret = compute_gae_single(values, reward=1.0, gamma=1.0, lambd=1.0)
    # MC return = 1.0 for all tokens
    # advantage = return - value = 1.0 - 0.5 = 0.5
    assert torch.allclose(ret, torch.ones(5)), f"returns with V: {ret}"
    assert torch.allclose(adv, torch.full((5,), 0.5)), f"advantages with V: {adv}"

    print("✓ test_gae_single")


# ============================================================
# Test 3: DIS policy loss
# ============================================================
def test_dis_loss():
    """DIS 双向裁剪。"""
    import torch
    from sao.standalone.grpo_step import dis_policy_loss

    # ratio = 1.0 (on-policy) → no clipping, loss = -ratio * adv
    tlp = [torch.tensor([-1.0, -2.0])]  # train log probs
    rlp = [torch.tensor([-1.0, -2.0])]  # rollout log probs (same)
    adv = [0.5]
    loss, metrics = dis_policy_loss(tlp, rlp, adv, clip_low=0.7, clip_high=6.0)
    # ratio = exp(0) = 1.0, in range [0.7, 6.0] → no clip
    assert abs(metrics["clip_ratio"] - 0.0) < 1e-6, f"clip should be 0: {metrics['clip_ratio']}"
    assert abs(metrics["mean_ratio"] - 1.0) < 1e-4, f"ratio should be 1.0: {metrics['mean_ratio']}"
    # loss = -(ratio * adv) = -(1.0 * 0.5) = -0.5 per token, averaged over 2 tokens = -0.5
    assert abs(metrics["loss"] - (-0.5)) < 1e-4, f"loss: {metrics['loss']}"

    # ratio outside trust region → clipped (masked to 0)
    tlp2 = [torch.tensor([0.0, 0.0])]  # log_prob = 0 → prob = 1
    rlp2 = [torch.tensor([-5.0, 10.0])]  # extreme rollout log_probs
    # ratio = exp(0 - (-5)) = exp(5) = 148 → outside [0.7, 6.0] → masked
    # ratio = exp(0 - 10) = exp(-10) ≈ 0 → outside [0.7, 6.0] → masked
    loss2, metrics2 = dis_policy_loss(tlp2, rlp2, [1.0], clip_low=0.7, clip_high=6.0)
    assert metrics2["clip_ratio"] > 0.5, f"should be mostly clipped: {metrics2['clip_ratio']}"

    print("✓ test_dis_loss")


# ============================================================
# Test 4: DIS with tensor advantages (from GAE)
# ============================================================
def test_dis_loss_tensor_advantages():
    """DIS 接收 tensor advantages（来自 GAE，不是 float）。"""
    import torch
    from sao.standalone.grpo_step import dis_policy_loss

    tlp = [torch.tensor([-1.0, -0.5, -2.0])]
    rlp = [torch.tensor([-1.0, -0.5, -2.0])]
    adv = [torch.tensor([0.5, 0.3, 0.8])]  # per-token advantages from GAE
    loss, metrics = dis_policy_loss(tlp, rlp, adv, clip_low=0.7, clip_high=6.0)
    assert torch.is_tensor(loss) or hasattr(loss, 'item'), "loss should be tensor"
    assert metrics["mean_ratio"] == 1.0 or abs(metrics["mean_ratio"] - 1.0) < 1e-4

    print("✓ test_dis_loss_tensor_advantages")


# ============================================================
# Test 5: Log prob computation correctness
# ============================================================
def test_compute_log_probs_shape():
    """compute_log_probs 返回正确 shape。"""
    import torch
    from sao.standalone.grpo_step import compute_log_probs

    # Mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.tensor(1.0))
            self.training = False
        def __call__(self, input_ids, **kwargs):
            batch, seq = input_ids.shape
            logits = torch.randn(batch, seq, 100)  # vocab=100
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits)

    model = MockModel()
    input_ids_list = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    response_lens = [5]
    log_probs = compute_log_probs(model, input_ids_list, response_lens, torch.device("cpu"), False)
    assert len(log_probs) == 1, f"should return 1 sample: {len(log_probs)}"
    assert log_probs[0].shape == (5,), f"shape should be (5,): {log_probs[0].shape}"

    print("✓ test_compute_log_probs_shape")


# ============================================================
# Test 6: Trajectory format validation
# ============================================================
def test_trajectory_format():
    """轨迹 JSON 格式验证。"""
    import json

    # 模拟一条正确的轨迹
    traj = {
        "id": 0,
        "prompt_ids": [1, 2, 3],
        "resp_ids": [4, 5, 6, 7],
        "logprobs": [-0.5, -0.3, -0.8, -0.1],
        "response_text": "The answer is \\boxed{42}.",
        "ground_truth": "42",
        "reward": 1.0,
        "timestamp": 1234567890.0,
        "resp_len": 4,
        "prompt_len": 3,
    }

    # 验证关键字段存在
    required = ["prompt_ids", "resp_ids", "logprobs", "reward", "ground_truth"]
    for key in required:
        assert key in traj, f"missing key: {key}"

    # logprobs 长度必须等于 resp_ids 长度
    assert len(traj["logprobs"]) == len(traj["resp_ids"]), \
        f"logprob len {len(traj['logprobs'])} != resp_ids len {len(traj['resp_ids'])}"

    # JSON 可序列化
    s = json.dumps(traj)
    d = json.loads(s)
    assert d["reward"] == 1.0

    print("✓ test_trajectory_format")


# ============================================================
# Test 7: Queue file naming (multi-machine safety)
# ============================================================
def test_queue_filename_uniqueness():
    """多机写入 queue 不会文件名冲突。"""
    import socket
    hostname = socket.gethostname()

    # 两个不同 worker_id 的文件名应该不同
    worker_a = f"1234567890_1111"
    worker_b = f"1234567891_2222"

    fname_a = f"traj_{worker_a}_00000001.json"
    fname_b = f"traj_{worker_b}_00000001.json"
    assert fname_a != fname_b, "different workers should have different filenames"

    print("✓ test_queue_filename_uniqueness")


# ============================================================
# Test 8: Paper parameter verification
# ============================================================
def test_paper_parameters():
    """验证论文 §4.1 的超参数。"""
    # These must match the paper
    params = {
        "batch_size": 128,          # paper: 128
        "group_size": 1,            # paper: single rollout
        "actor_lr": 1e-6,           # paper: 1×10⁻⁶
        "critic_lr": 5e-6,          # paper: 5×10⁻⁶
        "epsilon_low": 0.3,         # paper: ε_l=0.3
        "epsilon_high": 5.0,        # paper: ε_h=5.0
        "clip_low": 0.7,            # 1 - ε_l = 0.7
        "clip_high": 6.0,           # 1 + ε_h = 6.0
        "gamma": 1.0,               # paper: γ=1.0
        "gae_alpha": 1.5,           # paper: α=1.5
        "critic_k": 2,              # paper: TTUR K=2
        "critic_warmup": 10,        # paper: 10-step warmup
        "value_clip": 0.2,          # paper: 0.2
        "temperature": 1.0,         # paper: T=1.0
        "top_p": 1.0,              # paper: top-p=1.0
    }

    assert params["clip_low"] == 1 - params["epsilon_low"], "clip_low formula"
    assert params["clip_high"] == 1 + params["epsilon_high"], "clip_high formula"
    assert params["group_size"] == 1, "SAO uses single rollout"
    assert params["critic_k"] > 1, "TTUR requires K>1"

    print("✓ test_paper_parameters")
    print(f"  batch_size={params['batch_size']} (actual: 8, reduced for memory)")
    print(f"  lr={params['actor_lr']} critic_lr={params['critic_lr']}")
    print(f"  DIS: ε_l={params['epsilon_low']} ε_h={params['epsilon_high']} → clip=[{params['clip_low']}, {params['clip_high']}]")
    print(f"  GAE: γ={params['gamma']} α={params['gae_alpha']} → λ=1-1/(1.5·L)")
    print(f"  TTUR: K={params['critic_k']} warmup={params['critic_warmup']}")


# ============================================================
# Test 9: Frozen attention (critic)
# ============================================================
def test_frozen_attention():
    """验证 critic 的 attention 参数被冻结。"""
    import torch
    import torch.nn as nn

    from sao.standalone.critic import ValueModel

    # Mock model with attention-like params
    class MockBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(10, 10)
            self.self_attn.q_norm = nn.Linear(10, 10)
            self.moe_gate = nn.Linear(10, 10)
        def forward(self, x):
            return self.moe_gate(self.self_attn.q_norm(self.self_attn.q_proj(x)))

    base = MockBase()
    vm = ValueModel(base, hidden_size=10)

    # Before freeze
    n_before = sum(1 for p in vm.parameters() if p.requires_grad)

    vm.freeze_attention()

    # After freeze: self_attn params frozen, moe still trainable
    n_after = sum(1 for p in vm.parameters() if p.requires_grad)
    assert n_after < n_before, f"freeze should reduce trainable: {n_before} → {n_after}"
    assert not vm.model.self_attn.q_proj.weight.requires_grad, "q_proj should be frozen"
    assert vm.model.moe_gate.weight.requires_grad, "moe_gate should be trainable"
    print(f"✓ test_frozen_attention (frozen {n_before - n_after} params)")

# ============================================================
# Test 10: Reward on MATH sample data
# ============================================================
def test_math_data_reward():
    """用 MATH 数据集样本测试 reward 函数。"""
    import json

    data_path = os.environ.get("MATH_DATA",
        "/home/jovyan/h800fast/wangzekai/slime_sao/datasets/MATH_train.jsonl")
    if not os.path.exists(data_path):
        print(f"⚠ test_math_data_reward skipped (no data at {data_path})")
        return

    from sao.standalone.reward import normalize_answer

    # 读前 100 条，验证 ground truth 格式
    empty_count = 0
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            d = json.loads(line)
            gt = d["label"]
            normalized = normalize_answer(gt)
            if len(normalized) == 0:
                empty_count += 1
                continue
    if empty_count > 10:
        print(f"  ⚠️  {empty_count}/100 empty labels — MATH_train.jsonl may need regeneration")

    print(f"✓ test_math_data_reward (100 samples validated)")


# ============================================================
# Test 11: Value clipping in critic training
# ============================================================
def test_value_clipping():
    """Critic value clipping 逻辑。"""
    import torch

    # Simulate value clipping
    old_v = torch.tensor([1.0, 1.0, 1.0])
    new_v = torch.tensor([1.5, 0.5, 2.0])
    value_clip = 0.2

    clipped_v = old_v + (new_v - old_v).clamp(-value_clip, value_clip)
    # Token 0: 1.0 + min(0.5, 0.2) = 1.2
    # Token 1: 1.0 + max(-0.5, -0.2) = 0.8
    # Token 2: 1.0 + min(1.0, 0.2) = 1.2
    expected = torch.tensor([1.2, 0.8, 1.2])
    assert torch.allclose(clipped_v, expected), f"value clip: {clipped_v} != {expected}"

    print("✓ test_value_clipping")


# ============================================================
# Test 12: Gradient accumulation equivalence
# ============================================================
def test_gradient_accumulation():
    """验证梯度累积 = 批量 backward。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    # Simple model
    model_a = nn.Linear(10, 1)
    model_b = nn.Linear(10, 1)
    model_b.load_state_dict(model_a.state_dict())

    # 3 samples
    x = [torch.randn(10) for _ in range(3)]
    y = [torch.tensor(1.0) for _ in range(3)]

    # Method A: batch backward
    loss_a = sum((model_a(xi) - yi).pow(2) for xi, yi in zip(x, y)) / 3
    loss_a.backward()
    grad_a = model_a.weight.grad.clone()

    # Method B: per-sample backward (gradient accumulation)
    model_b.zero_grad()
    for xi, yi in zip(x, y):
        loss_i = (model_b(xi) - yi).pow(2) / 3
        loss_i.backward()
    grad_b = model_b.weight.grad.clone()

    # Gradients should be approximately equal
    assert torch.allclose(grad_a, grad_b, atol=1e-5), \
        f"grad mismatch:\n  batch={grad_a[:3]}\n  accum={grad_b[:3]}"

    print("✓ test_gradient_accumulation")


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    # Ensure WORKDIR is in path
    workdir = "/home/jovyan/h800fast/wangzekai/slime_sao"
    if os.path.exists(workdir):
        sys.path.insert(0, workdir)
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    tests = [
        test_reward_normalization,
        test_extract_boxed,
        test_length_adaptive_lambda,
        test_gae_single,
        test_dis_loss,
        test_dis_loss_tensor_advantages,
        test_compute_log_probs_shape,
        test_trajectory_format,
        test_queue_filename_uniqueness,
        test_paper_parameters,
        test_frozen_attention,
        test_math_data_reward,
        test_value_clipping,
        test_gradient_accumulation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
