#!/usr/bin/env python3
"""SAO 训练前 preflight 检查 — 在算力机上运行，必须全部通过才能开始训练。

用法:
  # 推理机 (ctm-05/01/02/04):
  BASH_ENV= python3 sao/standalone/preflight.py inference

  # 训练机 (ctm-06):
  BASH_ENV= python3 sao/standalone/preflight.py train
"""
from __future__ import annotations

import os, sys, time, json, glob

# ============================================================
# Config
# ============================================================
WORKDIR = "/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS  = "/home/jovyan/h800fast/wangzekai/slime_rootfs"
MODEL   = f"{WORKDIR}/models/Qwen3-30B-A3B-Thinking-2507"
DATA    = f"{WORKDIR}/datasets/MATH_train.jsonl"
PORT    = 30000

def setup():
    os.environ["LD_LIBRARY_PATH"] = f"{ROOTFS}/usr/local/cuda/lib64:{ROOTFS}/usr/local/nvidia/lib64"
    os.environ["PYTHONPATH"] = f"{WORKDIR}:{ROOTFS}/usr/local/lib/python3.12/dist-packages"
    os.environ["no_proxy"] = "*"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, WORKDIR)


class CheckResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def ok(self, msg):
        print(f"  ✅ {msg}")
        self.passed += 1

    def fail(self, msg):
        print(f"  ❌ {msg}")
        self.failed += 1

    def warn(self, msg):
        print(f"  ⚠️  {msg}")
        self.warnings += 1

    def summary(self):
        print(f"\n{'='*50}")
        print(f"Preflight: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        print(f"{'='*50}")
        return self.failed == 0


# ============================================================
# Shared checks
# ============================================================
def check_env(cr: CheckResult):
    """环境检查。"""
    print("\n[1] Environment")
    import torch
    cr.ok(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    cr.ok(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        cr.ok(f"  GPU {i}: {name} ({mem:.0f} GB)")

    try:
        import bitsandbytes
        cr.ok(f"bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        cr.warn("bitsandbytes not found (will use standard AdamW)")

    try:
        import transformers
        cr.ok(f"transformers {transformers.__version__}")
    except ImportError:
        cr.fail("transformers not found")

    try:
        import flash_attn
        cr.ok(f"flash_attn {flash_attn.__version__}")
    except ImportError:
        cr.warn("flash_attn not found")


def check_data(cr: CheckResult):
    """数据集检查。"""
    print("\n[2] Dataset")
    if not os.path.exists(DATA):
        cr.fail(f"Training data not found: {DATA}")
        return

    with open(DATA) as f:
        lines = f.readlines()
    cr.ok(f"MATH training data: {len(lines)} problems")

    # 验证前 10 条格式
    from sao.standalone.reward import normalize_answer, math_reward
    for i, line in enumerate(lines[:10]):
        d = json.loads(line)
        assert "input" in d, f"missing 'input' in line {i}"
        assert "label" in d, f"missing 'label' in line {i}"
        norm = normalize_answer(d["label"])
        assert len(norm) > 0, f"empty normalized answer: {d['label']}"

    # 测试 reward 函数: 构造一个正确的回答
    sample = json.loads(lines[0])
    gt = sample["label"]
    fake_response = f"The answer is $\\\\boxed{{{gt}}}$."
    r = math_reward(fake_response, gt)
    if r == 1.0:
        cr.ok(f"Reward function: correct answer detected (gt={gt})")
    else:
        cr.fail(f"Reward function FAILED: gt={gt}, reward={r}")

    # 测试 reward 函数: 错误答案
    r_wrong = math_reward(f"$\\\\boxed{{999999}}$", gt)
    if r_wrong == 0.0:
        cr.ok("Reward function: wrong answer rejected")
    else:
        cr.fail(f"Reward function FAILED: wrong answer got reward={r_wrong}")


def check_model_path(cr: CheckResult):
    """模型文件检查。"""
    print("\n[3] Model")
    if not os.path.exists(f"{MODEL}/config.json"):
        cr.fail(f"Model not found: {MODEL}")
        return
    cr.ok(f"Model exists: {MODEL}")

    import json
    with open(f"{MODEL}/config.json") as f:
        config = json.load(f)
    cr.ok(f"  model_type: {config.get('model_type')}")
    cr.ok(f"  hidden_size: {config.get('hidden_size')}")
    cr.ok(f"  num_hidden_layers: {config.get('num_hidden_layers')}")
    cr.ok(f"  num_experts: {config.get('num_experts', '?')}")

    # 检查权重文件
    safetensors = glob.glob(f"{MODEL}/*.safetensors")
    cr.ok(f"  weight files: {len(safetensors)}")


# ============================================================
# Inference checks
# ============================================================
def check_sglang(cr: CheckResult):
    """sglang 服务器检查。"""
    print("\n[4] sglang Server")
    import urllib.request
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        opener.open(req, timeout=5).read()
        cr.ok("sglang health check passed")
    except Exception as e:
        cr.fail(f"sglang not running on port {PORT}: {e}")
        return False
    return True


def check_sglang_generate(cr: CheckResult):
    """sglang /generate API 返回正确的 token IDs + logprobs。"""
    print("\n[5] sglang /generate API")
    import urllib.request

    payload = json.dumps({
        "input_ids": [151644, 872, 198, 108046, 151645, 198],
        "sampling_params": {"n": 1, "temperature": 1.0, "max_new_tokens": 8},
        "return_logprob": True,
        "logprob_start_len": 0,
    }).encode()

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(opener.open(req, timeout=30).read())
    except Exception as e:
        cr.fail(f"/generate request failed: {e}")
        return

    meta = resp.get("meta_info", {})
    logprobs_data = meta.get("output_token_logprobs", [])

    if not logprobs_data:
        cr.fail("output_token_logprobs is empty — logprobs not returned")
        return

    if not isinstance(logprobs_data[0], list):
        cr.fail(f"unexpected logprob format: {type(logprobs_data[0])}")
        return

    # 格式: [[logprob, token_id, top_logprobs], ...]
    token_ids = [entry[1] for entry in logprobs_data]
    logprobs = [entry[0] if entry[0] is not None else 0.0 for entry in logprobs_data]

    cr.ok(f"Got {len(token_ids)} tokens with logprobs")
    cr.ok(f"  token_ids[0:3]: {token_ids[:3]}")
    cr.ok(f"  logprobs[0:3]: {logprobs[:3]}")

    # 验证 token IDs 合法
    assert all(isinstance(tid, int) for tid in token_ids), "token IDs should be integers"
    assert all(lp is not None for lp in logprobs), "logprobs should not be None"
    cr.ok("Token IDs are integers, logprobs valid")


def check_token_alignment(cr: CheckResult):
    """sglang token IDs 和 HF tokenizer 一致。"""
    print("\n[6] Token Alignment")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        cr.ok(f"Tokenizer loaded: vocab={tokenizer.vocab_size}")
    except Exception as e:
        cr.fail(f"Tokenizer load failed: {e}")
        return

    # 构造一个 prompt
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cr.ok(f"Prompt: {len(prompt_ids)} tokens")

    # 通过 sglang 生成
    import urllib.request
    payload = json.dumps({
        "input_ids": prompt_ids,
        "sampling_params": {"n": 1, "temperature": 0.0, "max_new_tokens": 16},
        "return_logprob": True,
        "logprob_start_len": len(prompt_ids),
    }).encode()

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(opener.open(req, timeout=60).read())
    except Exception as e:
        cr.fail(f"Generation failed: {e}")
        return

    meta = resp.get("meta_info", {})
    lp_data = meta.get("output_token_logprobs", [])
    if lp_data:
        sglang_ids = [entry[1] for entry in lp_data]
        text = tokenizer.decode(sglang_ids, skip_special_tokens=False)
        cr.ok(f"sglang generated {len(sglang_ids)} tokens: {text[:80]}")

        # Re-tokenize: should match
        re_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if re_ids == sglang_ids:
            cr.ok("Token IDs match: sglang ↔ HF tokenizer ✓")
        else:
            cr.warn(f"Token mismatch: sglang={sglang_ids[:5]} vs re-tokenized={re_ids[:5]}")
    else:
        cr.fail("No output token logprobs")


def check_rollout_reward(cr: CheckResult):
    """生成一条真实轨迹，验证 reward。"""
    print("\n[7] End-to-End Rollout + Reward")
    import urllib.request

    # 从 MATH 数据取一道题
    with open(DATA) as f:
        sample = json.loads(f.readline())

    prompt_text = sample["input"]
    gt = sample["label"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    payload = json.dumps({
        "input_ids": prompt_ids,
        "sampling_params": {"n": 1, "temperature": 1.0, "max_new_tokens": 4096},
        "return_logprob": True,
        "logprob_start_len": len(prompt_ids),
    }).encode()

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(opener.open(req, timeout=300).read())
    except Exception as e:
        cr.fail(f"Generation failed: {e}")
        return

    meta = resp.get("meta_info", {})
    lp_data = meta.get("output_token_logprobs", [])
    if not lp_data:
        cr.fail("No output tokens")
        return

    token_ids = [entry[1] for entry in lp_data]
    logprobs = [entry[0] if entry[0] is not None else 0.0 for entry in lp_data]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)

    from sao.standalone.reward import math_reward, extract_boxed
    reward = math_reward(text, gt)
    boxed = extract_boxed(text)

    cr.ok(f"Generated {len(token_ids)} tokens, reward={reward}")
    cr.ok(f"  GT: {gt}")
    cr.ok(f"  Extracted: {boxed}")
    cr.ok(f"  Logprobs aligned: {len(logprobs) == len(token_ids)}")

    if reward == 0.0 and boxed is not None:
        from sao.standalone.reward import normalize_answer
        cr.warn(f"  Answer mismatch: '{normalize_answer(str(boxed))}' vs '{normalize_answer(gt)}'")


# ============================================================
# Training checks
# ============================================================
def check_model_loading(cr: CheckResult):
    """模型加载检查。"""
    print("\n[4] Model Loading")
    import torch
    from transformers import AutoModelForCausalLM

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "78GB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GB"

    print("  Loading actor (30B MoE)...", end=" ", flush=True)
    t0 = time.time()
    actor = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    actor.gradient_checkpointing_enable()
    actor.config.use_cache = False
    print(f"{time.time()-t0:.0f}s")
    cr.ok(f"Actor loaded: {sum(p.numel() for p in actor.parameters())/1e9:.1f}B params")

    # 记录每张 GPU 的显存
    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        cr.ok(f"  GPU {i}: {alloc:.1f} GB allocated")

    return actor


def check_critic_loading(cr: CheckResult, actor):
    """Critic + ValueModel 加载检查。"""
    print("\n[5] Critic + ValueModel")
    import torch
    from transformers import AutoModelForCausalLM
    from sao.standalone.critic import ValueModel

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "78GB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GB"

    print("  Loading critic base...", end=" ", flush=True)
    t0 = time.time()
    base_critic = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_critic.gradient_checkpointing_enable()
    base_critic.config.use_cache = False
    print(f"{time.time()-t0:.0f}s")

    critic = ValueModel(base_critic, hidden_size=actor.config.hidden_size)
    critic.freeze_attention()
    cr.ok(f"Critic: value_head on {critic.value_head.weight.device}")
    cr.ok(f"  dtype: {critic.value_head.weight.dtype}")

    n_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in critic.parameters())
    cr.ok(f"  trainable: {n_trainable/1e6:.0f}M / {n_total/1e6:.0f}M total")

    # 检查显存
    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        cr.ok(f"  GPU {i}: {alloc:.1f} GB (actor + critic)")

    return critic


def check_mini_training_step(cr: CheckResult, actor, critic):
    """迷你训练步骤: forward + backward + optimizer step。"""
    print("\n[6] Mini Training Step")
    import torch
    from sao.standalone.grpo_step import compute_log_probs, dis_policy_loss
    from sao.standalone.critic import compute_values, compute_gae_batch, train_critic_step

    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()

    # 构造假数据: 2 条短轨迹
    input_ids_list = [
        torch.tensor([1, 100, 200, 300, 400, 500], dtype=torch.long),  # 3 prompt + 3 response
        torch.tensor([1, 100, 200, 300, 400, 500, 600, 700], dtype=torch.long),
    ]
    response_lens = [3, 5]
    rollout_log_probs = [
        torch.tensor([-0.5, -0.3, -0.8], dtype=torch.float32),
        torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=torch.float32),
    ]
    rewards = [1.0, 0.0]

    # 1. Critic forward
    print("  Critic forward...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        values_list = compute_values(critic, input_ids_list, response_lens, device)
    torch.cuda.empty_cache()
    print(f"{time.time()-t0:.1f}s")
    cr.ok(f"Critic forward: {len(values_list)} samples, shapes={[v.shape for v in values_list]}")

    # 2. GAE
    advantages_list, returns_list = compute_gae_batch(
        values_list, rewards, response_lens,
        gamma=1.0, alpha=1.5,
    )
    cr.ok(f"GAE: advantages computed, sample 0 mean={advantages_list[0].mean():.4f}")

    # 3. Actor forward (per-sample, gradient accumulation)
    print("  Actor forward...", end=" ", flush=True)
    t0 = time.time()
    actor.train()
    actor_optimizer = _create_optimizer(actor, cr)
    actor_optimizer.zero_grad()
    total_tokens = sum(response_lens)
    for i in range(len(input_ids_list)):
        tlp = compute_log_probs(actor, [input_ids_list[i]], [response_lens[i]], device, True)
        sample_loss, _ = dis_policy_loss(
            tlp, [rollout_log_probs[i]], [advantages_list[i]],
            clip_low=0.7, clip_high=6.0,
        )
        (sample_loss * response_lens[i] / total_tokens).backward()
        del tlp, sample_loss
    torch.cuda.empty_cache()
    print(f"{time.time()-t0:.1f}s")

    # 4. Check gradients
    has_grad = sum(1 for p in actor.parameters() if p.requires_grad and p.grad is not None)
    cr.ok(f"Actor backward: {has_grad} params have gradients")

    # 5. Optimizer step
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    actor_optimizer.step()
    cr.ok("Actor optimizer step completed")

    # 6. Check memory
    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        peak = torch.cuda.max_memory_allocated(i) / 1e9
        if peak > 75:
            cr.warn(f"  GPU {i} peak: {peak:.1f} GB (close to limit)")
        else:
            cr.ok(f"  GPU {i} peak: {peak:.1f} GB (safe)")

    # 7. Critic training step (TTUR K=2)
    print("  Critic training (K=2)...", end=" ", flush=True)
    t0 = time.time()
    critic.train()
    critic_optimizer = _create_optimizer(
        [p for p in critic.parameters() if p.requires_grad], cr
    )
    critic_loss, _ = train_critic_step(
        critic, critic_optimizer, input_ids_list, response_lens,
        returns_list, device, value_clip=0.2, k_epochs=2,
    )
    torch.cuda.empty_cache()
    print(f"{time.time()-t0:.1f}s")
    cr.ok(f"Critic training: loss={critic_loss:.4f}")


def _create_optimizer(params, cr: CheckResult):
    """Create 8-bit AdamW if available, else standard."""
    import torch
    try:
        import bitsandbytes as bnb
        cr.ok("Using 8-bit AdamW (bitsandbytes)")
        if hasattr(params, 'parameters'):
            params = list(params.parameters())
        return bnb.optim.AdamW8bit(params, lr=1e-6, weight_decay=0.1, betas=(0.9, 0.98))
    except ImportError:
        cr.warn("bitsandbytes not found, using standard AdamW")
        if hasattr(params, 'parameters'):
            params = list(params.parameters())
        return torch.optim.AdamW(params, lr=1e-6, weight_decay=0.1, betas=(0.9, 0.98))


def check_checkpoint(cr: CheckResult):
    """检查已有 checkpoint。"""
    print("\n[7] Checkpoints")
    ckpt_dir = f"{WORKDIR}/checkpoints/sao"
    if not os.path.exists(ckpt_dir):
        cr.ok("No checkpoints (fresh start)")
        return

    ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("step_")])
    if not ckpts:
        cr.ok("No checkpoints (fresh start)")
        return

    cr.ok(f"Found {len(ckpts)} checkpoints: {ckpts[-3:]}")
    latest = os.path.join(ckpt_dir, ckpts[-1])
    if os.path.exists(f"{latest}/config.json"):
        cr.ok(f"Latest checkpoint valid: {ckpts[-1]}")
    else:
        cr.warn(f"Latest checkpoint may be corrupted: {ckpts[-1]}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"
    setup()
    cr = CheckResult()

    print(f"\n{'='*50}")
    print(f"SAO Preflight Check — Mode: {mode}")
    print(f"{'='*50}")

    # Shared checks
    check_env(cr)
    check_data(cr)
    check_model_path(cr)

    if mode == "inference":
        # Inference machine checks
        if check_sglang(cr):
            check_sglang_generate(cr)
            check_token_alignment(cr)
            check_rollout_reward(cr)

    elif mode == "train":
        # Training machine checks
        actor = check_model_loading(cr)
        if actor:
            critic = check_critic_loading(cr, actor)
            if critic:
                check_mini_training_step(cr, actor, critic)
        check_checkpoint(cr)

    else:
        print(f"Usage: python3 preflight.py [inference|train]")
        sys.exit(1)

    ok = cr.summary()
    if ok:
        print("\n✅ All checks passed. Ready to train.")
    else:
        print("\n❌ Some checks failed. Fix before training.")
    sys.exit(0 if ok else 1)
