#!/usr/bin/env python3
"""训练前 rollout 验证 — 随机抽 N 道题，完整展示生成过程和 reward。

在推理机上跑（需要 sglang 已启动），确认：
  1. 模型能正确生成
  2. reward 函数正确匹配
  3. TIR 代码执行正常（如果开启）
  4. token IDs + logprobs 对齐

用法:
  # 纯推理模式（5题）
  BASH_ENV= python3 sao/standalone/sample_rollout.py

  # TIR 模式（5题，带 Python 工具）
  ENABLE_TIR=1 BASH_ENV= python3 sao/standalone/sample_rollout.py

  # 自定义题数
  BASH_ENV= python3 sao/standalone/sample_rollout.py --n 10
"""
from __future__ import annotations

import os, sys, json, random, time

WORKDIR = "/home/jovyan/h800fast/wangzekai/slime_sao"
ROOTFS  = "/home/jovyan/h800fast/wangzekai/slime_rootfs"
MODEL   = f"{WORKDIR}/models/Qwen3-30B-A3B-Thinking-2507"
DATA    = f"{WORKDIR}/datasets/MATH_train.jsonl"
PORT    = 30000

def setup():
    os.environ["LD_LIBRARY_PATH"] = f"{ROOTFS}/usr/local/cuda/lib64:{ROOTFS}/usr/local/nvidia/lib64"
    os.environ["PYTHONPATH"] = f"{WORKDIR}:{ROOTFS}/usr/local/lib/python3.12/dist-packages"
    os.environ["no_proxy"] = "*"
    sys.path.insert(0, WORKDIR)


def main():
    setup()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of problems to sample")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--enable-tir", action="store_true",
                        default=os.environ.get("ENABLE_TIR", "0") == "1")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    from transformers import AutoTokenizer
    from sao.standalone.reward import math_reward, extract_boxed, normalize_answer

    # Load data
    with open(args.data) as f:
        all_problems = [json.loads(line) for line in f]
    print(f"Loaded {len(all_problems)} problems from {args.data}")

    # Sample N problems
    samples = random.sample(all_problems, min(args.n, len(all_problems)))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    enable_tir = args.enable_tir
    mode_str = "TIR (Python tools)" if enable_tir else "pure reasoning"
    print(f"Mode: {mode_str}")
    print(f"Sampling {len(samples)} problems\n")
    print("=" * 80)

    results = []

    for idx, sample in enumerate(samples):
        problem = sample["input"]
        gt = sample.get("label") or sample.get("answer") or ""
        level = sample.get("level", "?")
        ptype = sample.get("type", "?")

        print(f"\n{'─' * 80}")
        print(f"Problem {idx+1}/{len(samples)}  [{level}] [{ptype}]")
        print(f"GT: {gt}")
        print(f"Problem: {problem[:300]}...")
        print()

        # Build prompt
        if enable_tir:
            from sao.standalone.tir_rollout import TIR_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": TIR_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]
        else:
            messages = [{"role": "user", "content": problem}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        t0 = time.time()

        if enable_tir:
            from sao.standalone.tir_rollout import generate_tir_trajectory
            result = generate_tir_trajectory(
                port=args.port,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                max_turns=20,
                max_new_tokens=4096,
                temperature=1.0,
                top_p=1.0,
                code_timeout=10,
                context_limit=34000,
            )
            resp_ids = result["resp_ids"]
            text = result["text"]
            action_mask = result["action_mask"]
            logprobs = result["token_logprobs"]
            n_code = result["n_code_exec"]
            code_outputs = result["code_outputs"]
            n_turns = result["n_turns"]
        else:
            from sao.standalone.rollout_worker import generate_via_sglang
            result = generate_via_sglang(
                port=args.port,
                prompt_ids=prompt_ids,
                host="127.0.0.1",
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=32768,
            )
            resp_ids = result.get("output_ids", [])
            logprobs = result.get("output_logprobs", [])
            text = result.get("text", "")
            if not resp_ids and text:
                resp_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                logprobs = [0.0] * len(resp_ids)
            elif not text and resp_ids:
                text = tokenizer.decode(resp_ids, skip_special_tokens=False)
            action_mask = None
            n_code = 0
            code_outputs = []
            n_turns = 1

        elapsed = time.time() - t0
        reward = math_reward(text, gt)
        boxed = extract_boxed(text)

        # Show results
        print(f"⏱  Time: {elapsed:.1f}s  |  Tokens: {len(resp_ids)}  |  Turns: {n_turns}  |  Code exec: {n_code}")
        print(f"📊 Reward: {reward}  |  GT: {gt}  |  Extracted: {boxed}")

        if normalize_answer(str(boxed)) == normalize_answer(gt):
            print("✅ CORRECT")
        else:
            print("❌ WRONG")

        # Show action/observation breakdown (TIR)
        if action_mask:
            n_action = sum(action_mask)
            n_obs = len(action_mask) - n_action
            print(f"   Action tokens: {n_action}  |  Observation tokens: {n_obs}")

        # Show code execution details (TIR)
        if code_outputs:
            print(f"\n   Code executions:")
            for i, out in enumerate(code_outputs):
                print(f"   [{i+1}] output: {out[:200]}")

        # Logprobs alignment check
        if resp_ids and logprobs:
            aligned = len(logprobs) == len(resp_ids)
            if aligned:
                nonzero = sum(1 for lp in logprobs if lp != 0.0)
                print(f"   Logprobs: {len(logprobs)} tokens, {nonzero} nonzero  ✅ aligned")
            else:
                print(f"   ⚠️ Logprobs misaligned: {len(logprobs)} vs {len(resp_ids)} tokens")

        # Show response tail (last 500 chars)
        print(f"\n   Response (last 500 chars):")
        print(f"   ...{text[-500:]}")

        results.append({
            "idx": idx + 1,
            "reward": reward,
            "gt": gt,
            "extracted": str(boxed),
            "n_tokens": len(resp_ids),
            "n_code": n_code,
            "time": elapsed,
            "level": level,
        })

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'#':>3}  {'Reward':>6}  {'Tokens':>6}  {'Code':>4}  {'Time':>5}  {'Level':>8}  GT → Extracted")
    print(f"{'─'*80}")
    for r in results:
        correct = "✅" if r["reward"] == 1.0 else "❌"
        print(f"{r['idx']:3d}  {correct}     {r['n_tokens']:6d}  {r['n_code']:4d}  {r['time']:4.0f}s  {r['level']:>8}  {r['gt']} → {r['extracted']}")

    n_correct = sum(1 for r in results if r["reward"] == 1.0)
    avg_tokens = sum(r["n_tokens"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)
    total_code = sum(r["n_code"] for r in results)

    print(f"\n  Accuracy: {n_correct}/{len(results)} ({n_correct/len(results)*100:.0f}%)")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Avg time: {avg_time:.0f}s")
    print(f"  Total code executions: {total_code}")

    if n_correct == 0:
        print("\n⚠️  WARNING: All rewards are 0! Check:")
        print("  1. Is the reward function matching correctly?")
        print("  2. Is the model generating \\boxed{} answers?")
        print("  3. Are the ground truth answers in the right format?")
        print("\n  DO NOT start training until at least some rewards are non-zero.")
    elif n_correct / len(results) < 0.1:
        print("\n⚠️  WARNING: Very low accuracy (<10%). Model may be struggling.")
    else:
        print(f"\n✅ Looks good! {n_correct}/{len(results)} correct. Ready to train.")


if __name__ == "__main__":
    main()
