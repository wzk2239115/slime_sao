#!/usr/bin/env python3
"""用 Qwen3 base model 的 sglang 批量生成 TIR 轨迹，用于 SFT + value pretraining。

这是自蒸馏：用目标模型自己生成训练数据。
产出格式和 RL rollout 完全一致（prompt_ids, resp_ids, action_mask, logprobs, reward）。
sandbox 无 import 限制（sympy/numpy 都能用）。

用法（在算力机上）:
    # 1. 启动 sglang（base model）
    BASH_ENV= python3 sao/standalone/launch.py inference

    # 2. 跑批量 rollout（8卡H100 充分利用）
    BASH_ENV= python3 sao/standalone/collect_rollouts.py \
        --data datasets/MATH_train.jsonl \
        --output datasets/self_distill/rollouts.jsonl \
        --samples-per-problem 2 \
        --concurrency 20
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer
from tir_rollout import generate_tir_trajectory, TIR_SYSTEM_PROMPT
from reward import math_reward


def collect_one(
    port: int,
    tokenizer,
    problem: str,
    label: str,
    temperature: float,
    max_turns: int,
    max_new_tokens: int,
) -> dict | None:
    """对一道题跑一条 TIR 轨迹。"""
    messages = [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    try:
        result = generate_tir_trajectory(
            port=port,
            prompt_ids=prompt_ids,
            tokenizer=tokenizer,
            host="127.0.0.1",
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            code_timeout=10,
            context_limit=32000,
            logprobs=True,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    reward = math_reward(result["text"], label)

    return {
        "problem": problem[:200],
        "label": label,
        "prompt_ids": prompt_ids,
        "resp_ids": result["resp_ids"],
        "action_mask": result["action_mask"],
        "token_logprobs": result["token_logprobs"],
        "text": result["text"],
        "reward": reward,
        "n_code_exec": result["n_code_exec"],
        "n_turns": result["n_turns"],
        "n_resp_tokens": len(result["resp_ids"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MATH jsonl: {input, label}")
    ap.add_argument("--output", required=True, help="输出 jsonl")
    ap.add_argument("--model", default=None, help="tokenizer 路径（默认从 sglang 推断）")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--samples-per-problem", type=int, default=2, help="每题采样几条")
    ap.add_argument("--concurrency", type=int, default=20, help="并发线程数（8卡H100建议20-30）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=None, help="只处理前 N 题（调试）")
    ap.add_argument("--shard", type=int, default=0, help="分片: 当前第几片 (0-indexed)")
    ap.add_argument("--num-shards", type=int, default=1, help="分片: 总共几片（多机并行）")
    args = ap.parse_args()

    # tokenizer
    if args.model:
        tok_path = args.model
    else:
        # 尝试从环境推断
        tok_path = os.environ.get("MODEL_PATH", "/home/jovyan/h800fast/wangzekai/slime_sao/models/Qwen3-30B-A3B-Thinking-2507")
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # 读数据
    problems = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if args.max_samples:
        problems = problems[: args.max_samples]

    # 分片（多机并行）
    if args.num_shards > 1:
        problems = [p for i, p in enumerate(problems) if i % args.num_shards == args.shard]
        print(f"分片 {args.shard}/{args.num_shards}: 本机负责 {len(problems)} 题")

    # 展开成 samples_per_problem 条任务
    tasks = []
    for p in problems:
        for _ in range(args.samples_per_problem):
            tasks.append(p)
    print(f"题目: {len(problems)}, 采样: {args.samples_per_problem}, 总任务: {len(tasks)}")

    # 断点续跑
    done_problems = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done_problems.add(rec["problem"][:200])
                except Exception:
                    continue
        print(f"断点续跑: 已完成 {len(done_problems)} 条")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 并发跑
    fout = open(args.output, "a")
    lock = threading.Lock()

    success = 0
    correct = 0
    fail = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for i, task in enumerate(tasks):
            if task["input"][:200] in done_problems:
                continue
            fut = pool.submit(
                collect_one,
                args.port, tokenizer,
                task["input"], str(task.get("label", "")),
                args.temperature, args.max_turns, args.max_new_tokens,
            )
            futures[fut] = (i, task)

        total = len(futures)
        for fut in as_completed(futures):
            idx, task = futures[fut]
            try:
                result = fut.result()
                if result is None:
                    fail += 1
                    continue

                with lock:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

                success += 1
                if result["reward"] > 0:
                    correct += 1

                done = success + fail
                elapsed = time.time() - started
                rate = done / max(elapsed, 1)
                eta = (total - done) / max(rate, 0.01) / 3600
                tag = "✅" if result["reward"] > 0 else "❌"
                print(f"  [{done}/{total}] {tag} turns={result['n_turns']} "
                      f"code={result['n_code_exec']} tokens={result['n_resp_tokens']} "
                      f"acc={correct}/{success}={correct/max(success,1)*100:.0f}% "
                      f"rate={rate:.1f}/s eta={eta:.1f}h")
            except Exception as e:
                fail += 1
                print(f"  ERR: {e}")

    fout.close()
    elapsed = time.time() - started
    print(f"\n完成: {success} 成功 ({correct} 正确) / {fail} 失败, 耗时 {elapsed/3600:.1f}h")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
