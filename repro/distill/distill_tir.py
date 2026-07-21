"""用 360 API (GLM-5.2) + slime PythonSandbox 蒸馏 TIR 数学轨迹.

========================================================================
背景: 为什么需要这个脚本
========================================================================
SAO 论文 §4.1 用 GPT-OSS-120B 蒸馏了一批 TIR (Tool-Integrated Reasoning)
数据做 SFT 起步. 该数据未公开. 我们没有 GPT-OSS-120B, 但有:

  1. 360 API (https://api.360.cn/v1) 提供 GLM-5.2, 推理质量好, 支持 tool_calls;
  2. slime 自带的 examples/retool/tool_sandbox.py:PythonSandbox, 可本地跑 python.

本脚本把这两个粘起来: 用 GLM-5.2 当教师, PythonSandbox 当工具执行器,
跑出一条条带「thinking + python call + observation + ... + final answer」的
多轮 trajectory, 直接当 SAO 的 SFT 冷启动数据.

========================================================================
输出格式 (slime multi-turn SFT)
========================================================================
每条 jsonl 行:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "<题目>"},
        {"role": "assistant", "content": "<thinking>", "tool_calls": [
            {"name": "python", "arguments": {"code": "..."}}
        ]},
        {"role": "tool", "content": "<stdout>"},
        {"role": "assistant", "content": "<最终答案, 含 \\\\boxed{}>"}      ],
      "label": "70"
    }

slime ``sft_rollout.generate_rollout`` 会读 sample.prompt (=messages),
用 MultiTurnLossMaskGenerator 自动生成 token-level loss_mask
(assistant token=1, tool/user token=0).

========================================================================
用法
========================================================================
    # 1. 设置 API key (从 ~/.config/opencode/opencode.json 拷贝)
    export API_360_KEY="<your-360-api-key>"

    # 2. 蒸馏 AIME2025 (30 题, 用于 sanity check)
    python SAO/repro/distill/distill_tir.py \\
        --src /home/wzk/datasets/AIME2025/slime/aime2025-all.jsonl \\
        --dst /home/wzk/datasets/sao_sft/aime2025_distilled.jsonl \\
        --concurrency 2 --max-turns 8

    # 3. 蒸馏更大规模训练集 (例如 dapo-math-17k, 自备)
    python SAO/repro/distill/distill_tir.py \\
        --src /path/to/dapo-math-17k.jsonl \\
        --dst /home/wzk/datasets/sao_sft/dapo_distilled.jsonl \\
        --concurrency 2 --max-turns 8 --max-samples 5000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

# 让脚本能 import slime 包
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx

# 直接从 slime retool 示例复用 PythonSandbox, 避免重写
from examples.retool.tool_sandbox import PythonSandbox


# =========================================================================
# 1. 常量: 360 API 端点 + system prompt + python tool schema
# =========================================================================
API_URL = "https://api.360.cn/v1/chat/completions"
MODEL_NAME = "z-ai/glm-5.2"

# TIR system prompt: 鼓励模型「先思考, 不确定就调 python 验证, 最后给 \boxed{}」
SYSTEM_PROMPT = """You are a math expert. Solve the problem step by step.

Rules:
1. Think carefully before giving the answer.
2. When the computation is non-trivial (large numbers, modular arithmetic, \
combinatorics, geometry), USE the `python` tool to verify your reasoning.
3. After verification, put the final answer in \\boxed{}.
4. Keep your final response concise once you are confident.
"""

# OpenAI 风格 tool schema: 一个叫 python 的函数, 接 code 字符串
PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute python code and return stdout. Use for math verification.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."}
            },
            "required": ["code"],
        },
    },
}


# =========================================================================
# 2. 调用 360 API 的异步封装 (带重试)
# =========================================================================
async def call_glm52(
    client: httpx.AsyncClient,
    api_key: str,
    messages: list[dict],
    *,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    with_tool: bool = True,
    max_retries: int = 3,
) -> tuple[dict, str]:
    """调一次 GLM-5.2, 返回 (message dict, finish_reason).

    360 API 偶尔会超时或 5xx, 这里做指数退避重试.
    注意: GLM-5.2 是 thinking 模型, reasoning_content 可能很长 (12k+ 字符),
          max_tokens 至少 16k 才够 thinking + content. 否则 content 会被
          reasoning 挤掉, 拿不到最终答案.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if with_tool:
        payload["tools"] = [PYTHON_TOOL]

    last_err = None
    for attempt in range(max_retries):
        try:
            r = await client.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            return choice["message"], choice.get("finish_reason", "stop")
        except Exception as e:
            last_err = e
            # 指数退避: 2s, 4s, 8s
            wait = 2 ** (attempt + 1)
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}; sleep {wait}s")
            await asyncio.sleep(wait)

    raise RuntimeError(f"call_glm52 failed after {max_retries} retries: {last_err}")


# =========================================================================
# 3. 单条题目蒸馏: 多轮 tool-call 循环
# =========================================================================
async def distill_one(
    client: httpx.AsyncClient,
    api_key: str,
    sandbox: PythonSandbox,
    problem: str,
    label: str,
    *,
    max_turns: int = 8,
    max_tokens: int = 16384,
) -> dict | None:
    """蒸馏一条 trajectory.

    流程:
      user(题目) → assistant(thinking + tool_call)
                 → tool(stdout) → assistant(thinking + tool_call)
                 → ... 直到 assistant 不再调 tool, 给 final answer

    如果某轮被 max_tokens 截断 (finish_reason=length), 整条 trajectory 丢弃,
    因为 reasoning 没结束拿不到可信答案.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem + "\n\nPut your final answer in \\boxed{}."},
    ]
    trajectory = list(messages)  # 完整 trajectory (会被存到 jsonl)

    for turn in range(max_turns):
        # 调 GLM-5.2
        msg, finish_reason = await call_glm52(
            client, api_key, messages,
            max_tokens=max_tokens,
            with_tool=True,
        )

        # 被截断: reasoning 还没思考完, content 没了, 整条丢弃
        if finish_reason == "length":
            return None

        # GLM-5.2 把 thinking 放在 reasoning_content, 最终回复在 content.
        # 我们存的时候把 reasoning_content 合并到 content (slime SFT 只认 content).
        # 也可以保留为独立字段方便后续筛选.
        stored_msg = {"role": "assistant"}
        if msg.get("content"):
            stored_msg["content"] = msg["content"]
        else:
            # 模型只 think 没说话 (常见于第一轮, 仍在思考中)
            stored_msg["content"] = ""
        if msg.get("reasoning_content"):
            # 把 thinking 也存下来, 后续可选用作训练数据
            stored_msg["reasoning_content"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            stored_msg["tool_calls"] = msg["tool_calls"]

        trajectory.append(stored_msg)
        messages.append(msg)  # 给 API 的下一轮要带原始 message (含 tool_calls)

        # 没调 tool → 已经给最终答案了, 结束
        if not msg.get("tool_calls"):
            break

        # 执行 python tool, 把 stdout 灌回去
        for call in msg["tool_calls"]:
            fn = call["function"]
            if fn["name"] != "python":
                continue
            try:
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                code = args.get("code", "")
                # 用 slime 的 sandbox 执行 (有安全检查 + 超时 + 内存限制)
                obs = await sandbox.execute_code(code)
            except Exception as e:
                obs = f"Error executing code: {e}"

            # OpenAI 风格: tool 消息要带 tool_call_id
            tool_msg = {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": obs,
            }
            trajectory.append(tool_msg)
            messages.append(tool_msg)

    return {
        "messages": trajectory,
        "label": label,
        "num_turns": len([m for m in trajectory if m["role"] == "assistant"]),
    }


# =========================================================================
# 4. 并发蒸馏 (asyncio + semaphore)
# =========================================================================
async def distill_batch(
    items: list[dict],
    api_key: str,
    *,
    concurrency: int = 2,
    max_turns: int = 8,
    max_tokens: int = 4096,
    sink_path: str,
) -> tuple[int, int]:
    """并发蒸馏一批题目, 实时写入 sink_path.

    Returns:
        (成功数, 失败数)
    """
    sem = asyncio.Semaphore(concurrency)
    sandbox = PythonSandbox(timeout=30, memory_limit="100MB")
    success = 0
    fail = 0
    started = time.time()

    async with httpx.AsyncClient() as client:
        # 流式写入, 每条完成就 flush, 避免中途崩溃丢数据
        with open(sink_path, "w") as fout:
            async def _one(idx: int, item: dict):
                nonlocal success, fail
                async with sem:
                    tag = f"[{idx + 1}/{len(items)}]"
                    try:
                        rec = await distill_one(
                            client, api_key, sandbox,
                            problem=item["input"],
                            label=str(item.get("label", "")),
                            max_turns=max_turns,
                            max_tokens=max_tokens,
                        )
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        success += 1
                        elapsed = time.time() - started
                        print(f"  {tag} ✅ label={rec['label']} turns={rec['num_turns']} "
                              f"({elapsed:.0f}s elapsed, {success} ok / {fail} fail)")
                    except Exception as e:
                        fail += 1
                        print(f"  {tag} ❌ {type(e).__name__}: {e}")
                        traceback.print_exc()

            await asyncio.gather(*[_one(i, it) for i, it in enumerate(items)])

    return success, fail


# =========================================================================
# 5. CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="输入 jsonl, 字段: input, label")
    ap.add_argument("--dst", required=True, help="输出 jsonl, slime multi-turn SFT 格式")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="并发数. 360 API 5 并发会 timeout, 建议 2-3.")
    ap.add_argument("--max-turns", type=int, default=8,
                    help="单条 trajectory 最多多少轮 tool call")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="单轮 API 调用最大 token. GLM-5.2 thinking 长, 建议 >= 16k")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="只处理前 N 条 (调试用)")
    ap.add_argument("--api-key-env", default="API_360_KEY",
                    help="API key 环境变量名 (默认 API_360_KEY)")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        # 尝试从 opencode.json 读
        cfg_path = Path.home() / ".config/opencode/opencode.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            api_key = cfg.get("provider", {}).get("360-proxy", {}).get("options", {}).get("apiKey")
        if not api_key:
            sys.exit(f"❌ 未找到 API key. 请 export {args.api_key_env}=... 或配置 {cfg_path}")

    # 读输入
    items = []
    with open(args.src) as fin:
        for line in fin:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.max_samples:
        items = items[: args.max_samples]

    print(f"源数据: {args.src} ({len(items)} 条)")
    print(f"输出  : {args.dst}")
    print(f"并发  : {args.concurrency}, 单条最大轮数: {args.max_turns}")
    print(f"模型  : {MODEL_NAME} via {API_URL}")
    print()

    # 跑
    started = time.time()
    ok, fail = asyncio.run(distill_batch(
        items, api_key,
        concurrency=args.concurrency,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        sink_path=args.dst,
    ))
    elapsed = time.time() - started

    print()
    print("=" * 60)
    print(f"完成: {ok} 成功 / {fail} 失败, 耗时 {elapsed:.0f}s")
    print(f"平均单条: {elapsed / max(ok, 1):.1f}s")
    print(f"输出: {args.dst}")
    print()
    print("下一步: 用 slime sft_rollout 做 SFT 起步")
    print("  --rollout-function-path slime.rollout.sft_rollout.generate_rollout")
    print("  --loss-mask-type qwen3 (按目标模型选)")


if __name__ == "__main__":
    main()
