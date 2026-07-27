"""用 360 API (GLM-5.2) + slime PythonSandbox 蒸馏 TIR 数学轨迹.

SAO 论文 §4.1 用 GPT-OSS-120B 蒸馏 TIR 数据做 SFT 冷启动.
该数据未公开. 我们用 360 API 的 GLM-5.2 替代, 配合 slime PythonSandbox
执行 python 代码, 产出多轮 TIR trajectory.

改进版特性:
  - 高并发 (默认 15, API 实测支持 10+)
  - 断点续跑 (输出文件存在则跳过已完成的题目)
  - 内置正确性验证 (boxed answer vs label, 输出 correct 字段)
  - 扩展 sandbox 模块 (允许 numpy, sympy, scipy)

输出格式 (每行 jsonl):
    {
      "messages": [...],
      "label": "70",
      "correct": true,
      "num_turns": 2
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx

from examples.retool.tool_sandbox import PythonSandbox

# =========================================================================
# 1. 常量
# =========================================================================
API_URL = "https://api.360.cn/v1/chat/completions"
MODEL_NAME = "z-ai/glm-5.2"

SYSTEM_PROMPT = """You are a math expert. Solve the problem step by step.

Rules:
1. Think carefully before giving the answer.
2. When the computation is non-trivial (large numbers, modular arithmetic, \
combinatorics, geometry), USE the `python` tool to verify your reasoning.
3. After verification, put the final answer in \\boxed{}.
4. Keep your final response concise once you are confident.
"""

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
# 2. 扩展 sandbox: 允许 numpy, sympy, scipy
# =========================================================================
class ExtendedSandbox(PythonSandbox):
    def __init__(self, timeout=30, memory_limit="4GB"):
        super().__init__(timeout=timeout, memory_limit=memory_limit)
        self.allowed_modules = {
            "math", "random", "datetime", "collections", "itertools",
            "functools", "operator", "statistics", "decimal", "fractions",
            "numpy", "np", "sympy", "sp", "scipy", "re", "string",
            "typing", "dataclasses", "abc", "copy", "heapq",
            "bisect", "cmath", "numbers",
        }

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        dangerous = [
            r"import\s+os\b",
            r"import\s+sys\b",
            r"import\s+subprocess",
            r"import\s+shutil",
            r"import\s+socket",
            r"import\s+threading",
            r"import\s+multiprocessing",
            r"import\s+ctypes",
            r"__import__",
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"\bopen\s*\(",
            r"\bcompile\s*\(",
            r"__\w+__",
        ]
        for pat in dangerous:
            if re.search(pat, code):
                return False, f"Blocked pattern: {pat}"

        imports = re.findall(r"import\s+(\w+)", code) + re.findall(r"from\s+(\w+)", code)
        for imp in set(imports):
            if imp not in self.allowed_modules:
                return False, f"Module '{imp}' not allowed"
        return True, "ok"


# =========================================================================
# 3. 正确性验证: 提取 boxed + 归一化比较
# =========================================================================
def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return ""
    i = idx + 7
    depth = 1
    out = ""
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        out += text[i]
        i += 1
    return out.strip()


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = s.rstrip(".")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, TypeError):
        pass
    return s


def check_correct(predicted: str, label: str) -> bool:
    if not predicted or not label:
        return False
    return normalize_answer(predicted) == normalize_answer(label)


# =========================================================================
# 4. 调用 360 API (带重试)
# =========================================================================
async def call_glm52(
    client: httpx.AsyncClient,
    api_key: str,
    messages: list[dict],
    *,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    max_retries: int = 5,
) -> tuple[dict, str]:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tools": [PYTHON_TOOL],
    }

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
            wait = min(2 ** (attempt + 1), 30)
            print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}: {e}; sleep {wait}s")
            await asyncio.sleep(wait)

    raise RuntimeError(f"call_glm52 failed after {max_retries} retries: {last_err}")


# =========================================================================
# 5. 单条蒸馏
# =========================================================================
async def distill_one(
    client: httpx.AsyncClient,
    api_key: str,
    sandbox: ExtendedSandbox,
    problem: str,
    label: str,
    *,
    max_turns: int = 8,
    max_tokens: int = 16384,
    temperature: float = 0.7,
) -> dict | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem + "\n\nPut your final answer in \\boxed{}."},
    ]
    trajectory = list(messages)

    for turn in range(max_turns):
        msg, finish_reason = await call_glm52(
            client, api_key, messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if finish_reason == "length":
            return None

        stored_msg = {"role": "assistant"}
        stored_msg["content"] = msg.get("content") or ""
        if msg.get("reasoning_content"):
            stored_msg["reasoning_content"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            stored_msg["tool_calls"] = msg["tool_calls"]

        trajectory.append(stored_msg)
        messages.append(msg)

        if not msg.get("tool_calls"):
            break

        for call in msg["tool_calls"]:
            fn = call["function"]
            if fn["name"] != "python":
                continue
            try:
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                code = args.get("code", "")
                obs = await sandbox.execute_code(code)
            except Exception as e:
                obs = f"Error: {e}"

            tool_msg = {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": obs,
            }
            trajectory.append(tool_msg)
            messages.append(tool_msg)

    last_content = ""
    for m in reversed(trajectory):
        if m["role"] == "assistant" and m.get("content"):
            last_content = m["content"]
            break

    predicted = extract_boxed(last_content)
    correct = check_correct(predicted, label)

    return {
        "messages": trajectory,
        "label": label,
        "predicted": predicted,
        "correct": correct,
        "num_turns": len([m for m in trajectory if m["role"] == "assistant"]),
    }


# =========================================================================
# 6. 并发蒸馏 (断点续跑)
# =========================================================================
async def distill_batch(
    items: list[dict],
    api_key: str,
    *,
    concurrency: int = 15,
    max_turns: int = 8,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    sink_path: str,
    done_keys: set[str],
) -> tuple[int, int, int]:
    sem = asyncio.Semaphore(concurrency)
    sandbox = ExtendedSandbox(timeout=30, memory_limit="4GB")
    success = 0
    fail = 0
    correct_count = 0
    started = time.time()
    total = len(items)
    lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:
        with open(sink_path, "a") as fout:
            async def _one(idx: int, item: dict):
                nonlocal success, fail, correct_count
                problem_key = item["input"][:200]
                if problem_key in done_keys:
                    return

                async with sem:
                    tag = f"[{idx+1}/{total}]"
                    try:
                        rec = await distill_one(
                            client, api_key, sandbox,
                            problem=item["input"],
                            label=str(item.get("label", "")),
                            max_turns=max_turns,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        if rec is None:
                            fail += 1
                            print(f"  {tag} SKIP (truncated)")
                            return

                        async with lock:
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fout.flush()

                        success += 1
                        if rec["correct"]:
                            correct_count += 1

                        elapsed = time.time() - started
                        done = success + fail
                        eta = elapsed / max(done, 1) * (total - done - len(done_keys))
                        rate = done / max(elapsed, 1)
                        print(f"  {tag} {'✅' if rec['correct'] else '❌'} "
                              f"pred={rec['predicted'][:20]:20s} label={rec['label'][:20]:20s} "
                              f"turns={rec['num_turns']} "
                              f"| {success}ok {fail}skip "
                              f"acc={correct_count}/{success} "
                              f"rate={rate:.1f}/s eta={eta/3600:.1f}h")
                    except Exception as e:
                        fail += 1
                        print(f"  {tag} ERR {type(e).__name__}: {e}")

            await asyncio.gather(*[_one(i, it) for i, it in enumerate(items)])

    return success, fail, correct_count


# =========================================================================
# 7. CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="360 API GLM-5.2 TIR 蒸馏")
    ap.add_argument("--src", required=True, help="输入 jsonl: {input, label}")
    ap.add_argument("--dst", required=True, help="输出 jsonl")
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--api-key", default=None, help="API key (默认从 opencode.json 读)")
    args = ap.parse_args()

    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("API_360_KEY")
    if not api_key:
        cfg_path = Path.home() / ".config/opencode/opencode.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            api_key = cfg.get("provider", {}).get("360-proxy", {}).get("options", {}).get("apiKey")
    if not api_key:
        sys.exit("❌ 未找到 API key. 用 --api-key 或 export API_360_KEY=...")

    items = []
    with open(args.src) as fin:
        for line in fin:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.max_samples:
        items = items[: args.max_samples]

    done_keys: set[str] = set()
    if os.path.exists(args.dst):
        with open(args.dst) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    msgs = rec.get("messages", [])
                    for m in msgs:
                        if m.get("role") == "user":
                            done_keys.add(m["content"][:200])
                            break
                except json.JSONDecodeError:
                    continue
        print(f"断点续跑: 已完成 {len(done_keys)} 条, 跳过")

    todo = [it for it in items if it["input"][:200] not in done_keys]

    print(f"源数据 : {args.src} ({len(items)} 条)")
    print(f"待蒸馏 : {len(todo)} 条 (跳过 {len(done_keys)})")
    print(f"输出   : {args.dst}")
    print(f"并发   : {args.concurrency}, 温度 {args.temperature}")
    print()

    started = time.time()
    ok, fail, correct = asyncio.run(distill_batch(
        todo, api_key,
        concurrency=args.concurrency,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sink_path=args.dst,
        done_keys=done_keys,
    ))
    elapsed = time.time() - started

    print()
    print("=" * 60)
    print(f"完成: {ok} 成功 ({correct} 正确) / {fail} 失败")
    print(f"正确率: {correct}/{ok} = {correct/max(ok,1)*100:.1f}%")
    print(f"耗时: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"输出: {args.dst}")


if __name__ == "__main__":
    main()
