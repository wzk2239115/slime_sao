"""TIR (Tool-Integrated Reasoning) rollout.

Multi-turn generation: model writes reasoning + Python code → execute →
feed output back → model continues. Tracks action vs observation tokens
for Skip-Obs GAE and masked policy gradient.

Paper §3.2: "a sample is immediately fed into training upon generation"
Paper §3.2 Skip-Obs GAE Eq.4-5: bridge advantage across observation gaps.

Usage:
    from sao.standalone.tir_rollout import generate_tir_trajectory
    result = generate_tir_trajectory(port, prompt_ids, tokenizer, ...)
    # result has: resp_ids, action_mask, logprobs, text, n_code_executions
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time

try:
    from .rollout import _post
except ImportError:
    from rollout import _post


# ============================================================
# Code Execution Sandbox
# ============================================================
def execute_python(code: str, timeout: int = 10) -> str:
    """Execute Python code, return stdout (last 2000 chars).

    Runs in a subprocess with timeout. No import restrictions (trust model output).
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            f.flush()
            fname = f.name

        result = subprocess.run(
            ["python3", fname],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        os.unlink(fname)

        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            stderr = result.stderr.strip()[-500:]
            output = f"{output}\n{stderr}" if output else stderr
        return output[-2000:] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# Code Block Extraction
# ============================================================
CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[str]:
    """Extract ```python ... ``` blocks from model output."""
    return CODE_BLOCK_RE.findall(text)


def has_unclosed_code_block(text: str) -> bool:
    """Check if text has an unclosed ```python block (model mid-generation)."""
    count = text.count("```python")
    closed = text.count("```") - count  # closing ``` after each block
    return count > closed


# ============================================================
# Observation Formatting
# ============================================================
def format_observation(output: str) -> str:
    """Format code execution output for model feedback."""
    return f"\n```output\n{output}\n```\n"


# ============================================================
# Multi-Turn TIR Generation
# ============================================================
TIR_SYSTEM_PROMPT = """You are a mathematical reasoning assistant. You solve problems using Python code for computation.

## Instructions
1. Think step by step about the problem.
2. Write Python code in ```python blocks for ALL calculations — never do arithmetic by hand.
3. After each code block, you will see the output. Use it to continue.
4. Put your final answer in \\boxed{}.

## Example
User: Find the remainder when $7^{100}$ is divided by $13$.

```python
print(pow(7, 100, 13))
```
```output
9
```
The remainder is $\\boxed{9}$.

## Your Turn
Solve the given problem. ALWAYS use Python for calculations."""


def generate_tir_trajectory(
    port: int,
    prompt_ids: list[int],
    tokenizer,
    host: str = "127.0.0.1",
    max_turns: int = 20,
    max_new_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    code_timeout: int = 10,
    context_limit: int = 34000,
    logprobs: bool = True,
) -> dict:
    """Generate a multi-turn TIR trajectory.

    Each turn:
      1. Generate model action (reasoning + optional code)
      2. Extract & execute code blocks
      3. Format output as observation tokens
      4. Append to context, continue

    Returns:
        resp_ids: all response token IDs (action + observation interleaved)
        action_mask: [1=action, 0=observation] per token
        token_logprobs: rollout logprob per token (0.0 for observation)
        text: decoded text
        n_code_exec: number of code executions
        code_outputs: list of execution outputs
    """
    all_resp_ids: list[int] = []
    all_action_mask: list[int] = []
    all_logprobs: list[float] = []
    all_text = ""
    code_outputs: list[str] = []

    current_context = list(prompt_ids)
    n_code_exec = 0

    for turn in range(max_turns):
        remaining_budget = context_limit - len(current_context)
        if remaining_budget < max_new_tokens:
            break

        gen_tokens = min(max_new_tokens, remaining_budget)

        payload = {
            "input_ids": current_context,
            "sampling_params": {
                "n": 1,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": gen_tokens,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": len(current_context),
        }

        try:
            resp = _post(port, "/generate", payload, host=host, timeout=600)
        except Exception as e:
            break

        meta = resp.get("meta_info", {})
        lp_data = meta.get("output_token_logprobs", [])

        if lp_data and isinstance(lp_data[0], list):
            action_ids = [entry[1] for entry in lp_data]
            action_lps = [
                entry[0] if entry[0] is not None else 0.0 for entry in lp_data
            ]
        else:
            action_ids = resp.get("output_ids", [])
            action_lps = [0.0] * len(action_ids)

        if not action_ids:
            break

        action_text = tokenizer.decode(action_ids, skip_special_tokens=False)

        all_resp_ids.extend(action_ids)
        all_action_mask.extend([1] * len(action_ids))
        all_logprobs.extend(action_lps)
        all_text += action_text
        current_context.extend(action_ids)

        code_blocks = extract_code_blocks(action_text)
        if not code_blocks:
            break

        for code in code_blocks:
            output = execute_python(code, timeout=code_timeout)
            code_outputs.append(output)
            n_code_exec += 1

            obs_text = format_observation(output)
            obs_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]

            all_resp_ids.extend(obs_ids)
            all_action_mask.extend([0] * len(obs_ids))
            all_logprobs.extend([0.0] * len(obs_ids))
            all_text += obs_text
            current_context.extend(obs_ids)

            if len(current_context) >= context_limit:
                break

    return {
        "resp_ids": all_resp_ids,
        "action_mask": all_action_mask,
        "token_logprobs": all_logprobs,
        "text": all_text,
        "n_code_exec": n_code_exec,
        "code_outputs": code_outputs,
        "n_turns": turn + 1,
    }
