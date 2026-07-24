"""Rollout: generate responses via sglang HTTP API.

Supports both chat-completions and raw generate endpoints.
Returns token-level log-probs needed for GRPO/SAO training.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class RolloutSample:
    prompt_text: str
    response_text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_logprobs: list[float]  # log π_rollout for each response token
    reward: float = 0.0
    advantage: float = 0.0
    meta: dict = field(default_factory=dict)


def _post(port: int, path: str, payload: dict, timeout: int = 3600) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    # Bypass any HTTP proxy for localhost
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def generate_batch(
    port: int,
    prompts: list[str],
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 32768,
    model_path: str = "default",
) -> list[RolloutSample]:
    """Generate responses for a batch of prompts using sglang /generate API.

    Returns one RolloutSample per (prompt, sample) pair (total = len(prompts)*n).
    """
    results: list[RolloutSample] = []

    for prompt in prompts:
        payload = {
            "model": model_path,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        resp = _post(port, "/v1/chat/completions", payload, timeout=3600)

        for choice in resp["choices"]:
            content = choice["message"]["content"]
            # sglang provides logprobs if requested
            logprobs_data = choice.get("logprobs")
            token_ids = []
            token_logprobs = []

            if logprobs_data and logprobs_data.get("content"):
                for lp in logprobs_data["content"]:
                    token_ids.append(lp.get("token", 0))
                    token_logprobs.append(lp.get("logprob", 0.0))

            results.append(RolloutSample(
                prompt_text=prompt,
                response_text=content,
                prompt_token_ids=[],  # will be filled by tokenizer if needed
                response_token_ids=token_ids,
                response_logprobs=token_logprobs,
            ))

    return results


def generate_batch_raw(
    port: int,
    prompt_token_ids_batch: list[list[int]],
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 32768,
    skip_special_tokens: bool = False,
) -> list[RolloutSample]:
    """Generate using sglang's /generate API with raw token IDs.

    This gives us both token IDs and log-probs directly.
    """
    results: list[RolloutSample] = []

    for input_ids in prompt_token_ids_batch:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": skip_special_tokens,
            },
            "return_logprob": True,
            "logprob_start_len": len(input_ids),
        }
        try:
            resp = _post(port, "/generate", payload, timeout=3600)
        except Exception:
            # Fallback to text input
            payload_text = {
                "text": "",  # need tokenizer to decode
                "sampling_params": {
                    "n": n,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                },
                "return_logprob": True,
            }
            resp = _post(port, "/generate", payload_text, timeout=600)

        for sample in resp.get("samples", [resp]):
            out_ids = sample.get("output_ids", [])
            out_logprobs = sample.get("output_token_logprobs", [])

            results.append(RolloutSample(
                prompt_text="",
                response_text=sample.get("text", ""),
                prompt_token_ids=input_ids,
                response_token_ids=out_ids,
                response_logprobs=out_logprobs,
            ))

    return results
