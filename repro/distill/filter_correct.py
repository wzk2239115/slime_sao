"""过滤蒸馏数据: 只保留 answer 正确的样本.

用法:
    python SAO/repro/distill/filter_correct.py \
        --src /home/wzk/datasets/sao_sft/distilled_20260721_001201.jsonl \
        --dst /home/wzk/datasets/sao_sft/distilled_correct.jsonl
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/wzk/projects/slime")
from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def check_one(rec: dict) -> bool:
    """返回 rec 是否答案正确."""
    # 找最后一个非空 assistant content
    final = None
    for m in reversed(rec["messages"]):
        if m["role"] == "assistant" and (m.get("content") or "").strip():
            final = m["content"]
            break
    if not final:
        return False

    label = str(rec["label"])
    model_ans = extract_answer(final) or ""
    if not model_ans:
        return False

    return bool(
        grade_answer_mathd(model_ans, label)
        or grade_answer_sympy(model_ans, label)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.src) if l.strip()]
    correct = [r for r in records if check_one(r)]

    with open(args.dst, "w") as fout:
        for r in correct:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"输入: {len(records)} 条")
    print(f"保留: {len(correct)} 条 ({len(correct) / len(records) * 100:.1f}%)")
    print(f"输出: {args.dst}")

    # 简单统计
    turn_dist = {}
    for r in correct:
        n = r["num_turns"]
        turn_dist[n] = turn_dist.get(n, 0) + 1
    print(f"turns 分布: {dict(sorted(turn_dist.items()))}")

    # 平均 trajectory 长度
    avg_turns = sum(r["num_turns"] for r in correct) / len(correct)
    avg_msgs = sum(len(r["messages"]) for r in correct) / len(correct)
    print(f"平均 turns: {avg_turns:.1f}, 平均 messages: {avg_msgs:.1f}")


if __name__ == "__main__":
    main()
