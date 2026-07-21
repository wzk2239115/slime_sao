"""蒸馏数据质量检查: 用 deepscaler RM 对比蒸馏轨迹的最终答案 vs 标准 label."""
import json
import sys
sys.path.insert(0, "/home/wzk/projects/slime")
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/wzk/datasets/sao_sft/distilled_20260720_231756.jsonl"

records = [json.loads(l) for l in open(PATH) if l.strip()]
print(f"总条数: {len(records)}")
print()

correct = 0
wrong_samples = []
turn_buckets = {}

for i, rec in enumerate(records):
    # 找最后一个 assistant message 作为最终答案
    final = None
    for m in reversed(rec["messages"]):
        if m["role"] == "assistant" and m.get("content"):
            final = m["content"]
            break
    label = str(rec["label"])

    # deepscaler 要求 response 含 </think> 或 ###Response 才评分;
    # 我们的蒸馏数据没这些标记, 直接用 math_utils 的 extract_answer 拿模型答案
    from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
    # 兜底: content 没答案时, 从 reasoning_content 最后段找 \boxed{}
    cand = final or ""
    model_ans = extract_answer(cand) if cand else None
    if not model_ans:
        # 找所有 assistant message 的 reasoning_content, 取最后一次出现的 \boxed{}
        for m in reversed(rec["messages"]):
            if m["role"] == "assistant" and m.get("reasoning_content"):
                rc = m["reasoning_content"]
                if "\\boxed" in rc:
                    model_ans = extract_answer(rc)
                    break
    model_ans = model_ans or ""
    ok = bool(model_ans) and (
        grade_answer_mathd(model_ans, label) or grade_answer_sympy(model_ans, label)
    )

    n_turns = rec["num_turns"]
    turn_buckets.setdefault(n_turns, {"total": 0, "correct": 0})
    turn_buckets[n_turns]["total"] += 1
    if ok:
        correct += 1
        turn_buckets[n_turns]["correct"] += 1
    else:
        wrong_samples.append((i, label, model_ans[:50], n_turns))

print(f"正确: {correct}/{len(records)} = {correct/len(records)*100:.1f}%")
print()
print("按 turns 分布:")
for n in sorted(turn_buckets):
    b = turn_buckets[n]
    print(f"  turns={n}: {b['correct']}/{b['total']} 正确 ({b['correct']/b['total']*100:.0f}%)")
print()
if wrong_samples:
    print(f"错误样本 ({len(wrong_samples)} 条):")
    for idx, label, ans, n in wrong_samples[:10]:
        print(f"  [#{idx}] label={label!r:20s} model={ans!r:30s} turns={n}")
