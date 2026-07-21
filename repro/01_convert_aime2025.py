"""把 AIME2025 原始 jsonl 转成 slime eval 期望的 schema.

slime 默认字段是 input/label, AIME2025 原始是 question/answer.
两种方案都行, 这里我们做一次转换 (更直观, 不依赖 eval_config 的字段重映射).

用法:
    python SAO/repro/01_convert_aime2025.py \
        --src /home/wzk/datasets/AIME2025 \
        --dst /home/wzk/datasets/AIME2025/slime
"""
import argparse
import json
import os


def convert_one(in_path: str, out_path: str) -> int:
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # slime 字段: input (prompt 内容), label (ground truth)
            # question → input, answer → label
            rec = {
                "input": obj["question"],
                "label": str(obj["answer"]),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/wzk/datasets/AIME2025")
    ap.add_argument("--dst", default="/home/wzk/datasets/AIME2025/slime")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    total = 0
    for split in ("aime2025-I", "aime2025-II"):
        in_path = os.path.join(args.src, f"{split}.jsonl")
        out_path = os.path.join(args.dst, f"{split}.jsonl")
        n = convert_one(in_path, out_path)
        total += n
        print(f"  {split}: {n} 条 → {out_path}")

    # 合并一个全集 (AIME2025 I + II, 30 题)
    merged = os.path.join(args.dst, "aime2025-all.jsonl")
    with open(merged, "w") as fout:
        for split in ("aime2025-I", "aime2025-II"):
            with open(os.path.join(args.dst, f"{split}.jsonl")) as fin:
                fout.write(fin.read())
    print(f"  合并: {total} 条 → {merged}")
    print(f"\n✅ 转换完成, 共 {total} 题")


if __name__ == "__main__":
    main()
