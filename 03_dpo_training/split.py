"""
Step 3: Split capped DPO pairs into train/validation by qid (no data leakage).

Input:  cleaned_pairs_capped_top2.jsonl
Output: train_pairs.jsonl, val_pairs.jsonl

Usage:
  python split.py --input cleaned_pairs_capped_top2.jsonl --train_ratio 0.8 --seed 42
"""

import json
import random
import argparse
from collections import Counter
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize_split(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    qid_counter = Counter()
    dataset_counter = Counter()
    pair_type_counter = Counter()

    for rec in records:
        qid_counter[rec["qid"]] += 1
        dataset_counter[rec["dataset"]] += 1
        pair_type_counter[rec["pair_type"]] += 1

    pairs_per_qid = list(qid_counter.values())
    if pairs_per_qid:
        avg_ppq = sum(pairs_per_qid) / len(pairs_per_qid)
        min_ppq = min(pairs_per_qid)
        max_ppq = max(pairs_per_qid)
    else:
        avg_ppq, min_ppq, max_ppq = 0.0, 0, 0

    return {
        "n_records": len(records),
        "n_unique_qids": len(qid_counter),
        "avg_pairs_per_qid": avg_ppq,
        "min_pairs_per_qid": min_ppq,
        "max_pairs_per_qid": max_ppq,
        "dataset_distribution": dict(dataset_counter),
        "pair_type_distribution": dict(pair_type_counter),
    }


def print_split_summary(name: str, summary: Dict[str, Any]) -> None:
    print(f"\n========== {name} ==========")
    print(f"Records            : {summary['n_records']}")
    print(f"Unique qids        : {summary['n_unique_qids']}")
    print(f"Avg pairs / qid    : {summary['avg_pairs_per_qid']:.3f}")
    print(f"Min pairs / qid    : {summary['min_pairs_per_qid']}")
    print(f"Max pairs / qid    : {summary['max_pairs_per_qid']}")

    print("\nDataset distribution:")
    for k, v in summary["dataset_distribution"].items():
        print(f"  {k}: {v}")

    print("\nPair-type distribution:")
    for k, v in summary["pair_type_distribution"].items():
        print(f"  {k}: {v}")
    print("=" * (12 + len(name)))


def split_by_qid(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set, set]:
    unique_qids = sorted({rec["qid"] for rec in records})
    rng = random.Random(seed)
    rng.shuffle(unique_qids)

    n_train = int(len(unique_qids) * train_ratio)
    train_qids = set(unique_qids[:n_train])
    val_qids = set(unique_qids[n_train:])

    train_records = [rec for rec in records if rec["qid"] in train_qids]
    val_records = [rec for rec in records if rec["qid"] in val_qids]

    return train_records, val_records, train_qids, val_qids


def verify_split(all_records, train_records, val_records, train_qids, val_qids):
    all_qids = {rec["qid"] for rec in all_records}
    overlap = train_qids.intersection(val_qids)
    assert len(overlap) == 0, f"Leakage detected: overlapping qids: {overlap}"
    assert train_qids.union(val_qids) == all_qids, "Some qids are missing after split."
    assert len(train_records) + len(val_records) == len(all_records), "Record count mismatch."
    print("[OK] No qid overlap between train and val.")
    print("[OK] All qids are assigned.")
    print("[OK] Record counts match.")


def main():
    parser = argparse.ArgumentParser(description="Split DPO pairs into train/val by qid.")
    parser.add_argument("--input", type=str, default="cleaned_pairs_capped_top2.jsonl",
                        help="Path to capped input JSONL")
    parser.add_argument("--train_output", type=str, default="train_pairs.jsonl")
    parser.add_argument("--val_output", type=str, default="val_pairs.jsonl")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} capped records.")

    train_records, val_records, train_qids, val_qids = split_by_qid(
        records, train_ratio=args.train_ratio, seed=args.seed
    )
    verify_split(records, train_records, val_records, train_qids, val_qids)

    print_split_summary("TRAIN SPLIT", summarize_split(train_records))
    print_split_summary("VALIDATION SPLIT", summarize_split(val_records))

    save_jsonl(train_records, args.train_output)
    save_jsonl(val_records, args.val_output)
    print(f"\nSaved: {args.train_output}, {args.val_output}")


if __name__ == "__main__":
    main()
