"""
Step 2: Per-qid capped sampling of DPO preference pairs.

Input:  cleaned_pairs.jsonl
Output: cleaned_pairs_capped_top2.jsonl, cleaned_pairs_capped_top2_summary.json

Usage:
  python cap_dpo_pairs.py --input cleaned_pairs.jsonl --output cleaned_pairs_capped_top2.jsonl
  # or simply run without args to use default paths
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {line_idx}: {e}")
    return data


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_summary(summary: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def group_by_qid(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["qid"]].append(rec)
    return grouped


def select_pairs_for_one_qid(
    qid_records: List[Dict[str, Any]],
    max_pairs_per_qid: int = 2
) -> List[Dict[str, Any]]:
    same_pairs = sorted(
        [r for r in qid_records if r["pair_type"] == "same_answer"],
        key=lambda x: x["score_margin"], reverse=True
    )
    cross_pairs = sorted(
        [r for r in qid_records if r["pair_type"] == "cross_answer"],
        key=lambda x: x["score_margin"], reverse=True
    )

    selected = []

    if max_pairs_per_qid == 2:
        if same_pairs and cross_pairs:
            selected.append(cross_pairs[0])
            selected.append(same_pairs[0])
        elif same_pairs:
            selected.extend(same_pairs[:2])
        elif cross_pairs:
            selected.extend(cross_pairs[:2])
    else:
        used_ids = set()
        if same_pairs and cross_pairs:
            selected.append(cross_pairs[0])
            selected.append(same_pairs[0])
            used_ids.add(id(cross_pairs[0]))
            used_ids.add(id(same_pairs[0]))

        remaining_slots = max_pairs_per_qid - len(selected)
        if remaining_slots > 0:
            all_pairs_sorted = sorted(
                qid_records, key=lambda x: x["score_margin"], reverse=True
            )
            for rec in all_pairs_sorted:
                if id(rec) not in used_ids:
                    selected.append(rec)
                    used_ids.add(id(rec))
                    if len(selected) >= max_pairs_per_qid:
                        break

    return selected


def cap_dataset_by_qid(
    records: List[Dict[str, Any]],
    max_pairs_per_qid: int = 2
) -> List[Dict[str, Any]]:
    grouped = group_by_qid(records)
    capped_records = []
    for qid, qid_records in grouped.items():
        selected = select_pairs_for_one_qid(qid_records, max_pairs_per_qid)
        capped_records.extend(selected)
    return capped_records


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    dataset_counter = Counter()
    pair_type_counter = Counter()
    qid_counter = Counter()

    for rec in records:
        dataset_counter[rec["dataset"]] += 1
        pair_type_counter[rec["pair_type"]] += 1
        qid_counter[rec["qid"]] += 1

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
        "top_repeated_qids": qid_counter.most_common(20),
    }


def build_comparison_summary(
    original_records: List[Dict[str, Any]],
    capped_records: List[Dict[str, Any]],
    max_pairs_per_qid: int
) -> Dict[str, Any]:
    return {
        "strategy": {
            "name": "per_qid_capped_sampling",
            "max_pairs_per_qid": max_pairs_per_qid,
            "rule": (
                "If both same_answer and cross_answer exist, keep best 1 cross + best 1 same; "
                "otherwise keep top-k by score_margin within the available type."
            )
        },
        "before": summarize_records(original_records),
        "after": summarize_records(capped_records),
    }


def print_summary_block(title: str, summary: Dict[str, Any]) -> None:
    print(f"\n========== {title} ==========")
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

    print("\nTop repeated qids:")
    for qid, count in summary["top_repeated_qids"][:20]:
        print(f"  {qid}: {count}")
    print("=" * (12 + len(title)))


def print_comparison_summary(comparison: Dict[str, Any]) -> None:
    print("\n==========================================")
    print("CAPPED SAMPLING STRATEGY")
    print("==========================================")
    print(f"Name               : {comparison['strategy']['name']}")
    print(f"Max pairs per qid  : {comparison['strategy']['max_pairs_per_qid']}")
    print(f"Rule               : {comparison['strategy']['rule']}")
    print_summary_block("BEFORE CAPPING", comparison["before"])
    print_summary_block("AFTER CAPPING", comparison["after"])


def main():
    parser = argparse.ArgumentParser(description="Cap DPO preference pairs by qid.")
    parser.add_argument("--input", type=str, default="cleaned_pairs.jsonl",
                        help="Path to cleaned input JSONL")
    parser.add_argument("--output", type=str, default="cleaned_pairs_capped_top2.jsonl",
                        help="Path to capped output JSONL")
    parser.add_argument("--max_pairs_per_qid", type=int, default=2,
                        help="Maximum number of pairs to keep per qid")
    parser.add_argument("--summary", type=str, default=None,
                        help="Optional path to save comparison summary JSON")
    args = parser.parse_args()

    print(f"[INFO] Loading cleaned JSONL: {args.input}")
    original_records = load_jsonl(args.input)
    print(f"[INFO] Loaded {len(original_records)} cleaned records")

    capped_records = cap_dataset_by_qid(original_records, args.max_pairs_per_qid)

    print(f"[INFO] Saving capped JSONL to: {args.output}")
    save_jsonl(capped_records, args.output)

    comparison = build_comparison_summary(original_records, capped_records, args.max_pairs_per_qid)

    if args.summary is None:
        base, _ = os.path.splitext(args.output)
        summary_path = base + "_summary.json"
    else:
        summary_path = args.summary

    print(f"[INFO] Saving comparison summary to: {summary_path}")
    save_summary(comparison, summary_path)
    print_comparison_summary(comparison)


if __name__ == "__main__":
    main()
