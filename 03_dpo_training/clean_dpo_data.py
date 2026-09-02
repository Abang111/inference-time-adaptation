"""
Step 1: Clean and validate raw DPO preference pairs.

Input:  dpo_pairs_dedup.jsonl
Output: cleaned_pairs.jsonl, cleaned_pairs_summary.json

Usage:
  python clean_dpo_data.py --input dpo_pairs_dedup.jsonl --output cleaned_pairs.jsonl
  # or simply run without args to use default paths
"""

import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ========================= Utilities =========================

def safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


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


# ========================= Text Normalization =========================

FINAL_ANSWER_PATTERNS = [
    re.compile(r"Final\s*Answer\s*:\s*\(([A-Z])\)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Final\s*Answer\s*:\s*([A-Z])\s*\.?\s*$", re.IGNORECASE | re.MULTILINE),
]

STEP_PATTERN = re.compile(r"^\s*Step\s+\d+\s*:", re.IGNORECASE | re.MULTILINE)


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_spaces(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    text = normalize_newlines(text)
    tail = text[-1000:] if len(text) > 1000 else text

    for pattern in FINAL_ANSWER_PATTERNS:
        matches = list(pattern.finditer(tail))
        if matches:
            return matches[-1].group(1).upper()

    loose = re.search(r"Final\s*Answer\s*:\s*\(?\s*([A-Z])\s*\)?\s*\.?\s*$",
                      tail, re.IGNORECASE | re.MULTILINE)
    if loose:
        return loose.group(1).upper()
    return None


def normalize_final_answer_format(text: str) -> str:
    text = normalize_newlines(text)
    text = normalize_spaces(text)
    ans = extract_final_answer(text)
    if ans is None:
        return text

    text = re.sub(
        r"Final\s*Answer\s*:\s*\(?\s*[A-Z]\s*\)?\s*\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    ).rstrip()

    if text:
        text = f"{text}\n\nFinal Answer: ({ans})"
    else:
        text = f"Final Answer: ({ans})"
    return text


def normalize_response_text(text: str) -> str:
    text = safe_strip(text)
    text = normalize_newlines(text)
    text = normalize_spaces(text)
    text = normalize_final_answer_format(text)
    text = normalize_spaces(text)
    return text


def normalize_prompt_text(text: str) -> str:
    text = safe_strip(text)
    text = normalize_newlines(text)
    text = normalize_spaces(text)
    return text


# ========================= Validation =========================

VALID_PAIR_TYPES = {"same_answer", "cross_answer"}


def get_valid_option_set(dataset_name: str) -> Optional[set]:
    dataset_name = safe_strip(dataset_name).lower()
    if dataset_name == "medmcqa":
        return {"A", "B", "C", "D"}
    if dataset_name == "pubmedqa":
        return {"A", "B", "C"}
    return None


def has_step_structure(text: str) -> bool:
    return bool(STEP_PATTERN.search(text))


def safe_float(x: Any) -> Optional[float]:
    try:
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def length_ratio_ok(chosen: str, rejected: str,
                    min_chars: int = 20, max_ratio: float = 8.0) -> bool:
    lc, lr = len(chosen), len(rejected)
    if lc < min_chars or lr < min_chars:
        return False
    short_len = max(1, min(lc, lr))
    return max(lc, lr) / short_len <= max_ratio


def validate_and_clean_record(rec: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    reasons = []

    required_keys = ["qid", "prompt", "chosen", "rejected", "score_margin", "pair_type"]
    for key in required_keys:
        if key not in rec:
            reasons.append(f"missing_key:{key}")
    if reasons:
        return None, reasons

    qid = safe_strip(rec.get("qid"))
    dataset = safe_strip(rec.get("dataset"))
    prompt = safe_strip(rec.get("prompt"))
    chosen = safe_strip(rec.get("chosen"))
    rejected = safe_strip(rec.get("rejected"))
    pair_type = safe_strip(rec.get("pair_type"))

    if not qid:
        reasons.append("empty_qid")
    if not is_nonempty_string(prompt):
        reasons.append("empty_prompt")
    if not is_nonempty_string(chosen):
        reasons.append("empty_chosen")
    if not is_nonempty_string(rejected):
        reasons.append("empty_rejected")

    score_margin = safe_float(rec.get("score_margin"))
    if score_margin is None:
        reasons.append("invalid_score_margin")

    chosen_score = safe_float(rec.get("chosen_score"))
    rejected_score = safe_float(rec.get("rejected_score"))

    if pair_type not in VALID_PAIR_TYPES:
        reasons.append("invalid_pair_type")

    if reasons:
        return None, reasons

    # Normalize text
    prompt = normalize_prompt_text(prompt)
    chosen = normalize_response_text(chosen)
    rejected = normalize_response_text(rejected)

    # Parse answers
    parsed_chosen_answer = extract_final_answer(chosen)
    parsed_rejected_answer = extract_final_answer(rejected)

    if parsed_chosen_answer is None:
        reasons.append("cannot_parse_chosen_answer")
    if parsed_rejected_answer is None:
        reasons.append("cannot_parse_rejected_answer")
    if reasons:
        return None, reasons

    # Compare with recorded answers
    recorded_chosen_answer = safe_strip(rec.get("chosen_answer")).upper()
    recorded_rejected_answer = safe_strip(rec.get("rejected_answer")).upper()
    gold_answer = safe_strip(rec.get("gold_answer")).upper()

    if recorded_chosen_answer and parsed_chosen_answer != recorded_chosen_answer:
        reasons.append("chosen_answer_mismatch")
    if recorded_rejected_answer and parsed_rejected_answer != recorded_rejected_answer:
        reasons.append("rejected_answer_mismatch")

    # Dataset-specific validation
    valid_option_set = get_valid_option_set(dataset)
    if valid_option_set is not None:
        if parsed_chosen_answer not in valid_option_set:
            reasons.append("chosen_answer_invalid_for_dataset")
        if parsed_rejected_answer not in valid_option_set:
            reasons.append("rejected_answer_invalid_for_dataset")
        if gold_answer and gold_answer not in valid_option_set:
            reasons.append("gold_answer_invalid_for_dataset")

    # Preference sanity checks
    if score_margin is not None and score_margin <= 0:
        reasons.append("non_positive_margin")
    if chosen == rejected:
        reasons.append("chosen_equals_rejected")
    if not length_ratio_ok(chosen, rejected):
        reasons.append("abnormal_length_ratio_or_too_short")
    if not has_step_structure(chosen):
        reasons.append("chosen_missing_step_structure")
    if not has_step_structure(rejected):
        reasons.append("rejected_missing_step_structure")
    if chosen_score is not None and rejected_score is not None:
        if chosen_score <= rejected_score:
            reasons.append("chosen_score_not_greater_than_rejected_score")
    if pair_type == "same_answer" and parsed_chosen_answer != parsed_rejected_answer:
        reasons.append("pair_type_same_but_answers_differ")
    if pair_type == "cross_answer" and parsed_chosen_answer == parsed_rejected_answer:
        reasons.append("pair_type_cross_but_answers_same")

    if reasons:
        return None, reasons

    cleaned = {
        "qid": qid,
        "dataset": dataset,
        "pair_type": pair_type,
        "score_margin": score_margin,
        "gold_answer": gold_answer if gold_answer else None,
        "chosen_answer": parsed_chosen_answer,
        "rejected_answer": parsed_rejected_answer,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    return cleaned, []


# ========================= Dataset-level Processing =========================

def clean_dataset(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cleaned_records = []
    rejection_counter = Counter()
    dataset_counter = Counter()
    pair_type_counter = Counter()

    for rec in records:
        cleaned, reasons = validate_and_clean_record(rec)
        if cleaned is not None:
            cleaned_records.append(cleaned)
            dataset_counter[cleaned["dataset"]] += 1
            pair_type_counter[cleaned["pair_type"]] += 1
        else:
            for r in reasons:
                rejection_counter[r] += 1

    qid_counter = Counter(rec["qid"] for rec in cleaned_records)
    pairs_per_qid = list(qid_counter.values())

    if pairs_per_qid:
        avg_pairs_per_qid = sum(pairs_per_qid) / len(pairs_per_qid)
        max_pairs_per_qid = max(pairs_per_qid)
        min_pairs_per_qid = min(pairs_per_qid)
    else:
        avg_pairs_per_qid = 0.0
        max_pairs_per_qid = 0
        min_pairs_per_qid = 0

    summary = {
        "n_input_records": len(records),
        "n_cleaned_records": len(cleaned_records),
        "n_removed_records": len(records) - len(cleaned_records),
        "dataset_distribution": dict(dataset_counter),
        "pair_type_distribution": dict(pair_type_counter),
        "rejection_reasons": dict(rejection_counter),
        "n_unique_qids": len(qid_counter),
        "avg_pairs_per_qid": avg_pairs_per_qid,
        "min_pairs_per_qid": min_pairs_per_qid,
        "max_pairs_per_qid": max_pairs_per_qid,
        "top_repeated_qids": qid_counter.most_common(20),
    }
    return cleaned_records, summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n========== CLEANING SUMMARY ==========")
    print(f"Input records      : {summary['n_input_records']}")
    print(f"Cleaned records    : {summary['n_cleaned_records']}")
    print(f"Removed records    : {summary['n_removed_records']}")
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

    print("\nTop rejection reasons:")
    sorted_reasons = sorted(summary["rejection_reasons"].items(),
                            key=lambda x: x[1], reverse=True)
    for k, v in sorted_reasons[:20]:
        print(f"  {k}: {v}")

    print("\nTop repeated qids:")
    for qid, count in summary["top_repeated_qids"][:20]:
        print(f"  {qid}: {count}")
    print("======================================\n")


def save_summary(summary: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description="Clean JSONL preference pairs for DPO.")
    parser.add_argument("--input", type=str, default="dpo_pairs_dedup.jsonl",
                        help="Path to raw input JSONL")
    parser.add_argument("--output", type=str, default="cleaned_pairs.jsonl",
                        help="Path to cleaned output JSONL")
    parser.add_argument("--summary", type=str, default=None,
                        help="Optional path to save summary JSON")
    args = parser.parse_args()

    print(f"[INFO] Loading raw JSONL: {args.input}")
    raw_records = load_jsonl(args.input)
    print(f"[INFO] Loaded {len(raw_records)} raw records")

    cleaned_records, summary = clean_dataset(raw_records)

    print(f"[INFO] Saving cleaned JSONL to: {args.output}")
    save_jsonl(cleaned_records, args.output)

    if args.summary is None:
        base, _ = os.path.splitext(args.output)
        summary_path = base + "_summary.json"
    else:
        summary_path = args.summary

    print(f"[INFO] Saving summary JSON to: {summary_path}")
    save_summary(summary, summary_path)
    print_summary(summary)


if __name__ == "__main__":
    main()
