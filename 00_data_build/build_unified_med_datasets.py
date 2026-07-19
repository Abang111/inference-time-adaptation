# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# =========================================================
# Paths
# =========================================================
MEDQA_DIR = Path("")
MEDMCQA_DIR = Path("")
PUBMEDQA_DIR = Path("")

OUTPUT_DIR = Path("")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Basic helpers
# =========================================================
def to_python(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas objects into pure Python types
    so they can be safely dumped to JSON.
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if np is not None and isinstance(obj, np.generic):
        return obj.item()

    if np is not None and isinstance(obj, np.ndarray):
        return [to_python(x) for x in obj.tolist()]

    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_python(x) for x in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    try:
        return obj.item()
    except Exception:
        return str(obj)


def normalize_text(x: Any) -> str:
    x = to_python(x)
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def idx_to_letter(i: int) -> str:
    return chr(ord("A") + i)


def safe_get(row: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def read_parquet_records(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean_row = to_python(row)
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")


# =========================================================
# Unified format
# =========================================================
def build_standard_record(
    *,
    qid: str,
    dataset: str,
    split: str,
    question: str,
    options: Dict[str, str],
    answer: str,
    context: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    options = to_python(options)
    options = {
        str(k): normalize_text(v)
        for k, v in options.items()
        if normalize_text(v) != ""
    }
    answer_text = options.get(answer, "") if answer else ""

    return {
        "qid": str(qid),
        "dataset": str(dataset),
        "split": str(split),
        "question": normalize_text(question),
        "options": options,
        "answer": str(answer) if answer else "",
        "answer_text": normalize_text(answer_text),
        "context": normalize_text(context),
        "meta": to_python(meta),
    }


# =========================================================
# MedQA
# Actual observed keys:
# ['answer', 'answer_idx', 'meta_info', 'metamap_phrases', 'options', 'question']
# =========================================================
def convert_medqa_row(row: Dict[str, Any], split: str, i: int, source_path: str) -> Optional[Dict[str, Any]]:
    question = safe_get(row, ["question"])
    if question is None:
        return None

    raw_options = row.get("options", None)
    raw_options = to_python(raw_options)
    options: Dict[str, str] = {}

    # Case 1: options is dict
    if isinstance(raw_options, dict):
        direct = {}
        numeric = {}
        for k, v in raw_options.items():
            kk = str(k).strip()
            vv = normalize_text(v)
            if vv == "":
                continue

            if kk in ["A", "B", "C", "D", "E", "F"]:
                direct[kk] = vv
            elif kk.isdigit():
                numeric[idx_to_letter(int(kk))] = vv

        if len(direct) >= 2:
            options = direct
        elif len(numeric) >= 2:
            options = numeric

        if len(options) < 2 and "text" in raw_options:
            text_list = to_python(raw_options.get("text"))
            label_list = to_python(raw_options.get("label"))
            if isinstance(text_list, list):
                tmp = {}
                for j, opt in enumerate(text_list):
                    if normalize_text(opt) == "":
                        continue
                    label = idx_to_letter(j)
                    if isinstance(label_list, list) and j < len(label_list):
                        lab = normalize_text(label_list[j])
                        if lab:
                            label = lab
                    tmp[label] = normalize_text(opt)
                if len(tmp) >= 2:
                    options = tmp

    # Case 2: options is list
    elif isinstance(raw_options, list):
        # Expected actual structure:
        # [{'key': 'A', 'value': 'Ampicillin'}, ...]
        if raw_options and isinstance(raw_options[0], dict) and "key" in raw_options[0] and "value" in raw_options[0]:
            for item in raw_options:
                k = normalize_text(item.get("key"))
                v = normalize_text(item.get("value"))
                if k and v:
                    options[k] = v
        else:
            for j, opt in enumerate(raw_options):
                if normalize_text(opt) != "":
                    options[idx_to_letter(j)] = normalize_text(opt)

    if len(options) < 2:
        return None

    # answer
    answer = ""
    raw_answer = to_python(row.get("answer", None))
    raw_answer_idx = to_python(row.get("answer_idx", None))

    # Priority 1: answer_idx
    if raw_answer_idx is not None:
        ans_idx = normalize_text(raw_answer_idx)

        # In your MedQA, answer_idx is already A/B/C/D
        if ans_idx in options:
            answer = ans_idx
        else:
            try:
                idx = int(ans_idx)
                letter = idx_to_letter(idx)
                if letter in options:
                    answer = letter
            except Exception:
                pass

    # Priority 2: answer is letter
    if not answer and raw_answer is not None:
        ans = normalize_text(raw_answer)
        if ans in options:
            answer = ans

    # Priority 3: answer is option text
    if not answer and raw_answer is not None:
        ans_text = normalize_text(raw_answer).lower()
        for k, v in options.items():
            if normalize_text(v).lower() == ans_text:
                answer = k
                break

    original_id = f"{split}_{i:08d}"

    return build_standard_record(
        qid=f"medqa_{split}_{i:08d}",
        dataset="medqa",
        split=split,
        question=question,
        options=options,
        answer=answer,
        context="",
        meta={
            "source_dataset": "medqa",
            "source_split": split,
            "original_id": original_id,
            "source_path": source_path,
            "meta_info": to_python(row.get("meta_info", None)),
            "metamap_phrases": to_python(row.get("metamap_phrases", None)),
        },
    )


def load_medqa() -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    split_to_file = {
        "train": MEDQA_DIR / "train-00000-of-00001.parquet",
        "validation": MEDQA_DIR / "validation-00000-of-00001.parquet",
        "test": MEDQA_DIR / "test-00000-of-00001.parquet",
    }

    for split, path in split_to_file.items():
        if not path.exists():
            print(f"[WARN] MedQA split file missing: {path}")
            continue

        raw_rows = read_parquet_records(path)
        rows = []
        for i, row in enumerate(raw_rows):
            rec = convert_medqa_row(row, split, i, str(path))
            if rec is not None:
                rows.append(rec)

        result[split] = rows
        print(f"[INFO] MedQA {split}: raw={len(raw_rows)} kept={len(rows)}")

        if raw_rows:
            print(f"[INFO] MedQA {split} first-row keys: {sorted(raw_rows[0].keys())}")
            print(f"[DEBUG] MedQA {split} first raw options type: {type(raw_rows[0].get('options'))}")
            print(f"[DEBUG] MedQA {split} first raw options: {to_python(raw_rows[0].get('options'))}")
            print(f"[DEBUG] MedQA {split} first raw answer: {to_python(raw_rows[0].get('answer'))}")
            print(f"[DEBUG] MedQA {split} first raw answer_idx: {to_python(raw_rows[0].get('answer_idx'))}")

    return result


# =========================================================
# MedMCQA
# =========================================================
def convert_medmcqa_row(row: Dict[str, Any], split: str, i: int, source_path: str) -> Optional[Dict[str, Any]]:
    question = safe_get(row, ["question", "query", "stem"])
    if question is None:
        return None

    options = {
        "A": normalize_text(safe_get(row, ["opa", "option_a", "A"], "")),
        "B": normalize_text(safe_get(row, ["opb", "option_b", "B"], "")),
        "C": normalize_text(safe_get(row, ["opc", "option_c", "C"], "")),
        "D": normalize_text(safe_get(row, ["opd", "option_d", "D"], "")),
    }
    options = {k: v for k, v in options.items() if v != ""}
    if len(options) < 2:
        return None

    raw_answer = to_python(safe_get(row, ["cop", "answer", "label", "target"]))
    answer = ""
    if raw_answer is not None:
        s = normalize_text(raw_answer)
        if s in ["1", "2", "3", "4"]:
            answer = idx_to_letter(int(s) - 1)
        elif s in options:
            answer = s

    original_id = normalize_text(safe_get(row, ["id"], f"{split}_{i:08d}"))

    return build_standard_record(
        qid=f"medmcqa_{split}_{i:08d}",
        dataset="medmcqa",
        split=split,
        question=question,
        options=options,
        answer=answer,
        context="",
        meta={
            "source_dataset": "medmcqa",
            "source_split": split,
            "original_id": original_id,
            "source_path": source_path,
            "subject_name": normalize_text(safe_get(row, ["subject_name"], "")),
            "topic_name": normalize_text(safe_get(row, ["topic_name"], "")),
        },
    )


def load_medmcqa() -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    split_to_file = {
        "train": MEDMCQA_DIR / "train-00000-of-00001.parquet",
        "validation": MEDMCQA_DIR / "validation-00000-of-00001.parquet",
        "test": MEDMCQA_DIR / "test-00000-of-00001.parquet",
    }

    for split, path in split_to_file.items():
        if not path.exists():
            print(f"[WARN] MedMCQA split file missing: {path}")
            continue

        raw_rows = read_parquet_records(path)
        rows = []
        for i, row in enumerate(raw_rows):
            rec = convert_medmcqa_row(row, split, i, str(path))
            if rec is not None:
                rows.append(rec)

        result[split] = rows
        print(f"[INFO] MedMCQA {split}: raw={len(raw_rows)} kept={len(rows)}")

        if raw_rows:
            print(f"[INFO] MedMCQA {split} first-row keys: {sorted(raw_rows[0].keys())}")

    return result


# =========================================================
# PubMedQA
# =========================================================
def convert_pubmedqa_row(row: Dict[str, Any], split: str, i: int, source_path: str) -> Optional[Dict[str, Any]]:
    question = safe_get(row, ["question", "query"])
    if question is None:
        return None

    options = {
        "A": "yes",
        "B": "no",
        "C": "maybe",
    }

    context = ""
    ctx = safe_get(row, ["context", "long_answer", "abstract", "content"])
    ctx = to_python(ctx)

    if ctx is not None:
        if isinstance(ctx, list):
            context = " ".join([normalize_text(x) for x in ctx if normalize_text(x) != ""])
        elif isinstance(ctx, dict):
            if "contexts" in ctx and isinstance(ctx["contexts"], list):
                context = " ".join([normalize_text(x) for x in ctx["contexts"] if normalize_text(x) != ""])
            else:
                context = json.dumps(to_python(ctx), ensure_ascii=False)
        else:
            context = normalize_text(ctx)

    raw_answer = to_python(safe_get(row, ["final_decision", "answer", "label"]))
    answer = ""
    if raw_answer is not None:
        s = normalize_text(raw_answer).lower()
        if s == "yes":
            answer = "A"
        elif s == "no":
            answer = "B"
        elif s in ["maybe", "unknown"]:
            answer = "C"

    original_id = to_python(safe_get(row, ["pubid", "id"], f"{split}_{i:08d}"))

    return build_standard_record(
        qid=f"pubmedqa_{split}_{i:08d}",
        dataset="pubmedqa",
        split=split,
        question=question,
        options=options,
        answer=answer,
        context=context,
        meta={
            "source_dataset": "pubmedqa",
            "source_split": split,
            "original_id": original_id,
            "source_path": source_path,
        },
    )


def load_pubmedqa() -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    train_path = PUBMEDQA_DIR / "train-00000-of-00001.parquet"

    if not train_path.exists():
        print(f"[WARN] PubMedQA train file missing: {train_path}")
        return result

    raw_rows = read_parquet_records(train_path)
    rows = []
    for i, row in enumerate(raw_rows):
        rec = convert_pubmedqa_row(row, "train", i, str(train_path))
        if rec is not None:
            rows.append(rec)

    result["train"] = rows
    print(f"[INFO] PubMedQA train: raw={len(raw_rows)} kept={len(rows)}")

    if raw_rows:
        print(f"[INFO] PubMedQA train first-row keys: {sorted(raw_rows[0].keys())}")
        sample_context = to_python(raw_rows[0].get("context", None))
        print(f"[DEBUG] PubMedQA first raw context type: {type(sample_context)}")

    return result


# =========================================================
# Save outputs
# =========================================================
def save_dataset_outputs(dataset_name: str, split_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    all_rows = []
    for split, rows in split_map.items():
        out_path = OUTPUT_DIR / f"{dataset_name}_{split}.jsonl"
        write_jsonl(out_path, rows)
        print(f"[OK] saved {len(rows):>7} rows -> {out_path}")
        all_rows.extend(rows)
    return all_rows


def main():
    merged_all = []

    print("\n========== Loading MedQA ==========")
    try:
        medqa = load_medqa()
        merged_all.extend(save_dataset_outputs("medqa", medqa))
    except Exception as e:
        print(f"[WARN] Failed to load MedQA: {e}")

    print("\n========== Loading MedMCQA ==========")
    try:
        medmcqa = load_medmcqa()
        merged_all.extend(save_dataset_outputs("medmcqa", medmcqa))
    except Exception as e:
        print(f"[WARN] Failed to load MedMCQA: {e}")

    print("\n========== Loading PubMedQA ==========")
    try:
        pubmedqa = load_pubmedqa()
        merged_all.extend(save_dataset_outputs("pubmedqa", pubmedqa))
    except Exception as e:
        print(f"[WARN] Failed to load PubMedQA: {e}")

    merged_path = OUTPUT_DIR / "all_unified.jsonl"
    write_jsonl(merged_path, merged_all)
    print(f"\n[OK] saved merged {len(merged_all)} rows -> {merged_path}")

    stats = {}
    for row in merged_all:
        key = f"{row['dataset']}/{row['split']}"
        stats[key] = stats.get(key, 0) + 1

    print("\n========== Stats ==========")
    for k in sorted(stats.keys()):
        print(f"{k:24s} : {stats[k]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
