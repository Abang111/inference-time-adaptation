#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_medqa_doc_supported_subset.py

Purpose
-------



Behavior
--------
1. Read medqa_train_docs.jsonl
2. Keep only samples with retrieved docs (same spirit as previous code)
3. Save:
   - all doc-supported samples
   - a fixed random subset (default 1000)
   - qid list for the subset
4. Write outputs into the CURRENT working directory

Outputs
-------
./medqa_train_docs_with_evidence_all.jsonl
./medqa_train_docs_with_evidence_subset_1000.jsonl
./medqa_train_docs_with_evidence_subset_1000_qids.json
./medqa_train_docs_with_evidence_stats.json
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# User config
# =============================================================================

INPUT_JSONL = Path(
    #train_docs
).resolve()

OUT_DIR = Path(".").resolve()

SEED = 42
SUBSET_SIZE = 1000

KEEP_ONLY_WITH_DOCS = True
MAX_DOCS_PER_QUESTION = 6
MAX_TITLE_CHARS_PER_DOC = 300
MAX_ABSTRACT_CHARS_PER_DOC = 1800


# =============================================================================
# IO helpers
# =============================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# Normalization helpers
# =============================================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def shorten_text(text: str, max_chars: Optional[int]) -> str:
    text = normalize_text(text)
    if not max_chars or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_mcq_question(question: str, options: Dict[str, str]) -> str:
    question = normalize_text(question)
    lines = [question, ""]
    for k in sorted(options.keys()):
        lines.append(f"({k}) {normalize_text(options[k])}")
    return "\n".join(lines).strip()


def docs_from_retrieved_record(rec: Dict[str, Any], max_docs: int = 6) -> List[str]:
    docs = []
    retrieved_docs = rec.get("retrieved_docs", []) or []

    for i, d in enumerate(retrieved_docs[:max_docs], start=1):
        pmid = normalize_text(d.get("pmid", ""))
        title = shorten_text(d.get("title", ""), MAX_TITLE_CHARS_PER_DOC)
        abstract = shorten_text(d.get("abstract", ""), MAX_ABSTRACT_CHARS_PER_DOC)
        journal = normalize_text(d.get("journal", ""))
        year = normalize_text(d.get("year", ""))

        parts = []
        if pmid:
            parts.append(f"PMID {pmid}")
        if title:
            parts.append(f"Title: {title}")
        if journal or year:
            src = " ".join([x for x in [journal, year] if x])
            parts.append(f"Source: {src}")
        if abstract:
            parts.append(f"Abstract: {abstract}")

        text = " | ".join(parts).strip()
        if text:
            docs.append(f"Document {i}: {text}")

    return docs


# =============================================================================
# Core extraction
# =============================================================================

def convert_row(r: Dict[str, Any]) -> Dict[str, Any]:
    docs = docs_from_retrieved_record(r, max_docs=MAX_DOCS_PER_QUESTION)

    converted = {
        "qid": r.get("qid", ""),
        "dataset": r.get("dataset", "medqa"),
        "split": r.get("split", "train"),
        "question": format_mcq_question(r.get("question", ""), r.get("options", {}) or {}),
        "docs": docs,
        "num_docs": len(docs),
        "gold_answer": normalize_text(r.get("answer", "")),
        "gold_answer_text": normalize_text(r.get("answer_text", "")),
        "raw_record": {
            "question": r.get("question", ""),
            "options": r.get("options", {}),
            "query": r.get("query", ""),
            "query_primary": r.get("query_primary", ""),
            "query_fallback": r.get("query_fallback", ""),
            "pmids": r.get("pmids", []),
        },
    }
    return converted


def main():
    random.seed(SEED)

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    print(f"[INFO] Reading: {INPUT_JSONL}")
    raw_rows = read_jsonl(INPUT_JSONL)
    print(f"[INFO] Raw rows loaded: {len(raw_rows)}")

    converted_rows = []
    skipped_no_docs = 0

    for r in raw_rows:
        row = convert_row(r)

        if KEEP_ONLY_WITH_DOCS and len(row["docs"]) == 0:
            skipped_no_docs += 1
            continue

        converted_rows.append(row)

    print(f"[INFO] Rows kept after doc filter: {len(converted_rows)}")
    print(f"[INFO] Rows skipped for no docs:    {skipped_no_docs}")

    # sort by qid first to make sampling deterministic across environments
    converted_rows = sorted(converted_rows, key=lambda x: str(x["qid"]))

    all_out = OUT_DIR / "medqa_train_docs_with_evidence_all.jsonl"
    write_jsonl(all_out, converted_rows)
    print(f"[SAVE] All doc-supported rows -> {all_out}")

    # subset
    if SUBSET_SIZE is None or SUBSET_SIZE <= 0 or SUBSET_SIZE >= len(converted_rows):
        subset_rows = list(converted_rows)
    else:
        subset_rows = random.sample(converted_rows, SUBSET_SIZE)
        subset_rows = sorted(subset_rows, key=lambda x: str(x["qid"]))

    subset_name = f"medqa_train_docs_with_evidence_subset_{len(subset_rows)}"
    subset_out = OUT_DIR / f"{subset_name}.jsonl"
    subset_qids_out = OUT_DIR / f"{subset_name}_qids.json"
    stats_out = OUT_DIR / "medqa_train_docs_with_evidence_stats.json"

    write_jsonl(subset_out, subset_rows)
    write_json(subset_qids_out, {
        "dataset": "medqa",
        "source_file": str(INPUT_JSONL),
        "keep_only_with_docs": KEEP_ONLY_WITH_DOCS,
        "seed": SEED,
        "subset_size": len(subset_rows),
        "qids": [x["qid"] for x in subset_rows],
    })

    stats = {
        "source_file": str(INPUT_JSONL),
        "raw_rows": len(raw_rows),
        "kept_rows_after_doc_filter": len(converted_rows),
        "skipped_no_docs": skipped_no_docs,
        "subset_size": len(subset_rows),
        "seed": SEED,
        "max_docs_per_question": MAX_DOCS_PER_QUESTION,
        "max_title_chars_per_doc": MAX_TITLE_CHARS_PER_DOC,
        "max_abstract_chars_per_doc": MAX_ABSTRACT_CHARS_PER_DOC,
        "outputs": {
            "all_doc_supported_jsonl": str(all_out),
            "subset_jsonl": str(subset_out),
            "subset_qids_json": str(subset_qids_out),
        },
    }
    write_json(stats_out, stats)

    print(f"[SAVE] Fixed subset rows         -> {subset_out}")
    print(f"[SAVE] Fixed subset qids         -> {subset_qids_out}")
    print(f"[SAVE] Stats                    -> {stats_out}")

    print("\n[DONE]")
    print(f"All later G experiments can use: {subset_out}")


if __name__ == "__main__":
    main()
