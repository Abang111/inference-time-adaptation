# -*- coding: utf-8 -*-

import json
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================================================
# Config
# =========================================================
UNIFIED_DIR = Path("")
OUTPUT_DIR = Path("")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMAIL = "" #your email


# =========================================================
# IO
# =========================================================
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# Text helpers
# =========================================================
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen and x:
            seen.add(x)
            out.append(x)
    return out


GENERAL_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "with",
    "and", "or", "by", "is", "are", "was", "were", "be", "been", "being",
    "which", "what", "who", "whom", "whose", "this", "that", "these", "those",
    "most", "least", "likely", "following", "except", "not", "does", "do",
    "did", "can", "could", "would", "should", "may", "might", "it", "as",
    "into", "than", "then", "also", "often", "usually", "because", "about",
    "between", "under", "over", "all", "none", "each", "such", "their",
    "there", "here", "after", "before", "during", "within", "without",
    "using", "used", "use", "has", "have", "had", "having", "been",
    "being", "due", "secondary", "associated", "undergoing"
}

CASE_VIGNETTE_STOPWORDS = {
    "man", "woman", "male", "female", "boy", "girl", "child", "children",
    "patient", "patients", "year", "years", "old", "month", "months",
    "presents", "present", "presented", "comes", "come", "brought", "reports",
    "report", "states", "history", "known", "past", "medical", "family",
    "visit", "clinic", "office", "department", "emergency", "hospital",
    "underwent", "seen", "follow", "follow-up", "evaluation", "exam", "examination",
    "days", "weeks", "hours", "months", "today", "yesterday",
    "noted", "note", "shows", "showed", "show", "found", "findings",
    "returned", "initial", "stable", "monitoring", "close", "currently"
}

NOISY_EXAM_WORDS = {
    "aiims", "aipg", "neet", "dnb", "pgi", "year", "newer", "hypothesis",
    "except", "incorrect", "correct", "true", "false", "choose", "best",
    "following", "regarding", "according"
}

MEDICAL_KEEP_SHORT = {
    "hiv", "tb", "copd", "bph", "uti", "dna", "rna", "mrna", "rrna",
    "trna", "ecg", "eeg", "mri", "ct", "pet", "csf", "abg", "psa",
    "fsh", "lh", "tsh", "acth", "hcg", "pt", "aptt", "hbv", "hcv",
    "cmv", "ebv", "hsv", "ivf", "pcos", "osa", "gerd", "ibd", "ibs"
}


def tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return re.findall(r"[a-zA-Z0-9\-]+", text)


def is_age_or_number_token(tok: str) -> bool:
    if tok.isdigit():
        return True
    if re.fullmatch(r"\d+\-year\-old", tok):
        return True
    if re.fullmatch(r"\d+\-month\-old", tok):
        return True
    return False


def clean_tokens(
    tokens: List[str],
    extra_stopwords: Optional[set] = None,
    keep_short_set: Optional[set] = None,
) -> List[str]:
    extra_stopwords = extra_stopwords or set()
    keep_short_set = keep_short_set or set()

    out = []
    for tok in tokens:
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in keep_short_set:
            out.append(tok)
            continue
        if tok in GENERAL_STOPWORDS:
            continue
        if tok in extra_stopwords:
            continue
        if is_age_or_number_token(tok):
            continue
        if len(tok) <= 2:
            continue
        out.append(tok)
    return out


def option_texts_sorted(options: Dict[str, str]) -> List[str]:
    vals = []
    for k, v in sorted((options or {}).items()):
        v = normalize_text(v)
        if v:
            vals.append(v)
    return vals


def extract_medical_option_tokens(options: Dict[str, str], max_options: int = 4) -> List[str]:
    tokens = []
    for opt in option_texts_sorted(options)[:max_options]:
        toks = tokenize(opt)
        toks = clean_tokens(toks, extra_stopwords=CASE_VIGNETTE_STOPWORDS, keep_short_set=MEDICAL_KEEP_SHORT)
        tokens.extend(toks)
    return dedupe_keep_order(tokens)


def flatten_nested_strings(x: Any) -> List[str]:
    out = []

    def _walk(obj):
        if obj is None:
            return
        if isinstance(obj, str):
            s = normalize_text(obj)
            if s:
                out.append(s)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
            return
        s = normalize_text(obj)
        if s:
            out.append(s)

    _walk(x)
    return dedupe_keep_order(out)


def tokenize_phrase_list(phrases: List[str]) -> List[str]:
    toks = []
    for p in phrases:
        toks.extend(tokenize(p))
    return dedupe_keep_order(toks)


def looks_medical_token(tok: str) -> bool:
    if tok in MEDICAL_KEEP_SHORT:
        return True
    if len(tok) >= 5:
        return True
    if tok.endswith(("itis", "osis", "emia", "uria", "pathy", "oma", "genic", "tomy", "scopy")):
        return True
    return False


def filter_medical_content_tokens(tokens: List[str]) -> List[str]:
    toks = clean_tokens(
        tokens,
        extra_stopwords=CASE_VIGNETTE_STOPWORDS,
        keep_short_set=MEDICAL_KEEP_SHORT,
    )
    return [t for t in toks if looks_medical_token(t)]


# =========================================================
# MedQA query builder
# =========================================================
def medqa_option_concept_phrases(options: Dict[str, str]) -> List[str]:
    phrases = []
    for opt in option_texts_sorted(options):
        if normalize_text(opt):
            phrases.append(normalize_text(opt))
    return dedupe_keep_order(phrases)


def build_query_candidates_medqa(record: Dict[str, Any]) -> List[str]:
    question = normalize_text(record.get("question", ""))
    options = record.get("options", {}) or {}
    meta = record.get("meta", {}) or {}

    metamap_phrases = flatten_nested_strings(meta.get("metamap_phrases", None))
    meta_info_phrases = flatten_nested_strings(meta.get("meta_info", None))
    option_phrases = medqa_option_concept_phrases(options)

    metamap_tokens = filter_medical_content_tokens(tokenize_phrase_list(metamap_phrases))
    meta_info_tokens = filter_medical_content_tokens(tokenize_phrase_list(meta_info_phrases))
    q_tokens = filter_medical_content_tokens(tokenize(question))
    opt_tokens = filter_medical_content_tokens(tokenize_phrase_list(option_phrases))

    q1 = " ".join(dedupe_keep_order(metamap_tokens + opt_tokens)[:10]).strip()
    q2 = " ".join(dedupe_keep_order(meta_info_tokens + opt_tokens)[:10]).strip()
    q3 = " ".join(dedupe_keep_order(q_tokens + opt_tokens)[:10]).strip()
    q4 = " ".join(dedupe_keep_order(opt_tokens)[:8]).strip()
    q5 = " ".join(dedupe_keep_order(metamap_tokens)[:8]).strip()

    cands = [q1, q2, q3, q4, q5]
    cands = [c for c in cands if c]
    return dedupe_keep_order(cands)


# =========================================================
# MedMCQA query builder
# =========================================================
def build_query_medmcqa_primary(record: Dict[str, Any], max_terms: int = 12) -> str:
    question = normalize_text(record.get("question", ""))
    options = record.get("options", {}) or {}
    meta = record.get("meta", {}) or {}

    q_tokens = clean_tokens(
        tokenize(question),
        extra_stopwords=NOISY_EXAM_WORDS,
        keep_short_set=MEDICAL_KEEP_SHORT,
    )

    subject_tokens = clean_tokens(
        tokenize(normalize_text(meta.get("subject_name", ""))),
        extra_stopwords=NOISY_EXAM_WORDS,
        keep_short_set=MEDICAL_KEEP_SHORT,
    )

    topic_tokens = clean_tokens(
        tokenize(normalize_text(meta.get("topic_name", ""))),
        extra_stopwords=NOISY_EXAM_WORDS,
        keep_short_set=MEDICAL_KEEP_SHORT,
    )

    opt_tokens = extract_medical_option_tokens(options, max_options=3)

    tokens = dedupe_keep_order(q_tokens + topic_tokens + subject_tokens + opt_tokens)
    return " ".join(tokens[:max_terms]).strip()


def build_query_medmcqa_fallback(record: Dict[str, Any], max_terms: int = 8) -> str:
    options = record.get("options", {}) or {}
    meta = record.get("meta", {}) or {}

    opt_tokens = extract_medical_option_tokens(options, max_options=4)
    topic_tokens = clean_tokens(
        tokenize(normalize_text(meta.get("topic_name", ""))),
        extra_stopwords=NOISY_EXAM_WORDS,
        keep_short_set=MEDICAL_KEEP_SHORT,
    )
    question_tokens = clean_tokens(
        tokenize(normalize_text(record.get("question", ""))),
        extra_stopwords=NOISY_EXAM_WORDS,
        keep_short_set=MEDICAL_KEEP_SHORT,
    )

    tokens = dedupe_keep_order(topic_tokens + opt_tokens + question_tokens[:4])
    return " ".join(tokens[:max_terms]).strip()


# =========================================================
# PubMedQA query builder
# =========================================================
def extract_pubmedqa_context_text(context: Any) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return normalize_text(context)
    if isinstance(context, list):
        return normalize_text(" ".join([normalize_text(x) for x in context if normalize_text(x)]))
    if isinstance(context, dict):
        if "contexts" in context and isinstance(context["contexts"], list):
            return normalize_text(" ".join([normalize_text(x) for x in context["contexts"] if normalize_text(x)]))
        return normalize_text(json.dumps(context, ensure_ascii=False))
    return normalize_text(context)


def build_query_pubmedqa_primary(record: Dict[str, Any], max_q_terms: int = 8, max_c_terms: int = 10) -> str:
    question = normalize_text(record.get("question", ""))
    context = extract_pubmedqa_context_text(record.get("context", ""))

    q_tokens = clean_tokens(tokenize(question), keep_short_set=MEDICAL_KEEP_SHORT)
    c_tokens = clean_tokens(tokenize(context), keep_short_set=MEDICAL_KEEP_SHORT)

    tokens = dedupe_keep_order(q_tokens[:max_q_terms] + c_tokens[:max_c_terms])
    return " ".join(tokens).strip()


def build_query_pubmedqa_fallback(record: Dict[str, Any], max_terms: int = 10) -> str:
    question = normalize_text(record.get("question", ""))
    q_tokens = clean_tokens(tokenize(question), keep_short_set=MEDICAL_KEEP_SHORT)
    return " ".join(dedupe_keep_order(q_tokens)[:max_terms]).strip()


# =========================================================
# Dispatcher
# =========================================================
def build_queries_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
    dataset = normalize_text(record.get("dataset", "")).lower()

    if dataset == "medqa":
        candidates = build_query_candidates_medqa(record)
        primary = candidates[0] if candidates else ""
        fallback = candidates[1] if len(candidates) > 1 else primary
        return {
            "query": primary,
            "query_primary": primary,
            "query_fallback": fallback,
            "query_candidates": candidates,
        }

    if dataset == "medmcqa":
        primary = build_query_medmcqa_primary(record)
        fallback = build_query_medmcqa_fallback(record)
        candidates = dedupe_keep_order([primary, fallback])
        return {
            "query": primary,
            "query_primary": primary,
            "query_fallback": fallback,
            "query_candidates": [c for c in candidates if c],
        }

    if dataset == "pubmedqa":
        primary = build_query_pubmedqa_primary(record)
        fallback = build_query_pubmedqa_fallback(record)
        candidates = dedupe_keep_order([primary, fallback])
        return {
            "query": primary,
            "query_primary": primary,
            "query_fallback": fallback,
            "query_candidates": [c for c in candidates if c],
        }

    q = normalize_text(record.get("question", ""))
    toks = clean_tokens(tokenize(q), keep_short_set=MEDICAL_KEEP_SHORT)
    primary = " ".join(dedupe_keep_order(toks)[:12]).strip()
    return {
        "query": primary,
        "query_primary": primary,
        "query_fallback": primary,
        "query_candidates": [primary] if primary else [],
    }


# =========================================================
# Output record
# =========================================================
def build_query_record(record: Dict[str, Any], contact_email: str) -> Dict[str, Any]:
    qs = build_queries_for_record(record)

    return {
        "qid": record.get("qid", ""),
        "dataset": record.get("dataset", ""),
        "split": record.get("split", ""),
        "question": record.get("question", ""),
        "options": record.get("options", {}),
        "answer": record.get("answer", ""),
        "answer_text": record.get("answer_text", ""),
        "context": record.get("context", ""),
        "meta": record.get("meta", {}),
        "query": qs["query"],
        "query_primary": qs["query_primary"],
        "query_fallback": qs["query_fallback"],
        "query_candidates": qs["query_candidates"],
        "retrieval_meta": {
            "source": "pubmed",
            "contact_email": contact_email,
        },
    }


# =========================================================
# Sampling
# =========================================================
def sample_rows(rows: List[Dict[str, Any]], n: Optional[int], seed: int = 42) -> List[Dict[str, Any]]:
    if n is None or n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:n])
    return [rows[i] for i in chosen]


# =========================================================
# Main file processing
# =========================================================
def process_one_file(input_path: Path, output_path: Path, email: str, sample_n: Optional[int] = None) -> None:
    rows = read_jsonl(input_path)
    original_n = len(rows)

    if sample_n is not None:
        rows = sample_rows(rows, sample_n)

    out_rows = [build_query_record(r, email) for r in rows]
    write_jsonl(output_path, out_rows)

    print(f"[OK] {input_path.name:28s} -> {output_path.name:32s} | original={original_n:7d} | saved={len(out_rows):7d}")

    if out_rows:
        ex = out_rows[0]
        print(f"     sample qid            : {ex['qid']}")
        print(f"     sample primary query  : {ex['query_primary'][:180]}")
        print(f"     sample fallback query : {ex['query_fallback'][:180]}")
        print(f"     sample candidates     : {ex['query_candidates'][:3]}")


# =========================================================
# Main
# =========================================================
def main():
    email = DEFAULT_EMAIL

    # small prototype
    small_plan = {
        "medqa_train.jsonl": 200,
        "medqa_validation.jsonl": 100,
        "medqa_test.jsonl": 100,
        "medmcqa_train.jsonl": 200,
        "medmcqa_validation.jsonl": 100,
        "medmcqa_test.jsonl": 100,
        "pubmedqa_train.jsonl": 200,
    }

    full_files = [
        "medqa_train.jsonl",
        "medqa_validation.jsonl",
        "medqa_test.jsonl",
        "medmcqa_train.jsonl",
        "medmcqa_validation.jsonl",
        "medmcqa_test.jsonl",
        "pubmedqa_train.jsonl",
    ]

    print("\n========== Building SMALL retrieval inputs ==========")
    for fn, n in small_plan.items():
        input_path = UNIFIED_DIR / fn
        if not input_path.exists():
            print(f"[WARN] missing input: {input_path}")
            continue

        output_path = OUTPUT_DIR / fn.replace(".jsonl", "_queries_small.jsonl")
        process_one_file(input_path, output_path, email=email, sample_n=n)

    print("\n========== Building FULL retrieval inputs ==========")
    for fn in full_files:
        input_path = UNIFIED_DIR / fn
        if not input_path.exists():
            print(f"[WARN] missing input: {input_path}")
            continue

        output_path = OUTPUT_DIR / fn.replace(".jsonl", "_queries.jsonl")
        process_one_file(input_path, output_path, email=email, sample_n=None)

    print("\nDone.")


if __name__ == "__main__":
    main()
