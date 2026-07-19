#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
edprm_policy_sampling_rerank_dataset_v4.py

Purpose
-------
Use FULL retrieved PubMed docs as dataset input, then:
1. sample policy traces from TWO policy models
   - Llama-3.1-8B-Instruct-finetuned
   - Llama-3.1-8B-Instruct
2. score step-level and final-answer-level with reward model
3. rerank traces jointly
4. construct DPO preference pairs
5. save incrementally after EACH question


Input scope
-----------
This script uses FULL TRAIN docs only for:
- medqa
- medmcqa
- pubmedqa

This matches the training-data construction usage of the previous script.
"""

import gc
import os
import re
import json
import math
import time
import shutil
import random
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict, Counter
from contextlib import contextmanager

import torch
import transformers.modeling_utils as hf_modeling_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# =============================================================================
# User config
# =============================================================================

# ---------- Reward model ----------
MEDPRM_DIR = Path("llama-3.1-medprm-reward-v1.0_old").resolve()

# Only used if MEDPRM_DIR missing tokenizer files
BASE_TOKENIZER_DIR = Path("Meta-Llama-3.1-8B-Instruct").resolve()
BASE_TOKENIZER_SP_DIR = Path("Meta-Llama-3.1-8B-Instruct/original").resolve()

# ---------- Policy models ----------
# NOTE:
# If your actual fine-tuned 8B directory has a different name, only change this line.
POLICY_FT_8B_MODEL_DIR = Path("Meta-Llama-3.1-8B-Instruct").resolve()
POLICY_8B_MODEL_DIR = Path("Meta-Llama-3.1-8B-Instruct").resolve()

# ---------- Input FULL docs ----------
FULL_DOCS_ROOT = Path("med_retrieval_project/retrieval_full_docs_run").resolve()
DOCS_DIR = FULL_DOCS_ROOT / "docs"

# FULL train docs only, same three datasets as before
INPUT_FILES = [
    DOCS_DIR / "medqa_train_docs.jsonl",
    DOCS_DIR / "medmcqa_train_docs.jsonl",
    DOCS_DIR / "pubmedqa_train_docs.jsonl",
]

# Optional debug cap. None means use all selected questions.
MAX_QUESTIONS = None

# Skip questions with zero retrieved docs
SKIP_IF_NO_EVIDENCE = True

# Max docs used in prompt and RM scoring
MAX_DOCS_PER_QUESTION = 6

# To keep prompts manageable when using full abstracts
MAX_ABSTRACT_CHARS_PER_DOC = 1800
MAX_TITLE_CHARS_PER_DOC = 300

# ---------- Runtime ----------
RM_DTYPE = torch.bfloat16
POLICY_DTYPE = torch.bfloat16

VISIBLE_GPU_INDICES = [0,1,2,3]
GPU_MEMORY_LIMIT_GB = 20
GPU_TOTAL_MEMORY_GB_ASSUMED = 24
GPU_MEMORY_FRACTION = GPU_MEMORY_LIMIT_GB / GPU_TOTAL_MEMORY_GB_ASSUMED

RM_DEVICE = "cuda:0"

# Keep one model per device for stability.
# Extra GPUs remain available in the 6-card machine.
POLICY_FT_8B_DEVICE = "cuda:1"
POLICY_FT_8B_DEVICE_MAP = {"": POLICY_FT_8B_DEVICE}

POLICY_8B_DEVICE = "cuda:2"
POLICY_8B_DEVICE_MAP = {"": POLICY_8B_DEVICE}

SEED = 42

# ---------- Sampling ----------
TARGET_VALID_SAMPLES_PER_QUESTION = 8
TARGET_VALID_SAMPLES_PER_MODEL = TARGET_VALID_SAMPLES_PER_QUESTION // 2  # 4 + 4
MAX_ATTEMPTS_PER_QUESTION = 48
MAX_ATTEMPTS_PER_MODEL = MAX_ATTEMPTS_PER_QUESTION // 2  # 24 + 24
MAX_NEW_TOKENS = 384

DO_SAMPLE = True
TEMPERATURE = 1.2
TOP_P = 0.98
TOP_K = 100
REPETITION_PENALTY = 1.03

# ---------- Valid trace constraints ----------
MIN_SCORED_STEPS = 3
MAX_SCORED_STEPS = 8

# ---------- Step score aggregation ----------
PREFIX_HIGH_THRESHOLD = 0.5

# Make final answer dominate more than before
TRACE_ALPHA_MIN = 0.05
TRACE_BETA_MEAN = 0.20
TRACE_GAMMA_PREFIX = 0.05
TRACE_DELTA_FINAL = 0.70

# ---------- Pair construction ----------
REQUIRE_CHOSEN_MATCH_GOLD = True
PREFER_REJECTED_NOT_GOLD = True

MIN_FINAL_ANSWER_SCORE_FOR_CHOSEN = 0.55
MAX_FINAL_ANSWER_SCORE_FOR_REJECTED = 0.45

PAIR_MARGIN_ABS = 0.03
PAIR_MARGIN_FRAC_OF_GAP = 0.25
TOP_K_CHOSEN = 4
BOTTOM_K_REJECTED = 4
MAX_PAIRS_PER_QUESTION = 12

# ---------- Prompt mode ----------
RM_STEP_PROMPT_VARIANT = "v4_lenient_supported_or_reasonable"
RM_FINAL_PROMPT_VARIANT = "v_final_answer_best_supported"

POLICY_INCLUDE_DOCS = True

# ---------- Output ----------
OUT_DIR = Path("").resolve()#modify to your output address
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_QUESTION_DIR = OUT_DIR / "per_question_results"
PER_QUESTION_DIR.mkdir(parents=True, exist_ok=True)

RAW_DPO_JSONL = OUT_DIR / "dpo_pairs_raw.jsonl"
DEDUP_DPO_JSONL = OUT_DIR / "dpo_pairs_dedup.jsonl"
PROGRESS_JSONL = OUT_DIR / "progress_summary.jsonl"
SKIP_JSONL = OUT_DIR / "skipped_questions.jsonl"
FINAL_REPORT_JSON = OUT_DIR / "final_report.json"


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# GPU helpers
# =============================================================================
def configure_gpu_memory_limits() -> None:

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available. Skipping per-GPU memory fraction setup.")
        return

    n = torch.cuda.device_count()
    print(f"[INFO] CUDA device count detected: {n}")
    print(f"[INFO] Target visible GPU indices: {VISIBLE_GPU_INDICES}")
    print(
        f"[INFO] Setting per-process memory fraction to {GPU_MEMORY_FRACTION:.6f} "
        f"(~{GPU_MEMORY_LIMIT_GB}GB / {GPU_TOTAL_MEMORY_GB_ASSUMED}GB)"
    )

    for idx in VISIBLE_GPU_INDICES:
        if idx >= n:
            print(f"[WARN] cuda:{idx} not present on this machine. Skipping.")
            continue
        try:
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=idx)
            print(
                f"[INFO] Applied memory fraction to cuda:{idx}: "
                f"{GPU_MEMORY_FRACTION:.6f}"
            )
        except Exception as e:
            print(f"[WARN] Failed to set memory fraction for cuda:{idx}: {e}")


# =============================================================================
# Small helper: temporarily disable HF allocator warmup during model loading
# =============================================================================
@contextmanager
def disable_hf_caching_allocator_warmup():
    original = hf_modeling_utils.caching_allocator_warmup

    def _no_op(*args, **kwargs):
        return None

    hf_modeling_utils.caching_allocator_warmup = _no_op
    try:
        yield
    finally:
        hf_modeling_utils.caching_allocator_warmup = original


# =============================================================================
# Helper: ensure reward tokenizer files exist
# =============================================================================
def ensure_medprm_tokenizer_files(medprm_dir: Path) -> None:
    medprm_dir.mkdir(parents=True, exist_ok=True)
    if (medprm_dir / "tokenizer.json").exists() or (medprm_dir / "tokenizer.model").exists():
        return

    candidates = [
        ("tokenizer.json", BASE_TOKENIZER_DIR / "tokenizer.json"),
        ("tokenizer.model", BASE_TOKENIZER_SP_DIR / "tokenizer.model"),
        ("tokenizer.model", BASE_TOKENIZER_DIR / "tokenizer.model"),
    ]

    copied_any = False
    for dst_name, src in candidates:
        if src.exists():
            shutil.copy2(src, medprm_dir / dst_name)
            print(f"[INFO] Copied {src} -> {medprm_dir / dst_name}")
            copied_any = True
            break

    for fname in ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]:
        src = BASE_TOKENIZER_DIR / fname
        dst = medprm_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[INFO] Copied {src} -> {dst}")

    if not copied_any:
        raise FileNotFoundError("Could not make MEDPRM_DIR tokenizer self-contained.")


# =============================================================================
# IO helpers
# =============================================================================
def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_completed_qids(per_question_dir: Path) -> set:
    done = set()
    if not per_question_dir.exists():
        return done
    for fp in per_question_dir.glob("*.json"):
        done.add(fp.stem)
    return done


# =============================================================================
# Dataset / docs loading
# =============================================================================
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_mcq_question(question: str, options: Dict[str, str]) -> str:
    question = normalize_text(question)
    lines = [question, ""]
    for k in sorted(options.keys()):
        lines.append(f"({k}) {normalize_text(options[k])}")
    return "\n".join(lines).strip()


def shorten_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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
            jy = " ".join([x for x in [journal, year] if x])
            parts.append(f"Source: {jy}")
        if abstract:
            parts.append(f"Abstract: {abstract}")

        text = " | ".join(parts).strip()
        if text:
            docs.append(f"Document {i}: {text}")

    return docs


def selected_input_files() -> List[Path]:
    return [p for p in INPUT_FILES if p.exists()]


def load_questions_from_doc_files(paths: List[Path], max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    items = []

    for path in paths:
        rows = read_jsonl(path)
        print(f"[INFO] Reading full docs file: {path} | rows={len(rows)}")

        for r in rows:
            docs = docs_from_retrieved_record(r, max_docs=MAX_DOCS_PER_QUESTION)

            item = {
                "qid": r["qid"],
                "dataset": r.get("dataset", ""),
                "split": r.get("split", ""),
                "question": format_mcq_question(r.get("question", ""), r.get("options", {}) or {}),
                "docs": docs,
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

            if SKIP_IF_NO_EVIDENCE and len(docs) == 0:
                append_jsonl(SKIP_JSONL, {
                    "qid": item["qid"],
                    "dataset": item["dataset"],
                    "reason": "no_retrieved_docs",
                })
                continue

            items.append(item)

            if max_questions is not None and len(items) >= max_questions:
                return items

    return items


def build_dpo_prompt(question: str, docs: List[str]) -> str:
    doc_block = "\n".join(docs).strip()
    return (
        "You are a medical reasoning assistant.\n"
        "Use the provided evidence documents to solve the multiple-choice question step by step.\n\n"
        f"Documents:\n{doc_block}\n\n"
        f"Question:\n{question}\n\n"
        "Please answer in the format:\n"
        "Step 1: ...\n"
        "Step 2: ...\n"
        "Step 3: ...\n"
        "Final Answer: (X)"
    )


# =============================================================================
# RM +/- token scoring utilities
# =============================================================================
def logsumexp(vals: List[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def get_plus_minus_candidate_ids(tokenizer) -> Tuple[List[int], List[int], Dict[str, Any]]:
    plus_a = tokenizer("+", add_special_tokens=False)["input_ids"]
    plus_b = tokenizer(" +", add_special_tokens=False)["input_ids"]
    minus_a = tokenizer("-", add_special_tokens=False)["input_ids"]
    minus_b = tokenizer(" -", add_special_tokens=False)["input_ids"]

    plus_ids = [ids[0] for ids in [plus_a, plus_b] if len(ids) == 1]
    minus_ids = [ids[0] for ids in [minus_a, minus_b] if len(ids) == 1]

    dbg = {
        "plus_tokenization": [plus_a, plus_b],
        "minus_tokenization": [minus_a, minus_b],
        "plus_single_token_ids": plus_ids,
        "minus_single_token_ids": minus_ids,
    }

    if not plus_ids or not minus_ids:
        raise RuntimeError(f"Could not get single-token +/- candidates: {dbg}")
    return plus_ids, minus_ids, dbg


def score_next_token_plus_minus(
    model,
    tokenizer,
    prompt_text: str,
    plus_ids: List[int],
    minus_ids: List[int],
) -> Dict[str, Any]:
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    last_logits = logits[0, -1, :]
    plus_logits = [float(last_logits[i].item()) for i in plus_ids]
    minus_logits = [float(last_logits[i].item()) for i in minus_ids]

    pl = logsumexp(plus_logits)
    ml = logsumexp(minus_logits)

    pair = torch.tensor([pl, ml], dtype=torch.float32, device=last_logits.device)
    probs = torch.softmax(pair, dim=0)
    p_plus = float(probs[0].item())
    p_minus = float(probs[1].item())

    return {
        "p_plus": p_plus,
        "p_minus": p_minus,
        "margin": p_plus - p_minus,
    }


# =============================================================================
# Prompt builders
# =============================================================================
def build_step_prompt_variant(
    tokenizer,
    variant_name: str,
    docs: List[str],
    question: str,
    step_text: str,
    step_idx_1based: int = 1,
) -> str:
    doc_block = "\n\n".join(docs).strip()

    if variant_name == "v4_lenient_supported_or_reasonable":
        system = (
            "You are a medical step verifier.\n"
            "You will be given documents, a medical multiple-choice question, and ONE current reasoning step.\n"
            "Judge whether the CURRENT step should be accepted as reasonably correct.\n"
            "Accept '+' when the step is medically reasonable, consistent with the documents, and a plausible intermediate inference or summary.\n"
            "Minor paraphrases, short intermediate conclusions, and logically necessary bridging statements are allowed.\n"
            "Reject '-' only if the step clearly contradicts the documents, contains a wrong medical claim, or is seriously illogical.\n"
            "Do not be overly strict just because the wording is not explicitly copied from the documents.\n"
            "Output exactly one character: '+' or '-'."
        )
        user = (
            f"Documents:\n{doc_block}\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"CURRENT STEP #{step_idx_1based}:\n"
            f"```text\n{step_text.strip()}\n```\n\n"
            "Output one symbol (+ or -)."
        )
    else:
        raise ValueError(f"Unknown step prompt variant: {variant_name}")

    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True
    )


def build_final_answer_prompt_variant(
    tokenizer,
    variant_name: str,
    docs: List[str],
    question: str,
    reasoning_steps: List[str],
    final_answer: Optional[str],
) -> str:
    doc_block = "\n\n".join(docs).strip()
    step_block = "\n".join([f"- {s}" for s in reasoning_steps]).strip()

    if variant_name == "v_final_answer_best_supported":
        system = (
            "You are a medical answer verifier.\n"
            "You will be given documents, a multiple-choice medical question, a reasoning trace, and the chosen final answer.\n"
            "Judge whether the chosen final answer is the BEST supported option.\n"
            "Accept '+' only if the chosen option is well supported and more compatible with the documents than the main alternatives.\n"
            "Reject '-' if the chosen option is contradicted by the documents, or if another option is clearly better supported.\n"
            "Be stricter here than for individual reasoning steps.\n"
            "Output exactly one character: '+' or '-'."
        )
        user = (
            f"Documents:\n{doc_block}\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"Reasoning Trace:\n{step_block}\n\n"
            f"Chosen Final Answer: ({final_answer})\n\n"
            "Output one symbol (+ or -)."
        )
    else:
        raise ValueError(f"Unknown final prompt variant: {variant_name}")

    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True
    )


# =============================================================================
# Policy prompt + sampling
# =============================================================================
def build_policy_prompt(policy_tokenizer, question: str, docs: List[str]) -> str:
    if POLICY_INCLUDE_DOCS:
        doc_block = "\n\n".join(docs).strip()
        system = (
            "You are a medical reasoning assistant.\n"
            "Use the provided evidence documents to solve the multiple-choice clinical question step by step.\n"
            "If the documents conflict with your prior intuition, follow the documents.\n"
            "Your final answer must be the option best supported by the provided evidence.\n"
            "Use the EXACT format below:\n"
            "Step 1: ...\n"
            "Step 2: ...\n"
            "Step 3: ...\n"
            "Final Answer: (X)\n"
            "Rules:\n"
            "- Use 3 to 6 reasoning steps.\n"
            "- Each step must be a medically meaningful claim or inference.\n"
            "- Prefer evidence-grounded reasoning.\n"
            "- Do not write markdown bullets or extra headings.\n"
            "- Do not leave out the final answer line.\n"
            "- X must be one of A, B, C, D, E."
        )
        user = (
            f"Documents:\n{doc_block}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Please follow the required format exactly."
        )
    else:
        system = (
            "You are a medical reasoning assistant.\n"
            "Solve the multiple-choice clinical question step by step.\n"
            "Use the EXACT format below:\n"
            "Step 1: ...\n"
            "Step 2: ...\n"
            "Step 3: ...\n"
            "Final Answer: (X)\n"
            "Rules:\n"
            "- Use 3 to 6 reasoning steps.\n"
            "- Each step must be a medically meaningful claim or inference.\n"
            "- Do not leave out the final answer line.\n"
            "- X must be one of A, B, C, D, E."
        )
        user = f"Question:\n{question.strip()}\n\nPlease follow the required format exactly."

    return policy_tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True
    )


def sample_one_policy_trace(
    model,
    tokenizer,
    question: str,
    docs: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    prompt = build_policy_prompt(tokenizer, question, docs)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    gen_cfg = GenerationConfig(
        do_sample=DO_SAMPLE,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

    new_tokens = gen[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "prompt_text": prompt,
        "raw_generation": text.strip(),
    }


# =============================================================================
# Trace parsing
# =============================================================================
ANSWER_PATTERNS = [
    re.compile(r"Final\s*Answer\s*:\s*\(([A-E])\)", re.IGNORECASE),
    re.compile(r"Final\s*Answer\s*:\s*([A-E])", re.IGNORECASE),
    re.compile(r"Answer\s*:\s*\(([A-E])\)", re.IGNORECASE),
    re.compile(r"Answer\s*:\s*([A-E])", re.IGNORECASE),
    re.compile(r"\bthe answer is\s*\(?([A-E])\)?", re.IGNORECASE),
]


def extract_final_answer(text: str) -> Optional[str]:
    for pat in ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def normalize_trace_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_step_text(step: str) -> str:
    step = step.strip()
    step = re.sub(r"^\*+\s*", "", step)
    step = re.sub(r"^\#+\s*", "", step)
    step = re.sub(r"^Step\s*\d+\s*[:\-]\s*", "", step, flags=re.IGNORECASE)
    step = re.sub(r"^\d+[\.\)]\s*", "", step)
    step = re.sub(r"^[-*]\s*", "", step)
    return step.strip("*").strip()


def is_structural_or_low_value_step(step: str) -> bool:
    s = step.strip()
    if not s:
        return True
    s_lower = s.lower()

    if len(s) < 25:
        return True

    bad_prefixes = [
        "given information",
        "understanding the question",
        "evaluating each option",
        "let's evaluate",
        "let us evaluate",
        "let's consider",
        "let us consider",
        "analysis:",
        "reasoning:",
        "conclusion:",
        "final reasoning:",
        "we need to determine",
        "the question asks",
        "this question asks",
    ]
    if any(s_lower.startswith(x) for x in bad_prefixes):
        return True

    if re.fullmatch(r"\(?[A-E]\)?", s, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"[A-Za-z ]{1,30}:", s):
        return True

    return False


def split_long_sentence_into_substeps(sent: str) -> List[str]:
    sent = sent.strip()
    if len(sent) <= 220:
        return [sent]

    parts = re.split(
        r"\s*;\s*|\s+therefore,\s+|\s+however,\s+|\s+but\s+|\s+because\s+|\s+which means\s+",
        sent,
        flags=re.IGNORECASE
    )
    parts = [p.strip(" ,;:") for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts
    return [sent]


def split_reasoning_steps(text: str) -> List[str]:
    text = normalize_trace_text(text)
    text_wo_final = re.sub(r"Final\s*Answer\s*:.*", "", text, flags=re.IGNORECASE).strip()

    candidates = []

    if re.search(r"\bStep\s*\d+\s*[:\-]", text_wo_final, flags=re.IGNORECASE):
        parts = re.split(r"(?=\bStep\s*\d+\s*[:\-])", text_wo_final, flags=re.IGNORECASE)
        candidates.extend([p.strip() for p in parts if p.strip()])
    else:
        lines = [ln.strip() for ln in text_wo_final.split("\n") if ln.strip()]
        numbered_lines = []
        normal_lines = []

        for ln in lines:
            if re.match(r"^(\d+[\.\)]|[-*])\s+", ln):
                numbered_lines.append(re.sub(r"^(\d+[\.\)]|[-*])\s+", "", ln).strip())
            else:
                normal_lines.append(ln)

        if len(numbered_lines) >= 2:
            candidates.extend(numbered_lines)
        else:
            merged = " ".join(normal_lines) if normal_lines else text_wo_final
            sents = re.split(r"(?<=[\.\!\?])\s+", merged)
            sents = [s.strip() for s in sents if s.strip()]
            candidates.extend(sents)

    processed = []
    for c in candidates:
        c = clean_step_text(c)
        if not c:
            continue
        for ss in split_long_sentence_into_substeps(c):
            ss = clean_step_text(ss)
            if not ss:
                continue
            if is_structural_or_low_value_step(ss):
                continue
            processed.append(ss)

    deduped = []
    seen = set()
    for p in processed:
        k = p.lower().strip()
        if k not in seen:
            deduped.append(p)
            seen.add(k)

    if deduped:
        return deduped

    fallback = clean_step_text(text_wo_final)
    if fallback and not is_structural_or_low_value_step(fallback):
        return [fallback]
    return []


# =============================================================================
# RM scoring
# =============================================================================
def score_trace_steps(
    rm_model,
    rm_tokenizer,
    plus_ids: List[int],
    minus_ids: List[int],
    docs: List[str],
    question: str,
    steps: List[str],
    variant_name: str,
) -> Dict[str, Any]:
    step_results = []

    for i, step in enumerate(steps, start=1):
        prompt = build_step_prompt_variant(
            tokenizer=rm_tokenizer,
            variant_name=variant_name,
            docs=docs,
            question=question,
            step_text=step,
            step_idx_1based=i,
        )
        out = score_next_token_plus_minus(
            model=rm_model,
            tokenizer=rm_tokenizer,
            prompt_text=prompt,
            plus_ids=plus_ids,
            minus_ids=minus_ids,
        )

        step_results.append({
            "step_index": i,
            "step_text": step,
            **out,
        })

    if not step_results:
        return {
            "step_scores": [],
            "num_scored_steps": 0,
            "min_step_score": 0.0,
            "mean_step_score": 0.0,
            "prefix_high_len": 0,
            "prefix_high_ratio": 0.0,
        }

    scores = [x["p_plus"] for x in step_results]
    min_score = min(scores)
    mean_score = sum(scores) / len(scores)

    prefix_len = 0
    for s in scores:
        if s >= PREFIX_HIGH_THRESHOLD:
            prefix_len += 1
        else:
            break

    prefix_ratio = prefix_len / max(len(scores), 1)

    return {
        "step_scores": step_results,
        "num_scored_steps": len(step_results),
        "min_step_score": float(min_score),
        "mean_step_score": float(mean_score),
        "prefix_high_len": int(prefix_len),
        "prefix_high_ratio": float(prefix_ratio),
    }


def score_final_answer(
    rm_model,
    rm_tokenizer,
    plus_ids: List[int],
    minus_ids: List[int],
    docs: List[str],
    question: str,
    reasoning_steps: List[str],
    final_answer: Optional[str],
    variant_name: str,
) -> Dict[str, Any]:
    if final_answer is None:
        return {
            "final_answer_score": 0.0,
            "final_answer_margin": -1.0,
        }

    prompt = build_final_answer_prompt_variant(
        tokenizer=rm_tokenizer,
        variant_name=variant_name,
        docs=docs,
        question=question,
        reasoning_steps=reasoning_steps,
        final_answer=final_answer,
    )
    out = score_next_token_plus_minus(
        model=rm_model,
        tokenizer=rm_tokenizer,
        prompt_text=prompt,
        plus_ids=plus_ids,
        minus_ids=minus_ids,
    )

    return {
        "final_answer_score": float(out["p_plus"]),
        "final_answer_margin": float(out["margin"]),
    }


# =============================================================================
# Ranking + SC+RM
# =============================================================================
def rank_traces(traces: List[Dict[str, Any]], key_name: str = "trace_score") -> List[Dict[str, Any]]:
    return sorted(traces, key=lambda x: x.get(key_name, -1e9), reverse=True)


def compute_best_of_n(traces: List[Dict[str, Any]], key_name: str = "trace_score") -> Optional[Dict[str, Any]]:
    ranked = rank_traces(traces, key_name=key_name)
    return ranked[0] if ranked else None


def compute_sc_rm(traces: List[Dict[str, Any]], key_name: str = "trace_score") -> Dict[str, Any]:
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tr in traces:
        ans = tr.get("final_answer")
        if ans is None:
            ans = "UNKNOWN"
        groups[ans].append(tr)

    cluster_stats = []
    for ans, arr in groups.items():
        total_score = sum(float(x.get(key_name, 0.0)) for x in arr)
        mean_score = total_score / max(len(arr), 1)
        best_score = max(float(x.get(key_name, 0.0)) for x in arr)
        mean_final_answer_score = sum(float(x.get("final_answer_score", 0.0)) for x in arr) / max(len(arr), 1)
        cluster_stats.append({
            "answer": ans,
            "count": len(arr),
            "total_score": float(total_score),
            "mean_score": float(mean_score),
            "best_score": float(best_score),
            "mean_final_answer_score": float(mean_final_answer_score),
        })

    cluster_stats = sorted(
        cluster_stats,
        key=lambda x: (x["total_score"], x["mean_final_answer_score"]),
        reverse=True
    )
    best_cluster = cluster_stats[0] if cluster_stats else None

    return {
        "clusters": cluster_stats,
        "selected_answer_by_sc_rm": best_cluster["answer"] if best_cluster else None,
    }


# =============================================================================
# Pair construction
# =============================================================================
def build_dpo_pairs_for_question(
    qid: str,
    traces: List[Dict[str, Any]],
    gold_answer: str,
    score_key: str = "trace_score",
    top_k: int = TOP_K_CHOSEN,
    bottom_k: int = BOTTOM_K_REJECTED,
    max_pairs: int = MAX_PAIRS_PER_QUESTION,
) -> List[Dict[str, Any]]:
    valid_traces = [
        t for t in traces
        if t.get("final_answer") is not None
        and t.get("num_scored_steps", 0) >= MIN_SCORED_STEPS
    ]
    if len(valid_traces) < 2:
        return []

    scores = [float(t.get(score_key, 0.0)) for t in valid_traces]
    score_gap = max(scores) - min(scores) if scores else 0.0
    adaptive_margin = max(PAIR_MARGIN_ABS, PAIR_MARGIN_FRAC_OF_GAP * score_gap)

    chosen_pool = []
    rejected_pool = []

    for t in valid_traces:
        ans = t.get("final_answer")
        ans_score = float(t.get("final_answer_score", 0.0))

        if REQUIRE_CHOSEN_MATCH_GOLD:
            if ans == gold_answer and ans_score >= MIN_FINAL_ANSWER_SCORE_FOR_CHOSEN:
                chosen_pool.append(t)
        else:
            if ans_score >= MIN_FINAL_ANSWER_SCORE_FOR_CHOSEN:
                chosen_pool.append(t)

        if PREFER_REJECTED_NOT_GOLD:
            if ans != gold_answer:
                rejected_pool.append(t)
            elif ans_score <= MAX_FINAL_ANSWER_SCORE_FOR_REJECTED:
                rejected_pool.append(t)
        else:
            if ans_score <= MAX_FINAL_ANSWER_SCORE_FOR_REJECTED:
                rejected_pool.append(t)

    chosen_pool = rank_traces(chosen_pool, key_name=score_key)[:top_k]
    rejected_pool = sorted(rejected_pool, key=lambda x: x.get(score_key, 1e9))[:bottom_k]

    pairs = []
    seen = set()

    for same_answer_only in [True, False]:
        for ch in chosen_pool:
            for rj in rejected_pool:
                if ch["trace_id"] == rj["trace_id"]:
                    continue

                if same_answer_only and ch.get("final_answer") != rj.get("final_answer"):
                    continue
                if (not same_answer_only) and ch.get("final_answer") == rj.get("final_answer"):
                    continue

                diff = float(ch.get(score_key, 0.0)) - float(rj.get(score_key, 0.0))
                if diff < adaptive_margin:
                    continue

                final_diff = float(ch.get("final_answer_score", 0.0)) - float(rj.get("final_answer_score", 0.0))
                if final_diff < 0.05:
                    continue

                key = (ch["trace_id"], rj["trace_id"])
                if key in seen:
                    continue
                seen.add(key)

                pairs.append({
                    "qid": qid,
                    "chosen_trace_id": ch["trace_id"],
                    "rejected_trace_id": rj["trace_id"],
                    "chosen_score": float(ch.get(score_key, 0.0)),
                    "rejected_score": float(rj.get(score_key, 0.0)),
                    "score_margin": float(diff),
                    "chosen_final_answer_score": float(ch.get("final_answer_score", 0.0)),
                    "rejected_final_answer_score": float(rj.get("final_answer_score", 0.0)),
                    "chosen_answer": ch.get("final_answer"),
                    "rejected_answer": rj.get("final_answer"),
                    "pair_type": "same_answer" if same_answer_only else "cross_answer",
                    "chosen_text": ch.get("raw_generation", ""),
                    "rejected_text": rj.get("raw_generation", ""),
                })

                if len(pairs) >= max_pairs:
                    return pairs

    return pairs


# =============================================================================
# Diagnostics helpers
# =============================================================================
def normalize_for_signature(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_signature(text: str, keep_chars: int = 300) -> str:
    norm = normalize_for_signature(text)
    norm = norm[:keep_chars]
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def pair_signature(chosen_text: str, rejected_text: str) -> str:
    a = text_signature(chosen_text, keep_chars=250)
    b = text_signature(rejected_text, keep_chars=250)
    return f"{a}__{b}"


def compute_pair_diagnostics(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not pairs:
        return {
            "pair_type_distribution": {},
            "unique_chosen_count": 0,
            "unique_rejected_count": 0,
            "dedup_pair_count": 0,
            "dedup_pairs": [],
        }

    pair_type_distribution = dict(Counter([p["pair_type"] for p in pairs]))

    chosen_sigs = [text_signature(p["chosen_text"]) for p in pairs]
    rejected_sigs = [text_signature(p["rejected_text"]) for p in pairs]
    unique_chosen_count = len(set(chosen_sigs))
    unique_rejected_count = len(set(rejected_sigs))

    dedup_map = {}
    for p in pairs:
        sig = pair_signature(p["chosen_text"], p["rejected_text"])
        if sig not in dedup_map:
            dedup_map[sig] = p
        else:
            if p["score_margin"] > dedup_map[sig]["score_margin"]:
                dedup_map[sig] = p

    dedup_pairs = list(dedup_map.values())

    return {
        "pair_type_distribution": pair_type_distribution,
        "unique_chosen_count": unique_chosen_count,
        "unique_rejected_count": unique_rejected_count,
        "dedup_pair_count": len(dedup_pairs),
        "dedup_pairs": dedup_pairs,
    }


# =============================================================================
# Loading models
# =============================================================================
def load_reward_model():
    ensure_medprm_tokenizer_files(MEDPRM_DIR)
    print(f"[INFO] Loading reward model: {MEDPRM_DIR}")
    print(f"[INFO] Reward model device: {RM_DEVICE}")

    safetensors_files = list(MEDPRM_DIR.glob("*.safetensors"))
    bin_files = list(MEDPRM_DIR.glob("*.bin"))
    has_index_json = (MEDPRM_DIR / "model.safetensors.index.json").exists()
    has_adapter_cfg = (MEDPRM_DIR / "adapter_config.json").exists()

    has_full_model = (
        (len(safetensors_files) > 0 and has_index_json)
        or (MEDPRM_DIR / "model.safetensors").exists()
        or (MEDPRM_DIR / "pytorch_model.bin").exists()
    )

    has_adapter = (
        has_adapter_cfg
        or any("adapter_model" in p.name for p in safetensors_files + bin_files)
    )

    rm_device_map = {"": RM_DEVICE}

    if has_full_model and not has_adapter:
        print("[INFO] Detected full reward model.")
        model = AutoModelForCausalLM.from_pretrained(
            str(MEDPRM_DIR),
            torch_dtype=RM_DTYPE,
            device_map=rm_device_map,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MEDPRM_DIR), use_fast=True)

    elif has_adapter:
        print("[INFO] Detected adapter-style reward model.")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_TOKENIZER_DIR),
            torch_dtype=RM_DTYPE,
            device_map=rm_device_map,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, str(MEDPRM_DIR))
        tokenizer = AutoTokenizer.from_pretrained(str(BASE_TOKENIZER_DIR), use_fast=True)

    else:
        raise FileNotFoundError(
            f"No recognizable full-model or adapter weights found in {MEDPRM_DIR}. "
            f"Found files: {[p.name for p in MEDPRM_DIR.iterdir()]}"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    plus_ids, minus_ids, tok_dbg = get_plus_minus_candidate_ids(tokenizer)
    return model, tokenizer, plus_ids, minus_ids, tok_dbg


def load_policy_model_ft8b():
    print(f"[INFO] Loading fine-tuned 8B policy model: {POLICY_FT_8B_MODEL_DIR}")
    print(f"[INFO] Fine-tuned 8B policy device_map: {POLICY_FT_8B_DEVICE_MAP}")

    safetensors_files = list(POLICY_FT_8B_MODEL_DIR.glob("*.safetensors"))
    bin_files = list(POLICY_FT_8B_MODEL_DIR.glob("*.bin"))
    has_index_json = (POLICY_FT_8B_MODEL_DIR / "model.safetensors.index.json").exists()
    has_adapter_cfg = (POLICY_FT_8B_MODEL_DIR / "adapter_config.json").exists()

    has_full_model = (
        (len(safetensors_files) > 0 and has_index_json)
        or (POLICY_FT_8B_MODEL_DIR / "model.safetensors").exists()
        or (POLICY_FT_8B_MODEL_DIR / "pytorch_model.bin").exists()
    )

    has_adapter = (
        has_adapter_cfg
        or any("adapter_model" in p.name for p in safetensors_files + bin_files)
    )

    if has_full_model and not has_adapter:
        print("[INFO] Detected full fine-tuned 8B policy model.")
        with disable_hf_caching_allocator_warmup():
            model = AutoModelForCausalLM.from_pretrained(
                str(POLICY_FT_8B_MODEL_DIR),
                torch_dtype=POLICY_DTYPE,
                device_map=POLICY_FT_8B_DEVICE_MAP,
                low_cpu_mem_usage=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(str(POLICY_FT_8B_MODEL_DIR), use_fast=True)

    elif has_adapter:
        print("[INFO] Detected adapter-style fine-tuned 8B policy model.")
        from peft import PeftModel

        with disable_hf_caching_allocator_warmup():
            base_model = AutoModelForCausalLM.from_pretrained(
                str(BASE_TOKENIZER_DIR),
                torch_dtype=POLICY_DTYPE,
                device_map=POLICY_FT_8B_DEVICE_MAP,
                low_cpu_mem_usage=True,
            )
        model = PeftModel.from_pretrained(base_model, str(POLICY_FT_8B_MODEL_DIR))
        tokenizer = AutoTokenizer.from_pretrained(str(BASE_TOKENIZER_DIR), use_fast=True)

    else:
        raise FileNotFoundError(
            f"No recognizable full-model or adapter weights found in {POLICY_FT_8B_MODEL_DIR}. "
            f"Found files: {[p.name for p in POLICY_FT_8B_MODEL_DIR.iterdir()] if POLICY_FT_8B_MODEL_DIR.exists() else 'DIR_NOT_FOUND'}"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "hf_device_map"):
        print("[INFO] Fine-tuned 8B hf_device_map:")
        for k, v in model.hf_device_map.items():
            print(f"  {k} -> {v}")

    return model, tokenizer


def load_policy_model_8b():
    print(f"[INFO] Loading base 8B policy model: {POLICY_8B_MODEL_DIR}")
    print(f"[INFO] Base 8B policy device_map: {POLICY_8B_DEVICE_MAP}")

    with disable_hf_caching_allocator_warmup():
        model = AutoModelForCausalLM.from_pretrained(
            str(POLICY_8B_MODEL_DIR),
            torch_dtype=POLICY_DTYPE,
            device_map=POLICY_8B_DEVICE_MAP,
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(POLICY_8B_MODEL_DIR), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "hf_device_map"):
        print("[INFO] Base 8B hf_device_map:")
        for k, v in model.hf_device_map.items():
            print(f"  {k} -> {v}")

    return model, tokenizer


# =============================================================================
# Validity checks
# =============================================================================
def is_trace_valid(trace_obj: Dict[str, Any]) -> Tuple[bool, str]:
    if trace_obj.get("final_answer") is None:
        return False, "missing_final_answer"
    if trace_obj.get("num_scored_steps", 0) < MIN_SCORED_STEPS:
        return False, "too_few_scored_steps"
    if trace_obj.get("num_scored_steps", 0) > MAX_SCORED_STEPS:
        return False, "too_many_scored_steps"
    return True, "ok"


# =============================================================================
# Incremental exports
# =============================================================================
def export_raw_pairs_for_result(res: Dict[str, Any], out_path: Path) -> int:
    count = 0
    prompt = build_dpo_prompt(res["question"], res["docs"])

    for p in res["dpo_pairs"]:
        row = {
            "qid": p["qid"],
            "dataset": res.get("dataset", ""),
            "prompt": prompt,
            "chosen": p["chosen_text"],
            "rejected": p["rejected_text"],
            "gold_answer": res.get("gold_answer", ""),
            "gold_answer_text": res.get("gold_answer_text", ""),
            "chosen_score": p["chosen_score"],
            "rejected_score": p["rejected_score"],
            "score_margin": p["score_margin"],
            "chosen_final_answer_score": p["chosen_final_answer_score"],
            "rejected_final_answer_score": p["rejected_final_answer_score"],
            "chosen_answer": p["chosen_answer"],
            "rejected_answer": p["rejected_answer"],
            "pair_type": p["pair_type"],
        }
        append_jsonl(out_path, row)
        count += 1
    return count


def export_dedup_pairs_for_result(res: Dict[str, Any], out_path: Path) -> int:
    count = 0
    prompt = build_dpo_prompt(res["question"], res["docs"])

    for p in res["pair_diagnostics"]["dedup_pairs"]:
        row = {
            "qid": p["qid"],
            "dataset": res.get("dataset", ""),
            "prompt": prompt,
            "chosen": p["chosen_text"],
            "rejected": p["rejected_text"],
            "gold_answer": res.get("gold_answer", ""),
            "gold_answer_text": res.get("gold_answer_text", ""),
            "chosen_score": p["chosen_score"],
            "rejected_score": p["rejected_score"],
            "score_margin": p["score_margin"],
            "chosen_final_answer_score": p["chosen_final_answer_score"],
            "rejected_final_answer_score": p["rejected_final_answer_score"],
            "chosen_answer": p["chosen_answer"],
            "rejected_answer": p["rejected_answer"],
            "pair_type": p["pair_type"],
        }
        append_jsonl(out_path, row)
        count += 1
    return count


# =============================================================================
# Per-question sampling helpers
# =============================================================================
def make_trace_obj(
    *,
    qid: str,
    trace_seq_idx: int,
    model_tag: str,
    raw_text: str,
    final_answer: Optional[str],
    steps: List[str],
    step_pack: Dict[str, Any],
    final_pack: Dict[str, Any],
    gold_answer: str,
) -> Dict[str, Any]:
    base_trace_score = (
        TRACE_ALPHA_MIN * step_pack["min_step_score"]
        + TRACE_BETA_MEAN * step_pack["mean_step_score"]
        + TRACE_GAMMA_PREFIX * step_pack["prefix_high_ratio"]
        + TRACE_DELTA_FINAL * final_pack["final_answer_score"]
    )

    gold_match_bonus = 0.20 if final_answer == gold_answer else -0.50
    trace_score = base_trace_score + gold_match_bonus

    trace_obj = {
        "qid": qid,
        "trace_id": f"{qid}_{model_tag}_trace_{trace_seq_idx:03d}",
        "sample_index": trace_seq_idx,
        "policy_model_tag": model_tag,
        "raw_generation": raw_text,
        "final_answer": final_answer,
        "num_steps_raw": len(steps),
        "steps": steps,
        **step_pack,
        **final_pack,
        "base_trace_score": float(base_trace_score),
        "gold_match_bonus": float(gold_match_bonus),
        "trace_score": float(trace_score),
    }

    ok, reason = is_trace_valid(trace_obj)
    trace_obj["is_valid"] = ok
    trace_obj["invalid_reason"] = None if ok else reason
    return trace_obj


def collect_valid_traces_from_policy(
    *,
    item: Dict[str, Any],
    policy_model,
    policy_tokenizer,
    rm_model,
    rm_tokenizer,
    plus_ids: List[int],
    minus_ids: List[int],
    model_tag: str,
    target_valid: int,
    max_attempts: int,
    starting_trace_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    qid = item["qid"]
    docs = item["docs"]
    question = item["question"]
    gold_answer = item.get("gold_answer", "")

    valid_traces = []
    discarded_traces = []
    attempts_used = 0
    trace_seq_idx = starting_trace_idx

    while len(valid_traces) < target_valid and attempts_used < max_attempts:
        tr = sample_one_policy_trace(
            model=policy_model,
            tokenizer=policy_tokenizer,
            question=question,
            docs=docs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        )

        raw_text = tr["raw_generation"]
        final_answer = extract_final_answer(raw_text)
        steps = split_reasoning_steps(raw_text)

        step_pack = score_trace_steps(
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            plus_ids=plus_ids,
            minus_ids=minus_ids,
            docs=docs,
            question=question,
            steps=steps,
            variant_name=RM_STEP_PROMPT_VARIANT,
        )

        final_pack = score_final_answer(
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            plus_ids=plus_ids,
            minus_ids=minus_ids,
            docs=docs,
            question=question,
            reasoning_steps=steps,
            final_answer=final_answer,
            variant_name=RM_FINAL_PROMPT_VARIANT,
        )

        trace_obj = make_trace_obj(
            qid=qid,
            trace_seq_idx=trace_seq_idx,
            model_tag=model_tag,
            raw_text=raw_text,
            final_answer=final_answer,
            steps=steps,
            step_pack=step_pack,
            final_pack=final_pack,
            gold_answer=gold_answer,
        )

        ok = trace_obj["is_valid"]
        reason = trace_obj["invalid_reason"] or "ok"

        print(
            f"[{model_tag}][{attempts_used:03d}] "
            f"valid={str(ok):<5} "
            f"reason={reason:<22} "
            f"answer={str(final_answer):<8} "
            f"gold={gold_answer:<2} "
            f"steps_raw={trace_obj['num_steps_raw']:<2d} "
            f"steps_scored={trace_obj['num_scored_steps']:<2d} "
            f"min={trace_obj['min_step_score']:.4f} "
            f"mean={trace_obj['mean_step_score']:.4f} "
            f"prefix={trace_obj['prefix_high_len']:<2d} "
            f"ans_score={trace_obj['final_answer_score']:.4f} "
            f"trace={trace_obj['trace_score']:.4f}"
        )

        if ok:
            valid_traces.append(trace_obj)
        else:
            discarded_traces.append(trace_obj)

        attempts_used += 1
        trace_seq_idx += 1

    return valid_traces, discarded_traces, attempts_used, trace_seq_idx


def try_fill_remaining_with_alternating_policies(
    *,
    item: Dict[str, Any],
    policy_ft8b_model,
    policy_ft8b_tokenizer,
    policy8_model,
    policy8_tokenizer,
    rm_model,
    rm_tokenizer,
    plus_ids: List[int],
    minus_ids: List[int],
    existing_valid: List[Dict[str, Any]],
    existing_discarded: List[Dict[str, Any]],
    attempts_used_ft8b: int,
    attempts_used_8b: int,
    next_trace_seq_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int]:
    valid_traces = list(existing_valid)
    discarded_traces = list(existing_discarded)
    total_attempts = attempts_used_ft8b + attempts_used_8b
    turn = 0

    while len(valid_traces) < TARGET_VALID_SAMPLES_PER_QUESTION and total_attempts < MAX_ATTEMPTS_PER_QUESTION:
        can_use_ft8b = attempts_used_ft8b < MAX_ATTEMPTS_PER_MODEL
        can_use_8b = attempts_used_8b < MAX_ATTEMPTS_PER_MODEL

        if not can_use_ft8b and not can_use_8b:
            break

        if can_use_ft8b and can_use_8b:
            use_ft8b = (turn % 2 == 0)
        else:
            use_ft8b = can_use_ft8b

        if use_ft8b:
            new_valid, new_discarded, used, next_trace_seq_idx = collect_valid_traces_from_policy(
                item=item,
                policy_model=policy_ft8b_model,
                policy_tokenizer=policy_ft8b_tokenizer,
                rm_model=rm_model,
                rm_tokenizer=rm_tokenizer,
                plus_ids=plus_ids,
                minus_ids=minus_ids,
                model_tag="ft8b",
                target_valid=1,
                max_attempts=1,
                starting_trace_idx=next_trace_seq_idx,
            )
            attempts_used_ft8b += used
        else:
            new_valid, new_discarded, used, next_trace_seq_idx = collect_valid_traces_from_policy(
                item=item,
                policy_model=policy8_model,
                policy_tokenizer=policy8_tokenizer,
                rm_model=rm_model,
                rm_tokenizer=rm_tokenizer,
                plus_ids=plus_ids,
                minus_ids=minus_ids,
                model_tag="8b",
                target_valid=1,
                max_attempts=1,
                starting_trace_idx=next_trace_seq_idx,
            )
            attempts_used_8b += used

        valid_traces.extend(new_valid)
        discarded_traces.extend(new_discarded)
        total_attempts = attempts_used_ft8b + attempts_used_8b
        turn += 1

    return valid_traces, discarded_traces, attempts_used_ft8b, attempts_used_8b, next_trace_seq_idx


# =============================================================================
# Per-question pipeline
# =============================================================================
def process_one_question(
    item: Dict[str, Any],
    policy_ft8b_model,
    policy_ft8b_tokenizer,
    policy8_model,
    policy8_tokenizer,
    rm_model,
    rm_tokenizer,
    plus_ids: List[int],
    minus_ids: List[int],
) -> Dict[str, Any]:
    qid = item["qid"]
    docs = item["docs"]
    question = item["question"]
    gold_answer = item.get("gold_answer", "")
    gold_answer_text = item.get("gold_answer_text", "")

    print("\n" + "=" * 120)
    print(f"[QUESTION] {qid} | dataset={item.get('dataset')} | gold={gold_answer}")
    print("=" * 120)

    trace_seq_idx = 0

    valid_ft8b, discarded_ft8b, attempts_ft8b, trace_seq_idx = collect_valid_traces_from_policy(
        item=item,
        policy_model=policy_ft8b_model,
        policy_tokenizer=policy_ft8b_tokenizer,
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
        plus_ids=plus_ids,
        minus_ids=minus_ids,
        model_tag="ft8b",
        target_valid=TARGET_VALID_SAMPLES_PER_MODEL,
        max_attempts=MAX_ATTEMPTS_PER_MODEL,
        starting_trace_idx=trace_seq_idx,
    )

    valid_8b, discarded_8b, attempts_8b, trace_seq_idx = collect_valid_traces_from_policy(
        item=item,
        policy_model=policy8_model,
        policy_tokenizer=policy8_tokenizer,
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
        plus_ids=plus_ids,
        minus_ids=minus_ids,
        model_tag="8b",
        target_valid=TARGET_VALID_SAMPLES_PER_MODEL,
        max_attempts=MAX_ATTEMPTS_PER_MODEL,
        starting_trace_idx=trace_seq_idx,
    )

    valid_traces = valid_ft8b + valid_8b
    discarded_traces = discarded_ft8b + discarded_8b

    if len(valid_traces) < TARGET_VALID_SAMPLES_PER_QUESTION:
        print(
            f"[INFO] Need extra valid traces: have {len(valid_traces)} / {TARGET_VALID_SAMPLES_PER_QUESTION}. "
            f"Trying alternating backfill within remaining attempt budget."
        )
        valid_traces, discarded_traces, attempts_ft8b, attempts_8b, trace_seq_idx = try_fill_remaining_with_alternating_policies(
            item=item,
            policy_ft8b_model=policy_ft8b_model,
            policy_ft8b_tokenizer=policy_ft8b_tokenizer,
            policy8_model=policy8_model,
            policy8_tokenizer=policy8_tokenizer,
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            plus_ids=plus_ids,
            minus_ids=minus_ids,
            existing_valid=valid_traces,
            existing_discarded=discarded_traces,
            attempts_used_ft8b=attempts_ft8b,
            attempts_used_8b=attempts_8b,
            next_trace_seq_idx=trace_seq_idx,
        )

    all_attempts = attempts_ft8b + attempts_8b

    print(f"\n[SUMMARY] valid traces: {len(valid_traces)} / target {TARGET_VALID_SAMPLES_PER_QUESTION}")
    print(f"[SUMMARY] total attempts: {all_attempts}, discarded: {len(discarded_traces)}")
    print(f"[SUMMARY] attempts_ft8b={attempts_ft8b}, attempts_8b={attempts_8b}")
    print(
        f"[SUMMARY] valid_ft8b={len([t for t in valid_traces if t['policy_model_tag'] == 'ft8b'])}, "
        f"valid_8b={len([t for t in valid_traces if t['policy_model_tag'] == '8b'])}"
    )

    if not valid_traces:
        return {
            "qid": qid,
            "dataset": item.get("dataset", ""),
            "question": question,
            "docs": docs,
            "gold_answer": gold_answer,
            "gold_answer_text": gold_answer_text,
            "num_valid_traces": 0,
            "num_discarded_traces": len(discarded_traces),
            "all_attempts": all_attempts,
            "attempts_ft8b": attempts_ft8b,
            "attempts_8b": attempts_8b,
            "score_gap": 0.0,
            "answer_histogram": {},
            "ranked_traces": [],
            "best_of_n": None,
            "sc_rm": {"clusters": [], "selected_answer_by_sc_rm": None},
            "dpo_pairs": [],
            "pair_diagnostics": {},
            "discarded_traces": discarded_traces,
        }

    ranked_by_score = rank_traces(valid_traces, key_name="trace_score")
    best_of_n = compute_best_of_n(valid_traces, key_name="trace_score")
    sc_rm = compute_sc_rm(valid_traces, key_name="trace_score")

    dpo_pairs = build_dpo_pairs_for_question(
        qid=qid,
        traces=valid_traces,
        gold_answer=gold_answer,
        score_key="trace_score",
        top_k=TOP_K_CHOSEN,
        bottom_k=BOTTOM_K_REJECTED,
        max_pairs=MAX_PAIRS_PER_QUESTION,
    )

    pair_diagnostics = compute_pair_diagnostics(dpo_pairs)

    scores = [x["trace_score"] for x in valid_traces]
    score_gap = (max(scores) - min(scores)) if scores else 0.0

    answer_counter: DefaultDict[str, int] = defaultdict(int)
    model_counter: DefaultDict[str, int] = defaultdict(int)
    for tr in valid_traces:
        answer_counter[str(tr.get("final_answer"))] += 1
        model_counter[str(tr.get("policy_model_tag"))] += 1

    result = {
        "qid": qid,
        "dataset": item.get("dataset", ""),
        "question": question,
        "docs": docs,
        "gold_answer": gold_answer,
        "gold_answer_text": gold_answer_text,
        "num_valid_traces": len(valid_traces),
        "num_discarded_traces": len(discarded_traces),
        "all_attempts": all_attempts,
        "attempts_ft8b": attempts_ft8b,
        "attempts_8b": attempts_8b,
        "valid_traces_by_model": dict(model_counter),
        "score_gap": float(score_gap),
        "answer_histogram": dict(answer_counter),
        "ranked_traces": ranked_by_score,
        "best_of_n": best_of_n,
        "sc_rm": sc_rm,
        "dpo_pairs": dpo_pairs,
        "pair_diagnostics": pair_diagnostics,
        "discarded_traces": discarded_traces,
    }

    print("\n[TOP-5 VALID TRACE BY TRACE SCORE]")
    for rank_i, tr in enumerate(ranked_by_score[:5], start=1):
        print(
            f"  rank={rank_i:<2d} trace_id={tr['trace_id']} "
            f"model={tr.get('policy_model_tag')} "
            f"answer={tr.get('final_answer')} trace={tr['trace_score']:.4f} "
            f"ans_score={tr['final_answer_score']:.4f} "
            f"mean={tr['mean_step_score']:.4f} min={tr['min_step_score']:.4f}"
        )

    print(f"\n[Best-of-N] trace_id={best_of_n['trace_id'] if best_of_n else None} "
          f"answer={best_of_n['final_answer'] if best_of_n else None}")
    print(f"[SC+RM] selected answer={sc_rm['selected_answer_by_sc_rm']}")
    print(f"[DPO pairs] raw count={len(dpo_pairs)}")
    print(f"[Score gap] {score_gap:.4f}")

    pd = pair_diagnostics
    print("\n[PAIR DIAGNOSTICS]")
    print(f"  pair_type_distribution = {pd['pair_type_distribution']}")
    print(f"  unique_chosen_count    = {pd['unique_chosen_count']}")
    print(f"  unique_rejected_count  = {pd['unique_rejected_count']}")
    print(f"  dedup_pair_count       = {pd['dedup_pair_count']}")

    return result


# =============================================================================
# Main
# =============================================================================
def main():
    set_seed(SEED)

    print("[INFO] Suggested shell env before running:")
    print("  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5")
    print("  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    configure_gpu_memory_limits()

    input_files = selected_input_files()
    if not input_files:
        raise FileNotFoundError(f"No FULL docs files found under {DOCS_DIR}")

    questions = load_questions_from_doc_files(input_files, max_questions=MAX_QUESTIONS)
    print(f"[INFO] Loaded questions from FULL docs files: {len(questions)}")
    print(f"[INFO] Input docs dir = {DOCS_DIR}")

    completed_qids = load_completed_qids(PER_QUESTION_DIR)
    print(f"[INFO] Already completed qids: {len(completed_qids)}")

    rm_model, rm_tokenizer, plus_ids, minus_ids, tok_dbg = load_reward_model()
    policy_ft8b_model, policy_ft8b_tokenizer = load_policy_model_ft8b()
    policy8_model, policy8_tokenizer = load_policy_model_8b()

    summary_report: Dict[str, Any] = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "policy_ft8b_model_path": str(POLICY_FT_8B_MODEL_DIR),
            "policy_8b_model_path": str(POLICY_8B_MODEL_DIR),
            "reward_model_path": str(MEDPRM_DIR),
            "input_files": [str(p) for p in input_files],
        },
        "config": {
            "visible_gpu_indices": VISIBLE_GPU_INDICES,
            "gpu_memory_limit_gb": GPU_MEMORY_LIMIT_GB,
            "gpu_total_memory_gb_assumed": GPU_TOTAL_MEMORY_GB_ASSUMED,
            "gpu_memory_fraction": GPU_MEMORY_FRACTION,
            "target_valid_samples_per_question": TARGET_VALID_SAMPLES_PER_QUESTION,
            "target_valid_samples_per_model": TARGET_VALID_SAMPLES_PER_MODEL,
            "max_attempts_per_question": MAX_ATTEMPTS_PER_QUESTION,
            "max_attempts_per_model": MAX_ATTEMPTS_PER_MODEL,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "step_prompt_variant": RM_STEP_PROMPT_VARIANT,
            "final_prompt_variant": RM_FINAL_PROMPT_VARIANT,
            "prefix_high_threshold": PREFIX_HIGH_THRESHOLD,
            "trace_alpha_min": TRACE_ALPHA_MIN,
            "trace_beta_mean": TRACE_BETA_MEAN,
            "trace_gamma_prefix": TRACE_GAMMA_PREFIX,
            "trace_delta_final": TRACE_DELTA_FINAL,
            "min_scored_steps": MIN_SCORED_STEPS,
            "max_scored_steps": MAX_SCORED_STEPS,
            "pair_margin_abs": PAIR_MARGIN_ABS,
            "pair_margin_frac_of_gap": PAIR_MARGIN_FRAC_OF_GAP,
            "min_final_answer_score_for_chosen": MIN_FINAL_ANSWER_SCORE_FOR_CHOSEN,
            "max_final_answer_score_for_rejected": MAX_FINAL_ANSWER_SCORE_FOR_REJECTED,
            "require_chosen_match_gold": REQUIRE_CHOSEN_MATCH_GOLD,
            "prefer_rejected_not_gold": PREFER_REJECTED_NOT_GOLD,
            "policy_include_docs": POLICY_INCLUDE_DOCS,
            "rm_device": RM_DEVICE,
            "policy_ft8b_device_map": POLICY_FT_8B_DEVICE_MAP,
            "policy_8b_device_map": POLICY_8B_DEVICE_MAP,
            "max_docs_per_question": MAX_DOCS_PER_QUESTION,
            "max_abstract_chars_per_doc": MAX_ABSTRACT_CHARS_PER_DOC,
        },
        "tokenization_debug": tok_dbg,
        "total_loaded_questions": len(questions),
        "processed_questions": 0,
        "skipped_existing_questions": 0,
    }

    t0 = time.time()

    for item in questions:
        qid = item["qid"]

        if qid in completed_qids:
            print(f"[SKIP] already done: {qid}")
            summary_report["skipped_existing_questions"] += 1
            continue

        try:
            res = process_one_question(
                item=item,
                policy_ft8b_model=policy_ft8b_model,
                policy_ft8b_tokenizer=policy_ft8b_tokenizer,
                policy8_model=policy8_model,
                policy8_tokenizer=policy8_tokenizer,
                rm_model=rm_model,
                rm_tokenizer=rm_tokenizer,
                plus_ids=plus_ids,
                minus_ids=minus_ids,
            )

            per_q_path = PER_QUESTION_DIR / f"{qid}.json"
            write_json(per_q_path, res)

            raw_rows = export_raw_pairs_for_result(res, RAW_DPO_JSONL)
            dedup_rows = export_dedup_pairs_for_result(res, DEDUP_DPO_JSONL)

            append_jsonl(PROGRESS_JSONL, {
                "qid": qid,
                "dataset": res.get("dataset", ""),
                "gold_answer": res.get("gold_answer", ""),
                "num_valid_traces": res.get("num_valid_traces", 0),
                "num_discarded_traces": res.get("num_discarded_traces", 0),
                "all_attempts": res.get("all_attempts", 0),
                "attempts_ft8b": res.get("attempts_ft8b", 0),
                "attempts_8b": res.get("attempts_8b", 0),
                "valid_traces_by_model": res.get("valid_traces_by_model", {}),
                "score_gap": res.get("score_gap", 0.0),
                "raw_pair_count": len(res.get("dpo_pairs", [])),
                "dedup_pair_count": res.get("pair_diagnostics", {}).get("dedup_pair_count", 0),
                "raw_rows_appended": raw_rows,
                "dedup_rows_appended": dedup_rows,
            })

            summary_report["processed_questions"] += 1
            summary_report["elapsed_seconds"] = time.time() - t0
            write_json(FINAL_REPORT_JSON, summary_report)

            print(f"[SAVE] per-question result -> {per_q_path}")
            print(f"[SAVE] appended raw pairs -> {RAW_DPO_JSONL}")
            print(f"[SAVE] appended dedup pairs -> {DEDUP_DPO_JSONL}")
            print(f"[SAVE] appended progress -> {PROGRESS_JSONL}")

        except Exception as e:
            print(f"[ERROR] qid={qid} | error={e}")
            append_jsonl(SKIP_JSONL, {
                "qid": qid,
                "dataset": item.get("dataset", ""),
                "reason": "runtime_error",
                "error": str(e),
            })

    summary_report["elapsed_seconds"] = time.time() - t0
    write_json(FINAL_REPORT_JSON, summary_report)

    print("\n" + "=" * 120)
    print("[DONE]")
    print(f"[INFO] Final report: {FINAL_REPORT_JSON}")
    print(f"[INFO] Raw DPO pairs: {RAW_DPO_JSONL}")
    print(f"[INFO] Dedup DPO pairs: {DEDUP_DPO_JSONL}")
    print(f"[INFO] Per-question dir: {PER_QUESTION_DIR}")
    print("=" * 120)

    # cleanup hint, not strictly necessary
    try:
        del policy_ft8b_model, policy8_model, rm_model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
