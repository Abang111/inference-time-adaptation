"""
Step 4: DPO training and evaluation.

Input:  train_pairs.jsonl, val_pairs.jsonl
Output: dpo_outputs/final_model/

Usage:
  python train_validation.py --model sshleifer/tiny-gpt2
"""

import json
import re
import inspect
import argparse
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


# ========================= Utilities =========================

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


FINAL_ANSWER_PATTERNS = [
    re.compile(r"Final\s*Answer\s*:\s*\(([A-Z])\)", re.IGNORECASE),
    re.compile(r"Final\s*Answer\s*:\s*([A-Z])", re.IGNORECASE),
]


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).upper()
    return None


# ========================= Log-prob Evaluation =========================

def conditional_logprob(
    model, tokenizer, prompt: str, completion: str,
    device="cpu", max_length=None
) -> float:
    if max_length is None:
        max_length = getattr(model.config, "n_positions",
                             getattr(model.config, "max_position_embeddings", 1024))

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[-(max_length - 1):]

    max_prompt_len = max(1, max_length - len(completion_ids))
    prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_tensor).logits

    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    start = len(prompt_ids)
    for pos in range(start, len(input_ids)):
        total += log_probs[0, pos - 1, input_ids[pos]].item()
    return total


def evaluate_preference_accuracy(records, model, tokenizer, device="cpu"):
    correct = 0
    results = []

    for rec in records:
        lp_chosen = conditional_logprob(model, tokenizer, rec["prompt"], rec["chosen"], device=device)
        lp_rejected = conditional_logprob(model, tokenizer, rec["prompt"], rec["rejected"], device=device)
        delta = lp_chosen - lp_rejected
        is_correct = delta > 0

        results.append({
            "qid": rec.get("qid"),
            "pair_type": rec.get("pair_type"),
            "lp_chosen": lp_chosen,
            "lp_rejected": lp_rejected,
            "delta": delta,
            "correct": is_correct,
        })
        if is_correct:
            correct += 1

    acc = correct / len(records) if records else 0.0
    return acc, results


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description="DPO training and evaluation.")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2",
                        help="Base model name or path")
    parser.add_argument("--train_file", type=str, default="train_pairs.jsonl")
    parser.add_argument("--val_file", type=str, default="val_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, default="dpo_outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--use_cpu", action="store_true", default=True)
    args = parser.parse_args()

    model_name = args.model

    # ----- Load data -----
    data_files = {"train": args.train_file, "validation": args.val_file}
    raw_ds = load_dataset("json", data_files=data_files)
    print(raw_ds)
    print("Columns:", raw_ds["train"].column_names)

    # ----- Load model & tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")

    # ----- DPO Config -----
    supported_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    supported_params.discard("self")

    candidate_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "remove_unused_columns": False,
        "beta": args.beta,
        "report_to": "none",
        "eval_strategy": "epoch",
        "use_cpu": args.use_cpu,
        "bf16": False,
        "fp16": False,
        "do_train": True,
        "do_eval": True,
        "optim": "adamw_torch",
        "gradient_checkpointing": False,
        "max_length": args.max_length,
    }

    final_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported_params}
    print("Final DPOConfig kwargs:", final_kwargs)

    training_args = DPOConfig(**final_kwargs)

    # ----- Train -----
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=raw_ds["train"],
        eval_dataset=raw_ds["validation"],
        processing_class=tokenizer,
    )
    print("DPOTrainer initialized successfully.")

    train_result = trainer.train()
    print(train_result)

    save_dir = f"{args.output_dir}/final_model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to: {save_dir}")

    # ----- Evaluate preference accuracy -----
    val_records = load_jsonl(args.val_file)
    print(f"Number of validation pair records: {len(val_records)}")

    policy_tokenizer = AutoTokenizer.from_pretrained(save_dir)
    policy_model = AutoModelForCausalLM.from_pretrained(save_dir)
    ref_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ref_model_eval = AutoModelForCausalLM.from_pretrained(model_name)

    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token

    policy_model.eval()
    ref_model_eval.eval()

    device = torch.device("cpu")

    policy_pref_acc, _ = evaluate_preference_accuracy(
        val_records, policy_model, policy_tokenizer, device=device
    )
    ref_pref_acc, _ = evaluate_preference_accuracy(
        val_records, ref_model_eval, ref_tokenizer, device=device
    )

    print(f"Policy preference accuracy:  {policy_pref_acc:.4f}")
    print(f"Reference preference accuracy: {ref_pref_acc:.4f}")
    print(f"Improvement over reference: {policy_pref_acc - ref_pref_acc:+.4f}")


if __name__ == "__main__":
    main()
