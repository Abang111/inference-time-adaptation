#!/usr/bin/env python3
"""Train the paper's DPO policy as a LoRA adapter.

This entry point was reconstructed from the archived preprocessing/training
pipeline, the retained ``training_args.bin``/``trainer_state.json`` files, the
saved PEFT ``adapter_config.json``, and the paper's hyperparameter table. The
archived original entry point only covered a tiny-GPT2 CPU sanity check; it is
preserved separately as ``legacy_train_validation.py``.

Input JSONL rows must contain string-valued ``prompt``, ``chosen``, and
``rejected`` fields. Split by ``qid`` beforehand with ``split.py`` to avoid
question leakage between train and validation data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA DPO training for Meta-Llama-3.1-8B-Instruct."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model id or local base-model directory.",
    )
    parser.add_argument("--train_file", default="train_pairs.jsonl")
    parser.add_argument("--val_file", default="val_pairs.jsonl")
    parser.add_argument("--output_dir", default="dpo_outputs")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint directory, or omit to start a new run.",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bf16 (the paper run used bf16).",
    )
    return parser.parse_args()


def require_file(path_text: str, label: str) -> str:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return str(path)


def validate_dataset(dataset: Dataset, split_name: str) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in dataset.column_names]
    if missing:
        raise ValueError(
            f"{split_name} is missing required columns {missing}; "
            f"available columns are {dataset.column_names}."
        )
    if len(dataset) == 0:
        raise ValueError(f"{split_name} contains no records.")

    for row_index in range(min(100, len(dataset))):
        row = dataset[row_index]
        for name in REQUIRED_COLUMNS:
            value = row[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{split_name} row {row_index} has an empty/non-string {name!r}."
                )


def assert_disjoint_qids(train_dataset: Dataset, val_dataset: Dataset) -> None:
    if "qid" not in train_dataset.column_names or "qid" not in val_dataset.column_names:
        return
    train_qids = {str(value) for value in train_dataset["qid"]}
    val_qids = {str(value) for value in val_dataset["qid"]}
    overlap = train_qids & val_qids
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(f"qid leakage between train and validation: {preview}")


def write_run_config(path: Path, args: argparse.Namespace) -> None:
    payload = vars(args).copy()
    payload["lora_target_modules"] = list(LORA_TARGET_MODULES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_file = require_file(args.train_file, "Training JSONL")
    val_file = require_file(args.val_file, "Validation JSONL")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
    )
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]
    validate_dataset(train_dataset, "train")
    validate_dataset(val_dataset, "validation")
    assert_disjoint_qids(train_dataset, val_dataset)

    use_bf16 = not args.no_bf16
    if use_bf16 and not torch.cuda.is_available():
        raise RuntimeError("The paper configuration requires CUDA for bf16 training.")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bf16.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    # With a PEFT config and no explicit ref_model, DPOTrainer can compare
    # against the same base model with the trainable adapter disabled.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    write_run_config(output_dir / "run_config.json", args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final LoRA adapter and tokenizer to: {final_dir}")


if __name__ == "__main__":
    main()
