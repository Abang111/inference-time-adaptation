# DPO data preparation and LoRA training

This directory connects the deduplicated output of `02_dpo_pair_construction` to the DPO policy used by G2 and G6.

## Pipeline

1. `clean_dpo_data.py` validates the JSONL schema, normalizes answer formatting, and removes malformed or inconsistent preferences.
2. `cap_dpo_pairs.py` limits repeated preferences from the same question while retaining the strongest same-answer and cross-answer pairs where possible.
3. `split.py` performs a deterministic train/validation split by `qid`, preventing question leakage across splits.
4. `train_dpo_lora.py` trains and saves LoRA adapters with the paper/checkpoint settings.

The trainer expects JSONL rows with at least `prompt`, `chosen`, and `rejected` string fields. The preparation scripts also use `qid`, `dataset`, `pair_type`, and score metadata produced by the pair-construction stage.

## Paper/checkpoint defaults

| Setting | Value |
| --- | ---: |
| Base model | Meta-Llama-3.1-8B-Instruct |
| Precision | bf16 |
| Epochs | 3 |
| Learning rate | 5e-5 |
| DPO beta | 0.1 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Maximum sequence length | 2048 |
| Optimizer | AdamW (`adamw_torch`) |
| Warmup ratio | 0.1 |
| Scheduler | cosine |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Seed | 42 |

Example:

```bash
accelerate launch train_dpo_lora.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --train_file /path/to/train_pairs.jsonl \
  --val_file /path/to/val_pairs.jsonl \
  --output_dir /path/to/dpo_outputs
```

`legacy_train_validation.py` is the untouched tiny-GPT2/CPU sanity-check script found in the archived project. It is retained for provenance and should not be used as the claimed 8B paper configuration.

