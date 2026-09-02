# Inference-Time Adaptation for Medical QA

This repository contains the data preparation, PubMed retrieval, reward-guided DPO preference construction, LoRA training, and G1-G6 evaluation code used for the paper *Characterizing Inference-Time Adaptation in Medical QA: From Retrieval Augmentation to Reward-Guided Alignment*.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `00_data_build` | Convert MedQA, MedMCQA, and PubMedQA into a unified JSONL schema. |
| `01_retrieval` | Build question-visible PubMed queries and retrieve evidence. |
| `02_dpo_pair_construction` | Sample reasoning traces, score them with MedPRM, and write raw and deduplicated DPO pairs. |
| `03_dpo_training` | Clean, cap, split, and train a LoRA DPO policy from the constructed pairs. |
| `04_paper_experiments` | Run the controlled G1-G6 evaluation configurations and compute accuracy/calibration metrics. |

## End-to-end DPO workflow

First configure the model, data, GPU, and output paths in the user-config block of `02_dpo_pair_construction/build_dpo_pairs_all_datasets.py`, then run:

```bash
python 02_dpo_pair_construction/build_dpo_pairs_all_datasets.py

python 03_dpo_training/clean_dpo_data.py \
  --input dpo_pairs_dedup.jsonl \
  --output cleaned_pairs.jsonl

python 03_dpo_training/cap_dpo_pairs.py \
  --input cleaned_pairs.jsonl \
  --output cleaned_pairs_capped_top2.jsonl \
  --max_pairs_per_qid 2

python 03_dpo_training/split.py \
  --input cleaned_pairs_capped_top2.jsonl \
  --train_output train_pairs.jsonl \
  --val_output val_pairs.jsonl \
  --train_ratio 0.8 \
  --seed 42

python 03_dpo_training/train_dpo_lora.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_file train_pairs.jsonl \
  --val_file val_pairs.jsonl \
  --output_dir dpo_outputs
```

The reconstructed training entry point defaults to the retained experiment settings: bf16, 3 epochs, learning rate `5e-5`, DPO beta `0.1`, batch size 1, gradient accumulation 8, maximum length 2048, AdamW, cosine scheduling, warmup ratio 0.1, and LoRA rank/alpha/dropout `16/32/0.05`.

## Evaluation configurations

| Group | Configuration | Script |
| --- | --- | --- |
| G1 | Base only | `04_paper_experiments/G1_base_only.py` |
| G2 | DPO only | `04_paper_experiments/G2_dpo_only.py` |
| G3 | Base + processed retrieval | `04_paper_experiments/G3_processed_retrieval.py` |
| G4 | Base + reward-model verifier | `04_paper_experiments/G4_rm_verifier.py` |
| G5 | Base + processed retrieval + reward-model verifier | `04_paper_experiments/G5_processed_retrieval_rm.py` |
| G6 | DPO + processed retrieval + reward-model verifier | `04_paper_experiments/G6_dpo_processed_retrieval_rm.py` |

The additional `G3_raw_retrieval_ablation.py` script evaluates retrieval without cleaning/compression. Each evaluation script retains its original experiment-specific user-config block; set the local model, adapter, data, output, and CUDA paths before running it.

## Training-code provenance

The original project archive contains a complete DPO data-cleaning/splitting chain and a small-model CPU sanity-check trainer. The final 8B LoRA entry point itself was not retained as a standalone source file. `train_dpo_lora.py` is therefore explicitly reconstructed from that pipeline, the saved trainer state, the saved PEFT adapter configuration, and the hyperparameters reported in the paper. The untouched sanity-check source is preserved as `legacy_train_validation.py` for traceability; it is not the paper's final 8B training configuration.

See `03_dpo_training/README.md` and `04_paper_experiments/README.md` for details.
