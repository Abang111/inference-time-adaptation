# Inference-Time Adaptation for Medical Question Answering

Official implementation accompanying **“Characterizing Inference-Time Adaptation in Medical QA: From Retrieval Augmentation to Reward-Guided Alignment.”**

This repository provides the data-construction and policy-optimization pipeline needed to:

1. normalize MedQA, MedMCQA, and PubMedQA into one schema;
2. retrieve question-relevant PubMed evidence;
3. sample and reward-score reasoning traces;
4. convert those traces into DPO preference pairs; and
5. train a LoRA adapter with Direct Preference Optimization (DPO).

The repository contains code, not the original datasets or model weights. Dataset licenses and gated-model access must be handled separately.

## Method at a glance

```text
MedQA / MedMCQA / PubMedQA
              |
              v
       unified JSONL records
              |
              v
    question-visible PubMed queries -----> PubMed abstracts
              |                                  |
              +----------------+-----------------+
                               v
               sample policy reasoning traces
                               |
                               v
                score traces with MedPRM
                               |
                               v
                   (prompt, chosen, rejected)
                               |
                               v
                    clean -> cap -> split
                               |
                               v
                     LoRA DPO fine-tuning
```

## Repository layout

```text
.
|-- 00_data_build/
|   |-- build_unified_med_datasets.py
|   `-- extract_medqa_doc_supported_subset.py
|-- 01_retrieval/
|   |-- build_queries.py
|   `-- retrieve_pubmed_full.py
|-- 02_dpo_pair_construction/
|   |-- build_dpo_pairs_all_datasets.py
|   `-- build_dpo_pairs_medmcqa.py
|-- 03_dpo_training/
|   |-- clean_dpo_data.py
|   |-- cap_dpo_pairs.py
|   |-- split.py
|   |-- train_dpo_lora.py
|   |-- legacy_train_validation.py
|   `-- README.md
|-- requirements.txt
`-- README.md
```

The numbered directories follow the order in which the pipeline is normally run.

## Installation

### 1. Create a Python environment

Python 3.10 is recommended. The commands below assume Linux, which is the intended environment for multi-GPU training.

```bash
git clone https://github.com/Abang111/inference-time-adaptation.git
cd inference-time-adaptation

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch and the remaining dependencies

The paper environment used PyTorch 2.7.0 with CUDA 12.6. Install the CUDA wheel first, then install the repository requirements:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For a different CUDA version, install the appropriate PyTorch build for the host before running the second command. `deepspeed` is Linux-only in `requirements.txt`; the data-preparation stages can be run on Windows, but the reported DPO training setup is intended for Linux with NVIDIA GPUs.

### 3. Obtain model access

The full pipeline expects the following model components:

| Component | Model used in the paper | Used by |
| --- | --- | --- |
| Base policy | `meta-llama/Meta-Llama-3.1-8B-Instruct` | trace sampling and DPO training |
| Reward model | `llama-3.1-medprm-reward-v1.0` | preference scoring |
| DPO policy | base policy plus the trained LoRA adapter | output of the training pipeline |

Accept any required model licenses, authenticate with Hugging Face if necessary, and either use a model ID or download the weights locally. The pair-construction scripts use local `Path(...)` values and therefore require their configuration blocks to be edited before execution.

## Configuration model

Most data, retrieval, and pair-construction programs are research scripts with a clearly marked **User config** or **Paths** block near the top. Set the input, output, model, and CUDA values there before running a script. The DPO post-processing and training programs instead expose command-line arguments.

The most important configuration locations are:

| Stage | File | Values to set |
| --- | --- | --- |
| Dataset conversion | `00_data_build/build_unified_med_datasets.py` | `MEDQA_DIR`, `MEDMCQA_DIR`, `PUBMEDQA_DIR`, `OUTPUT_DIR` |
| Query construction | `01_retrieval/build_queries.py` | `UNIFIED_DIR`, `OUTPUT_DIR`, `DEFAULT_EMAIL` |
| PubMed retrieval | `01_retrieval/retrieve_pubmed_full.py` | `INPUT_DIR`, `RUN_ROOT`; environment variables `NCBI_EMAIL` and optionally `NCBI_API_KEY` |
| DPO-pair construction | `02_dpo_pair_construction/build_dpo_pairs_all_datasets.py` | policy/reward model paths, `FULL_DOCS_ROOT`, GPU assignment, `OUT_DIR` |

Empty `Path("")` values are placeholders, not usable defaults. Use dedicated output directories so that generated artifacts are not written into the source tree accidentally.

## Input data

Place the source datasets in separate directories. `build_unified_med_datasets.py` expects Hugging Face-style Parquet shards with these names:

```text
MedQA/
|-- train-00000-of-00001.parquet
|-- validation-00000-of-00001.parquet
`-- test-00000-of-00001.parquet

MedMCQA/
|-- train-00000-of-00001.parquet
|-- validation-00000-of-00001.parquet
`-- test-00000-of-00001.parquet

PubMedQA/
`-- train-00000-of-00001.parquet
```

Every normalized row has the same core structure:

```json
{
  "qid": "medqa_train_000001",
  "dataset": "medqa",
  "split": "train",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "B",
  "answer_text": "...",
  "context": "...",
  "meta": {}
}
```

## End-to-end reproduction

Run all commands from the repository root unless stated otherwise.

### Step 1: Normalize the medical QA datasets

Edit the path block in `00_data_build/build_unified_med_datasets.py`, then run:

```bash
python 00_data_build/build_unified_med_datasets.py
```

The program writes one JSONL file per available dataset/split plus `all_unified.jsonl`. Missing datasets are reported as warnings, allowing an individual dataset to be prepared independently.

### Step 2: Build PubMed queries

Set `UNIFIED_DIR`, `OUTPUT_DIR`, and `DEFAULT_EMAIL` in `01_retrieval/build_queries.py`:

```bash
python 01_retrieval/build_queries.py
```

For every supported split, the script writes a small debugging sample and a full query file. Query generation uses only information visible in the question/options and does not insert the gold answer into the search query.

### Step 3: Retrieve PubMed evidence

Set `INPUT_DIR` and `RUN_ROOT` in `01_retrieval/retrieve_pubmed_full.py`, then provide the NCBI identity variables:

```bash
export NCBI_EMAIL="name@example.org"
export NCBI_API_KEY="optional-api-key"
python 01_retrieval/retrieve_pubmed_full.py
```

PowerShell equivalent:

```powershell
$env:NCBI_EMAIL = "name@example.org"
$env:NCBI_API_KEY = "optional-api-key"
python 01_retrieval/retrieve_pubmed_full.py
```

Retrieved documents are written below `<RUN_ROOT>/docs/`; query and PMID responses are cached below `<RUN_ROOT>/cache/`, so an interrupted retrieval run can be resumed without repeating completed requests.

### Step 4: Optionally create a fixed document-supported MedQA subset

Set `INPUT_JSONL` in `00_data_build/extract_medqa_doc_supported_subset.py` to the retrieved `medqa_train_docs.jsonl`, choose an output directory, and run:

```bash
python 00_data_build/extract_medqa_doc_supported_subset.py
```

With the default settings, this retains questions with evidence and draws a deterministic 1,000-question subset using seed 42. The resulting JSONL is useful for fixed-size downstream analyses.

### Step 5: Construct reward-guided DPO pairs

Edit the complete User config block in `02_dpo_pair_construction/build_dpo_pairs_all_datasets.py`, particularly:

- `MEDPRM_DIR` and tokenizer fallbacks;
- `POLICY_FT_8B_MODEL_DIR` and `POLICY_8B_MODEL_DIR`;
- `FULL_DOCS_ROOT`;
- the GPU assignments; and
- `OUT_DIR`.

Then run:

```bash
python 02_dpo_pair_construction/build_dpo_pairs_all_datasets.py
```

For each question, the script samples reasoning traces from two policies, checks answer formatting, obtains step/final reward scores, ranks valid traces, and selects preferred/dispreferred responses. It saves progress after every question. The key outputs are:

| Output | Meaning |
| --- | --- |
| `dpo_pairs_raw.jsonl` | all accepted preference pairs before global deduplication |
| `dpo_pairs_dedup.jsonl` | deduplicated pairs used by the training-data pipeline |
| `per_question_results/` | traces, scores, and diagnostics for each question |
| `progress_summary.jsonl` | incremental run status |
| `skipped_questions.jsonl` | questions omitted and the reason |
| `final_report.json` | aggregate construction statistics and configuration |

The training-relevant fields are `prompt`, `chosen`, and `rejected`. Additional fields such as `qid`, `dataset`, `pair_type`, answer labels, reward scores, and score margin are retained for filtering and auditability.

### Step 6: Clean, cap, and split the pairs

```bash
python 03_dpo_training/clean_dpo_data.py \
  --input /path/to/dpo_pairs_dedup.jsonl \
  --output /path/to/cleaned_pairs.jsonl \
  --summary /path/to/cleaning_summary.json

python 03_dpo_training/cap_dpo_pairs.py \
  --input /path/to/cleaned_pairs.jsonl \
  --output /path/to/cleaned_pairs_capped_top2.jsonl \
  --max_pairs_per_qid 2 \
  --summary /path/to/capping_summary.json

python 03_dpo_training/split.py \
  --input /path/to/cleaned_pairs_capped_top2.jsonl \
  --train_output /path/to/train_pairs.jsonl \
  --val_output /path/to/val_pairs.jsonl \
  --train_ratio 0.8 \
  --seed 42
```

The split is performed by `qid`, so preference pairs from a single question cannot leak across training and validation sets.

### Step 7: Train the LoRA DPO policy

Single-process launch:

```bash
python 03_dpo_training/train_dpo_lora.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_file /path/to/train_pairs.jsonl \
  --val_file /path/to/val_pairs.jsonl \
  --output_dir /path/to/dpo_outputs
```

Multi-GPU launch through Accelerate:

```bash
accelerate config
accelerate launch 03_dpo_training/train_dpo_lora.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --train_file /path/to/train_pairs.jsonl \
  --val_file /path/to/val_pairs.jsonl \
  --output_dir /path/to/dpo_outputs
```

The final LoRA adapter and tokenizer are saved in `<output_dir>/final_model/`. Epoch checkpoints, `trainer_state.json`, and `run_config.json` make model selection and run auditing possible.

Default training settings:

| Setting | Value |
| --- | ---: |
| Precision | bf16 |
| Epochs | 3 |
| Learning rate | `5e-5` |
| DPO beta | `0.1` |
| Per-device train/eval batch size | 1 |
| Gradient accumulation | 8 |
| Maximum sequence length | 2048 |
| Optimizer | `adamw_torch` |
| Scheduler | cosine |
| Warmup ratio | 0.1 |
| Gradient checkpointing | enabled |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Seed | 42 |

Use `--help` to see every supported training override. bf16 is enabled by default and can be disabled with `--no_bf16` for a compatible debugging environment.

## Reproducibility notes

- Reported software environment: Ubuntu 22.04.5, Python 3.10.20, PyTorch 2.7.0 + CUDA 12.6, Transformers 4.46.3, Accelerate 1.1.1, PEFT 0.18.1, DeepSpeed 0.18.9, Datasets 4.8.4, NumPy 2.2.6, and Pandas 2.3.3.
- Reported hardware: six NVIDIA RTX 4070 Ti SUPER GPUs with 16 GB per GPU and 502 GiB system memory.
- The pair constructor and DPO trainer use seed 42 and record run metadata. Keep the generated summaries and run configuration with every trained adapter.
- The all-dataset pair builder supports MedQA, MedMCQA, and PubMedQA. When reporting a trained checkpoint, record the actual per-dataset counts in the generated summaries rather than inferring the mixture from the script configuration.
- PubMed content can change over time. Preserve the retrieval cache and output JSONL files when exact reruns are required.

## Code provenance

The complete DPO cleaning, capping, splitting, saved trainer state, and LoRA adapter configuration were retained in the project archive. The original final 8B training launcher was not retained as a standalone source file. `03_dpo_training/train_dpo_lora.py` is therefore a transparent reconstruction from those artifacts and the hyperparameters reported in the paper.

`03_dpo_training/legacy_train_validation.py` is the untouched tiny-GPT2/CPU validation script found in the archive. It is preserved for traceability and is not presented as the final 8B paper training configuration.

## Troubleshooting

- **`ModuleNotFoundError`**: activate the intended environment and run `pip install -r requirements.txt`.
- **bf16 or CUDA error**: use a bf16-capable NVIDIA GPU, correct the device mapping, or use `--no_bf16` only for a non-paper debugging run.
- **CUDA out of memory**: keep per-device batch size at 1, use gradient accumulation, reduce concurrent models/samples, or distribute policy and reward models across more GPUs.
- **Gated-model error**: accept the model license and authenticate with an account that has access.
- **PubMed retrieval refuses to start**: set `NCBI_EMAIL`; `NCBI_API_KEY` is optional.

## Citation

If this repository is useful in your work, please cite the accompanying paper. A final BibTeX entry can be added here once the paper has public bibliographic metadata.
