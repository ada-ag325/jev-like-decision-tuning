# jev-like-decision-tuning

Option-based sentiment decision tuning with Qwen3-1.7B and LoRA. Training optimizes cross-entropy over the next-token A/B/C logits, while the original model weights remain frozen. Jev is an optional external evaluation reference; this project does not claim to reproduce Jev's internal implementation or distill its outputs.

## Current status

The project supports **train -> save -> reload -> evaluate** through Python modules, command-line scripts, and a Colab notebook. Training and evaluation run independently; evaluation reads the saved adapter, tokenizer, base-model identity, and configuration from disk.

The original executed pilot is preserved in [`notebooks/archive/jev_20261018_pilot.ipynb`](notebooks/archive/jev_20261018_pilot.ipynb). The new workflow is in [`notebooks/jev.ipynb`](notebooks/jev.ipynb). Offline tests train an actual tiny randomly initialized Qwen3 model and verify adapter reload and evaluation in separate processes. The new pipeline has not yet been rerun on the full pretrained Qwen3 model with CUDA or the live Jev API.

## Install

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
python -m pytest -q
```

The `train` extra pins Transformers, PEFT, and Accelerate to the versions tested by this implementation. PyTorch and dataset dependencies have version ranges; each training/evaluation records the versions actually used. This is not a complete environment lockfile. Standard configurations require a CUDA GPU; CPU float32 is supported for small tests. No model or dataset is downloaded by the test suite.

For the optional Jev comparison, install `python -m pip install -e '.[train,jev]'`. API credentials are supplied at runtime and never saved into experiment configurations.

## Train, save, reload, evaluate

Prepare a local Hugging Face DatasetDict saved with `save_to_disk`, with `train`, `validation`, and `test` splits. The expected sentiment ClassLabel order is negative, neutral, positive. The dataset is not included in this repository and the loader does not download it.

Run these commands from the repository root on a CUDA machine:

```bash
python scripts/train.py \
  --config configs/attention_mlp.json \
  --data-path /absolute/path/to/tweet_eval_sentiment \
  --output-dir outputs/attention_mlp_seed42
```

The output directory must be new. This trains one epoch on 3,000 examples per class and saves the adapter and tokenizer. The base model may be downloaded from Hugging Face on first use. No Jev requests are made during training.

You may now stop Python or restart the runtime. Evaluate the saved adapter in a new process:

```bash
python scripts/evaluate.py \
  --run-dir outputs/attention_mlp_seed42 \
  --split validation \
  --backend lora

python scripts/evaluate.py \
  --run-dir outputs/attention_mlp_seed42 \
  --split validation \
  --backend base
```

Use `--limit 100` for a quick check; limited evaluations have distinct filenames. Use `--split test` after fixing the experiment design. `--split train` evaluates the full training split, not only the selected 9,000 examples. If the dataset moved, pass `--data-path /new/path`. For CPU testing, specify both `--device cpu --precision float32`.

Existing runs and evaluation names are not overwritten. To deliberately repeat an evaluation, provide a new `--name`, such as `validation_lora_repeat1`. This workflow reloads finished adapters for inference; it does not resume an interrupted optimizer or continue training from a previous adapter.

## Experiment artifacts

```text
outputs/attention_mlp_seed42/
├── config.json                  # Resolved settings and data/model paths
├── metadata.json                # Status, revisions, environment, code and data fingerprints
├── training_samples.jsonl       # Source row IDs, labels, answer positions, prompt hashes
├── training_log.json            # Loss history, parameter count, time and peak GPU memory
├── adapter/                     # LoRA weights, PEFT configuration, saved tokenizer
└── evaluation/
    ├── validation_lora.predictions.jsonl
    ├── validation_lora.metrics.json
    ├── validation_base.predictions.jsonl
    └── validation_base.metrics.json
```

Predictions contain row IDs, true/predicted labels, and class probabilities. Reports include accuracy, Macro-F1, Macro Recall, per-class scores, and a confusion matrix. Every report references its run and training manifest. `metadata.status` becomes `complete` only after adapter and training-log saving succeeds. Failed or unfinished runs cannot be evaluated as completed experiments.

Only adapter weights are saved, so reloading still requires the original base model, from its local path or the Hub/cache. A resolved Hub model revision is recorded when available. Keep the original base files unchanged when using a local base model.

## Colab

Open [`notebooks/jev.ipynb`](notebooks/jev.ipynb) in a GPU runtime. Set `REPO_ROOT` to the full checkout; the default is `/content/drive/MyDrive/jev_project/jev-like-decision-tuning`. The setup cell installs `[train]`, checks source files, and adds `src/` to the current kernel's search path.

Set `DATA_PATH` to your existing Drive dataset and choose a directory for `OUTPUT_ROOT`. The training cell prints the exact completed `RUN_DIR`. To evaluate after restarting, run setup, import `evaluate_experiment`, and set `RUN_DIR` to that saved path; skip the training cell.

Test evaluation and Jev API requests are disabled by default. To use Jev, install the `jev` extra, provide the Colab Secret `jev`, and explicitly enable `RUN_JEV`. New Jev checkpoints include a dataset/request manifest; legacy pilot caches without that manifest are not automatically reused.

## Configurations and scope

`configs/attention.json`, `configs/mlp.json`, and `configs/attention_mlp.json` differ only in LoRA target placement. `--seed` overrides the training seed; dataset selection and option-order seeds stay fixed. These configurations are ready for later ablations, but this refactor does not constitute completed ablation experiments.

See [workflow details](docs/training-and-evaluation.md) and [data/prompt contracts](docs/data-and-prompts.md). The repository's source, documentation, and runtime messages are in English.
