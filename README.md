# jev-like-decision-tuning

Option-based decision tuning experiments with Qwen3-1.7B and LoRA on TweetEval sentiment classification. The notebook compares the base model, the fine-tuned model, and the Jev API.

"Jev-like" describes the experimental direction; this repository does not claim to reproduce Jev's internal implementation.

## Current status

Data loading, sampling, prompt construction, and training tokenization have been extracted into an installable Python package. Model loading, batching, training, and evaluation orchestration remain in [`notebooks/jev.ipynb`](notebooks/jev.ipynb).

The notebook still requires Google Colab, a local TweetEval DatasetDict on Google Drive, a CUDA training environment, and a Jev API key stored under `jev` in Colab Secrets. Training and API evaluation have not been rerun as part of this refactor. Saved cell outputs are omitted.

## Repository layout

```text
jev-like-decision-tuning/
├── notebooks/                 # Colab experiment and interactive exploration
├── src/jev_decision_tuning/
│   ├── data.py                # Loading, sampling, formatting, tokenization
│   └── prompts.py             # Prompt text, option mapping, chat template
├── configs/                   # Reserved for experiment configurations
├── scripts/                   # Reserved for command-line entry points
├── tests/                     # Data and prompt regression tests
├── docs/                      # Method and workflow documentation
├── pyproject.toml             # Package and data/test dependencies
├── README.md
└── .gitignore
```

`src/` holds reusable implementation; future `scripts/` will call it. Local datasets belong in `data/`, and model weights, predictions, and logs belong in `outputs/`. Both directories are ignored by Git. Share curated experiment findings in `docs/`.

## Install and test the extracted utilities

From the repository root, with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

The package installs the data-processing dependency, `datasets`. The `dev` extra adds test and notebook-validation dependencies. It does not install the full model-training stack or download model weights or datasets. Dependency ranges are not a reproducibility lockfile.

## Use the data pipeline

```python
from jev_decision_tuning.data import (
    load_sentiment_dataset,
    sample_balanced,
    format_training_dataset,
)

# Load an existing DatasetDict saved with Hugging Face save_to_disk.
ds = load_sentiment_dataset("data/tweet_eval_sentiment")
train = sample_balanced(ds["train"], per_class=3000, seed=42)
formatted = format_training_dataset(train, seed=42)
print(formatted[0]["prompt"])
print(formatted[0]["target_letter"])
```

See [`docs/data-and-prompts.md`](docs/data-and-prompts.md) for input/output contracts and the distinction between a class ID and an answer position.

## Run the notebook in Colab

Keep a full repository checkout accessible to Colab. The first cell mounts Drive, installs the package from `REPO_ROOT`, and imports the extracted helpers. Set `REPO_ROOT` to your checkout; its default is `/content/drive/MyDrive/jev-like-decision-tuning`. Set the dataset path in the next cell to your saved TweetEval sentiment DatasetDict. Uploading the notebook alone is no longer sufficient.

The remaining cells retain the existing Colab training workflow and model dependency installation cells. Full environment pinning and command-line entry points are future work.

## Next steps

1. Extract model loading, option scoring, and LoRA training.
2. Extract Qwen / Jev evaluation and metric reporting.
3. Add experiment configurations, complete environment specifications, and runnable scripts.
