"""Dataset loading, reproducible sampling, formatting, and tokenization."""

import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from .prompts import DEFAULT_ORDER, LABEL_NAMES, format_example, render_chat_prompt


def validate_sentiment_dataset(dataset: Dataset) -> None:
    """Require text and the original TweetEval sentiment ClassLabel ordering."""
    if not {"text", "label"}.issubset(dataset.column_names):
        raise ValueError("dataset must contain text and label columns")
    names = getattr(dataset.features["label"], "names", None)
    if names != list(LABEL_NAMES):
        raise ValueError(f"label names must be {list(LABEL_NAMES)}, got {names}")


def load_sentiment_dataset(path: str | Path) -> DatasetDict:
    """Load a local DatasetDict saved with save_to_disk; do not download data."""
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise ValueError("expected a DatasetDict with train, validation, and test splits")
    for split in ("train", "validation", "test"):
        if split not in dataset:
            raise ValueError(f"missing required split: {split}")
        validate_sentiment_dataset(dataset[split])
    return dataset


def sample_examples(dataset: Dataset, count: int = 100, seed: int = 42) -> Dataset:
    """Select a reproducible shuffled subset, as in the initial baseline."""
    validate_sentiment_dataset(dataset)
    if count <= 0 or count > len(dataset):
        raise ValueError(f"count must be between 1 and {len(dataset)}, got {count}")
    return dataset.shuffle(seed=seed).select(range(count))


def sample_balanced(dataset: Dataset, per_class: int = 3000, seed: int = 42) -> Dataset:
    """Sample each class independently, concatenate, and shuffle the result."""
    validate_sentiment_dataset(dataset)
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    subsets = []
    for label_id, label_name in enumerate(LABEL_NAMES):
        subset = dataset.filter(lambda example: example["label"] == label_id)
        if len(subset) < per_class:
            raise ValueError(
                f"class {label_name} has {len(subset)} rows; requested {per_class}"
            )
        subsets.append(subset.shuffle(seed=seed).select(range(per_class)))
    return concatenate_datasets(subsets).shuffle(seed=seed)


def format_training_dataset(dataset: Dataset, seed: int = 42) -> Dataset:
    """Shuffle options once per row, preserving text and the original label.

    A fresh local RNG makes repeated calls reproducible for the same row order.
    Mapping stays sequential to preserve the notebook's random sequence.
    """
    validate_sentiment_dataset(dataset)
    rng = random.Random(seed)

    def format_row(example: dict[str, Any]) -> dict[str, Any]:
        order = list(DEFAULT_ORDER)
        rng.shuffle(order)
        return format_example(example, order)

    return dataset.map(format_row, load_from_cache_file=False)


def tokenize_example(example: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Encode only the prompt, keeping the answer position as a scalar label."""
    text = render_chat_prompt(tokenizer, example["prompt"])
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    encoded["labels"] = example["target"]
    return encoded


def tokenize_dataset(dataset: Dataset, tokenizer: Any) -> Dataset:
    """Replace formatted columns with model inputs and scalar answer labels."""
    return dataset.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
