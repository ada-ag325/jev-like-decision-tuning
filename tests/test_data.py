from collections import Counter

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from jev_decision_tuning.data import (
    format_training_dataset,
    load_sentiment_dataset,
    sample_balanced,
    sample_examples,
    tokenize_dataset,
    tokenize_example,
    validate_sentiment_dataset,
)


@pytest.fixture
def dataset():
    return Dataset.from_dict(
        {
            "text": [f"Example {i}" for i in range(18)],
            "label": [i % 3 for i in range(18)],
        },
        features=Features({
            "text": Value("string"),
            "label": ClassLabel(names=["negative", "neutral", "positive"]),
        }),
    )


def test_balanced_sampling_is_reproducible_and_keeps_source_labels(dataset):
    first = sample_balanced(dataset, per_class=3, seed=42)
    assert first.to_dict() == sample_balanced(dataset, per_class=3, seed=42).to_dict()
    assert Counter(first["label"]) == {0: 3, 1: 3, 2: 3}
    assert len(set(first["text"])) == 9
    source_labels = dict(zip(dataset["text"], dataset["label"]))
    assert all(source_labels[row["text"]] == row["label"] for row in first)


def test_balanced_sampling_rejects_insufficient_data(dataset):
    with pytest.raises(ValueError, match="negative has 6 rows; requested 7"):
        sample_balanced(dataset, per_class=7)
    with pytest.raises(ValueError, match="positive"):
        sample_balanced(dataset, per_class=0)


def test_validation_sampling_matches_original_subset(dataset):
    assert sample_examples(dataset, count=5).to_dict() == (
        dataset.shuffle(seed=42).select(range(5)).to_dict()
    )
    with pytest.raises(ValueError, match="count"):
        sample_examples(dataset, count=19)


def test_option_shuffling_is_repeatable_and_preserves_original_columns(dataset):
    first = format_training_dataset(dataset)
    assert first.to_dict() == format_training_dataset(dataset).to_dict()
    assert list(first["text"]) == list(dataset["text"])
    assert list(first["label"]) == list(dataset["label"])
    # Golden sequence from the original notebook's random.Random(42) shuffles.
    assert list(first["target_letter"][:6]) == ["B", "B", "B", "C", "A", "C"]
    assert any(row["target"] != row["label"] for row in first)


def test_loader_round_trip_and_missing_split(dataset, tmp_path):
    valid = DatasetDict({name: dataset for name in ("train", "validation", "test")})
    valid.save_to_disk(tmp_path / "valid")
    loaded = load_sentiment_dataset(tmp_path / "valid")
    assert loaded["train"].to_dict() == dataset.to_dict()

    DatasetDict({"train": dataset}).save_to_disk(tmp_path / "incomplete")
    with pytest.raises(ValueError, match="missing required split: validation"):
        load_sentiment_dataset(tmp_path / "incomplete")


def test_loader_rejects_a_single_dataset(dataset, tmp_path):
    dataset.save_to_disk(tmp_path / "single")
    with pytest.raises(ValueError, match="DatasetDict"):
        load_sentiment_dataset(tmp_path / "single")


def test_schema_rejects_reordered_labels_or_missing_text(dataset):
    wrong_labels = dataset.cast_column(
        "label", ClassLabel(names=["positive", "neutral", "negative"])
    )
    with pytest.raises(ValueError, match="label names"):
        validate_sentiment_dataset(wrong_labels)
    with pytest.raises(ValueError, match="text and label"):
        validate_sentiment_dataset(dataset.remove_columns("text"))


class RecordingTokenizer:
    """Record the actual model input without loading a remote tokenizer."""

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_options = kwargs
        return f'<user>{messages[0]["content"]}</user><assistant>'

    def __call__(self, text, **kwargs):
        self.text = text
        self.tokenizer_options = kwargs
        return {"input_ids": [11, 12, 13], "attention_mask": [1, 1, 1]}


def test_tokenization_does_not_leak_the_answer_into_model_input():
    tokenizer = RecordingTokenizer()
    result = tokenize_example({"prompt": "Question", "target": 2}, tokenizer)
    assert tokenizer.messages == [{"role": "user", "content": "Question"}]
    assert tokenizer.text == "<user>Question</user><assistant>"
    assert tokenizer.template_options == {
        "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
    }
    assert tokenizer.tokenizer_options == {"add_special_tokens": False, "truncation": False}
    assert result == {"input_ids": [11, 12, 13], "attention_mask": [1, 1, 1], "labels": 2}


def test_dataset_tokenization_keeps_scalar_option_targets(dataset):
    formatted = format_training_dataset(dataset)
    encoded = tokenize_dataset(formatted, RecordingTokenizer())
    assert set(encoded.column_names) == {"input_ids", "attention_mask", "labels"}
    assert list(encoded["labels"]) == list(formatted["target"])
