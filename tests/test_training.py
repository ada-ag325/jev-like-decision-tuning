"""Offline checks with a randomly initialized tiny Qwen3, never remote weights."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, set_seed

from jev_decision_tuning.modeling import (
    TARGET_GROUPS,
    add_lora,
    collate_examples,
    load_base_model,
    load_run,
    option_logits,
    option_token_ids,
    resolve_device_precision,
    save_adapter,
)
from jev_decision_tuning.training import train_model


def write_tiny_base(path):
    """Write a real local causal LM/tokenizer for workflow regression tests."""
    path.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
        "A": 3, "B": 4, "C": 5,
        "Text": 6, "Answer": 7, ":": 8,
        "negative": 9, "neutral": 10, "positive": 11,
        "good": 12, "bad": 13, "okay": 14, "assistant": 15,
    }
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=512,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant' }}{% endif %}"
    )
    set_seed(7)
    config = Qwen3Config(
        vocab_size=len(vocabulary),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=512,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def tiny_base_dir(tmp_path):
    return write_tiny_base(tmp_path / "base")


@pytest.fixture
def tokenized_examples():
    return [
        {"input_ids": [6, 12, 15], "attention_mask": [1, 1, 1], "labels": 2},
        {"input_ids": [6, 13, 7, 15], "attention_mask": [1, 1, 1, 1], "labels": 0},
        {"input_ids": [6, 14, 15], "attention_mask": [1, 1, 1], "labels": 1},
        {"input_ids": [12, 7, 15], "attention_mask": [1, 1, 1], "labels": 2},
    ]


def test_option_scores_use_last_nonpadding_token_and_only_three_options():
    logits = torch.arange(2 * 4 * 7, dtype=torch.float32).reshape(2, 4, 7)

    class FixedModel:
        def __call__(self, **kwargs):
            assert set(kwargs) == {"input_ids", "attention_mask", "use_cache"}
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=logits)

    scores = option_logits(
        FixedModel(),
        torch.tensor([[1, 2, 0, 0], [1, 2, 3, 4]]),
        torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
        [3, 4, 5],
    )
    torch.testing.assert_close(scores, torch.stack([logits[0, 1, 3:6], logits[1, 3, 3:6]]))
    assert scores.shape == (2, 3)


@pytest.mark.parametrize("mask", [[[0, 1, 1]], [[1, 0, 1]], [[0, 0, 0]], [[1, 2, 0]]])
def test_option_scoring_rejects_invalid_padding_before_model_forward(mask):
    with pytest.raises(ValueError):
        option_logits(None, torch.tensor([[1, 2, 3]]), torch.tensor(mask), [3, 4, 5])


def test_option_ids_reject_merged_tokens_and_collisions():
    class FakeTokenizer:
        def __init__(self, encoding):
            self.encoding = encoding

        def encode(self, letter, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return self.encoding[letter]

    with pytest.raises(ValueError, match="exactly one token"):
        option_token_ids(FakeTokenizer({"A": [3], "B": [4, 5], "C": [6]}))
    with pytest.raises(ValueError, match="distinct"):
        option_token_ids(FakeTokenizer({"A": [3], "B": [3], "C": [6]}))


@pytest.mark.parametrize("target_group", TARGET_GROUPS)
def test_only_requested_lora_parameters_are_trainable(tiny_base_dir, target_group):
    model, tokenizer = load_base_model(str(tiny_base_dir), device="cpu", precision="float32")
    assert not any(parameter.requires_grad for parameter in model.parameters())
    model = add_lora(model, target_group=target_group, rank=2, alpha=4, dropout=0.0)
    found_targets = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert ".lora_" in name
            found_targets.add(name.split(".lora_")[0].rsplit(".", 1)[-1])
            assert parameter.dtype == torch.float32
        else:
            assert ".lora_" not in name
    assert found_targets == set(TARGET_GROUPS[target_group])
    assert tokenizer.padding_side == "right"


def test_training_changes_only_lora_and_reload_preserves_probabilities(
    tiny_base_dir, tokenized_examples, tmp_path
):
    model, tokenizer = load_base_model(str(tiny_base_dir), device="cpu", precision="float32")
    set_seed(23)
    model = add_lora(model, rank=2, alpha=4, dropout=0.0)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    report = train_model(
        model, tokenizer, tokenized_examples,
        epochs=2, batch_size=2, learning_rate=0.01, seed=23,
        precision="float32", log_every=50,
    )
    changed_adapters = []
    for name, parameter in model.named_parameters():
        if ".lora_" in name:
            changed_adapters.append(not torch.equal(parameter, before[name]))
        else:
            assert torch.equal(parameter, before[name]), name
    assert any(changed_adapters)
    assert len(report["history"]) == 4
    assert report["samples"] == 4
    assert report["skipped_updates"] == 0
    assert report["peak_memory_gib"] == 0
    assert report["trainable_parameters"] > 0
    json.dumps(report, allow_nan=False)
    assert not model.training
    batch = collate_examples(tokenizer, tokenized_examples[:2])
    with torch.inference_mode():
        original = option_logits(
            model, batch["input_ids"], batch["attention_mask"], option_token_ids(tokenizer)
        ).softmax(dim=-1)
    run_dir = tmp_path / "run"
    # Ensure load_run restores the saved tokenizer, not just the base tokenizer.
    tokenizer.chat_template += " saved"
    saved_template = tokenizer.chat_template
    save_adapter(model, tokenizer, run_dir)
    (run_dir / "config.json").write_text(json.dumps({
        "model_name": str(tiny_base_dir), "model_revision": None,
        "device": "cpu", "precision": "float32",
    }))
    (run_dir / "metadata.json").write_text(json.dumps({"status": "complete"}))
    reloaded, restored_tokenizer = load_run(run_dir)
    assert restored_tokenizer.chat_template == saved_template
    assert not reloaded.training
    assert not any(parameter.requires_grad for parameter in reloaded.parameters())
    with torch.inference_mode():
        restored = option_logits(
            reloaded, batch["input_ids"], batch["attention_mask"], option_token_ids(restored_tokenizer)
        ).softmax(dim=-1)
    torch.testing.assert_close(original, restored, atol=1e-7, rtol=1e-6)
    with pytest.raises(FileExistsError):
        save_adapter(model, tokenizer, run_dir)


def test_training_is_repeatable_with_same_initialization_and_seed(tiny_base_dir, tokenized_examples):
    outcomes = []
    for _ in range(2):
        model, tokenizer = load_base_model(str(tiny_base_dir), device="cpu", precision="float32")
        set_seed(23)
        model = add_lora(model, rank=2, alpha=4, dropout=0.1)
        report = train_model(
            model, tokenizer, tokenized_examples,
            batch_size=2, learning_rate=0.01, seed=23, precision="float32",
        )
        outcomes.append((
            report["history"],
            {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad},
        ))
    assert outcomes[0][0] == outcomes[1][0]
    for name in outcomes[0][1]:
        torch.testing.assert_close(outcomes[0][1][name], outcomes[1][1][name], rtol=0, atol=0)


def test_unfrozen_base_weights_are_rejected(tiny_base_dir, tokenized_examples):
    model, tokenizer = load_base_model(str(tiny_base_dir), device="cpu", precision="float32")
    model.requires_grad_(True)
    with pytest.raises(ValueError, match="non-LoRA parameter"):
        train_model(model, tokenizer, tokenized_examples, precision="float32")


def test_incomplete_run_is_not_loaded(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"status": "training"}))
    with pytest.raises(ValueError, match="not complete"):
        load_run(tmp_path)


def test_cpu_execution_requires_float32():
    with pytest.raises(ValueError, match="CPU execution requires"):
        resolve_device_precision("cpu", "float16")
