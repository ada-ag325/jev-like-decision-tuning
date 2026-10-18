import json
import sys
from types import SimpleNamespace

import pytest
from datasets import ClassLabel, Dataset, Features, Value

from jev_decision_tuning.evaluation import evaluate_jev, evaluate_model, save_evaluation
from jev_decision_tuning.metrics import compute_metrics


@pytest.fixture
def dataset():
    return Dataset.from_dict(
        {"text": ["short", "a little longer", "a much longer example with several words"], "label": [2, 0, 1]},
        features=Features({
            "text": Value("string"),
            "label": ClassLabel(names=["negative", "neutral", "positive"]),
        }),
    )


def test_metrics_include_all_classes_and_true_by_predicted_matrix():
    records = [{"true_label": truth, "pred_label": prediction} for truth, prediction in [(0, 0), (0, 1), (1, 1), (2, 1)]]
    result = compute_metrics(records)
    assert result["examples"] == 4
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert result["accuracy"] == 0.5
    assert result["macro_recall"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 3)
    assert result["per_class"]["positive"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    json.dumps(result, allow_nan=False)


def test_metrics_reject_empty_or_invalid_class_ids():
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([])
    for invalid in (True, -1, 3, "0", 1.5):
        with pytest.raises(ValueError, match="class ID"):
            compute_metrics([{"true_label": invalid, "pred_label": 0}])


def test_batched_evaluation_matches_single_rows_with_mixed_lengths(dataset):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        padding_side = "left"

        def __init__(self):
            self.inputs = []

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [{"A": 3, "B": 4, "C": 5}[text]]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return messages[0]["content"] + "<assistant>"

        def __call__(self, prompts, **kwargs):
            assert self.padding_side == "right"
            assert kwargs == {"add_special_tokens": False, "truncation": False, "padding": True, "return_tensors": "pt"}
            self.inputs.extend(prompts)
            sequences = []
            choices = dict(zip(dataset["text"], (2, 0, 1)))
            for prompt in prompts:
                text = prompt.split("Text: ", 1)[1].split("\n\nA:", 1)[0]
                sequences.append([7] * len(text.split()) + [10 + choices[text]])
            width = max(map(len, sequences))
            return {
                "input_ids": torch.tensor([sequence + [0] * (width - len(sequence)) for sequence in sequences]),
                "attention_mask": torch.tensor([[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences]),
            }

    class Model(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, attention_mask, use_cache=False):
            assert not self.training
            assert not torch.is_grad_enabled()
            logits = torch.zeros((*input_ids.shape, 16))
            for row in range(input_ids.shape[0]):
                for column in range(input_ids.shape[1]):
                    marker = input_ids[row, column].item()
                    if marker >= 10:
                        logits[row, column, 3 + marker - 10] = 4.0
                    elif marker == 0:
                        logits[row, column, 3] = 20.0
            return SimpleNamespace(logits=logits)

    tokenizer = Tokenizer()
    model = Model()
    batched = evaluate_model(model, tokenizer, dataset, batch_size=3)
    individual = evaluate_model(model, tokenizer, dataset, batch_size=1)
    assert batched == individual
    assert [row["row_id"] for row in batched] == [0, 1, 2]
    assert [row["pred_label"] for row in batched] == [2, 0, 1]
    assert [row["text"] for row in batched] == list(dataset["text"])
    assert tokenizer.padding_side == "left"
    assert all(prompt.endswith("<assistant>") for prompt in tokenizer.inputs)
    assert all(row["p_positive"] >= 0 for row in batched)
    relabeled = dataset.map(lambda row: {"label": (row["label"] + 1) % 3})
    changed = evaluate_model(model, tokenizer, relabeled, batch_size=2)
    assert [row["pred_label"] for row in changed] == [row["pred_label"] for row in batched]


def test_save_evaluation_records_context_and_refuses_overwrite(tmp_path):
    records = [{"row_id": 0, "text": "example", "true_label": 0, "pred_label": 1}]
    report = save_evaluation(records, tmp_path, name="validation", metadata={"run_id": "example"})
    assert report["metadata"] == {"run_id": "example"}
    assert json.loads((tmp_path / "validation.predictions.jsonl").read_text()) == records[0]
    assert json.loads((tmp_path / "validation.metrics.json").read_text()) == report
    with pytest.raises(FileExistsError):
        save_evaluation(records, tmp_path, name="validation")
    save_evaluation(records, tmp_path, name="validation", overwrite=True)
    with pytest.raises(ValueError, match="filename stem"):
        save_evaluation(records, tmp_path, name="../escape")


@pytest.fixture
def fake_sdk(monkeypatch):
    class Client:
        def __init__(self, fail_after=None):
            self.calls = []
            self.fail_after = fail_after

        def system_one(self, **kwargs):
            if self.fail_after is not None and len(self.calls) == self.fail_after:
                raise RuntimeError("simulated interrupted evaluation")
            self.calls.append(kwargs)
            return SimpleNamespace(
                model=kwargs["model"],
                choices={"sentiment": SimpleNamespace(choice="B", probabilities={"A": 0.1, "B": 0.7, "C": 0.2})},
                usage=SimpleNamespace(input_tokens=11, output_tokens=1),
            )

    def unexpected_client(**kwargs):
        raise AssertionError("an injected client should be reused")

    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(Choice=lambda **kwargs: kwargs, TypeSafeClient=unexpected_client))
    return Client


def test_jev_resumes_interrupted_checkpoint_without_repeating_requests(dataset, tmp_path, fake_sdk):
    path = tmp_path / "jev.jsonl"
    first = fake_sdk(fail_after=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_jev(dataset, api_key="do-not-persist", checkpoint_path=path, client=first)
    assert path.with_suffix(".jsonl.manifest.json").exists()
    assert len(path.read_text().splitlines()) == 1
    second = fake_sdk()
    result = evaluate_jev(dataset, api_key=None, checkpoint_path=path, client=second)
    assert len(second.calls) == 2
    assert [row["row_id"] for row in result] == [0, 1, 2]
    assert second.calls[0]["questions"]["sentiment"]["criteria"] == {"A": "negative", "B": "neutral", "C": "positive"}
    assert "do-not-persist" not in path.read_text() + path.with_suffix(".jsonl.manifest.json").read_text()
    assert evaluate_jev(dataset, api_key=None, checkpoint_path=path) == result


def test_jev_manifest_checks_even_rows_not_cached_yet(dataset, tmp_path, fake_sdk):
    path = tmp_path / "jev.jsonl"
    with pytest.raises(RuntimeError):
        evaluate_jev(dataset, api_key=None, checkpoint_path=path, client=fake_sdk(fail_after=1))
    changed = dataset.map(lambda row, index: {"text": row["text"] + (" changed" if index == 2 else "")}, with_indices=True)
    client = fake_sdk()
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_jev(changed, api_key=None, checkpoint_path=path, client=client)
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_jev(dataset, api_key=None, checkpoint_path=path, model_name="different-version", client=client)
    assert not client.calls


def test_jev_rejects_corrupt_or_unidentified_cache(dataset, tmp_path, fake_sdk):
    path = tmp_path / "jev.jsonl"
    evaluate_jev(dataset, api_key=None, checkpoint_path=path, client=fake_sdk())
    with path.open("a") as stream:
        stream.write('{"partial":')
    with pytest.raises(ValueError, match="corrupt Jev checkpoint"):
        evaluate_jev(dataset, api_key=None, checkpoint_path=path, client=fake_sdk())
    path.with_suffix(".jsonl.manifest.json").unlink()
    with pytest.raises(ValueError, match="missing its manifest"):
        evaluate_jev(dataset, api_key=None, checkpoint_path=path, client=fake_sdk())
