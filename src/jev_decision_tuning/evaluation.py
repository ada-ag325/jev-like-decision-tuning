"""Batched model evaluation and resumable optional Jev API evaluation."""

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import nullcontext
from numbers import Integral
from pathlib import Path
from typing import Any

from .data import validate_sentiment_dataset
from .metrics import compute_metrics
from .prompts import INSTRUCTION, LABEL_NAMES, LETTERS, build_prompt, render_chat_prompt


def _probabilities(values: Any) -> list[float]:
    probabilities = [float(value) for value in values]
    if (
        len(probabilities) != len(LABEL_NAMES)
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-3)
    ):
        raise ValueError("expected three finite probabilities in [0, 1] that sum to one")
    return probabilities


def _record(row_id: int, example: dict[str, Any], values: Any, **extra: Any) -> dict[str, Any]:
    probabilities = _probabilities(values)
    return {
        "row_id": row_id,
        "text": example["text"],
        "true_label": int(example["label"]),
        "pred_label": max(range(len(LABEL_NAMES)), key=probabilities.__getitem__),
        **dict(zip((f"p_{name}" for name in LABEL_NAMES), probabilities)),
        **extra,
    }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    batch_size: int = 4,
    compact_options: bool = False,
) -> list[dict[str, Any]]:
    """Score A/B/C with fixed negative/neutral/positive options in row order.

    Only the rendered question is tokenized; ground-truth labels never enter
    the model. Padding is on the right, and scoring uses each row's final
    non-padding token, including when different prompt lengths share a batch.
    """
    import torch

    from .modeling import option_logits, option_token_ids

    validate_sentiment_dataset(dataset)
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not len(dataset):
        raise ValueError("cannot evaluate an empty dataset")
    option_ids = option_token_ids(tokenizer)
    model.eval()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    previous_padding_side = tokenizer.padding_side
    records = []
    try:
        tokenizer.padding_side = "right"
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                examples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
                prompts = [
                    render_chat_prompt(
                        tokenizer,
                        build_prompt(example["text"], compact_options=compact_options),
                    )
                    for example in examples
                ]
                inputs = tokenizer(
                    prompts,
                    add_special_tokens=False,
                    truncation=False,
                    padding=True,
                    return_tensors="pt",
                )
                scores = option_logits(
                    model,
                    inputs["input_ids"].to(device),
                    inputs["attention_mask"].to(device),
                    option_ids,
                )
                probabilities = torch.softmax(scores.float(), dim=-1).cpu().tolist()
                records.extend(
                    _record(start + offset, example, values)
                    for offset, (example, values) in enumerate(zip(examples, probabilities))
                )
    finally:
        tokenizer.padding_side = previous_padding_side
    return records


def _atomic_json(path: Path, value: Any) -> None:
    """Write a complete JSON file before exposing its final filename."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_evaluation_name(name: str) -> None:
    """Validate output names before expensive inference starts."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("name must be a filename stem containing letters, digits, underscores, dots, or hyphens")


def save_evaluation(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save predictions and metrics, refusing existing outputs by default."""
    validate_evaluation_name(name)
    output_dir = Path(output_dir)
    predictions_path = output_dir / f"{name}.predictions.jsonl"
    metrics_path = output_dir / f"{name}.metrics.json"
    report = {**compute_metrics(records), "name": name, "metadata": metadata or {}}
    predictions_json = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records)
    json.dumps(report, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and (predictions_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"evaluation {name!r} already exists in {output_dir}; choose another name or set overwrite=True")
    with predictions_path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(predictions_json)
    _atomic_json(metrics_path, report)
    return report


def _jev_manifest(dataset: Any, model_name: str) -> dict[str, Any]:
    rows = [{"text": row["text"], "label": int(row["label"])} for row in dataset]
    dataset_bytes = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "examples": len(rows),
        "model": model_name,
        "instruction": INSTRUCTION,
        "criteria": dict(zip(LETTERS, LABEL_NAMES)),
        "label_names": list(LABEL_NAMES),
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {**manifest, "fingerprint": hashlib.sha256(manifest_bytes).hexdigest()}


def _read_jev_checkpoint(path: Path, dataset: Any, model_name: str) -> dict[int, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
                row_id = record["row_id"]
                if not isinstance(row_id, int) or isinstance(row_id, bool) or not 0 <= row_id < len(dataset):
                    raise ValueError("invalid row_id")
                if row_id in records:
                    raise ValueError("duplicate row_id")
                example = dataset[row_id]
                if record["text"] != example["text"] or record["true_label"] != example["label"]:
                    raise ValueError("cached row does not match the evaluation dataset")
                if record["model"] != model_name:
                    raise ValueError("cached model does not match the requested model")
                expected = _record(row_id, example, [record[f"p_{label}"] for label in LABEL_NAMES])
                if record["pred_label"] != expected["pred_label"] or record["api_choice"] not in LETTERS:
                    raise ValueError("cached prediction is inconsistent")
                records[row_id] = record
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"corrupt Jev checkpoint {path}, line {line_number}: {error}") from error
    return records


def evaluate_jev(
    dataset: Any,
    *,
    api_key: str | None,
    checkpoint_path: str | Path,
    model_name: str = "jev-1.13.0",
    client: Any = None,
) -> list[dict[str, Any]]:
    """Resume compatible Jev predictions; never put the API key in artifacts.

    A manifest fingerprints the complete ordered dataset, model, instruction,
    and criteria before the first request. A checkpoint without its manifest
    or with malformed records is rejected instead of silently reused.
    Supplying a client transfers client lifecycle management to the caller.
    """
    validate_sentiment_dataset(dataset)
    if not len(dataset):
        raise ValueError("cannot evaluate an empty dataset")
    checkpoint = Path(checkpoint_path)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
    expected_manifest = _jev_manifest(dataset, model_name)
    if manifest_path.exists():
        try:
            actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"corrupt Jev checkpoint manifest: {manifest_path}") from error
        if actual_manifest != expected_manifest:
            raise ValueError("incompatible Jev checkpoint: dataset, model, instruction, or criteria changed; choose a new checkpoint path")
    elif checkpoint.exists():
        raise ValueError(f"Jev checkpoint is missing its manifest: {manifest_path}; choose a new checkpoint path")

    records = _read_jev_checkpoint(checkpoint, dataset, model_name)
    if len(records) == len(dataset):
        return [records[index] for index in range(len(dataset))]
    if client is None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("an API key is required for unfinished Jev evaluation")
    try:
        from typesafe_sdk import Choice, TypeSafeClient
    except ImportError as error:
        raise ImportError("install the jev extra to run Jev evaluation: pip install -e '.[jev]'") from error

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _atomic_json(manifest_path, expected_manifest)
    manager = nullcontext(client) if client is not None else TypeSafeClient(api_key=api_key)
    with manager as api_client:
        with checkpoint.open("a", encoding="utf-8") as stream:
            for row_id, example in enumerate(dataset):
                if row_id in records:
                    continue
                response = api_client.system_one(
                    model=model_name,
                    state=example["text"],
                    questions={
                        "sentiment": Choice(
                            instructions=INSTRUCTION,
                            criteria=dict(zip(LETTERS, LABEL_NAMES)),
                        )
                    },
                )
                if response.model != model_name:
                    raise ValueError(f"Jev returned model {response.model!r}; expected {model_name!r}")
                answer = response.choices["sentiment"]
                if answer.choice not in LETTERS:
                    raise ValueError(f"Jev returned an invalid choice: {answer.choice!r}")
                record = _record(
                    row_id,
                    example,
                    [answer.probabilities[letter] for letter in LETTERS],
                    api_choice=answer.choice,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records[row_id] = record
    return [records[index] for index in range(len(dataset))]
