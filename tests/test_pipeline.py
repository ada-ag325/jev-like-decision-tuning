"""Exercise the real training/reload command-line boundary entirely offline."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from jev_decision_tuning.config import ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_rejects_typos_and_invalid_cpu_precision(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"data_path": "data/example", "batch_szie": 4}))
    with pytest.raises(TypeError, match="batch_szie"):
        load_config(path)
    with pytest.raises(ValueError, match="float32"):
        ExperimentConfig(data_path="data", device="cpu", precision="float16")
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig(data_path="data", learning_rate=float("nan"))


def test_all_placement_configs_differ_only_in_target_group():
    configs = [json.loads((ROOT / "configs" / f"{group}.json").read_text()) for group in ("attention", "mlp", "attention_mlp")]
    assert {config.pop("target_group") for config in configs} == {"attention", "mlp", "attention_mlp"}
    assert configs[0] == configs[1] == configs[2]


def test_direct_training_api_records_absolute_paths_and_failed_status(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    from jev_decision_tuning.experiment import train_experiment

    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    config = ExperimentConfig(data_path="missing-data", model_name="base", device="cpu", precision="float32")
    with pytest.raises(FileNotFoundError):
        train_experiment(config, "failed-run")
    saved = json.loads((tmp_path / "failed-run/config.json").read_text())
    assert saved["data_path"] == str(tmp_path / "missing-data")
    assert saved["model_name"] == str(tmp_path / "base")
    metadata = json.loads((tmp_path / "failed-run/metadata.json").read_text())
    assert metadata["status"] == "failed"


def test_train_then_evaluate_in_new_processes(tmp_path):
    pytest.importorskip("peft")
    from test_training import write_tiny_base
    from jev_decision_tuning.evaluation import evaluate_model
    from jev_decision_tuning.modeling import load_run

    base_path = write_tiny_base(tmp_path / "tiny-base")
    features = Features({"text": Value("string"), "label": ClassLabel(names=["negative", "neutral", "positive"])})
    dataset = Dataset.from_dict(
        {"text": ["bad", "okay", "good", "bad bad", "okay okay", "good good"], "label": [0, 1, 2, 0, 1, 2]},
        features=features,
    )
    data_path = tmp_path / "dataset"
    DatasetDict({split: dataset for split in ("train", "validation", "test")}).save_to_disk(data_path)
    config = ExperimentConfig(
        data_path=str(data_path), model_name=str(base_path),
        per_class=2, rank=2, alpha=4, dropout=0.0,
        device="cpu", precision="float32", batch_size=2, log_every=1,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    run_dir = tmp_path / "run"
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }

    def command(script, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90,
        )

    trained = command("train.py", "--config", config_path, "--output-dir", run_dir)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["training_manifest_sha256"]
    assert (run_dir / "adapter" / "adapter_model.safetensors").is_file()
    log = json.loads((run_dir / "training_log.json").read_text())
    assert log["samples"] == 6
    assert len(log["history"]) == 3
    assert log["skipped_updates"] == 0
    assert len((run_dir / "training_samples.jsonl").read_text().splitlines()) == 6

    evaluated = command("evaluate.py", "--run-dir", run_dir, "--split", "test", "--batch-size", 2)
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    predictions_path = run_dir / "evaluation/test_lora.predictions.jsonl"
    records = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert [row["row_id"] for row in records] == list(range(6))
    assert [row["true_label"] for row in records] == dataset["label"]
    report = json.loads((run_dir / "evaluation/test_lora.metrics.json").read_text())
    assert report["examples"] == 6
    assert "macro_recall" in report
    assert report["metadata"]["training_manifest_sha256"] == metadata["training_manifest_sha256"]

    # A separate process reload must reproduce the same saved model's scores.
    model, tokenizer = load_run(run_dir)
    expected = evaluate_model(model, tokenizer, dataset, batch_size=2)
    for actual, reference in zip(records, expected):
        for label in ("negative", "neutral", "positive"):
            assert actual[f"p_{label}"] == pytest.approx(reference[f"p_{label}"], abs=1e-6)

    base = command("evaluate.py", "--run-dir", run_dir, "--split", "test", "--backend", "base")
    assert base.returncode == 0, base.stdout + base.stderr
    assert (run_dir / "evaluation/test_base.metrics.json").is_file()
    original_bytes = predictions_path.read_bytes()
    duplicate = command("evaluate.py", "--run-dir", run_dir, "--split", "test")
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert predictions_path.read_bytes() == original_bytes
    invalid_name = command("evaluate.py", "--run-dir", run_dir, "--name", "invalid name")
    assert invalid_name.returncode != 0
    assert "filename stem" in invalid_name.stderr
    duplicate_train = command("train.py", "--config", config_path, "--output-dir", run_dir)
    assert duplicate_train.returncode != 0
    assert (run_dir / "metadata.json").read_text() == json.dumps(metadata, indent=2) + "\n"
