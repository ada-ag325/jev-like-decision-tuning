"""Orchestrate training artifacts and evaluation in independent processes."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig


def write_json(path, value):
    """Atomically replace a JSON artifact within its destination directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment_metadata():
    versions = {}
    for package in ("torch", "transformers", "peft", "datasets", "accelerate", "scikit-learn", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "packages": versions, "git_commit": commit, "git_dirty": dirty}


def train_experiment(config: ExperimentConfig, run_dir: str | Path) -> Path:
    """Train one fresh adapter and save all artifacts under one new run directory."""
    import torch
    from transformers import set_seed

    from .data import load_sentiment_dataset, sample_balanced, format_training_dataset, tokenize_dataset
    from .modeling import load_base_model, add_lora, save_adapter
    from .training import train_model

    config = config.resolve_paths()
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config.to_dict())
    metadata = {
        "status": "training",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(),
        "loss": "cross_entropy_over_A_B_C",
    }
    write_json(run_dir / "metadata.json", metadata)
    try:
        datasets = load_sentiment_dataset(config.data_path)
        metadata["data_splits"] = {
            name: {"rows": len(split), "fingerprint": split._fingerprint}
            for name, split in datasets.items()
        }
        train = datasets["train"]
        if "source_row_id" in train.column_names:
            raise ValueError("source_row_id is reserved for the sampling manifest")
        train = train.add_column("source_row_id", list(range(len(train))))
        selected = sample_balanced(train, per_class=config.per_class, seed=config.data_seed)
        formatted = format_training_dataset(selected, seed=config.option_seed)
        manifest_path = run_dir / "training_samples.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in formatted:
                record = {
                    "source_row_id": row["source_row_id"],
                    "label": row["label"],
                    "target": row["target"],
                    "target_letter": row["target_letter"],
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                }
                handle.write(json.dumps(record) + "\n")
        metadata["training_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        # Seed before adapter initialization, then reset training RNG in train_model.
        set_seed(config.train_seed)
        model, tokenizer = load_base_model(
            config.model_name, revision=config.model_revision,
            device=config.device, precision=config.precision,
        )
        metadata["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
        metadata["device"] = str(model.device)
        metadata["hardware"] = torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else platform.machine()
        metadata["chat_template_sha256"] = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        model = add_lora(
            model, target_group=config.target_group, rank=config.rank,
            alpha=config.alpha, dropout=config.dropout,
        )
        metadata["trainable_parameter_names"] = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        write_json(run_dir / "metadata.json", metadata)
        tokenized = tokenize_dataset(formatted, tokenizer)
        log = train_model(
            model, tokenizer, tokenized, epochs=config.epochs,
            batch_size=config.batch_size, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, seed=config.train_seed,
            max_grad_norm=config.max_grad_norm, precision=config.precision,
            log_every=config.log_every,
        )
        save_adapter(model, tokenizer, run_dir)
        write_json(run_dir / "training_log.json", log)
        metadata.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        write_json(run_dir / "metadata.json", metadata)
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(run_dir / "metadata.json", metadata)
        raise
    return run_dir


def evaluate_experiment(
    run_dir: str | Path, *, split="validation", backend="lora",
    data_path=None, batch_size=4, device=None, precision=None, limit=None,
    name=None,
):
    """Load a saved run independently and write aligned predictions and metrics."""
    from .data import load_sentiment_dataset
    from .evaluation import evaluate_model, save_evaluation, validate_evaluation_name
    from .modeling import load_base_model, load_run

    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    if backend not in ("base", "lora"):
        raise ValueError("backend must be base or lora")
    run_dir = Path(run_dir).expanduser().resolve()
    config = ExperimentConfig(**json.loads((run_dir / "config.json").read_text()))
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("the run is incomplete; evaluate only a successfully saved run")
    dataset_path = str(Path(data_path or config.data_path).expanduser().resolve())
    dataset = load_sentiment_dataset(dataset_path)[split]
    if limit is not None:
        if type(limit) is not int or not 0 < limit <= len(dataset):
            raise ValueError(f"limit must be between 1 and {len(dataset)}")
        dataset = dataset.select(range(limit))
    name = name or f"{split}_{backend}" + (f"_first{limit}" if limit else "")
    # Reject duplicate outputs before loading a large model or doing inference.
    validate_evaluation_name(name)
    output_dir = run_dir / "evaluation"
    if any((output_dir / f"{name}.{suffix}").exists() for suffix in ("predictions.jsonl", "metrics.json")):
        raise FileExistsError(f"evaluation {name!r} already exists; choose another --name")
    device = device or config.device
    precision = precision or config.precision
    if backend == "lora":
        model, tokenizer = load_run(run_dir, device=device, precision=precision)
    else:
        # Use the saved tokenizer even for the base comparator.
        from transformers import AutoTokenizer
        model, _ = load_base_model(
            config.model_name,
            revision=metadata.get("resolved_model_revision") or config.model_revision,
            device=device, precision=precision,
        )
        tokenizer = AutoTokenizer.from_pretrained(run_dir / "adapter")
        tokenizer.padding_side = "right"
    records = evaluate_model(model, tokenizer, dataset, batch_size=batch_size)
    context = {
        "backend": backend, "run_dir": str(run_dir), "split": split,
        "data_path": dataset_path, "dataset_fingerprint": dataset._fingerprint,
        "limit": limit, "device": device, "precision": precision,
        "model_name": config.model_name,
        "model_revision": metadata.get("resolved_model_revision") or config.model_revision,
        "training_manifest_sha256": metadata.get("training_manifest_sha256"),
        "environment": environment_metadata(),
    }
    return save_evaluation(records, output_dir, name=name, metadata=context)
