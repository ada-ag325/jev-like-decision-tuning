"""Explicit, validated configuration for a single training experiment."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: str
    model_name: str = "Qwen/Qwen3-1.7B"
    model_revision: str | None = None
    target_group: str = "attention_mlp"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    per_class: int = 3000
    data_seed: int = 42
    option_seed: int = 42
    train_seed: int = 42
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cuda"
    precision: str = "float16"
    log_every: int = 50

    def __post_init__(self):
        for name in ("data_path", "model_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ValueError("model_revision must be a non-empty string or null")
        if self.target_group not in ("attention", "mlp", "attention_mlp"):
            raise ValueError("target_group must be attention, mlp, or attention_mlp")
        for name in ("rank", "alpha", "per_class", "epochs", "batch_size", "log_every"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("data_seed", "option_seed", "train_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.weight_decay, (int, float)) or not 0 <= self.weight_decay < float("inf"):
            raise ValueError("weight_decay must be finite and non-negative")
        if not isinstance(self.device, str) or not (
            self.device in ("cpu", "cuda")
            or (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError("device must be cpu, cuda, or cuda:N")
        if self.precision not in ("float16", "float32", "bfloat16"):
            raise ValueError("precision must be float16, float32, or bfloat16")
        if self.device == "cpu" and self.precision != "float32":
            raise ValueError("CPU execution requires precision=float32")

    def to_dict(self):
        return asdict(self)

    def resolve_paths(self):
        """Make a run portable across working directories on the same machine."""
        values = self.to_dict()
        values["data_path"] = str(Path(self.data_path).expanduser().resolve())
        model_path = Path(self.model_name).expanduser()
        if model_path.is_dir():
            values["model_name"] = str(model_path.resolve())
        return ExperimentConfig(**values)


def load_config(path: str | Path, **overrides) -> ExperimentConfig:
    """Resolve relative data/model paths from the current working directory."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("configuration must be a JSON object")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig(**values).resolve_paths()
