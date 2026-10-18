"""Model loading, LoRA placement, and option scoring shared by all entry points."""

import json
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import LETTERS

TARGET_GROUPS = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attention_mlp": (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
}
PRECISIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_device_precision(
    device: str | torch.device, precision: str
) -> tuple[torch.device, torch.dtype]:
    """Validate one explicit CPU/CUDA device and its requested precision."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {tuple(PRECISIONS)}")
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    if resolved.type == "cpu":
        if precision != "float32":
            raise ValueError("CPU execution requires precision='float32'")
        return torch.device("cpu"), PRECISIONS[precision]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a GPU runtime or cpu with float32")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    resolved = torch.device("cuda", index)
    if precision == "bfloat16":
        with torch.cuda.device(resolved):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"bfloat16 is not supported on {resolved}")
    return resolved, PRECISIONS[precision]


def prepare_tokenizer(tokenizer: Any) -> Any:
    """Use right padding and an existing EOS token when no pad token exists."""
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must have a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str,
    *,
    revision: str | None = None,
    device: str = "cuda",
    precision: str = "float16",
) -> tuple[Any, Any]:
    """Load a frozen causal LM and tokenizer on one device, without sharding."""
    resolved_device, dtype = resolve_device_precision(device, precision)
    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(model_name, revision=revision)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
    ).to(resolved_device)
    model.requires_grad_(False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    option_token_ids(tokenizer)
    return model, tokenizer


def add_lora(
    model: Any,
    *,
    target_group: str = "attention_mlp",
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> Any:
    """Attach LoRA only to the requested attention/MLP projection modules."""
    if target_group not in TARGET_GROUPS:
        raise ValueError(f"target_group must be one of {tuple(TARGET_GROUPS)}")
    if not isinstance(rank, Integral) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("alpha must be positive and dropout must be in [0, 1)")
    if getattr(model, "peft_config", None):
        raise ValueError("model already contains an adapter; start with a fresh base model")
    target_modules = TARGET_GROUPS[target_group]
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(target_modules) - available)
    if missing:
        raise ValueError(f"base model is missing target modules: {missing}")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=list(target_modules),
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
        ),
    )
    model.config.use_cache = False
    trainable = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"non-LoRA parameter is trainable: {name}")
            parameter.data = parameter.data.float()
            trainable += parameter.numel()
    if not trainable:
        raise ValueError("no trainable LoRA parameters were created")
    return model


def option_token_ids(tokenizer: Any) -> list[int]:
    """Require A, B, and C to each encode as one distinct token."""
    encoded = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError("A, B, and C must each encode as exactly one token")
    result = [int(ids[0]) for ids in encoded]
    if len(set(result)) != len(LETTERS):
        raise ValueError("A, B, and C must have distinct token IDs")
    return result


def option_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    option_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Score A/B/C at each prompt's last non-padding token, returning float32.

    The scalar training target is an option position. It is never passed to the
    model as a causal-language-model label or included in the input sequence.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have the same [batch, length] shape")
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("cannot score empty inputs")
    if torch.any((attention_mask != 0) & (attention_mask != 1)):
        raise ValueError("attention_mask must contain only zero and one")
    lengths = attention_mask.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("every input must contain at least one non-padding token")
    if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
        raise ValueError("option scoring requires right-padded attention masks")
    ids = torch.as_tensor(option_ids, dtype=torch.long, device=input_ids.device)
    if ids.ndim != 1 or ids.numel() != 3 or ids.unique().numel() != 3 or torch.any(ids < 0):
        raise ValueError("option_ids must contain three distinct nonnegative token IDs")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if torch.any(ids >= outputs.logits.shape[-1]):
        raise ValueError("option token ID exceeds model vocabulary size")
    rows = torch.arange(input_ids.shape[0], device=outputs.logits.device)
    positions = (lengths - 1).to(device=outputs.logits.device, dtype=torch.long)
    return outputs.logits[rows, positions][:, ids.to(outputs.logits.device)].float()


def collate_examples(tokenizer: Any, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Right-pad model inputs, keeping any scalar option targets separate."""
    if tokenizer.padding_side != "right":
        raise ValueError("tokenizer.padding_side must be 'right'")
    if not examples:
        raise ValueError("cannot collate an empty batch")
    features = [
        {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}
        for row in examples
    ]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    has_labels = ["labels" in row for row in examples]
    if any(has_labels):
        if not all(has_labels):
            raise ValueError("all examples in a labeled batch must contain labels")
        labels = [row["labels"] for row in examples]
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or value not in (0, 1, 2)
            for value in labels
        ):
            raise ValueError("labels must be scalar option positions: 0, 1, or 2")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch


def save_adapter(model: Any, tokenizer: Any, run_dir: str | Path) -> Path:
    """Save only adapter weights plus the tokenizer inside a new run directory."""
    if not getattr(model, "peft_config", None):
        raise ValueError("save_adapter requires a PEFT model")
    adapter_dir = Path(run_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir


def load_run(
    run_dir: str | Path,
    *,
    device: str | None = None,
    precision: str | None = None,
) -> tuple[Any, Any]:
    """Reload a completed run using its base model revision and saved tokenizer."""
    run_dir = Path(run_dir)
    with (run_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete":
        raise ValueError("run is not complete; reload requires metadata.status == 'complete'")
    with (run_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    adapter_dir = run_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing saved adapter: {adapter_dir}")
    revision = metadata.get("resolved_model_revision") or config.get("model_revision")
    model, _ = load_base_model(
        config["model_name"],
        revision=revision,
        device=config.get("device", "cuda") if device is None else device,
        precision=config.get("precision", "float16") if precision is None else precision,
    )
    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(str(adapter_dir), local_files_only=True)
    )
    option_token_ids(tokenizer)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.requires_grad_(False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    return model, tokenizer
