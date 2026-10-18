"""Train LoRA with cross-entropy over the next-token A/B/C option scores."""

import time
from functools import partial
from numbers import Integral
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import set_seed

from .modeling import collate_examples, option_logits, option_token_ids, resolve_device_precision


def train_model(
    model: Any,
    tokenizer: Any,
    train_tokenized: Any,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    max_grad_norm: float = 1.0,
    precision: str = "float16",
    log_every: int = 50,
) -> dict[str, Any]:
    """Train only LoRA parameters and return a JSON-serializable training log.

    Each call starts a fresh optimizer and seeded shuffle. Checkpoints are saved
    by the caller after training, not used for resuming an interrupted optimizer.
    """
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("log_every", log_every)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(train_tokenized) == 0:
        raise ValueError("training dataset must not be empty")
    if learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("learning_rate/max_grad_norm must be positive and weight_decay nonnegative")
    if tokenizer.padding_side != "right":
        raise ValueError("tokenizer.padding_side must be 'right'")
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise ValueError("training requires all parameters on one CPU or CUDA device")
    device, dtype = resolve_device_precision(devices.pop(), precision)
    trainable = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"non-LoRA parameter is trainable: {name}")
            parameter.data = parameter.data.float()
            trainable.append(parameter)
    if not trainable:
        raise ValueError("model has no trainable LoRA parameters")
    set_seed(seed)
    loader = DataLoader(
        train_tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_examples, tokenizer),
        generator=torch.Generator().manual_seed(seed),
    )
    option_ids = option_token_ids(tokenizer)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_cuda and precision == "float16", init_scale=1024
    )
    model.config.use_cache = False
    model.train()
    history = []
    skipped_updates = 0
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    step = 0
    for epoch in range(1, epochs + 1):
        window_losses = []
        for batch_number, batch in enumerate(loader, start=1):
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=use_cuda and precision != "float32",
            ):
                scores = option_logits(model, input_ids, attention_mask, option_ids)
                loss = F.cross_entropy(scores, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at batch {step}; stopping training")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, max_grad_norm, error_if_nonfinite=not scaler.is_enabled()
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            skipped_updates += int(skipped)
            loss_value = loss.item()
            history.append({
                "step": step,
                "epoch": epoch,
                "loss": loss_value,
                "update_skipped": skipped,
            })
            window_losses.append(loss_value)
            if batch_number % log_every == 0 or batch_number == len(loader):
                mean_loss = sum(window_losses) / len(window_losses)
                print(
                    f"Epoch {epoch}/{epochs}, batch {batch_number}/{len(loader)} "
                    f"| Mean loss: {mean_loss:.4f}"
                )
                window_losses.clear()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "samples": len(train_tokenized),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "loss": "cross_entropy_over_A_B_C",
        "history": history,
        "elapsed_seconds": elapsed,
        "skipped_updates": skipped_updates,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if use_cuda else 0.0,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
