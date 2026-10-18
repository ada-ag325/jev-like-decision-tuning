# Training and evaluation workflow

## Responsibilities

- `config.py`: validates a flat JSON configuration and resolves local paths.
- `modeling.py`: loads a base model, attaches one of three LoRA target groups, scores A/B/C, saves adapters/tokenizers, and reloads completed runs.
- `training.py`: right-padded batches, seeded shuffling, option cross-entropy, optimizer steps, and training statistics.
- `evaluation.py`: batched Qwen predictions, optional resumable Jev requests, and prediction/report export.
- `metrics.py`: metrics with a fixed negative/neutral/positive class order.
- `experiment.py`: connects the modules and manages a single run directory.
- `scripts/train.py` and `scripts/evaluate.py`: independent command-line entry points.

The notebook calls these same modules; it no longer contains its own training loop or evaluation implementation.

## Preserved training recipe

The default combined configuration uses Qwen3-1.7B, 3,000 examples per class, three seeds set to 42, batch size 4, one epoch, AdamW with learning rate 5e-5 and weight decay 0.01, gradient clipping at 1.0, LoRA rank 8 / alpha 16 / dropout 0.05, and CUDA float16 autocast with GradScaler. Trainable LoRA weights stay float32. Training uses only the A/B/C logits at each prompt's last non-padding token.

`data_seed` fixes which examples are selected. `option_seed` fixes the per-row option permutations. `train_seed` is applied before adapter initialization and reset for dropout and batch shuffling. Each experiment begins from a fresh base model. Same rank does not imply equal trainable-parameter budgets across target groups.

## Save and reload

The CLI creates a new output directory, stores configuration and metadata, then trains and saves `adapter/`. It writes the training log before marking the run complete. Reusing an existing run directory fails rather than overwriting results.

`load_run` requires a complete run, loads the base model at the recorded revision when available, loads the saved tokenizer, attaches the saved adapter in inference mode, and freezes all parameters. `evaluate_experiment` performs that reload on each call. The `base` comparator loads the same original base model and uses the same saved tokenizer and evaluation prompts.

The adapter depends on the original base model; it is not a full standalone copy of its weights. Runtime device/precision and the dataset location may be overridden during evaluation. Such overrides are included in report metadata. Changing hardware or precision can introduce numerical differences.

There is no optimizer-state resume in this version. An interrupted run should use a new output directory when restarted. The saved `training_samples.jsonl` records source IDs, class labels, option targets, and prompt hashes; full dataset fingerprints and a manifest hash are recorded in metadata.

## Evaluation details

All standard local evaluation uses the spaced prompt format, fixed negative/neutral/positive option order, the non-thinking chat template, `add_special_tokens=False`, and right padding. Batched scoring selects each row's final valid token, not the padded final column. Ground-truth labels are used only after scoring.

Evaluation writes a JSONL prediction file and a JSON report with accuracy, Macro-F1, Macro Recall, per-class precision/recall/F1/support, and a confusion matrix. Limited evaluations are explicitly named with their row limit. The full `train` split is different from the sampled training subset; do not interpret full-train scores as in-sample scores for all rows.

Jev evaluation sends raw text and structured choices, not Qwen's chat prompt. Its cache identity includes the full ordered dataset, model version, instruction, and criteria. A cache mismatch or corrupt record fails explicitly. Tests use mocked API responses; no live API requests were made during this refactor.

## Changes relative to the pilot

The executed pilot is preserved in `notebooks/archive/jev_20261018_pilot.ipynb`. The new workflow unifies training and evaluation output locations, permits independent adapter reloading, exposes the three target groups, and adds Macro Recall to the reports. Evaluation now explicitly uses `add_special_tokens=False`, matching training, and supports batches. The pilot's initial compact-format 100-example baseline is not part of the new default workflow; its prompt builder remains available.

The package pins the tested Transformers/PEFT/Accelerate versions instead of installing unbounded latest versions. These environment and evaluation changes mean the archived pilot is preliminary evidence, not an automatically reproduced score for the new pipeline. Run the fixed pipeline on CUDA before claiming full-model reproduction.

## Verification

Offline tests use a real tiny randomly initialized Qwen3 and a local tokenizer. They check that only the selected LoRA parameters change, padding and target mapping are correct, trained probabilities survive save/reload, and the training/evaluation CLIs work in separate Python processes. Model, data, and API downloads are not required for these tests.

For a real GPU check, train a new combined run, evaluate both base and adapter on validation, and retain its exact configuration and metadata. Restart the Colab runtime and evaluate that same run again under a new report name. Compare predictions before starting the planned ablation matrix.
