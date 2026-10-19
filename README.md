# Jev-Inspired Decision Tuning with Qwen3

A small research project investigating **whether task-specific post-training can improve a small language model's decisions among predefined candidates, and how the adapted model compares with Jev on the same task**. We adapt Qwen3-1.7B with LoRA on TweetEval sentiment. An initial experiment directly compares the unchanged Qwen checkpoint, its LoRA-adapted version, and **Jev 1.13.0** on the same validation samples.

Across three training seeds, all configurations improve validation performance over the base model. Attention-only achieves the highest mean Macro Recall with the fewest trainable parameters, although the differences between configurations are small.

## Research motivation

Many software applications require frequent judgments among a finite set of candidates. For these tasks, generating a full answer can introduce unnecessary output-decoding and parsing overhead. [Jev](https://docs.typesafe.ai/introduction), which emphasizes structured decisions and probabilities that software can use directly, motivates our exploration of a language-model interface that scores candidates without generating a full response.

Our central questions are: **can task-specific supervision improve an existing small language model's judgments among predefined candidates, and how does its task performance compare with Jev?** Without access to Jev's complete implementation and with limited compute, we investigate these questions using LoRA fine-tuning of [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). We restrict the study to sentiment classification with three fixed categories, making the task and its evaluation explicit.

Qwen3-1.7B has already undergone pretraining and post-training. Here, **base model** means the released checkpoint before our additional task-specific fine-tuning. We measure adaptation gains against that unchanged checkpoint, with Jev as an external task-level reference in the pilot. The LoRA placement ablation is a follow-up analysis of how attention-only, MLP-only, and joint adaptation affect classification quality and trainable-parameter cost.

## Method

The model receives a text and three candidate sentiment labels associated with A, B, and C. We extract those three token logits at the final prompt position and compute:

```text
candidate probabilities = softmax([logit_A, logit_B, logit_C])
loss = cross_entropy(candidate logits, correct option position)
```

Only LoRA parameters are trained; the original model weights remain frozen. The method reuses the language-model output head rather than adding a separate classification head. Correct answers are kept outside the input. Candidate order is shuffled once per training example, with the target remapped accordingly; validation uses a fixed negative/neutral/positive order.

Both the base and fine-tuned models use the same single-forward-pass candidate-scoring interface. Their comparison measures task adaptation; it does not demonstrate that LoRA accelerates inference. Inference latency and serving cost have not been benchmarked in this study.

Training uses the dataset's sentiment labels. This project does not reproduce Jev's internal implementation or RLCD training method, and does not train on Jev outputs. The Jev comparison is preserved in the [pilot notebook](notebooks/archive/jev_20261018_pilot.ipynb) and reported below, separately from the later three-seed ablation.

## Experimental setup

| Component | Setting |
|---|---|
| Base model | Qwen3-1.7B; the same base-model revision across runs |
| Dataset | TweetEval sentiment: negative, neutral, positive |
| Training data | 9,000 examples: 3,000 per class from the training split |
| Validation data | Full validation split: 2,000 examples |
| Training seeds | 42, 43, 44 for each configuration: nine runs total |
| Data and option-order seeds | Both fixed at 42 across all runs |
| LoRA | Rank 8, alpha 16, dropout 0.05 |
| Optimization | AdamW; learning rate 5e-5; weight decay 0.01; gradient clipping 1.0 |
| Training budget | One epoch; batch size 4; 2,250 steps per run |
| Precision | CUDA float16 autocast; float32 trainable LoRA weights |

Attention adapters target `q_proj`, `k_proj`, `v_proj`, and `o_proj`. MLP adapters target `gate_proj`, `up_proj`, and `down_proj`. The joint configuration targets both sets. Each run starts from the original base model.

**Macro Recall is the primary metric**, following the [official TweetEval sentiment evaluation](https://github.com/cardiffnlp/tweeteval/blob/main/evaluation_script.py). Macro-F1 and accuracy are reported alongside it. This is a fixed-rank comparison; the configurations have different trainable-parameter budgets.

## Validation results

### Direct comparison with Jev: initial pilot

The [executed pilot notebook](notebooks/archive/jev_20261018_pilot.ipynb) evaluates all three models on the same **2,000 validation examples**. The LoRA model uses joint attention + MLP adaptation with training seed 42. Qwen is scored through A/B/C token logits; Jev is queried through its API with the same sentiment instruction and class meanings.

| Model | Macro Recall ↑ | Macro-F1 ↑ | Accuracy ↑ |
|---|---:|---:|---:|
| Base Qwen3-1.7B | 0.5999 | 0.5039 | 0.5510 |
| Qwen3-1.7B + LoRA (attention + MLP) | 0.7239 | 0.6434 | 0.6645 |
| Jev 1.13.0 | 0.7168 | 0.6517 | 0.6660 |

Macro Recall is recomputed from the notebook's saved confusion matrices; Macro-F1 and accuracy match its saved summary. Post-training improves Qwen's accuracy from **55.10% to 66.45%**, close to Jev's **66.60%** in this run. LoRA has higher Macro Recall, while Jev has higher Macro-F1 and accuracy.

This is a **single-run task-level comparison**, with different model architectures and training histories; parameter budgets and inference costs are not matched. It does not establish equivalent general capability or superiority over Jev. The pilot predates the current workflow and is reported separately from the following ablation; its Qwen scores should not be substituted for the later runs' scores.

### LoRA placement ablation

Results from the completed Colab ablation summary on **18 October 2026**. LoRA scores are **mean ± sample standard deviation over three training seeds**; all metrics are on a 0–1 scale and rounded to four decimal places. The base model was evaluated once under the same validation protocol.

| Configuration | Trainable parameters | Macro Recall ↑ | Macro-F1 ↑ | Accuracy ↑ |
|---|---:|---:|---:|---:|
| Base Qwen3-1.7B | 0 | 0.6014 | 0.5063 | 0.5530 |
| Attention-only | **3,211,264** | **0.7370 ± 0.0111** | **0.6854 ± 0.0237** | **0.6993 ± 0.0203** |
| MLP-only | 5,505,024 | 0.7348 ± 0.0099 | 0.6813 ± 0.0237 | 0.6952 ± 0.0216 |
| Attention + MLP | 8,716,288 | 0.7330 ± 0.0099 | 0.6809 ± 0.0252 | 0.6943 ± 0.0238 |

These values were transcribed from the completed Colab summary. The underlying per-run prediction files and reports are not yet bundled with this repository.

### Findings and limitations

- **Adaptation helps on this validation set.** Mean Macro Recall improves by approximately 13 percentage points over the base model for all three configurations.
- **Joint adaptation shows no additional mean benefit under this recipe.** Attention-only uses about 63% fewer trainable parameters than the joint configuration while obtaining slightly higher mean scores.
- **A reliable ranking between placements is not established.** Mean Macro Recall differs by only 0.22–0.40 percentage points between attention-only and the alternatives, with only three training seeds. The comparison does not isolate placement from parameter budget.
- **Neutral classification remains a weakness.** Mean neutral recall is 0.5539 for attention-only, 0.5393 for MLP-only, and 0.5501 for joint adaptation. Error analysis and candidate-order robustness evaluation remain to be completed.

The findings are limited to one model, one dataset, and one training recipe. They do not establish general superiority over Jev, robustness to candidate permutations, or calibrated probabilities.

## Test-set results

**Pending.** The complete test-set comparison of the nine saved LoRA models and the base model will be added after evaluation is finished. No test-set conclusion is claimed for this ablation. Earlier exploratory test scores in the pilot notebook are not substituted for this comparison.

## Future outlook

A future system could use a small candidate-scoring model for frequent, well-scoped judgments and route complex cases to a larger reasoning model. This collaboration is a possible application direction, not an implemented or evaluated contribution of the current project. It would require evaluating the routing policy, uncertainty estimates, and end-to-end quality, latency, and cost.

## Reproduce the workflow

Use Python 3.10+ and a CUDA GPU for the standard configuration. From the repository root:

```bash
python -m pip install -e '.[train,dev]'
python -m pytest -q

python scripts/train.py \
  --config configs/attention.json \
  --seed 42 \
  --data-path /absolute/path/to/tweet_eval_sentiment \
  --output-dir outputs/attention_seed42

python scripts/evaluate.py \
  --run-dir outputs/attention_seed42 \
  --split validation --backend lora

python scripts/evaluate.py \
  --run-dir outputs/attention_seed42 \
  --split validation --backend base
```

The data directory must contain an existing Hugging Face DatasetDict saved with `save_to_disk`, including `train`, `validation`, and `test` splits. Data and model weights are not bundled. Repeat with `configs/mlp.json` and `configs/attention_mlp.json`, using seeds 42, 43, and 44 and a new output directory for each run. For exact replication, use the recorded model revision and run configuration rather than an unpinned Hub revision.

For Colab, open [the workflow notebook](notebooks/jev.ipynb) and set the repository and dataset paths. Saved run directories contain the adapter, tokenizer, configuration, environment metadata, sample manifest, training log, predictions, and metrics. Evaluation reloads the saved model in a separate process; no retraining is needed after a runtime restart. Existing runs and named evaluations are not overwritten.

The Colab smoke check reproduced identical predictions and probabilities after a runtime restart. Offline tests also cover LoRA-only updates, adapter save/reload parity, and independent training/evaluation processes. See [workflow details](docs/training-and-evaluation.md) and [data/prompt contracts](docs/data-and-prompts.md) for implementation and artifact formats.
