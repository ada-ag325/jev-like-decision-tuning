# Data processing and prompt construction

## Data flow

```text
Saved TweetEval DatasetDict
  -> load_sentiment_dataset(path)
  -> sample_balanced(ds["train"], per_class=3000, seed=42)
  -> format_training_dataset(train, seed=42)
  -> tokenize_dataset(formatted, tokenizer)
  -> training.py DataLoader and LoRA training
```

The loader requires `train`, `validation`, and `test` splits with `text` and `label` columns. Labels must use the ClassLabel names `negative`, `neutral`, `positive`, in that order. Loading reads existing local data; it does not download or resplit it.

`sample_balanced` retains the notebook's class-by-class filter, seeded shuffle, selection, concatenation, and final seeded shuffle. Its default is 3,000 examples per class (9,000 total). It raises an error if a class has too few rows. `sample_examples` implements the initial 100-example validation sample.

## Class IDs and answer positions

The original `label` is a class ID: 0 = negative, 1 = neutral, 2 = positive. The `order` argument lists which class is displayed at each answer position.

For `order=(2, 0, 1)`, the options are:

```text
A: positive
B: negative
C: neutral
```

A positive example still has `label=2`, but its `target` is now 0 and `target_letter` is `A`. `format_example` returns `prompt`, `target`, and `target_letter`. When applied to a dataset, the original `text` and `label` columns remain available.

`format_training_dataset` creates a fresh `random.Random(seed)` per call and shuffles options sequentially, once per row. Repeating it with the same seed and row order reproduces the same prompts and targets. Reordering rows changes the sequence's assignment to examples; this is not per-example-ID randomness.

## Prompt variants preserved

The archived pilot used two option formats. Both remain available through the prompt helpers:

- The initial 100-example baseline uses `build_prompt(text, compact_options=True)`, producing `A:negative`.
- Training and full Qwen evaluation use the default spacing, producing `A: negative`.
- The introductory display uses `format_example(..., compact_options=True, include_reply_instruction=False)` to show the original prompt ending at `Answer:`.

Model-facing prompts end with `Answer:\nReply with only A, B, or C.`. `render_chat_prompt` applies the tokenizer's chat template with `add_generation_prompt=True` and `enable_thinking=False`.

The Jev API continues to receive raw text plus a structured `Choice`, using the same instruction and fixed letter-to-class mapping. It does not receive the Qwen chat prompt.

## Tokenization and supervision

`tokenize_example` encodes the chat-formatted prompt with `add_special_tokens=False` and `truncation=False`, then stores `target` as the scalar `labels` value. The correct answer is never appended to the input.

`tokenize_dataset` removes the original formatted columns, leaving tokenizer outputs and `labels`. Right padding, batch collation, and last-valid-token scoring are implemented in `modeling.py`; `training.py` applies the cross-entropy over A/B/C logits.

## Refactor boundaries

The prompt strings, sampling defaults, option shuffle sequence, and training tokenization settings are preserved. Dataset schema checks and clear insufficient-sample errors are new. The package must now be installed before importing notebook helpers.

Comments, progress messages, and report column names are in English. Downstream consumers of the old report column names must update their field lookups. The current notebook calls the extracted modules; the earlier executed notebook is preserved under `notebooks/archive/`.

Tests use synthetic datasets, a recording tokenizer, and a tiny local Qwen3 model, so they do not establish full-model accuracy or GPU/API runtime compatibility. No model training or Jev requests are needed to run them.
