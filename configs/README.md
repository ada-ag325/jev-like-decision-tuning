# Experiment configurations

The JSON files `attention.json`, `mlp.json`, and `attention_mlp.json` differ only in `target_group`. All other defaults preserve the pilot training recipe. The data path is relative to the working directory; the CLI resolves it to an absolute path before saving the run configuration.

Use `--data-path` to point at an existing saved TweetEval DatasetDict. `--seed` changes only `train_seed`; data sampling and option permutation seeds remain fixed. Defaults use CUDA float16. For tiny CPU tests, set both device to `cpu` and precision to `float32`.

Model revisions may be pinned explicitly. Otherwise the resolved Hub revision is recorded when available. Never put API credentials in configuration files.
