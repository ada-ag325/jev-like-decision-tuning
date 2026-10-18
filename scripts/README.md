# Entry points

Install the package with `python -m pip install -e '.[train]'` and run from the repository root.

- `train.py --config CONFIG --output-dir NEW_DIRECTORY`: train a fresh adapter and save a completed run.
- `evaluate.py --run-dir DIRECTORY --backend lora --split validation`: independently reload and evaluate the adapter.
- `evaluate.py --run-dir DIRECTORY --backend base --split validation`: evaluate the corresponding original base model with the saved tokenizer.

Use `--help` for overrides. Existing runs and named reports are not overwritten. See the root README for complete commands and data requirements.
