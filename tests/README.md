# Tests

Run `python -m pytest` after installing `python -m pip install -e '.[dev]'` from the repository root.

Tests cover all option permutations, exact prompt text, seeded sampling and option shuffling, dataset schema checks, prompt-only tokenization, and notebook validity. They use small synthetic Hugging Face datasets and a recording tokenizer; they do not download models or call Jev.
