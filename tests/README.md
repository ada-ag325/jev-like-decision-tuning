# Tests

Run `python -m pytest -q` after installing `python -m pip install -e '.[train,dev]'` from the repository root.

The suite covers data/prompt contracts, option targets, batched scoring, metrics, LoRA target placement, frozen base weights, save/reload probability parity, and independent training/evaluation CLI processes. A real tiny randomly initialized Qwen3 model and synthetic datasets keep checks offline and CPU-friendly. Jev tests mock API responses.

No pretrained model downloads, production training, or live Jev requests are made. The tiny-model tests establish workflow correctness, not full-model CUDA compatibility or classification quality.
