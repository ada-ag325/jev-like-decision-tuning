# Notebooks

`jev.ipynb` contains the Colab workflow for sentiment data preparation, base-model evaluation, LoRA training, and Qwen / Jev comparison.

Data and prompt helpers are imported from `src/jev_decision_tuning`. The first cell mounts Google Drive and installs the package from a checkout specified by `REPO_ROOT`. Edit this path for your environment; the notebook requires the repository alongside it.

The original notebook was imported with execution outputs, execution counts, and associated widget state removed. Its data and prompt cells have since been refactored, and comments, messages, and report columns are now in English. Training and API results have not been reproduced during this refactor.
