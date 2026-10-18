# Notebooks

`jev.ipynb` is the Colab entry point for the shared train/save/reload/evaluate modules. Set the repository and dataset paths, train a new run, then independently load it for evaluation. Test and Jev API evaluations are disabled by default.

The executed notebook from before the training/evaluation extraction is preserved unchanged in `archive/jev_20261018_pilot.ipynb`, including its saved results. It is a historical snapshot and still contains the old inline implementations. Use the current notebook for new runs.

After restarting Colab, run setup and set `RUN_DIR` to a completed run's exact saved path. No retraining is needed for evaluation. The setup cell installs the package and adds its source path to the live kernel.
