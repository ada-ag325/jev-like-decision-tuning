"""Train one configured experiment; run from a checkout after package installation."""

import argparse

from jev_decision_tuning.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--output-dir", required=True, help="New, non-existing experiment directory")
    parser.add_argument("--data-path", help="Override the local DatasetDict directory")
    parser.add_argument("--seed", type=int, help="Override only the training seed")
    parser.add_argument("--device", help="cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=("float16", "float32", "bfloat16"))
    args = parser.parse_args()
    config = load_config(
        args.config, data_path=args.data_path, train_seed=args.seed,
        device=args.device, precision=args.precision,
    )
    from jev_decision_tuning.experiment import train_experiment
    run_dir = train_experiment(config, args.output_dir)
    print(f"Saved completed run to: {run_dir}")
    print("Use scripts/evaluate.py --run-dir with this directory to reload and evaluate.")


if __name__ == "__main__":
    main()
