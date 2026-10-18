"""Reload a saved experiment and evaluate the adapter or its base comparator."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--backend", choices=("lora", "base"), default="lora")
    parser.add_argument("--data-path", help="Override the DatasetDict path after moving a run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", help="cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows for a smoke check")
    parser.add_argument("--name", help="Unique output stem; existing predictions are never overwritten")
    args = parser.parse_args()
    from jev_decision_tuning.experiment import evaluate_experiment
    report = evaluate_experiment(**vars(args))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
