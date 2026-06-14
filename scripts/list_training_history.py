"""List historical training runs ordered by a comparison metric."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import Settings


METRICS = (
    "test_pr_auc",
    "test_fbeta",
    "test_recall",
    "test_precision",
    "validation_pr_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sort-by", choices=METRICS, default="test_pr_auc")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Print the most relevant fields from the history comparison index."""
    args = parse_args()
    path = Settings().training_history_index_path
    if not path.exists():
        raise FileNotFoundError("Nenhum historico de treinamento foi encontrado.")
    history = pd.read_csv(path).sort_values(args.sort_by, ascending=False).head(args.limit)
    columns = [
        "run_id",
        "model_name",
        "threshold",
        "test_pr_auc",
        "test_fbeta",
        "test_recall",
        "test_precision",
        "audit_status",
    ]
    print(history[columns].to_string(index=False))


if __name__ == "__main__":
    main()
