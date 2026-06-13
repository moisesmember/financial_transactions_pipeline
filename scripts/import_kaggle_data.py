"""Import the configured Kaggle dataset into local storage or MinIO."""

from __future__ import annotations

import argparse

from src.config.settings import Settings
from src.ingestion.import_service import DatasetImportService


def parse_args() -> argparse.Namespace:
    """Parse overwrite behavior from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Substitui arquivos existentes; por padrao usa KAGGLE_OVERWRITE.",
    )
    return parser.parse_args()


def main() -> None:
    """Import the Kaggle dataset into the configured storage backend."""
    args = parse_args()
    result = DatasetImportService(Settings()).import_data(overwrite=args.overwrite)
    print(f"Importados: {len(result.imported)} | Ignorados: {len(result.skipped)}")


if __name__ == "__main__":
    main()
