"""Kaggle dataset source implemented through the official Kaggle CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from src.ingestion.source import DatasetSource
from src.utils.logger import get_logger


logger = get_logger(__name__)
CommandRunner = Callable[[Sequence[str]], None]


class KaggleDatasetSource(DatasetSource):
    """Download and extract one Kaggle dataset."""

    def __init__(self, dataset: str, command_runner: CommandRunner | None = None) -> None:
        if "/" not in dataset:
            raise ValueError("KAGGLE_DATASET deve usar o formato 'proprietario/dataset'.")
        self.dataset = dataset
        self.command_runner = command_runner or self._run_command

    def describe(self) -> str:
        """Return the configured Kaggle dataset description."""
        return f"kaggle:{self.dataset}"

    def download(self, destination: Path) -> list[Path]:
        """Download and unzip the configured dataset into destination."""
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            "kaggle",
            "datasets",
            "download",
            self.dataset,
            "--path",
            str(destination),
            "--unzip",
            "--quiet",
        ]
        logger.info("Iniciando download do dataset Kaggle: %s", self.dataset)
        try:
            self.command_runner(command)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "CLI do Kaggle nao encontrada. Instale as dependencias com "
                "'pip install -r requirements.txt'."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Falha ao baixar o dataset do Kaggle. Verifique o dataset e as credenciais."
            ) from exc

        files = sorted(path for path in destination.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(f"O Kaggle nao retornou arquivos para o dataset '{self.dataset}'.")
        logger.info("Download do Kaggle concluido | dataset=%s | arquivos=%d", self.dataset, len(files))
        return files

    @staticmethod
    def _run_command(command: Sequence[str]) -> None:
        """Execute a Kaggle CLI command and fail on non-zero exit status."""
        subprocess.run(list(command), check=True)
