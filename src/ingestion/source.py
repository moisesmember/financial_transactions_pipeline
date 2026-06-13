"""Contracts for external dataset sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DatasetSource(ABC):
    """Download a dataset into an isolated local staging directory."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable source description."""

    @abstractmethod
    def download(self, destination: Path) -> list[Path]:
        """Download and return every extracted dataset file."""
