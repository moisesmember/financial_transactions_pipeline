"""Dataset, code and experiment version fingerprints."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from src.config.settings import Settings


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(settings: Settings) -> str:
    """Return a deterministic fingerprint for the configured raw dataset."""
    if settings.dataset_version_override:
        return settings.dataset_version_override
    digest = hashlib.sha256()
    for filename in sorted(settings.kaggle_expected_files):
        path = settings.raw_data_dir / filename
        digest.update(filename.encode("utf-8"))
        if path.exists():
            digest.update(sha256_file(path).encode("ascii"))
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def code_version(settings: Settings) -> str:
    """Return the configured code version or the current Git revision."""
    if settings.code_version_override:
        return settings.code_version_override
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=settings.project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def experiment_fingerprint(payload: dict) -> str:
    """Hash the reproducibility inputs of one experiment."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
