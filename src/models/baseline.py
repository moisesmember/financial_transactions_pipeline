"""Explicit promotion of trained artifacts to the official baseline."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2, rmtree
from typing import Any
from uuid import uuid4

from src.config.settings import Settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class BaselineRegistry:
    """Store an immutable local snapshot of the official baseline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backup_dir: Path | None = None
        self._promoted_dir: Path | None = None

    def promote(
        self,
        metadata: dict[str, Any],
        report_paths: list[Path] | None = None,
        overwrite: bool = False,
        audit_status: str | None = None,
        pipeline_path: Path | None = None,
    ) -> Path:
        """Promote current artifacts, requiring explicit overwrite."""
        baseline_dir = self.settings.baseline_dir
        if baseline_dir.exists() and any(baseline_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Baseline oficial ja existe em {baseline_dir}. Use overwrite para substitui-lo."
            )
        baseline_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = baseline_dir.parent / f".baseline-staging-{uuid4().hex}"
        backup_dir = baseline_dir.parent / f".baseline-backup-{uuid4().hex}"
        staging_dir.mkdir(parents=True)

        pipeline_target = staging_dir / self.settings.baseline_pipeline_filename
        source_pipeline = pipeline_path or self.settings.pipeline_path
        copy2(source_pipeline, pipeline_target)
        copied_reports: list[str] = []
        for report_path in report_paths or []:
            if report_path.exists():
                target = staging_dir / report_path.name
                copy2(report_path, target)
                copied_reports.append(target.name)

        baseline_metadata = {
            **metadata,
            "baseline": {
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline_sha256": self._sha256(pipeline_target),
                "pipeline_file": pipeline_target.name,
                "reports": copied_reports,
                "audit_status": audit_status or "not_available",
            },
        }
        metadata_path = staging_dir / self.settings.baseline_metadata_filename
        metadata_path.write_text(
            json.dumps(self._json_safe(baseline_metadata), indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        had_baseline = baseline_dir.exists()
        if had_baseline:
            baseline_dir.replace(backup_dir)
        try:
            staging_dir.replace(baseline_dir)
        except OSError:
            if had_baseline and backup_dir.exists():
                backup_dir.replace(baseline_dir)
            if staging_dir.exists():
                rmtree(staging_dir)
            raise
        self._backup_dir = backup_dir if backup_dir.exists() else None
        self._promoted_dir = baseline_dir
        metadata_path = baseline_dir / self.settings.baseline_metadata_filename
        logger.info(
            "Baseline oficial promovido | diretorio=%s | audit_status=%s",
            baseline_dir,
            baseline_metadata["baseline"]["audit_status"],
        )
        return metadata_path

    def commit_promotion(self) -> None:
        """Discard the rollback snapshot after external persistence succeeds."""
        if self._backup_dir is not None and self._backup_dir.exists():
            rmtree(self._backup_dir)
        self._backup_dir = None
        self._promoted_dir = None

    def rollback_promotion(self) -> None:
        """Restore the previous local baseline after an external failure."""
        if self._promoted_dir is not None and self._promoted_dir.exists():
            rmtree(self._promoted_dir)
        if self._backup_dir is not None and self._backup_dir.exists():
            self._backup_dir.replace(self.settings.baseline_dir)
        self._backup_dir = None
        self._promoted_dir = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
