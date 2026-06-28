"""Read-only access to the consolidated model run fact view."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import Settings


SCHEMA = "fraud_tracking"
VIEW = "fact_model_runs"
QUALIFIED_VIEW = f"{SCHEMA}.{VIEW}"


class ModelRunFactRepositoryError(RuntimeError):
    """Base error raised while reading the model run fact view."""


class ModelRunFactViewNotFoundError(ModelRunFactRepositoryError):
    """Raised when the required database view has not been migrated."""


class ModelRunFactRepository:
    """Export paginated rows from the model run analytical view."""

    def __init__(self, settings: Settings, engine: Engine | None = None) -> None:
        self._engine = engine or create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": settings.database_connect_timeout_seconds},
        )

    def export_page(self, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        """Return the total row count and a page ordered by newest run."""
        try:
            with self._engine.connect() as connection:
                views = set(inspect(connection).get_view_names(schema=SCHEMA))
                if VIEW not in views:
                    raise ModelRunFactViewNotFoundError(
                        f"View {QUALIFIED_VIEW} nao encontrada."
                    )

                total = connection.execute(
                    text(f"SELECT COUNT(*) FROM {QUALIFIED_VIEW}")
                ).scalar_one()
                rows = connection.execute(
                    text(
                        f"""
                        SELECT *
                        FROM {QUALIFIED_VIEW}
                        ORDER BY completed_at DESC, run_id DESC
                        LIMIT :limit OFFSET :offset
                        """
                    ),
                    {"limit": limit, "offset": offset},
                ).mappings()
                return int(total), [_json_safe(dict(row)) for row in rows]
        except ModelRunFactViewNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise ModelRunFactRepositoryError(
                f"Nao foi possivel consultar {QUALIFIED_VIEW}."
            ) from exc


def _json_safe(value: Any) -> Any:
    """Convert database-specific values into JSON-compatible primitives."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
