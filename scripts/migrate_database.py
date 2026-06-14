"""Run PostgreSQL schema migrations through Alembic."""

from __future__ import annotations

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("upgrade", "downgrade", "current", "history"),
        help="Migration action to execute.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Target revision; defaults to head for upgrade and -1 for downgrade.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the requested Alembic command."""
    args = parse_args()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    if args.action == "upgrade":
        command.upgrade(config, args.revision or "head")
    elif args.action == "downgrade":
        command.downgrade(config, args.revision or "-1")
    elif args.action == "current":
        command.current(config, verbose=True)
    else:
        command.history(config, verbose=True)


if __name__ == "__main__":
    main()
