"""Hermetic test environment defaults."""

from __future__ import annotations

import os


os.environ["STORAGE_BACKEND"] = "local"
os.environ["KEEP_LOCAL_ARTIFACTS"] = "true"
os.environ["KEEP_LOCAL_RAW_DATA"] = "true"
