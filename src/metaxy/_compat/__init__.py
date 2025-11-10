"""Compatibility helpers for third-party dependencies used by Metaxy."""

from __future__ import annotations

# Import compat modules for their side effects (e.g., monkey patches)
from . import sqlglot  # noqa: F401

__all__ = ["sqlglot"]
