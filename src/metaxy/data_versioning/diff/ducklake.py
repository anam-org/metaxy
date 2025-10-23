"""DuckLake diff resolver built on top of the Ibis implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from metaxy.data_versioning.diff.ibis import IbisDiffResolver

if TYPE_CHECKING:
    import ibis.expr.types as ir


class DuckLakeDiffResolver(IbisDiffResolver):
    """Ensures diff queries execute against the DuckLake catalog."""

    def __init__(
        self, backend: "ir.BaseBackend | None" = None, alias: str = "ducklake"
    ):
        super().__init__()
        self.backend = backend
        self.alias = alias
        if backend is not None:
            try:
                backend.raw_sql(f"USE {alias}")  # type: ignore[attr-defined]
            except Exception:
                pass
