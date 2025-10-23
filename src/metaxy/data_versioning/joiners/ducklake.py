"""DuckLake implementation of the upstream joiner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from metaxy.data_versioning.joiners.ibis import IbisJoiner

if TYPE_CHECKING:
    import ibis.expr.types as ir


class DuckLakeJoiner(IbisJoiner):
    """Wraps IbisJoiner while ensuring the DuckLake catalog is active."""

    def __init__(self, backend: "ir.BaseBackend", alias: str = "ducklake"):
        super().__init__(backend=backend)
        self.alias = alias
        try:
            backend.raw_sql(f"USE {alias}")  # type: ignore[attr-defined]
        except Exception:
            pass
