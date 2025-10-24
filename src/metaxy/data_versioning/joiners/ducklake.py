"""DuckLake implementation of the upstream joiner."""

from typing import TYPE_CHECKING

from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner

if TYPE_CHECKING:
    import ibis.expr.types as ir


class DuckLakeJoiner(NarwhalsJoiner):
    """Narwhals joiner that ensures DuckLake catalog context when available."""

    def __init__(self, backend: "ir.BaseBackend", alias: str = "ducklake"):
        super().__init__()
        self._backend = backend
        self.alias = alias
        self._ensure_ducklake_catalog()

    def _ensure_ducklake_catalog(self) -> None:
        try:
            self._backend.raw_sql(f"USE {self.alias}")  # type: ignore[attr-defined]
        except Exception:
            # Catalog already selected or backend doesn't expose raw_sql
            pass
