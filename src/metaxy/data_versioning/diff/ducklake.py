"""DuckLake diff resolver aligned with Narwhals implementation."""

from typing import TYPE_CHECKING

from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver

if TYPE_CHECKING:
    import ibis.expr.types as ir


class DuckLakeDiffResolver(NarwhalsDiffResolver):
    """Ensures diff queries execute against the DuckLake catalog when possible."""

    def __init__(
        self, backend: "ir.BaseBackend | None" = None, alias: str = "ducklake"
    ):
        super().__init__()
        self.backend = backend
        self.alias = alias
        if backend is not None:
            self._ensure_ducklake_catalog()

    def _ensure_ducklake_catalog(self) -> None:
        if self.backend is None:
            return
        try:
            self.backend.raw_sql(f"USE {self.alias}")  # type: ignore[attr-defined]
        except Exception:
            # Backend already pinned to catalog or raw_sql unsupported
            pass
