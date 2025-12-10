"""Vortex metadata store implementation.

Vortex is a next-generation columnar file format optimized for compression and
high-performance reads.

Currently only local filesystem storage is supported. Remote storage (S3, GCS, Azure)
is not yet available due to limitations in the vortex-data Python library.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field
from typing_extensions import Self

from metaxy._utils import switch_implementation_to_polars
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import vortex as _vortex  # pyright: ignore[reportMissingImports]  # noqa: F401

vortex: Any | None = None


def _ensure_vortex_dependency() -> Any:
    """Import the optional vortex dependency on first use."""
    global vortex
    if vortex is not None:
        return vortex

    try:
        import vortex as _vortex  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised via optional dep
        raise ImportError(
            "The VortexMetadataStore requires the vortex extra. "
            'Install with `pip install "metaxy[vortex]"` or `uv sync --extra vortex`.'
        ) from exc

    vortex = _vortex
    return _vortex


class VortexMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for VortexMetadataStore.

    Note:
        Currently only local filesystem storage is supported.

    Example:
        ```python
        config = VortexMetadataStoreConfig(
            root_path="./metaxy_data",
            layout="nested",
        )

        store = VortexMetadataStore.from_config(config)
        ```
    """

    root_path: str | Path = Field(
        description="Base directory where feature tables are stored (local paths only).",
    )
    layout: Literal["flat", "nested"] = Field(
        default="nested",
        description="Directory layout for feature tables ('nested' or 'flat').",
    )


class VortexMetadataStore(MetadataStore):
    """
    Vortex metadata store backed by [vortex-data](https://github.com/vortex-data/vortex).

    Vortex is a next-generation columnar file format optimized for compression and
    high-performance reads. It stores feature metadata in Vortex files located under
    ``root_path``. Uses the Polars versioning engine for provenance calculations.

    Note:
        Currently only **local filesystem storage** is supported.
        Remote storage (S3, GCS, Azure) is not yet available.

    Note:
        Vortex uses a read-concat-write pattern for appends, as it's an immutable file format.
        This means each append reads the existing file, concatenates new data, and rewrites.
        For high-frequency updates, consider using DuckDB or LanceDB instead.

    Example:
        ```py
        from metaxy.metadata_store.vortex import VortexMetadataStore

        store = VortexMetadataStore(root_path="./metaxy_data")
        ```
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        root_path: str | Path,
        *,
        fallback_stores: list[MetadataStore] | None = None,
        layout: Literal["flat", "nested"] = "nested",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Vortex metadata store.

        Args:
            root_path: Base directory where feature tables are stored.
                Only local filesystem paths are supported.
            fallback_stores: Ordered list of read-only fallback stores.
            layout: Directory layout for feature tables. Options:

                - `"nested"`: Feature tables stored in nested directories `{part1}/{part2}.vortex`

                - `"flat"`: Feature tables stored as `{part1}__{part2}.vortex`

            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][].

        Raises:
            NotImplementedError: If a remote path (s3://, gs://, az://) is provided.
        """
        if layout not in ("flat", "nested"):
            raise ValueError(f"Invalid layout: {layout}. Must be 'flat' or 'nested'.")
        self.layout = layout

        root_str = str(root_path)

        # Check for remote paths and raise clear error
        if not is_local_path(root_str):
            raise NotImplementedError(
                f"Remote storage paths are not yet supported by VortexMetadataStore. "
                f"Got: {root_str!r}. "
                f"The vortex-data Python library currently only supports local filesystem writes. "
                f"For remote storage, consider using DeltaMetadataStore instead."
            )

        # Local path (including file:// and local:// URLs)
        if root_str.startswith("file://"):
            root_str = root_str[7:]
        elif root_str.startswith("local://"):
            root_str = root_str[8:]
        local_path = Path(root_str).expanduser().resolve()
        self._root_uri = str(local_path)

        super().__init__(
            fallback_stores=fallback_stores,
            versioning_engine_cls=PolarsVersioningEngine,
            versioning_engine="polars",
            **kwargs,
        )

    # ===== MetadataStore abstract methods =====

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Check if feature exists in Vortex store."""
        feature_key = self._resolve_feature_key(feature)
        return self._table_exists(self._feature_uri(feature_key))

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH64 by default to match other non-SQL stores."""
        return HashAlgorithm.XXHASH64

    @contextmanager
    def _create_versioning_engine(
        self, plan: FeaturePlan
    ) -> Iterator[PolarsVersioningEngine]:
        """Create Polars versioning engine for Vortex store."""
        with self._create_polars_versioning_engine(plan) as engine:
            yield engine

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:  # noqa: ARG002
        """Open the Vortex store.

        Creates the root directory if it doesn't exist.
        Vortex files are opened on-demand per operation.

        Args:
            mode: Access mode (accepted for API consistency).

        Yields:
            Self: The store instance with connection open
        """
        self._context_depth += 1

        try:
            if self._context_depth == 1:
                Path(self._root_uri).mkdir(parents=True, exist_ok=True)
                self._is_open = True
                self._validate_after_open()

            yield self
        finally:
            self._context_depth -= 1
            if self._context_depth == 0:
                self._is_open = False

    # ===== Internal helpers =====

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        """Return the path used by Vortex for this feature."""
        if self.layout == "nested":
            table_path = "/".join(part for part in feature_key.parts if part)
        else:
            table_path = feature_key.table_name
        return f"{self._root_uri}/{table_path}.vortex"

    def _table_exists(self, table_uri: str) -> bool:
        """Check whether the provided path contains a Vortex file."""
        _vortex = _ensure_vortex_dependency()
        try:
            _ = _vortex.open(table_uri)
            return True
        except (FileNotFoundError, OSError, Exception):
            return False

    # ===== Storage operations =====

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Append metadata to the Vortex file for a feature.

        Uses read-concat-write pattern since Vortex is an immutable file format.
        """
        table_uri = self._feature_uri(feature_key)

        df_polars = switch_implementation_to_polars(df)

        if isinstance(df_polars, nw.LazyFrame):
            df_native = df_polars.collect().to_native()
        else:
            df_native = df_polars.to_native()

        assert isinstance(df_native, pl.DataFrame)

        # Cast Enum columns to String to avoid serialization issues
        df_native = df_native.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

        if self._table_exists(table_uri):
            existing_df = self._read_vortex_file(table_uri)
            existing_columns = existing_df.columns
            df_native_aligned = df_native.select(existing_columns)
            combined_df = pl.concat([existing_df, df_native_aligned])
            self._write_vortex_file(combined_df, table_uri)
        else:
            Path(table_uri).parent.mkdir(parents=True, exist_ok=True)
            self._write_vortex_file(df_native, table_uri)

    def _write_vortex_file(self, df: pl.DataFrame, uri: str) -> None:
        """Write DataFrame to local Vortex file."""
        _vortex = _ensure_vortex_dependency()
        arrow_table = df.to_arrow()
        _vortex.io.write(arrow_table, uri)

    def _read_vortex_file(self, uri: str) -> pl.DataFrame:
        """Read local Vortex file into Polars DataFrame."""
        _vortex = _ensure_vortex_dependency()
        vortex_file = _vortex.open(uri)
        return vortex_file.scan().read_all().to_polars_dataframe()

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Vortex file for the specified feature."""
        table_uri = self._feature_uri(feature_key)

        if not self._table_exists(table_uri):
            return

        Path(table_uri).unlink(missing_ok=True)

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata stored in Vortex for a single feature."""
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        table_uri = self._feature_uri(feature_key)
        if not self._table_exists(table_uri):
            return None

        _vortex = _ensure_vortex_dependency()
        vortex_file = _vortex.open(table_uri)

        if columns is not None:
            scan_iter = vortex_file.scan(list(columns))
        else:
            scan_iter = vortex_file.scan()

        df = scan_iter.read_all().to_polars_dataframe()
        lf = df.lazy()
        nw_lazy = nw.from_native(lf)

        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        return nw_lazy

    def display(self) -> str:
        """Return human-readable representation of the store."""
        return f"VortexMetadataStore(path={self._root_uri}, layout={self.layout})"

    def get_store_metadata(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        """Return store-specific metadata for a feature."""
        return {"uri": self._feature_uri(self._resolve_feature_key(feature_key))}

    @classmethod
    def config_model(cls) -> type[VortexMetadataStoreConfig]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the configuration model class for VortexMetadataStore."""
        return VortexMetadataStoreConfig
