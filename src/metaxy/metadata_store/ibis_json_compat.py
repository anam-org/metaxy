"""JSON-compatible Ibis metadata store for databases without native struct support.

This module provides a base class for Ibis stores that serialize struct columns to JSON.
Uses dict-based versioning engine to avoid struct field access operations.

Suitable for:
- PostgreSQL (uses JSONB)
- SQLite (uses TEXT with JSON functions)
- MySQL (uses JSON)
- Any SQL database with JSON support but without struct field access in Ibis

Key differences from IbisMetadataStore:
- Uses IbisDictBasedVersioningEngine instead of IbisVersioningEngine
- Flattens provenance structs to columns during computation
- Packs flattened columns to JSON/JSONB on write
- Unpacks JSON/JSONB to flattened columns on read
- Stays in Ibis lazy world (no Polars materialization)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.base import VersioningEngineOptions
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.metadata_store.json_struct_serializer import JsonStructSerializerMixin
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature import BaseFeature
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.flat_engine import (
    IbisFlatVersioningEngine,
)
from metaxy.versioning.types import Increment, LazyIncrement

if TYPE_CHECKING:
    from narwhals.typing import Frame

    from metaxy.versioning.ibis import IbisVersioningEngine


class _PostProcessingLazyIncrement(LazyIncrement):
    """LazyIncrement that applies post-processing when collected.

    Used by IbisJsonCompatStore to rebuild struct columns from flattened
    columns after collecting Ibis frames to Polars.
    """

    def __init__(
        self,
        *,
        added: nw.LazyFrame[Any],
        changed: nw.LazyFrame[Any],
        removed: nw.LazyFrame[Any],
        post_process_fn: Callable[[nw.DataFrame[Any]], nw.DataFrame[Any]],
        input: nw.LazyFrame[Any] | None = None,
    ) -> None:
        super().__init__(added=added, changed=changed, removed=removed, input=input)
        self._post_process_fn = post_process_fn

    def collect(self, **kwargs: Any) -> Increment:
        """Collect lazy frames and apply post-processing."""
        base_result = super().collect(**kwargs)

        # Apply post-processing to rebuild struct columns from flattened columns
        return Increment(
            added=self._post_process_fn(
                nw.from_native(collect_to_polars(base_result.added))
            ),
            changed=self._post_process_fn(
                nw.from_native(collect_to_polars(base_result.changed))
            ),
            removed=self._post_process_fn(
                nw.from_native(collect_to_polars(base_result.removed))
            ),
        )


class IbisJsonCompatStore(JsonStructSerializerMixin, IbisMetadataStore, ABC):
    """Ibis metadata store for databases without native struct support.

    Uses dict-based versioning engine with JSON serialization at database boundaries.
    Automatically handles packing/unpacking of provenance columns:
    - During computation: Uses flattened columns (e.g., "metaxy_provenance_by_field__field1")
    - In database: Stores as JSON/JSONB (e.g., {"field1": "...", "field2": "..."})

    Subclasses must implement backend-specific JSON functions:
    - _get_json_unpack_exprs(): JSON → flattened columns
    - _get_json_pack_expr(): flattened columns → JSON

    Operations stay in Ibis lazy world until JSON-compat post-processing,
    which may materialize into Polars for struct reconstruction.

    Example:
        ```python
        class PostgresMetadataStore(IbisJsonCompatStore):
            def _get_json_unpack_exprs(self, json_column, field_names):
                # Use PostgreSQL jsonb_extract_path_text()
                ...

            def _get_json_pack_expr(self, struct_name, field_columns):
                # Use PostgreSQL jsonb_build_object()
                ...
        ```
    """

    def __init__(
        self,
        versioning_engine: VersioningEngineOptions = "auto",
        connection_string: str | None = None,
        *,
        backend: str | None = None,
        connection_params: dict[str, Any] | None = None,
        table_prefix: str | None = None,
        **kwargs: Any,
    ):
        """Initialize JSON-compatible Ibis store.

        Forces dict-based versioning engine for struct-free operations.

        Args:
            **kwargs: Passed to IbisMetadataStore.__init__
        """
        super().__init__(
            versioning_engine=versioning_engine,
            connection_string=connection_string,
            backend=backend,
            connection_params=connection_params,
            table_prefix=table_prefix,
            **kwargs,
        )

        # Override versioning engine to use dict-based implementation
        # This is required because we need struct-free field access
        self.versioning_engine_cls = IbisFlatVersioningEngine

    @contextmanager
    def _create_versioning_engine(
        self, plan: FeaturePlan
    ) -> Iterator[IbisVersioningEngine]:
        """Create dict-based versioning engine for Ibis backends."""
        if self._conn is None:
            raise RuntimeError(
                "Cannot create provenance engine: store is not open. "
                "Ensure store is used as context manager."
            )

        hash_functions = self._create_hash_functions()
        engine: Any = self.versioning_engine_cls(
            plan=plan,
            hash_functions=hash_functions,
        )
        try:
            yield cast("IbisVersioningEngine", engine)
        finally:
            pass

    def _post_process_resolve_update_result(
        self, result: Increment | LazyIncrement, *, lazy: bool
    ) -> Increment | LazyIncrement:
        """Materialize structs from flattened columns for returned results.

        This converts Ibis results to Polars (eager) so we can rebuild struct
        columns from flattened provenance/data_version fields for user-facing
        results and test comparisons.

        This method expects `_current_feature_plan` to be set by `resolve_update`.
        """
        plan = getattr(self, "_current_feature_plan", None)
        field_names = (
            self._get_field_names(plan, include_dependencies=False)
            if plan is not None
            else None
        )
        if field_names is None:
            raise RuntimeError(
                "Missing feature plan for struct reconstruction in Polars post-processing."
            )
        if lazy:
            # For lazy results, return a custom LazyIncrement that will
            # call _post_process_polars_frame when collected.
            # We don't call _ensure_struct_from_flattened here because
            # Ibis structs become JSON strings when collected to Polars.
            def ensure(
                frame: nw.LazyFrame[Any] | nw.DataFrame[Any],
            ) -> nw.LazyFrame[Any]:
                return self._ensure_ibis_lazy_frame(frame)

            return LazyIncrement(
                added=ensure(result.added),
                changed=ensure(result.changed),
                removed=ensure(result.removed),
                post_process_fn=lambda df: self._post_process_polars_frame(
                    df, field_names=field_names
                ),
            )

        frames = {}
        for name in ["added", "changed", "removed"]:
            frame = getattr(result, name)
            if frame.implementation != nw.Implementation.POLARS:
                polars_df = collect_to_polars(frame)
                frame = nw.from_native(polars_df)
            if isinstance(frame, nw.LazyFrame):
                frame = frame.collect()
            frame = self._post_process_polars_frame(frame, field_names=field_names)
            frames[name] = frame

        return Increment(
            added=frames["added"],
            changed=frames["changed"],
            removed=frames["removed"],
        )

    def _post_process_polars_frame(self, frame: Frame) -> Frame:
        """Rebuild struct columns from flattened provenance fields in Polars."""
        import polars as pl

        native = frame.to_native()
        if not isinstance(native, (pl.DataFrame, pl.LazyFrame)):
            return frame

        is_lazy = isinstance(native, pl.LazyFrame)

        def add_struct_column(
            df: pl.DataFrame | pl.LazyFrame, prefix: str
        ) -> pl.DataFrame | pl.LazyFrame:
            cols = [c for c in df.columns if c.startswith(f"{prefix}__")]
            if not cols:
                return df

            struct_fields = [
                pl.col(col).alias(col.split("__", 1)[1]) for col in sorted(cols)
            ]
            if not struct_fields:
                return df
            struct_expr = pl.struct(struct_fields).alias(prefix)
            return df.with_columns(struct_expr)

        native = frame.to_native()
        if not isinstance(native, (pl.DataFrame, pl.LazyFrame)):
            return frame

        if isinstance(native, pl.LazyFrame):
            native = native.collect()

        if field_names is None:
            plan = getattr(self, "_current_feature_plan", None)
            field_names = (
                self._get_field_names(plan, include_dependencies=False)
                if plan is not None
                else None
            )
        if field_names is None:
            raise RuntimeError(
                "Missing feature plan for struct reconstruction in Polars post-processing."
            )
        native = self._restore_struct_polars(
            native,
            METAXY_PROVENANCE_BY_FIELD,
            field_names=field_names,
        )
        native = self._restore_struct_polars(
            native,
            METAXY_DATA_VERSION_BY_FIELD,
            field_names=field_names,
        )

        schema = native.schema
        datetime_cols = [
            col for col, dtype in schema.items() if isinstance(dtype, pl.Datetime)
        ]
        if datetime_cols:
            native = native.with_columns(
                [
                    pl.col(col)
                    .dt.replace_time_zone("UTC")
                    .dt.cast_time_unit("us")
                    .alias(col)
                    for col in datetime_cols
                ]
            )

        return nw.from_native(native.lazy() if is_lazy else native)

    def _preprocess_samples_for_resolve_update(
        self,
        *,
        feature: type[BaseFeature],
        plan: FeaturePlan,
        samples: Frame | None,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
        lazy: bool,
    ) -> Frame | None:
        """Flatten provenance structs in user-provided samples before resolving."""
        if samples is None:
            return None
        _ = (feature, filters, lazy)

        import polars as pl

        if samples.implementation == nw.Implementation.POLARS:
            samples_native = samples.to_native()
        else:
            samples_native = collect_to_polars(samples)

        polars_df: pl.DataFrame | pl.LazyFrame
        field_names = self._get_field_names(plan, include_dependencies=False)
        flattened_columns = self._get_flattened_field_columns(
            METAXY_PROVENANCE_BY_FIELD, field_names
        )
        # Resolving schema can be non-trivial for some backends; keep this as a fast
        # pre-check to avoid unnecessary Polars materialization.
        if df.implementation == nw.Implementation.POLARS:
            native = df.to_native()
            columns = (
                native.collect_schema().names()
                if isinstance(native, pl.LazyFrame)
                else list(native.schema.keys())
            )
        else:
            columns = df.collect_schema().names()
        has_flattened = any(
            flattened in columns for flattened in flattened_columns.values()
        )
        if METAXY_PROVENANCE_BY_FIELD not in columns and not has_flattened:
            return samples

        field_names = self._get_field_names(plan, include_dependencies=False)
        exprs = []
        for field_name in field_names:
            flattened = FlatVersioningMixin._get_flattened_column_name(
                METAXY_PROVENANCE_BY_FIELD, field_name
            )
            if (
                flattened not in polars_df.columns
                and METAXY_PROVENANCE_BY_FIELD in polars_df.columns
            ):
                exprs.append(
                    pl.col(METAXY_PROVENANCE_BY_FIELD)
                    .struct.field(field_name)
                    .alias(flattened)
                )
        if exprs:
            polars_df = polars_df.with_columns(exprs)

        polars_frame = (
            samples_native
            if isinstance(samples_native, pl.LazyFrame)
            else samples_native.lazy()
        )
        polars_frame = nw.from_native(polars_frame, eager_only=False)
        return self._ensure_ibis_lazy_frame(polars_frame)

    @abstractmethod
    def _get_json_unpack_exprs(
        self,
        json_column: str,
        field_names: list[str],
    ) -> dict[str, Any]:
        """Get backend-specific Ibis expressions to unpack JSON fields.

        This method must return Ibis expressions that extract fields from a JSON column
        and create flattened columns following the naming convention.

        Args:
            json_column: Name of the JSON column to unpack (e.g., "metaxy_provenance_by_field")
            field_names: List of field names to extract from the JSON object

        Returns:
            Dict mapping flattened column names to Ibis expressions.
            Keys should follow the pattern: "{json_column}__{field_name}"

        Example for PostgreSQL:
            ```python
            {
                "metaxy_provenance_by_field__field1": jsonb_extract_path_text(
                    ibis.col("metaxy_provenance_by_field"),
                    ibis.lit("field1")
                ),
                "metaxy_provenance_by_field__field2": jsonb_extract_path_text(
                    ibis.col("metaxy_provenance_by_field"),
                    ibis.lit("field2")
                )
            }
            ```

        Note:
            This method is called during read operations and must return Ibis expressions
            (not execute them). The expressions will be added to an Ibis table mutation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_json_unpack_exprs()"
        )

    @abstractmethod
    def _get_json_pack_expr(
        self,
        struct_name: str,
        field_columns: Mapping[str, str],
    ) -> Any:
        """Get backend-specific Ibis expression to pack columns into JSON.

        This method must return an Ibis expression that builds a JSON object from
        flattened columns.

        Args:
            struct_name: Name of the virtual struct (e.g., "metaxy_provenance_by_field")
            field_columns: Mapping from field names to flattened column names
                Example: {"field1": "metaxy_provenance_by_field__field1"}

        Returns:
            Ibis expression that creates a JSON object

        Example for PostgreSQL:
            ```python
            jsonb_build_object(
                'field1', ibis.col('metaxy_provenance_by_field__field1'),
                'field2', ibis.col('metaxy_provenance_by_field__field2')
            )
            ```

        Note:
            This method is called during write operations and must return an Ibis expression
            (not execute it). The expression will be used in an Ibis table mutation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_json_pack_expr()"
        )

    def _unpack_json_columns(
        self,
        lazy_frame: nw.LazyFrame[Any],
        feature_plan: FeaturePlan,
    ) -> nw.LazyFrame[Any]:
        return JsonStructSerializerMixin._unpack_json_columns(
            self, lazy_frame, feature_plan
        )

    def _pack_json_columns(
        self,
        df: nw.LazyFrame[Any] | nw.DataFrame[Any],
        feature_plan: FeaturePlan,
    ) -> nw.LazyFrame[Any] | nw.DataFrame[Any]:
        return JsonStructSerializerMixin._pack_json_columns(self, df, feature_plan)

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata, unpacking JSON to flattened columns.

        Process:
        1. Read from parent (Ibis LazyFrame with JSON columns)
        2. Unpack JSON → flattened columns (stays lazy!)
        3. Return as Narwhals LazyFrame (Ibis-backed)

        Args:
            feature: Feature to read metadata for
            filters: Optional filter expressions
            columns: Optional column selection
            **kwargs: Additional arguments

        Returns:
            Narwhals LazyFrame with flattened columns, or None if no data

        Note:
            This stays in Ibis lazy world - no collection/materialization!
        """
        # Read from parent (returns Ibis LazyFrame with JSON columns)
        lazy_frame = super().read_metadata_in_store(
            feature, filters=filters, columns=columns, **kwargs
        )

        if lazy_frame is None:
            return None

        # Get feature plan to determine field names
        feature_key = self._resolve_feature_key(feature)
        feature_plan = self._resolve_feature_plan(feature_key)

        # Unpack JSON columns to flattened columns (stays lazy!)
        lazy_frame = self._unpack_json_columns(lazy_frame, feature_plan)

        return lazy_frame

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Pack flattened columns to JSON, write to DB.

        Process:
        1. Pack flattened columns → JSON (stays lazy if Ibis-backed)
        2. Write via parent (Ibis insert)

        Args:
            feature_key: Feature to write metadata for
            df: Narwhals DataFrame/LazyFrame to write
            **kwargs: Additional arguments

        Note:
            If df is Ibis-backed, this stays lazy until the actual insert.
            If df is Polars-backed, parent will handle collection.
        """
        # Get feature plan to determine field names
        feature_plan = self._resolve_feature_plan(feature_key)

        # Pack flattened columns into JSON columns (stays lazy if Ibis)
        df = self._pack_json_columns(df, feature_plan)

        # Write via parent (Ibis insert)
        super().write_metadata_to_store(feature_key, df, **kwargs)
