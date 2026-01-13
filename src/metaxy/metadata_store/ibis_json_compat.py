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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import narwhals as nw
import polars as pl

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.base import VersioningEngineOptions
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.metadata_store.json_struct_serializer import JsonStructSerializerMixin
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
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
        store: IbisJsonCompatStore,
        plan: FeaturePlan,
        input: nw.LazyFrame[Any] | None = None,
    ) -> None:
        super().__init__(added=added, changed=changed, removed=removed, input=input)
        self._store = store
        self._plan = plan

    def collect(self, **kwargs: Any) -> Increment:
        """Collect lazy frames and apply post-processing."""
        base_result = super().collect(**kwargs)

        # Apply post-processing to rebuild struct columns from flattened columns
        return Increment(
            added=self._store._post_process_polars_frame(
                nw.from_native(collect_to_polars(base_result.added)), plan=self._plan
            ),
            changed=self._store._post_process_polars_frame(
                nw.from_native(collect_to_polars(base_result.changed)), plan=self._plan
            ),
            removed=self._store._post_process_polars_frame(
                nw.from_native(collect_to_polars(base_result.removed)), plan=self._plan
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
            versioning_engine: Which versioning engine to use.
                - "auto": Prefer the store's native engine, fall back to Polars if needed
                - "native": Always use the store's native engine, raise `VersioningEngineMismatchError`
                    if provided dataframes are incompatible
                - "polars": Always use the Polars engine
            connection_string: Ibis connection string (e.g., "clickhouse://host:9000/db")
                If provided, backend and connection_params are ignored.
            backend: Ibis backend name (e.g., "clickhouse", "postgres", "duckdb")
                Used with connection_params for more control.
            connection_params: Backend-specific connection parameters
                e.g., {"host": "localhost", "port": 9000, "database": "default"}
            table_prefix: Optional prefix applied to all feature and system table names.
                Useful for logically separating environments (e.g., "prod_"). Must form a valid SQL
                identifier when combined with the generated table name.
            **kwargs: Passed to MetadataStore.__init__ (e.g., fallback_stores, hash_algorithm)
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
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[IbisVersioningEngine]:
        """Create dict-based versioning engine for Ibis backends."""
        if self._conn is None:
            raise RuntimeError(
                "Cannot create provenance engine: store is not open. Ensure store is used as context manager."
            )

        hash_functions = self._create_hash_functions()
        engine: Any = self.versioning_engine_cls(
            plan=plan,
            hash_functions=hash_functions,
        )
        yield cast("IbisVersioningEngine", engine)

    @overload
    def _post_process_resolve_update_result(
        self, result: Increment, plan: FeaturePlan, *, lazy: Literal[False]
    ) -> Increment: ...

    @overload
    def _post_process_resolve_update_result(
        self, result: LazyIncrement, plan: FeaturePlan, *, lazy: Literal[True]
    ) -> LazyIncrement: ...

    def _post_process_resolve_update_result(
        self, result: Increment | LazyIncrement, plan: FeaturePlan, *, lazy: bool
    ) -> Increment | LazyIncrement:
        """Materialize structs from flattened columns for returned results.

        This converts Ibis results to Polars (eager) so we can rebuild struct
        columns from flattened provenance/data_version fields for user-facing
        results and test comparisons.
        """
        if lazy:
            # For lazy results, return a custom LazyIncrement that will
            # call _post_process_polars_frame when collected.
            # We don't call _ensure_struct_from_flattened here because
            # Ibis structs become JSON strings when collected to Polars.
            return _PostProcessingLazyIncrement(
                added=self._ensure_ibis_lazy_frame(result.added),
                changed=self._ensure_ibis_lazy_frame(result.changed),
                removed=self._ensure_ibis_lazy_frame(result.removed),
                store=self,
                plan=plan,
            )

        return self._post_process_increment_frames(cast("Increment", result), plan=plan)

    def _post_process_increment_frames(
        self,
        result: Increment,
        *,
        plan: FeaturePlan,
    ) -> Increment:
        def process_frame(frame: nw.DataFrame[Any] | nw.LazyFrame[Any]) -> nw.DataFrame[Any]:
            """Convert to Polars and post-process frame."""
            if frame.implementation != nw.Implementation.POLARS:
                polars_df = collect_to_polars(frame)
                frame = nw.from_native(polars_df)
            if isinstance(frame, nw.LazyFrame):
                frame = frame.collect()
            return self._post_process_polars_frame(frame, plan=plan)

        return Increment(
            added=process_frame(result.added),
            changed=process_frame(result.changed),
            removed=process_frame(result.removed),
        )

    def _post_process_polars_frame(
        self,
        frame: nw.DataFrame[Any],
        *,
        plan: FeaturePlan | None = None,
    ) -> nw.DataFrame[Any]:
        """Rebuild struct columns from flattened provenance fields in Polars.

        Expects a Polars-backed DataFrame and returns an eager result.

        Args:
            frame: Polars-backed DataFrame to post-process
            plan: Feature plan for determining field names for struct reconstruction
        """

        if frame.implementation != nw.Implementation.POLARS:
            raise RuntimeError(
                "_post_process_polars_frame expects a Polars-backed DataFrame; "
                "callers should use Polars fallback or collect via "
                "_PostProcessingLazyIncrement."
            )

        native = frame.to_native()
        if not isinstance(native, (pl.DataFrame, pl.LazyFrame)):
            return frame

        if isinstance(native, pl.LazyFrame):
            native = native.collect()

        if plan is None:
            raise RuntimeError("Missing feature plan for struct reconstruction in Polars post-processing.")

        native = self._restore_struct_columns_polars(native, plan=plan)

        schema = native.schema
        datetime_cols = [col for col, dtype in schema.items() if isinstance(dtype, pl.Datetime)]
        if datetime_cols:
            native = native.with_columns(
                [pl.col(col).dt.replace_time_zone("UTC").dt.cast_time_unit("us").alias(col) for col in datetime_cols]
            )

        return cast("nw.DataFrame[Any]", nw.from_native(native))

    @overload
    def _preprocess_samples_for_resolve_update(
        self,
        *,
        feature: type[BaseFeature],
        df: nw.DataFrame[Any],
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
        lazy: bool,
    ) -> nw.DataFrame[Any]: ...

    @overload
    def _preprocess_samples_for_resolve_update(
        self,
        *,
        feature: type[BaseFeature],
        df: nw.LazyFrame[Any],
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
        lazy: bool,
    ) -> nw.LazyFrame[Any]: ...

    @overload
    def _preprocess_samples_for_resolve_update(
        self,
        *,
        feature: type[BaseFeature],
        df: None,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
        lazy: bool,
    ) -> None: ...

    def _preprocess_samples_for_resolve_update(
        self,
        *,
        feature: type[BaseFeature],
        df: nw.DataFrame[Any] | nw.LazyFrame[Any] | None,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,  # noqa: ARG002
        lazy: bool,  # noqa: ARG002
    ) -> nw.DataFrame[Any] | nw.LazyFrame[Any] | None:
        """Flatten provenance structs in user-provided samples before resolving."""
        if df is None:
            return None

        plan = self._resolve_feature_plan(feature)
        field_names = self._get_field_names(plan, include_dependencies=False)
        flattened_columns = self._get_flattened_field_columns(METAXY_PROVENANCE_BY_FIELD, field_names)
        # Resolving schema can be non-trivial for some backends; keep this as a fast
        # pre-check to avoid unnecessary Polars materialization.
        if df.implementation == nw.Implementation.POLARS:
            native = df.to_native()
            columns = (
                native.collect_schema().names() if isinstance(native, pl.LazyFrame) else list(native.schema.keys())
            )
        else:
            columns = df.collect_schema().names()
        has_flattened = any(flattened in columns for flattened in flattened_columns.values())
        if METAXY_PROVENANCE_BY_FIELD not in columns and not has_flattened:
            return df

        samples_native = df.to_native() if df.implementation == nw.Implementation.POLARS else collect_to_polars(df)
        if not isinstance(samples_native, (pl.DataFrame, pl.LazyFrame)):
            return df

        # Restore struct from flattened columns or existing struct column
        samples_native = self._restore_struct_polars(
            samples_native,
            METAXY_PROVENANCE_BY_FIELD,
            field_names=field_names,
        )

        # If struct column exists, expand to flattened columns for downstream processing
        if not has_flattened and METAXY_PROVENANCE_BY_FIELD in columns:
            exprs = [
                pl.col(METAXY_PROVENANCE_BY_FIELD).struct.field(field_name).alias(flattened)
                for field_name, flattened in flattened_columns.items()
                if flattened not in columns
            ]
            if exprs:
                samples_native = samples_native.with_columns(exprs)

        # Convert to Ibis LazyFrame
        polars_frame = samples_native if isinstance(samples_native, pl.LazyFrame) else samples_native.lazy()
        return self._ensure_ibis_lazy_frame(nw.from_native(polars_frame, eager_only=False))

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
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_json_unpack_exprs()")

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
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_json_pack_expr()")

    def _unpack_json_columns(
        self,
        lazy_frame: nw.LazyFrame[Any],
        feature_plan: FeaturePlan,
    ) -> nw.LazyFrame[Any]:
        return JsonStructSerializerMixin._unpack_json_columns(self, lazy_frame, feature_plan)

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
        lazy_frame = super().read_metadata_in_store(feature, filters=filters, columns=columns, **kwargs)

        if lazy_frame is None:
            return None

        feature_key = self._resolve_feature_key(feature)
        if self._is_system_table(feature_key):
            return lazy_frame

        # Get feature plan to determine field names
        feature_plan = self._resolve_feature_plan(feature_key)

        # Unpack JSON columns to flattened columns (stays lazy!)
        lazy_frame = self._unpack_json_columns(lazy_frame, feature_plan)

        return lazy_frame

    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        current_only: bool,
    ) -> None:
        """Hard delete implementation for JSON compat stores.

        Overrides parent to use base table (without JSON unpacking) for WHERE clause
        extraction, since JSON unpacking creates complex SQL that can't be parsed.

        Args:
            feature_key: Feature to delete from
            filters: Narwhals expressions to filter records
            current_only: Not used here - version filtering handled by base class
        """
        import ibis.expr.types

        from metaxy.metadata_store.exceptions import TableNotFoundError
        from metaxy.metadata_store.utils import (
            _extract_where_expression,
            _strip_table_qualifiers,
        )

        table_name = self.get_table_name(feature_key)
        filter_list = list(filters or [])

        # Handle empty filters - truncate entire table
        if not filter_list:
            if table_name not in self.conn.list_tables():
                raise TableNotFoundError(f"Table '{table_name}' does not exist for feature {feature_key.to_string()}.")
            self.conn.truncate_table(table_name)  # ty: ignore[unresolved-attribute]
            return

        # Read from parent's parent (IbisMetadataStore) to avoid JSON unpacking
        # This gives us a simpler SQL for WHERE clause extraction
        filtered = IbisMetadataStore.read_metadata_in_store(self, feature_key, filters=filter_list)
        if filtered is None:
            from metaxy.metadata_store.exceptions import FeatureNotFoundError

            raise FeatureNotFoundError(f"Feature {feature_key.to_string()} not found in store")

        # Extract WHERE clause from compiled SELECT statement
        ibis_filtered = cast(ibis.expr.types.Table, filtered.to_native())
        select_sql = str(ibis_filtered.compile())

        dialect = self._sql_dialect
        predicate = _extract_where_expression(select_sql, dialect=dialect)
        if predicate is None:
            raise ValueError(f"Cannot extract WHERE clause for DELETE on {self.__class__.__name__}")

        # Generate and execute DELETE statement
        predicate = predicate.transform(_strip_table_qualifiers())
        where_clause = predicate.sql(dialect=dialect) if dialect else predicate.sql()

        delete_stmt = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.conn.raw_sql(delete_stmt)  # ty: ignore[unresolved-attribute]

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
        resolved_key = self._resolve_feature_key(feature_key)
        if self._is_system_table(resolved_key):
            return IbisMetadataStore.write_metadata_to_store(self, resolved_key, df, **kwargs)

        # Get feature plan to determine field names
        feature_plan = self._resolve_feature_plan(resolved_key)

        # Pack flattened columns into JSON columns (stays lazy if Ibis)
        df = self._pack_json_columns(df, feature_plan)

        # Write via parent (Ibis insert)
        super().write_metadata_to_store(resolved_key, df, **kwargs)
