"""Abstract base class for metadata storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import narwhals as nw
from narwhals.typing import Frame, FrameT, IntoFrame
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from metaxy._utils import switch_implementation_to_polars
from metaxy.config import MetaxyConfig
from metaxy.metadata_store.exceptions import (
    FeatureNotFoundError,
    StoreNotOpenError,
    SystemDataNotFoundError,
    VersioningEngineMismatchError,
)
from metaxy.metadata_store.system.keys import METAXY_SYSTEM_KEY_PREFIX
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import (
    _suppress_feature_version_warning,
    allow_feature_version_override,
    empty_frame_like,
)
from metaxy.metadata_store.warnings import (
    MetaxyColumnMissingWarning,
    PolarsMaterializationWarning,
)
from metaxy.models.constants import (
    ALL_SYSTEM_COLUMNS,
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.feature import FeatureGraph, current_graph
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import (
    CoercibleToFeatureKey,
    FeatureKey,
    ValidatedFeatureKeyAdapter,
)
from metaxy.versioning import VersioningEngine
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm, Increment, LazyIncrement

if TYPE_CHECKING:
    pass


# TypeVar for config types - used for typing from_config method
MetadataStoreConfigT = TypeVar("MetadataStoreConfigT", bound="MetadataStoreConfig")


class MetadataStoreConfig(BaseSettings):
    """Base configuration class for metadata stores.

    This class defines common configuration fields shared by all metadata store types.
    Store-specific config classes should inherit from this and add their own fields.

    Example:
        ```python
        from metaxy.metadata_store.duckdb import DuckDBMetadataStoreConfig

        config = DuckDBMetadataStoreConfig(
            database="metadata.db",
            hash_algorithm=HashAlgorithm.MD5,
        )

        store = DuckDBMetadataStore.from_config(config)
        ```
    """

    model_config = SettingsConfigDict(frozen=True, extra="forbid")

    fallback_stores: list[str] = Field(
        default_factory=list,
        description="List of fallback store names to search when features are not found in the current store.",
    )

    hash_algorithm: HashAlgorithm | None = Field(
        default=None,
        description="Hash algorithm for versioning. If None, uses store's default.",
    )

    versioning_engine: Literal["auto", "native", "polars"] = Field(
        default="auto",
        description="Which versioning engine to use: 'auto' (prefer native), 'native', or 'polars'.",
    )


VersioningEngineOptions = Literal["auto", "native", "polars"]

# Mapping of system columns to their expected Narwhals dtypes
# Used to cast Null-typed columns to correct types
# Note: Struct columns (METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD) are not cast
_SYSTEM_COLUMN_DTYPES = {
    METAXY_PROVENANCE: nw.String,
    METAXY_FEATURE_VERSION: nw.String,
    METAXY_FEATURE_SPEC_VERSION: nw.String,
    METAXY_SNAPSHOT_VERSION: nw.String,
    METAXY_DATA_VERSION: nw.String,
    METAXY_CREATED_AT: nw.Datetime,
    METAXY_MATERIALIZATION_ID: nw.String,
}


def _cast_present_system_columns(
    df: nw.DataFrame[Any] | nw.LazyFrame[Any],
) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
    """Cast system columns with Null/Unknown dtype to their correct types.

    This handles edge cases where empty DataFrames or certain operations
    result in Null-typed columns (represented as nw.Unknown in Narwhals)
    that break downstream processing.

    Args:
        df: Narwhals DataFrame or LazyFrame

    Returns:
        DataFrame with system columns cast to correct types
    """
    schema = df.collect_schema()
    columns_to_cast = []

    for col_name, expected_dtype in _SYSTEM_COLUMN_DTYPES.items():
        if col_name in schema and schema[col_name] == nw.Unknown:
            columns_to_cast.append(nw.col(col_name).cast(expected_dtype))

    if columns_to_cast:
        df = df.with_columns(columns_to_cast)

    return df


class MetadataStore(ABC):
    """
    Abstract base class for metadata storage backends.
    """

    # Subclasses can override this to disable auto_create_tables warning
    # Set to False for stores where table creation is not applicable (e.g., InMemoryMetadataStore)
    _should_warn_auto_create_tables: bool = True

    # Subclasses must define the versioning engine class to use
    versioning_engine_cls: type[VersioningEngine]

    def __init__(
        self,
        *,
        hash_algorithm: HashAlgorithm | None = None,
        versioning_engine: VersioningEngineOptions = "auto",
        fallback_stores: list[MetadataStore] | None = None,
        auto_create_tables: bool | None = None,
        materialization_id: str | None = None,
    ):
        """
        Initialize the metadata store.

        Args:
            hash_algorithm: Hash algorithm to use for the versioning engine.

            versioning_engine: Which versioning engine to use.

                - "auto": Prefer the store's native engine, fall back to Polars if needed

                - "native": Always use the store's native engine, raise `VersioningEngineMismatchError`
                    if provided dataframes are incompatible

                - "polars": Always use the Polars engine

            fallback_stores: Ordered list of read-only fallback stores.
                Used when upstream features are not in this store.
                `VersioningEngineMismatchError` is not raised when reading from fallback stores.
            auto_create_tables: If True, automatically create tables when opening the store.
                If None (default), reads from global MetaxyConfig (which reads from METAXY_AUTO_CREATE_TABLES env var).
                If False, never auto-create tables.

                !!! warning
                    Auto-create is intended for development/testing only.
                    Use proper database migration tools like Alembic for production deployments.

            materialization_id: Optional external orchestration ID.
                If provided, all metadata writes will include this ID in the `metaxy_materialization_id` column.
                Can be overridden per [`MetadataStore.write_metadata`][metaxy.MetadataStore.write_metadata] call.

        Raises:
            ValueError: If fallback stores use different hash algorithms or truncation lengths
            VersioningEngineMismatchError: If a user-provided dataframe has a wrong implementation
                and versioning_engine is set to `native`
        """
        # Initialize state early so properties can check it
        self._is_open = False
        self._context_depth = 0
        self._versioning_engine = versioning_engine
        self._allow_cross_project_writes = False
        self._materialization_id = materialization_id
        self._open_cm: AbstractContextManager[Self] | None = (
            None  # Track the open() context manager
        )

        # Resolve auto_create_tables from global config if not explicitly provided
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            self.auto_create_tables = MetaxyConfig.get().auto_create_tables
        else:
            self.auto_create_tables = auto_create_tables

        # Use store's default algorithm if not specified
        if hash_algorithm is None:
            hash_algorithm = self._get_default_hash_algorithm()

        self.hash_algorithm = hash_algorithm

        self.fallback_stores = fallback_stores or []

    @overload
    def resolve_update(
        self,
        feature: CoercibleToFeatureKey,
        *,
        samples: IntoFrame | Frame | None = None,
        filters: Mapping[CoercibleToFeatureKey, Sequence[nw.Expr]] | None = None,
        global_filters: Sequence[nw.Expr] | None = None,
        lazy: Literal[False] = False,
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        skip_comparison: bool = False,
        **kwargs: Any,
    ) -> Increment: ...

    @overload
    def resolve_update(
        self,
        feature: CoercibleToFeatureKey,
        *,
        samples: IntoFrame | Frame | None = None,
        filters: Mapping[CoercibleToFeatureKey, Sequence[nw.Expr]] | None = None,
        global_filters: Sequence[nw.Expr] | None = None,
        lazy: Literal[True],
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        skip_comparison: bool = False,
        **kwargs: Any,
    ) -> LazyIncrement: ...

    def resolve_update(
        self,
        feature: CoercibleToFeatureKey,
        *,
        samples: IntoFrame | Frame | None = None,
        filters: Mapping[CoercibleToFeatureKey, Sequence[nw.Expr]] | None = None,
        global_filters: Sequence[nw.Expr] | None = None,
        lazy: bool = False,
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        skip_comparison: bool = False,
        **kwargs: Any,
    ) -> Increment | LazyIncrement:
        """Calculate an incremental update for a feature.

        This is the main workhorse in Metaxy.

        Args:
            feature: Feature class to resolve updates for
            samples: A dataframe with joined upstream metadata and `"metaxy_provenance_by_field"` column set.
                When provided, `MetadataStore` skips loading upstream feature metadata and provenance calculations.

                !!! info "Required for root features"
                    Metaxy doesn't know how to populate input metadata for root features,
                    so `samples` argument for **must** be provided for them.

                !!! tip
                    For non-root features, use `samples` to customize the automatic upstream loading and field provenance calculation.
                    For example, it can be used to requires processing for specific sample IDs.

                Setting this parameter during normal operations is not required.

            filters: A mapping from feature keys to lists of Narwhals filter expressions.
                Keys can be feature classes, FeatureKey objects, or string paths.
                Applied at read-time. May filter the current feature,
                in this case it will also be applied to `samples` (if provided).
                Example: `{UpstreamFeature: [nw.col("x") > 10], ...}`
            global_filters: A list of Narwhals filter expressions applied to all features.
                These filters are combined with any feature-specific filters from `filters`.
                Useful for filtering by common columns like `sample_uid` across all features.
                Example: `[nw.col("sample_uid").is_in(["s1", "s2"])]`
            lazy: Whether to return a [metaxy.versioning.types.LazyIncrement][] or a [metaxy.versioning.types.Increment][].
            versioning_engine: Override the store's versioning engine for this operation.
            skip_comparison: If True, skip the increment comparison logic and return all
                upstream samples in `Increment.added`. The `changed` and `removed` frames will
                be empty.

        Raises:
            ValueError: If no `samples` dataframe has been provided when resolving an update for a root feature.
            VersioningEngineMismatchError: If `versioning_engine` has been set to `"native"`
                and a dataframe of a different implementation has been encountered during `resolve_update`.

        !!! example "With a root feature"

            ```py
            samples = pl.DataFrame({
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [{"field": "h1"}, {"field": "h2"}, {"field": "h3"}],
            })
            result = store.resolve_update(RootFeature, samples=nw.from_native(samples))
            ```
        """
        import narwhals as nw

        # Convert samples to Narwhals frame if not already
        samples_nw: nw.DataFrame[Any] | nw.LazyFrame[Any] | None = None
        if samples is not None:
            if isinstance(samples, (nw.DataFrame, nw.LazyFrame)):
                samples_nw = samples
            else:
                samples_nw = nw.from_native(samples)  # ty: ignore[invalid-assignment]

        # Normalize filter keys to FeatureKey
        normalized_filters: dict[FeatureKey, list[nw.Expr]] = {}
        if filters:
            for key, exprs in filters.items():
                feature_key = self._resolve_feature_key(key)
                normalized_filters[feature_key] = list(exprs)

        # Convert global_filters to a list for easy concatenation
        global_filter_list = list(global_filters) if global_filters else []

        feature_key = self._resolve_feature_key(feature)
        graph = current_graph()
        plan = graph.get_feature_plan(feature_key)

        # Root features without samples: error (samples required)
        if not plan.deps and samples_nw is None:
            raise ValueError(
                f"Feature {feature_key} has no upstream dependencies (root feature). "
                f"Must provide 'samples' parameter with sample_uid and {METAXY_PROVENANCE_BY_FIELD} columns. "
                f"Root features require manual {METAXY_PROVENANCE_BY_FIELD} computation."
            )

        # Combine feature-specific filters with global filters
        current_feature_filters = [
            *normalized_filters.get(feature_key, []),
            *global_filter_list,
        ]

        # Read current metadata with deduplication (latest_only=True by default)
        # Use allow_fallback=False since we only want metadata from THIS store
        # to determine what needs to be updated locally
        try:
            current_metadata: nw.LazyFrame[Any] | None = self.read_metadata(
                feature_key,
                filters=current_feature_filters if current_feature_filters else None,
                allow_fallback=False,
                current_only=True,  # filters by current feature_version
                latest_only=True,  # deduplicates by id_columns, keeping latest
            )
        except FeatureNotFoundError:
            current_metadata = None

        upstream_by_key: dict[FeatureKey, nw.LazyFrame[Any]] = {}
        filters_by_key: dict[FeatureKey, list[nw.Expr]] = {}

        # if samples are provided, use them as source of truth for upstream data
        if samples_nw is not None:
            # Apply filters to samples if any
            filtered_samples = samples_nw
            if current_feature_filters:
                filtered_samples = samples_nw.filter(current_feature_filters)

            # fill in METAXY_PROVENANCE column if it's missing (e.g. for root features)
            samples_nw = self.hash_struct_version_column(
                plan,
                df=filtered_samples,
                struct_column=METAXY_PROVENANCE_BY_FIELD,
                hash_column=METAXY_PROVENANCE,
            )

            # For root features, add data_version columns if they don't exist
            # (root features have no computation, so data_version equals provenance)
            # Use collect_schema().names() to avoid PerformanceWarning on lazy frames
            if METAXY_DATA_VERSION_BY_FIELD not in samples_nw.collect_schema().names():
                samples_nw = samples_nw.with_columns(
                    nw.col(METAXY_PROVENANCE_BY_FIELD).alias(
                        METAXY_DATA_VERSION_BY_FIELD
                    ),
                    nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
                )
        else:
            for upstream_spec in plan.deps or []:
                # Combine feature-specific filters with global filters for upstream
                upstream_filters = [
                    *normalized_filters.get(upstream_spec.key, []),
                    *global_filter_list,
                ]
                upstream_feature_metadata = self.read_metadata(
                    upstream_spec.key,
                    filters=upstream_filters,
                )
                if upstream_feature_metadata is not None:
                    upstream_by_key[upstream_spec.key] = upstream_feature_metadata

        # determine which implementation to use for resolving the increment
        # consider (1) whether all upstream metadata has been loaded with the native implementation
        # (2) if samples have native implementation

        # Use parameter if provided, otherwise use store default
        engine_mode = (
            versioning_engine
            if versioning_engine is not None
            else self._versioning_engine
        )

        # If "polars" mode, force Polars immediately
        if engine_mode == "polars":
            implementation = nw.Implementation.POLARS
            switched_to_polars = True
        else:
            implementation = self.native_implementation()
            switched_to_polars = False

            for upstream_key, df in upstream_by_key.items():
                if df.implementation != implementation:
                    switched_to_polars = True
                    # Only raise error in "native" mode if no fallback stores configured.
                    # If fallback stores exist, the implementation mismatch indicates data came
                    # from fallback (different implementation), which is legitimate fallback access.
                    # If data were local, it would have the native implementation.
                    if engine_mode == "native" and not self.fallback_stores:
                        raise VersioningEngineMismatchError(
                            f"versioning_engine='native' but upstream feature `{upstream_key.to_string()}` "
                            f"has implementation {df.implementation}, expected {self.native_implementation()}"
                        )
                    elif engine_mode == "auto" or (
                        engine_mode == "native" and self.fallback_stores
                    ):
                        PolarsMaterializationWarning.warn_on_implementation_mismatch(
                            expected=self.native_implementation(),
                            actual=df.implementation,
                            message=f"Using Polars for resolving the increment instead. This was caused by upstream feature `{upstream_key.to_string()}`.",
                        )
                    implementation = nw.Implementation.POLARS
                    break

            if (
                samples_nw is not None
                and samples_nw.implementation != self.native_implementation()
            ):
                if not switched_to_polars:
                    if engine_mode == "native":
                        # Always raise error for samples with wrong implementation, regardless
                        # of fallback stores, because samples come from user argument, not from fallback
                        raise VersioningEngineMismatchError(
                            f"versioning_engine='native' but provided `samples` have implementation {samples_nw.implementation}, "
                            f"expected {self.native_implementation()}"
                        )
                    elif engine_mode == "auto":
                        PolarsMaterializationWarning.warn_on_implementation_mismatch(
                            expected=self.native_implementation(),
                            actual=samples_nw.implementation,
                            message=f"Provided `samples` have implementation {samples_nw.implementation}. Using Polars for resolving the increment instead.",
                        )
                implementation = nw.Implementation.POLARS
                switched_to_polars = True

        if switched_to_polars:
            if current_metadata:
                current_metadata = switch_implementation_to_polars(current_metadata)
            if samples_nw:
                samples_nw = switch_implementation_to_polars(samples_nw)
            for upstream_key, df in upstream_by_key.items():
                upstream_by_key[upstream_key] = switch_implementation_to_polars(df)

        with self.create_versioning_engine(
            plan=plan, implementation=implementation
        ) as engine:
            if skip_comparison:
                # Skip comparison: return all upstream samples as added
                if samples_nw is not None:
                    # Root features or user-provided samples: use samples directly
                    # Note: samples already has metaxy_provenance computed
                    added = samples_nw.lazy()
                    input_df = None  # Root features have no upstream input
                else:
                    # Non-root features: load all upstream with provenance
                    added = engine.load_upstream_with_provenance(
                        upstream=upstream_by_key,
                        hash_algo=self.hash_algorithm,
                        filters=filters_by_key,
                    )
                    input_df = (
                        added  # Input is the same as added when skipping comparison
                    )
                changed = None
                removed = None
            else:
                added, changed, removed, input_df = (
                    engine.resolve_increment_with_provenance(
                        current=current_metadata,
                        upstream=upstream_by_key,
                        hash_algorithm=self.hash_algorithm,
                        filters=filters_by_key,
                        sample=samples_nw.lazy() if samples_nw is not None else None,
                    )
                )

        # Convert None to empty DataFrames
        if changed is None:
            changed = empty_frame_like(added)
        if removed is None:
            removed = empty_frame_like(added)

        if lazy:
            return LazyIncrement(
                added=added
                if isinstance(added, nw.LazyFrame)
                else nw.from_native(added),
                changed=changed
                if isinstance(changed, nw.LazyFrame)
                else nw.from_native(changed),
                removed=removed
                if isinstance(removed, nw.LazyFrame)
                else nw.from_native(removed),
                input=input_df
                if input_df is None or isinstance(input_df, nw.LazyFrame)
                else nw.from_native(input_df),
            )
        else:
            return Increment(
                added=added.collect() if isinstance(added, nw.LazyFrame) else added,
                changed=changed.collect()
                if isinstance(changed, nw.LazyFrame)
                else changed,
                removed=removed.collect()
                if isinstance(removed, nw.LazyFrame)
                else removed,
            )

    def compute_provenance(
        self,
        feature: CoercibleToFeatureKey,
        df: FrameT,
    ) -> FrameT:
        """Compute provenance columns for a DataFrame with pre-joined upstream data.

        !!! note
            This method may be useful in very rare cases.
            Rely on [`MetadataStore.resolve_update`][metaxy.metadata_store.base.MetadataStore.resolve_update] instead.

        Use this method when you perform custom joins outside of Metaxy's auto-join
        system but still want Metaxy to compute provenance. The method computes
        metaxy_provenance_by_field, metaxy_provenance, metaxy_data_version_by_field,
        and metaxy_data_version columns based on the upstream metadata.

        !!! info
            The input DataFrame must contain the renamed metaxy_data_version_by_field
            columns from each upstream feature. The naming convention follows the pattern
            `metaxy_data_version_by_field__<feature_key.to_column_suffix()>`. For example, for an
            upstream feature with key `["video", "raw"]`, the column should be named
            `metaxy_data_version_by_field__video_raw`.

        Args:
            feature: The feature to compute provenance for.
            df: A DataFrame containing pre-joined upstream data with renamed
                metaxy_data_version_by_field columns from each upstream feature.

        Returns:
            The input DataFrame with provenance columns added. Returns the same
            frame type as the input, either an eager DataFrame or a LazyFrame.

        Raises:
            StoreNotOpenError: If the store is not open.
            ValueError: If required upstream `metaxy_data_version_by_field` columns
                are missing from the DataFrame.

        Example:
            ```py

                # Read upstream metadata
                video_df = store.read_metadata(VideoFeature).collect()
                audio_df = store.read_metadata(AudioFeature).collect()

                # Rename data_version_by_field columns to the expected convention
                video_df = video_df.rename({
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__video_raw"
                })
                audio_df = audio_df.rename({
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__audio_raw"
                })

                # Perform custom join
                joined = video_df.join(audio_df, on="sample_uid", how="inner")

                # Compute provenance
                with_provenance = store.compute_provenance(MyFeature, joined)

                # Pass to resolve_update
                increment = store.resolve_update(MyFeature, samples=with_provenance)
            ```
        """
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        graph = current_graph()
        plan = graph.get_feature_plan(feature_key)

        # Use native implementation if DataFrame matches, otherwise fall back to Polars
        implementation = self.native_implementation()
        if df.implementation != implementation:
            implementation = nw.Implementation.POLARS
            df = switch_implementation_to_polars(df)  # ty: ignore[no-matching-overload]

        with self.create_versioning_engine(
            plan=plan, implementation=implementation
        ) as engine:
            # Validate required upstream columns exist
            expected_columns = {
                dep.feature: engine.get_renamed_data_version_by_field_col(dep.feature)
                for dep in (plan.feature_deps or [])
            }

            df_columns = set(df.collect_schema().names())  # ty: ignore[invalid-argument-type]
            missing_columns = [
                f"{col} (from upstream feature {key.to_string()})"
                for key, col in expected_columns.items()
                if col not in df_columns
            ]

            if missing_columns:
                raise ValueError(
                    f"DataFrame is missing required upstream columns for computing "
                    f"provenance of feature {feature_key.to_string()}. "
                    f"Missing columns: {missing_columns}. "
                    f"Make sure to rename metaxy_data_version_by_field columns from "
                    f"each upstream feature using the pattern "
                    f"metaxy_data_version_by_field__<feature_key.table_name>."
                )

            return engine.compute_provenance_columns(df, hash_algo=self.hash_algorithm)  # ty: ignore[invalid-argument-type]

    def read_metadata(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        allow_fallback: bool = True,
        current_only: bool = True,
        latest_only: bool = True,
    ) -> nw.LazyFrame[Any]:
        """
        Read metadata with optional fallback to upstream stores.

        Args:
            feature: Feature to read metadata for
            feature_version: Explicit feature_version to filter by (mutually exclusive with current_only=True)
            filters: Sequence of Narwhals filter expressions to apply to this feature.
                Example: `[nw.col("x") > 10, nw.col("y") < 5]`
            columns: Subset of columns to include. Metaxy's system columns are always included.
            allow_fallback: If `True`, check fallback stores on local miss
            current_only: If `True`, only return rows with current feature_version
            latest_only: Whether to deduplicate samples within `id_columns` groups ordered by `metaxy_created_at`.

        Returns:
            Narwhals LazyFrame with metadata

        Raises:
            FeatureNotFoundError: If feature not found in any store
            SystemDataNotFoundError: When attempting to read non-existent Metaxy system data
            ValueError: If both feature_version and current_only=True are provided

        !!! info
            When this method is called with default arguments, it will return the latest (by `metaxy_created_at`)
            metadata for the current feature version. Therefore, it's perfectly suitable for most use cases.

        !!! warning
            The order of rows is not guaranteed.
        """
        self._check_open()

        filters = filters or []
        columns = columns or []

        feature_key = self._resolve_feature_key(feature)
        is_system_table = self._is_system_table(feature_key)

        # Validate mutually exclusive parameters
        if feature_version is not None and current_only:
            raise ValueError(
                "Cannot specify both feature_version and current_only=True. "
                "Use current_only=False with feature_version parameter."
            )

        # Add feature_version filter only when needed
        if current_only or feature_version is not None and not is_system_table:
            version_filter = nw.col(METAXY_FEATURE_VERSION) == (
                current_graph().get_feature_version(feature_key)
                if current_only
                else feature_version
            )
            filters = [version_filter, *filters]

        if columns and not is_system_table:
            # Add only system columns that aren't already in the user's columns list
            columns_set = set(columns)
            missing_system_cols = [
                c for c in ALL_SYSTEM_COLUMNS if c not in columns_set
            ]
            read_columns = [*columns, *missing_system_cols]
        else:
            read_columns = None

        lazy_frame = None
        try:
            lazy_frame = self.read_metadata_in_store(
                feature, filters=filters, columns=read_columns
            )
        except FeatureNotFoundError as e:
            # do not read system features from fallback stores
            if is_system_table:
                raise SystemDataNotFoundError(
                    f"System Metaxy data with key {feature_key} is missing in {self.display()}. Invoke `metaxy graph push` before attempting to read system data."
                ) from e

        # Handle case where read_metadata_in_store returns None (no exception raised)
        if lazy_frame is None and is_system_table:
            raise SystemDataNotFoundError(
                f"System Metaxy data with key {feature_key} is missing in {self.display()}. Invoke `metaxy graph push` before attempting to read system data."
            )

        if lazy_frame is not None and not is_system_table and latest_only:
            from metaxy.models.constants import METAXY_CREATED_AT

            # Apply deduplication
            lazy_frame = self.versioning_engine_cls.keep_latest_by_group(
                df=lazy_frame,
                group_columns=list(
                    self._resolve_feature_plan(feature_key).feature.id_columns
                ),
                timestamp_column=METAXY_CREATED_AT,
            )

        if lazy_frame is not None:
            # After dedup, filter to requested columns if specified
            if columns:
                lazy_frame = lazy_frame.select(columns)

            return lazy_frame

        # Try fallback stores (opened on demand)
        if allow_fallback:
            for store in self.fallback_stores:
                try:
                    # Open fallback store on demand for reading
                    with store:
                        # Use full read_metadata to handle nested fallback chains
                        return store.read_metadata(
                            feature,
                            feature_version=feature_version,
                            filters=filters,
                            columns=columns,
                            allow_fallback=True,
                            current_only=current_only,
                            latest_only=latest_only,
                        )
                except FeatureNotFoundError:
                    # Try next fallback store
                    continue

        # Not found anywhere
        raise FeatureNotFoundError(
            f"Feature {feature_key.to_string()} not found in store"
            + (" or fallback stores" if allow_fallback else "")
        )

    def write_metadata(
        self,
        feature: CoercibleToFeatureKey,
        df: IntoFrame,
        materialization_id: str | None = None,
    ) -> None:
        """
        Write metadata for a feature (append-only by design).

        Automatically adds the Metaxy system columns, unless they already exist in the DataFrame.

        Args:
            feature: Feature to write metadata for
            df: Metadata DataFrame of any type supported by [Narwhals](https://narwhals-dev.github.io/narwhals/).
                Must have `metaxy_provenance_by_field` column of type Struct with fields matching feature's fields.
                Optionally, may also contain `metaxy_data_version_by_field`.
            materialization_id: Optional external orchestration ID for this write.
                Overrides the store's default `materialization_id` if provided.
                Useful for tracking which orchestration run produced this metadata.

        Raises:
            MetadataSchemaError: If DataFrame schema is invalid
            StoreNotOpenError: If store is not open
            ValueError: If writing to a feature from a different project than expected

        Note:
            - Must be called within a `MetadataStore.open(mode="write")` context manager.

            - Metaxy always performs an "append" operation. Metadata is never deleted or mutated.

            - Fallback stores are never used for writes.

            - Features from other Metaxy projects cannot be written to, unless project validation has been disabled with [MetadataStore.allow_cross_project_writes][].

        """
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        is_system_table = self._is_system_table(feature_key)

        # Validate project for non-system tables
        if not is_system_table:
            self._validate_project_write(feature)

        # Convert Polars to Narwhals to Polars if needed
        # if isinstance(df_nw, (pl.DataFrame, pl.LazyFrame)):
        df_nw = nw.from_native(df)

        assert isinstance(df_nw, (nw.DataFrame, nw.LazyFrame)), (
            f"df must be a Narwhals DataFrame, got {type(df_nw)}"
        )

        # For system tables, write directly without feature_version tracking
        if is_system_table:
            self._validate_schema_system_table(df_nw)
            self.write_metadata_to_store(feature_key, df_nw)
            return

        # Use collect_schema().names() to avoid PerformanceWarning on lazy frames
        if METAXY_PROVENANCE_BY_FIELD not in df_nw.collect_schema().names():
            from metaxy.metadata_store.exceptions import MetadataSchemaError

            raise MetadataSchemaError(
                f"DataFrame must have '{METAXY_PROVENANCE_BY_FIELD}' column"
            )

        # Add all required system columns
        # warning: for dataframes that do not match the native MetadataStore implementation
        # and are missing the METAXY_DATA_VERSION column, this call will lead to materializing the equivalent Polars DataFrame
        # while calculating the missing METAXY_DATA_VERSION column
        df_nw = self._add_system_columns(
            df_nw, feature, materialization_id=materialization_id
        )

        self._validate_schema(df_nw)
        self.write_metadata_to_store(feature_key, df_nw)

    def write_metadata_multi(
        self,
        metadata: Mapping[Any, IntoFrame],
        materialization_id: str | None = None,
    ) -> None:
        """
        Write metadata for multiple features in reverse topological order.

        Processes features so that dependents are written before their dependencies.
        This ordering ensures that downstream features are written first, which can
        be useful for certain data consistency requirements or when features need
        to be processed in a specific order.

        Args:
            metadata: Mapping from feature keys to metadata DataFrames.
                Keys can be any type coercible to FeatureKey (string, sequence,
                FeatureKey, or BaseFeature class). Values must be DataFrames
                compatible with Narwhals, containing required system columns.
            materialization_id: Optional external orchestration ID for all writes.
                Overrides the store's default `materialization_id` if provided.
                Applied to all feature writes in this batch.

        Raises:
            MetadataSchemaError: If any DataFrame schema is invalid
            StoreNotOpenError: If store is not open
            ValueError: If writing to a feature from a different project than expected

        Note:
            - Must be called within a `MetadataStore.open(mode="write")` context manager.
            - Empty mappings are handled gracefully (no-op).
            - Each feature's metadata is written via `write_metadata`, so all
              validation and system column handling from that method applies.

        Example:
            ```py
            with store.open(mode="write"):
                store.write_metadata_multi({
                    ChildFeature: child_df,
                    ParentFeature: parent_df,
                })
            # Features are written in reverse topological order:
            # ChildFeature first, then ParentFeature
            ```
        """
        if not metadata:
            return

        # Build mapping from resolved keys to dataframes in one pass
        resolved_metadata = {
            self._resolve_feature_key(key): df for key, df in metadata.items()
        }

        # Get reverse topological order (dependents first)
        graph = current_graph()
        sorted_keys = graph.topological_sort_features(
            list(resolved_metadata.keys()), descending=True
        )

        # Write metadata in reverse topological order
        for feature_key in sorted_keys:
            self.write_metadata(
                feature_key,
                resolved_metadata[feature_key],
                materialization_id=materialization_id,
            )

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[MetadataStoreConfig]:
        """Return the configuration model class for this store type.

        Subclasses must override this to return their specific config class.

        Returns:
            The config class type (e.g., DuckDBMetadataStoreConfig)

        Note:
            Subclasses override this with a more specific return type.
            Type checkers may show a warning about incompatible override,
            but this is intentional - each store returns its own config type.
        """
        ...

    @classmethod
    def from_config(cls, config: MetadataStoreConfig, **kwargs: Any) -> Self:
        """Create a store instance from a configuration object.

        This method creates a store by:
        1. Converting the config to a dict
        2. Resolving fallback store names to actual store instances
        3. Calling the store's __init__ with the config parameters

        Args:
            config: Configuration object (should be the type returned by config_model())
            **kwargs: Additional arguments passed directly to the store constructor
                (e.g., materialization_id for runtime parameters not in config)

        Returns:
            A new store instance configured according to the config object

        Example:
            ```python
            from metaxy.metadata_store.duckdb import (
                DuckDBMetadataStore,
                DuckDBMetadataStoreConfig,
            )

            config = DuckDBMetadataStoreConfig(
                database="metadata.db",
                fallback_stores=["prod"],
            )

            store = DuckDBMetadataStore.from_config(config)
            ```
        """
        # Convert config to dict, excluding unset values
        config_dict = config.model_dump(exclude_unset=True)

        # Pop and resolve fallback store names to actual store instances
        fallback_store_names = config_dict.pop("fallback_stores", [])
        fallback_stores = [
            MetaxyConfig.get().get_store(name) for name in fallback_store_names
        ]

        # Create store with resolved fallback stores, config, and extra kwargs
        return cls(fallback_stores=fallback_stores, **config_dict, **kwargs)

    @property
    def hash_truncation_length(self) -> int:
        return MetaxyConfig.get().hash_truncation_length or 64

    @property
    def materialization_id(self) -> str | None:
        """The external orchestration ID for this store instance.

        If set, all metadata writes include this ID in the `metaxy_materialization_id` column,
        allowing filtering of rows written during a specific materialization run.
        """
        return self._materialization_id

    @abstractmethod
    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get the default hash algorithm for this store type.

        Returns:
            Default hash algorithm
        """
        pass

    def native_implementation(self) -> nw.Implementation:
        """Get the native Narwhals implementation for this store's backend."""
        return self.versioning_engine_cls.implementation()

    @abstractmethod
    @contextmanager
    def _create_versioning_engine(
        self, plan: FeaturePlan
    ) -> Iterator[VersioningEngine]:
        """Create provenance engine for this store as a context manager.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            VersioningEngine instance appropriate for this store's backend.
            - For SQL stores (DuckDB, ClickHouse): Returns IbisVersioningEngine
            - For in-memory/Polars stores: Returns PolarsVersioningEngine

        Raises:
            NotImplementedError: If provenance tracking not supported by this store

        Example:
            ```python
            with self._create_versioning_engine(plan) as engine:
                result = engine.resolve_update(...)
            ```
        """
        ...

    @contextmanager
    def _create_polars_versioning_engine(
        self, plan: FeaturePlan
    ) -> Iterator[PolarsVersioningEngine]:
        yield PolarsVersioningEngine(plan=plan)

    @contextmanager
    def create_versioning_engine(
        self, plan: FeaturePlan, implementation: nw.Implementation
    ) -> Iterator[VersioningEngine | PolarsVersioningEngine]:
        """
        Creates an appropriate provenance engine.

        Falls back to Polars implementation if the required implementation differs from the store's native implementation.

        Args:
            plan: The feature plan.
            implementation: The desired engine implementation.

        Returns:
            An appropriate provenance engine.
        """

        if implementation == nw.Implementation.POLARS:
            cm = self._create_polars_versioning_engine(plan)
        elif implementation == self.native_implementation():
            cm = self._create_versioning_engine(plan)
        else:
            cm = self._create_polars_versioning_engine(plan)

        with cm as engine:
            yield engine

    def hash_struct_version_column(
        self,
        plan: FeaturePlan,
        df: Frame,
        struct_column: str,
        hash_column: str,
    ) -> Frame:
        with self.create_versioning_engine(plan, df.implementation) as engine:
            if (
                isinstance(engine, PolarsVersioningEngine)
                and df.implementation != nw.Implementation.POLARS
            ):
                PolarsMaterializationWarning.warn_on_implementation_mismatch(
                    self.native_implementation(),
                    df.implementation,
                    message=f"`{hash_column}` will be calculated in Polars.",
                )
                df = nw.from_native(df.lazy().collect().to_polars())

            return cast(
                Frame,
                engine.hash_struct_version_column(
                    df,  # ty: ignore[invalid-argument-type]
                    hash_algorithm=self.hash_algorithm,
                    struct_column=struct_column,
                    hash_column=hash_column,
                ),
            )

    @abstractmethod
    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:
        """Open/initialize the store for operations.

        Context manager that opens the store with specified access mode.
        Called internally by `__enter__`.
        Child classes should implement backend-specific connection setup/teardown here.

        Args:
            mode: Access mode for this connection session.

        Yields:
            Self: The store instance with connection open

        Note:
            Users should prefer using `with store:` pattern except when write access mode is needed.
        """
        ...

    def __enter__(self) -> Self:
        """Enter context manager - opens store in READ mode by default.

        Use [`MetadataStore.open`][metaxy.metadata_store.base.MetadataStore.open] for write access mode instead.

        Returns:
            Self: The opened store instance
        """
        # Determine mode based on auto_create_tables
        mode = "write" if self.auto_create_tables else "read"

        # Open the store (open() manages _context_depth internally)
        self._open_cm = self.open(mode)  # ty: ignore[invalid-assignment]
        self._open_cm.__enter__()  # ty: ignore[possibly-missing-attribute]

        return self

    def _validate_after_open(self) -> None:
        """Validate configuration after store is opened.

        Called automatically by __enter__ after open().
        Validates hash algorithm compatibility and fallback store consistency.
        """
        # Validate hash algorithm compatibility with components
        self.validate_hash_algorithm(check_fallback_stores=True)

        # Validate fallback stores use the same hash algorithm
        for i, fallback_store in enumerate(self.fallback_stores):
            if fallback_store.hash_algorithm != self.hash_algorithm:
                raise ValueError(
                    f"Fallback store {i} uses hash_algorithm='{fallback_store.hash_algorithm.value}' "
                    f"but this store uses '{self.hash_algorithm.value}'. "
                    f"All stores in a fallback chain must use the same hash algorithm."
                )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Delegate to open()'s context manager (which manages _context_depth)
        if self._open_cm is not None:
            self._open_cm.__exit__(exc_type, exc_val, exc_tb)
            self._open_cm = None

    def _check_open(self) -> None:
        """Check if store is open, raise error if not.

        Raises:
            StoreNotOpenError: If store is not open
        """
        if not self._is_open:
            raise StoreNotOpenError(
                f"{self.__class__.__name__} must be opened before use. "
                'Use it as a context manager: `with store: ...` or `with store.open(mode="write"): ...`'
            )

    # ========== Hash Algorithm Validation ==========

    def validate_hash_algorithm(
        self,
        check_fallback_stores: bool = True,
    ) -> None:
        """Validate that hash algorithm is supported by this store's components.

        Public method - can be called to verify hash compatibility.

        Args:
            check_fallback_stores: If True, also validate hash is supported by
                fallback stores (ensures compatibility for future cross-store operations)

        Raises:
            ValueError: If hash algorithm not supported by components or fallback stores
        """
        # Validate hash algorithm support without creating a full engine
        # (engine creation requires a graph which isn't available during store init)
        self._validate_hash_algorithm_support()

        # Check fallback stores
        if check_fallback_stores:
            for fallback in self.fallback_stores:
                fallback.validate_hash_algorithm(check_fallback_stores=False)

    def _validate_hash_algorithm_support(self) -> None:
        """Validate that the configured hash algorithm is supported.

        Default implementation does nothing (assumes all algorithms supported).
        Subclasses can override to check algorithm support.

        Raises:
            Exception: If hash algorithm is not supported
        """
        # Default: no validation (assume all algorithms supported)
        pass

    # ========== Helper Methods ==========

    def _is_system_table(self, feature_key: FeatureKey) -> bool:
        """Check if feature key is a system table."""
        return len(feature_key) >= 1 and feature_key[0] == METAXY_SYSTEM_KEY_PREFIX

    def _resolve_feature_key(self, feature: CoercibleToFeatureKey) -> FeatureKey:
        """Resolve various types to FeatureKey.

        Accepts types that can be converted into a FeatureKey.

        Args:
            feature: Feature to resolve to FeatureKey

        Returns:
            FeatureKey instance
        """
        return ValidatedFeatureKeyAdapter.validate_python(feature)

    def _resolve_feature_plan(self, feature: CoercibleToFeatureKey) -> FeaturePlan:
        """Resolve to FeaturePlan for dependency resolution."""
        # First resolve to FeatureKey
        feature_key = self._resolve_feature_key(feature)
        # Then get the plan
        graph = current_graph()
        return graph.get_feature_plan(feature_key)

    # ========== Core CRUD Operations ==========

    @contextmanager
    def allow_cross_project_writes(self) -> Iterator[None]:
        """Context manager to temporarily allow cross-project writes.

        This is an escape hatch for legitimate cross-project operations like migrations,
        where metadata needs to be written to features from different projects.

        Example:
            ```py
            # During migration, allow writing to features from different projects
            with store.allow_cross_project_writes():
                store.write_metadata(feature_from_project_a, metadata_a)
                store.write_metadata(feature_from_project_b, metadata_b)
            ```

        Yields:
            None: The context manager temporarily disables project validation
        """
        previous_value = self._allow_cross_project_writes
        try:
            self._allow_cross_project_writes = True
            yield
        finally:
            self._allow_cross_project_writes = previous_value

    def _validate_project_write(self, feature: CoercibleToFeatureKey) -> None:
        """Validate that writing to a feature matches the expected project from config.

        Args:
            feature: Feature to validate project for

        Raises:
            ValueError: If feature's project doesn't match the global config project
        """
        # Skip validation if cross-project writes are allowed
        if self._allow_cross_project_writes:
            return

        # Get the expected project from global config
        from metaxy.config import MetaxyConfig

        config = MetaxyConfig.get()
        expected_project = config.project

        # Use existing method to resolve to FeatureKey
        feature_key = self._resolve_feature_key(feature)

        # Get the Feature class from the graph

        graph = FeatureGraph.get_active()
        if feature_key not in graph.features_by_key:
            # Feature not in graph - can't validate, skip
            return

        feature_cls = graph.features_by_key[feature_key]
        feature_project = feature_cls.project

        # Validate the project matches
        if feature_project != expected_project:
            raise ValueError(
                f"Cannot write to feature {feature_key.to_string()} from project '{feature_project}' "
                f"when the global configuration expects project '{expected_project}'. "
                f"Use store.allow_cross_project_writes() context manager for legitimate "
                f"cross-project operations like migrations."
            )

    @abstractmethod
    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """
        Internal write implementation (backend-specific).

        Backends may convert to their specific type if needed (e.g., Polars, Ibis).

        Args:
            feature_key: Feature key to write to
            df: [Narwhals](https://narwhals-dev.github.io/narwhals/)-compatible DataFrame with metadata to write
            **kwargs: Backend-specific parameters

        Note: Subclasses implement this for their storage backend.
        """
        pass

    def _add_system_columns(
        self,
        df: Frame,
        feature: CoercibleToFeatureKey,
        materialization_id: str | None = None,
    ) -> Frame:
        """Add all required system columns to the DataFrame.

        Args:
            df: Narwhals DataFrame/LazyFrame
            feature: Feature class or key
            materialization_id: Optional external orchestration ID for this write.
                Overrides the store's default if provided.

        Returns:
            DataFrame with all system columns added
        """
        feature_key = self._resolve_feature_key(feature)

        # Use collect_schema().names() to avoid PerformanceWarning on lazy frames
        columns = df.collect_schema().names()

        # Check if version columns already exist in DataFrame
        has_feature_version = METAXY_FEATURE_VERSION in columns
        has_snapshot_version = METAXY_SNAPSHOT_VERSION in columns
        has_feature_spec_version = METAXY_FEATURE_SPEC_VERSION in columns

        # In suppression mode (migrations), use existing values as-is
        if (
            _suppress_feature_version_warning.get()
            and has_feature_version
            and has_snapshot_version
            and has_feature_spec_version
        ):
            pass  # Use existing values for migrations
        else:
            # Drop any existing version columns (e.g., from SQLModel with null values)
            # and add current versions
            columns_to_drop = []
            if has_feature_version:
                columns_to_drop.append(METAXY_FEATURE_VERSION)
            if has_snapshot_version:
                columns_to_drop.append(METAXY_SNAPSHOT_VERSION)
            if has_feature_spec_version:
                columns_to_drop.append(METAXY_FEATURE_SPEC_VERSION)
            if columns_to_drop:
                df = df.drop(*columns_to_drop)

            # Get current feature version, feature_spec_version, and snapshot_version from code
            # Use duck typing to avoid Ray serialization issues with issubclass
            if (
                isinstance(feature, type)
                and hasattr(feature, "feature_version")
                and callable(feature.feature_version)
            ):
                current_feature_version = feature.feature_version()
                current_feature_spec_version = feature.feature_spec_version()  # ty: ignore[possibly-missing-attribute]
            else:
                from metaxy import get_feature_by_key

                feature_cls = get_feature_by_key(feature_key)
                current_feature_version = feature_cls.feature_version()
                current_feature_spec_version = feature_cls.feature_spec_version()

            # Get snapshot_version from active graph
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
            current_snapshot_version = graph.snapshot_version

            df = df.with_columns(
                [
                    nw.lit(current_feature_version).alias(METAXY_FEATURE_VERSION),  # ty: ignore[invalid-argument-type]
                    nw.lit(current_snapshot_version).alias(METAXY_SNAPSHOT_VERSION),
                    nw.lit(current_feature_spec_version).alias(
                        METAXY_FEATURE_SPEC_VERSION
                    ),
                ]
            )

        # These should normally be added by the provenance engine during resolve_update
        from metaxy.models.constants import (
            METAXY_CREATED_AT,
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
        )

        # Re-fetch columns since df may have been modified above
        columns = df.collect_schema().names()

        if METAXY_PROVENANCE_BY_FIELD not in columns:
            raise ValueError(
                f"Metadata is missing a required column `{METAXY_PROVENANCE_BY_FIELD}`. It should have been created by a prior `MetadataStore.resolve_update` call. Did you drop it on the way?"
            )

        if METAXY_PROVENANCE not in columns:
            plan = self._resolve_feature_plan(feature_key)

            # Only warn for non-root features (features with dependencies).
            # Root features don't have upstream dependencies, so they don't go through
            # resolve_update() - they just need metaxy_provenance_by_field to be set.
            if plan.deps:
                MetaxyColumnMissingWarning.warn_on_missing_column(
                    expected=METAXY_PROVENANCE,
                    df=df,
                    message=f"It should have been created by a prior `MetadataStore.resolve_update` call. Re-crearing it from `{METAXY_PROVENANCE_BY_FIELD}` Did you drop it on the way?",
                )

            df = self.hash_struct_version_column(
                plan=plan,
                df=df,
                struct_column=METAXY_PROVENANCE_BY_FIELD,
                hash_column=METAXY_PROVENANCE,
            )

        # Re-fetch columns since df may have been modified
        columns = df.collect_schema().names()

        if METAXY_CREATED_AT not in columns:
            from datetime import datetime, timezone

            df = df.with_columns(
                nw.lit(datetime.now(timezone.utc)).alias(METAXY_CREATED_AT)
            )

        # Add materialization_id if not already present
        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        df = df.with_columns(
            nw.lit(
                materialization_id or self._materialization_id, dtype=nw.String
            ).alias(METAXY_MATERIALIZATION_ID)
        )

        # Check for missing data_version columns (should come from resolve_update but it's acceptable to just use provenance columns if they are missing)
        # Re-fetch columns since df may have been modified
        columns = df.collect_schema().names()

        if METAXY_DATA_VERSION_BY_FIELD not in columns:
            df = df.with_columns(
                nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD)
            )
            df = df.with_columns(nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION))
        elif METAXY_DATA_VERSION not in columns:
            df = self.hash_struct_version_column(
                plan=self._resolve_feature_plan(feature_key),
                df=df,
                struct_column=METAXY_DATA_VERSION_BY_FIELD,
                hash_column=METAXY_DATA_VERSION,
            )

        # Cast system columns with Null dtype to their correct types
        # This handles edge cases where empty DataFrames or certain operations
        # result in Null-typed columns that break downstream processing
        df = _cast_present_system_columns(df)

        return df

    def _validate_schema(self, df: Frame) -> None:
        """
        Validate that DataFrame has required schema.

        Args:
            df: Narwhals DataFrame or LazyFrame to validate

        Raises:
            MetadataSchemaError: If schema is invalid
        """
        from metaxy.metadata_store.exceptions import MetadataSchemaError

        schema = df.collect_schema()

        # Check for metaxy_provenance_by_field column
        if METAXY_PROVENANCE_BY_FIELD not in schema.names():
            raise MetadataSchemaError(
                f"DataFrame must have '{METAXY_PROVENANCE_BY_FIELD}' column"
            )

        # Check that metaxy_provenance_by_field is a struct
        provenance_dtype = schema[METAXY_PROVENANCE_BY_FIELD]
        if not isinstance(provenance_dtype, nw.Struct):
            raise MetadataSchemaError(
                f"'{METAXY_PROVENANCE_BY_FIELD}' column must be a Struct, got {provenance_dtype}"
            )

        # Note: metaxy_provenance is auto-computed if missing, so we don't validate it here

        # Check for feature_version column
        if METAXY_FEATURE_VERSION not in schema.names():
            raise MetadataSchemaError(
                f"DataFrame must have '{METAXY_FEATURE_VERSION}' column"
            )

        # Check for snapshot_version column
        if METAXY_SNAPSHOT_VERSION not in schema.names():
            raise MetadataSchemaError(
                f"DataFrame must have '{METAXY_SNAPSHOT_VERSION}' column"
            )

    def _validate_schema_system_table(self, df: Frame) -> None:
        """Validate schema for system tables (minimal validation).

        Args:
            df: Narwhals DataFrame to validate
        """
        # System tables don't need metaxy_provenance_by_field column
        pass

    @abstractmethod
    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop/delete all metadata for a feature.

        Backend-specific implementation for dropping feature metadata.

        Args:
            feature_key: The feature key to drop metadata for
        """
        pass

    def drop_feature_metadata(self, feature: CoercibleToFeatureKey) -> None:
        """Drop all metadata for a feature.

        This removes all stored metadata for the specified feature from the store.
        Useful for cleanup in tests or when re-computing feature metadata from scratch.

        Warning:
            This operation is irreversible and will **permanently delete all metadata** for the specified feature.

        Args:
            feature: Feature class or key to drop metadata for

        Example:
            ```py
            store.drop_feature_metadata(MyFeature)
            assert not store.has_feature(MyFeature)
            ```
        """
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        self._drop_feature_metadata_impl(feature_key)

    @abstractmethod
    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """
        Read metadata from THIS store only without using any fallbacks stores.

        Args:
            feature: Feature to read metadata for
            filters: List of Narwhals filter expressions for this specific feature.
            columns: Subset of columns to return
            **kwargs: Backend-specific parameters

        Returns:
            Narwhals LazyFrame with metadata, or None if feature not found in the store
        """
        pass

    # ========== Feature Existence ==========

    def has_feature(
        self,
        feature: CoercibleToFeatureKey,
        *,
        check_fallback: bool = False,
    ) -> bool:
        """
        Check if feature exists in store.

        Args:
            feature: Feature to check
            check_fallback: If True, also check fallback stores

        Returns:
            True if feature exists, False otherwise
        """
        self._check_open()

        if self.read_metadata_in_store(feature) is not None:
            return True

        # Check fallback stores
        if not check_fallback:
            return self._has_feature_impl(feature)
        else:
            for store in self.fallback_stores:
                if store.has_feature(feature, check_fallback=True):
                    return True

        return False

    @abstractmethod
    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Implementation of _has_feature.

        Args:
            feature: Feature to check

        Returns:
            True if feature exists, False otherwise
        """
        pass

    @abstractmethod
    def display(self) -> str:
        """Return a human-readable display string for this store.

        Used in warnings, logs, and CLI output to identify the store.

        Returns:
            Display string (e.g., "DuckDBMetadataStore(database=/path/to/db.duckdb)")
        """
        pass

    def find_store_for_feature(
        self,
        feature_key: CoercibleToFeatureKey,
        *,
        check_fallback: bool = True,
    ) -> MetadataStore | None:
        """Find the store that contains the given feature.

        Args:
            feature_key: Feature to find
            check_fallback: Whether to check fallback stores when the feature
                is not found in the current store

        Returns:
            The store containing the feature, or None if not found
        """
        self._check_open()

        # Check if feature exists in this store
        if self.has_feature(feature_key):
            return self

        # Try fallback stores if enabled (opened on demand)
        if check_fallback:
            for store in self.fallback_stores:
                with store:
                    found = store.find_store_for_feature(
                        feature_key, check_fallback=True
                    )
                    if found is not None:
                        return found

        return None

    def get_store_metadata(
        self,
        feature_key: CoercibleToFeatureKey,
        *,
        check_fallback: bool = True,
    ) -> dict[str, Any]:
        """Arbitrary key-value pairs with useful metadata for logging purposes (like a path in storage).

        This method should not expose sensitive information.

        Args:
            feature_key: Feature to get metadata for
            check_fallback: Whether to check fallback stores when the feature
                is not found in the current store

        Returns:
            Dictionary with store-specific metadata (e.g., `"display"`, `"table_name"`, `"uri"`)
        """
        store = self.find_store_for_feature(feature_key, check_fallback=check_fallback)
        if store is None:
            return {}
        return {
            "display": store.display(),
            **store._get_store_metadata_impl(feature_key),
        }

    def _get_store_metadata_impl(
        self, feature_key: CoercibleToFeatureKey
    ) -> dict[str, Any]:
        """Implementation of get_store_metadata for this specific store type.

        Override in subclasses to return store-specific metadata.

        Args:
            feature_key: Feature to get metadata for

        Returns:
            Dictionary with store-specific metadata
        """
        return {}

    def calculate_input_progress(
        self,
        lazy_increment: LazyIncrement,
        feature_key: CoercibleToFeatureKey,
    ) -> float | None:
        """Calculate progress percentage from lazy increment.

        Uses the `input` field from LazyIncrement to count total input units
        and compares with `added` to determine how many are missing.

        Progress represents the percentage of input units that have been processed
        at least once. Stale samples (in `changed`) are counted as processed since
        they have existing metadata, even though they may need re-processing due to
        upstream changes.

        Args:
            lazy_increment: The lazy increment containing input and added dataframes.
            feature_key: The feature key to look up lineage information.

        Returns:
            Progress percentage (0-100), or None if input is not available.
        """
        if lazy_increment.input is None:
            return None

        key = self._resolve_feature_key(feature_key)
        graph = current_graph()
        plan = graph.get_feature_plan(key)

        # Get the columns that define input units from the feature plan
        input_id_columns = plan.input_id_columns

        # Count distinct input units using two separate queries
        # We can't use concat because input and added may have different schemas
        # (e.g., nullable vs non-nullable columns)
        total_units: int = (
            lazy_increment.input.select(input_id_columns)
            .unique()
            .select(nw.len())
            .collect()
            .item()
        )

        if total_units == 0:
            return None  # No input available from upstream

        missing_units: int = (
            lazy_increment.added.select(input_id_columns)
            .unique()
            .select(nw.len())
            .collect()
            .item()
        )

        processed_units = total_units - missing_units
        return (processed_units / total_units) * 100

    def copy_metadata(
        self,
        from_store: MetadataStore,
        features: list[CoercibleToFeatureKey] | None = None,
        *,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        global_filters: Sequence[nw.Expr] | None = None,
        current_only: bool = False,
        latest_only: bool = True,
    ) -> dict[str, int]:
        """Copy metadata from another store with fine-grained filtering.

        This is a reusable method that can be called programmatically or from CLI/migrations.
        Copies metadata for specified features, preserving the original snapshot_version.

        Args:
            from_store: Source metadata store to copy from (must be opened for reading)
            features: List of features to copy. Can be:
                - None: copies all features from active graph
                - List of FeatureKey or Feature classes: copies specified features
            filters: Dict mapping feature keys (as strings) to sequences of Narwhals filter expressions.
                These filters are applied when reading from the source store.
                Example: {"feature/key": [nw.col("x") > 10], "other/feature": [...]}
            global_filters: Sequence of Narwhals filter expressions applied to all features.
                These filters are combined with any feature-specific filters from `filters`.
                Example: [nw.col("sample_uid").is_in(["s1", "s2"])]
            current_only: If True, only copy rows with the current feature_version (as defined
                in the loaded feature graph). Defaults to False to copy all versions.
            latest_only: If True (default), deduplicate samples within `id_columns` groups
                by keeping only the latest row per group (ordered by `metaxy_created_at`).

        Returns:
            Dict with statistics: {"features_copied": int, "rows_copied": int}

        Raises:
            ValueError: If source or destination store is not open
            FeatureNotFoundError: If a specified feature doesn't exist in source store

        Examples:
            ```py
            # Copy all features
            with source_store.open("read"), dest_store.open("write"):
                stats = dest_store.copy_metadata(from_store=source_store)
            ```

            ```py
            # Copy specific features
            with source_store.open("read"), dest_store.open("write"):
                stats = dest_store.copy_metadata(
                    from_store=source_store,
                    features=[FeatureKey(["my_feature"])],
                )
            ```

            ```py
            # Copy with global filters applied to all features
            with source_store.open("read"), dest_store.open("write"):
                stats = dest_store.copy_metadata(
                    from_store=source_store,
                    global_filters=[nw.col("sample_uid").is_in(["s1", "s2"])],
                )
            ```

            ```py
            # Copy specific features with per-feature filters
            with source_store.open("read"), dest_store.open("write"):
                stats = dest_store.copy_metadata(
                    from_store=source_store,
                    features=[
                        FeatureKey(["feature_a"]),
                        FeatureKey(["feature_b"]),
                    ],
                    filters={
                        "feature_a": [nw.col("field_a") > 10],
                        "feature_b": [nw.col("field_b") < 30],
                    },
                )
            ```
        """
        import logging

        logger = logging.getLogger(__name__)

        # Validate both stores are open
        if not self._is_open:
            raise ValueError(
                'Destination store must be opened with store.open("write") before use'
            )
        if not from_store._is_open:
            raise ValueError(
                'Source store must be opened with store.open("read") before use'
            )

        return self._copy_metadata_impl(
            from_store=from_store,
            features=features,
            filters=filters,
            global_filters=global_filters,
            current_only=current_only,
            latest_only=latest_only,
            logger=logger,
        )

    def _copy_metadata_impl(
        self,
        from_store: MetadataStore,
        features: list[CoercibleToFeatureKey] | None,
        filters: Mapping[str, Sequence[nw.Expr]] | None,
        global_filters: Sequence[nw.Expr] | None,
        current_only: bool,
        latest_only: bool,
        logger,
    ) -> dict[str, int]:
        """Internal implementation of copy_metadata."""
        # Determine which features to copy
        features_to_copy: list[FeatureKey]
        if features is None:
            # Copy all features from active graph (features defined in current project)
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
            features_to_copy = graph.list_features(only_current_project=True)
            logger.info(
                f"Copying all features from active graph: {len(features_to_copy)} features"
            )
        else:
            # Convert all to FeatureKey using the adapter
            features_to_copy = [self._resolve_feature_key(item) for item in features]
            logger.info(f"Copying {len(features_to_copy)} specified features")

        # Copy metadata for each feature
        total_rows = 0
        features_copied = 0

        with allow_feature_version_override():
            for feature_key in features_to_copy:
                try:
                    # Build combined filters for this feature
                    feature_filters: list[nw.Expr] = []

                    # Add global filters
                    if global_filters:
                        feature_filters.extend(global_filters)

                    # Add feature-specific filters
                    if filters:
                        feature_key_str = feature_key.to_string()
                        if feature_key_str in filters:
                            feature_filters.extend(filters[feature_key_str])

                    # Read metadata from source with all filters applied
                    source_lazy = from_store.read_metadata(
                        feature_key,
                        filters=feature_filters if feature_filters else None,
                        allow_fallback=False,
                        current_only=current_only,
                        latest_only=latest_only,
                    )

                    # Collect to narwhals DataFrame to get row count
                    source_df = source_lazy.collect()
                    row_count = len(source_df)

                    if row_count == 0:
                        logger.warning(
                            f"No rows found for {feature_key.to_string()}, skipping"
                        )
                        continue

                    # Write to destination (preserving snapshot_version and feature_version)
                    self.write_metadata(feature_key, source_df)

                    features_copied += 1
                    total_rows += row_count
                    logger.info(
                        f"Copied {row_count} rows for {feature_key.to_string()}"
                    )

                except FeatureNotFoundError:
                    logger.warning(
                        f"Feature {feature_key.to_string()} not found in source store, skipping"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Error copying {feature_key.to_string()}: {e}", exc_info=True
                    )
                    raise

        logger.info(
            f"Copy complete: {features_copied} features, {total_rows} total rows"
        )

        return {"features_copied": features_copied, "rows_copied": total_rows}
