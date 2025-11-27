"""Abstract base class for metadata storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import narwhals as nw
from narwhals.typing import Frame, IntoFrame
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
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.feature import BaseFeature, FeatureGraph, current_graph
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


VersioningEngineT = TypeVar("VersioningEngineT", bound=VersioningEngine)
VersioningEngineOptions = Literal["auto", "native", "polars"]

# Mapping of system columns to their expected Narwhals dtypes
# Used to cast Null-typed columns to correct types
# Note: Struct columns (METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD) are not cast
_SYSTEM_COLUMN_DTYPES = {
    METAXY_PROVENANCE: nw.String,
    METAXY_FEATURE_VERSION: nw.String,
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


class ErrorContext:
    """Context object for manual error logging within catch_errors().

    This class is returned by the catch_errors() context manager and provides
    a log_error() method for manually recording errors during data processing.

    Attributes:
        feature_key: The feature key associated with this error context
        id_columns: Tuple of id column names from the feature spec
        errors: List accumulating error records

    Example:
        ```py
        with store.catch_errors(MyFeature, autoflush=False) as ctx:
            for sample in samples:
                try:
                    process(sample)
                except ValueError as e:
                    ctx.log_error(
                        message=str(e),
                        error_type="ValueError",
                        sample_uid=sample['id']
                    )
        ```
    """

    def __init__(
        self,
        feature_key: FeatureKey,
        id_columns: tuple[str, ...],
        errors: list[dict[str, Any]],
    ):
        """Initialize ErrorContext.

        Args:
            feature_key: Feature key for this context
            id_columns: ID column names from feature spec
            errors: Shared list to accumulate errors (mutated in place)
        """
        self.feature_key = feature_key
        self.id_columns = id_columns
        self.errors = errors  # Shared reference - mutations visible to caller

    def log_error(
        self,
        message: str,
        error_type: str,
        **id_column_values: Any,
    ) -> None:
        """Log an error for a specific sample.

        Args:
            message: Error message describing what went wrong
            error_type: Type/category of error (e.g., exception class name)
            **id_column_values: Values for each id_column to identify the sample.
                Must provide values for ALL id_columns from the feature spec.

        Raises:
            ValueError: If provided id_column_values don't match feature's id_columns

        Example:
            ```py
            # Single id_column
            ctx.log_error(
                message="Invalid value",
                error_type="ValueError",
                sample_uid="123"
            )

            # Multiple id_columns
            ctx.log_error(
                message="Processing failed",
                error_type="RuntimeError",
                sample_uid="123",
                timestamp="2024-01-01"
            )
            ```
        """
        # Validate that all id_columns are provided
        provided_keys = set(id_column_values.keys())
        expected_keys = set(self.id_columns)

        if provided_keys != expected_keys:
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys

            error_parts = []
            if missing:
                error_parts.append(f"missing: {sorted(missing)}")
            if extra:
                error_parts.append(f"unexpected: {sorted(extra)}")

            raise ValueError(
                f"ID column mismatch for feature {self.feature_key.to_string()}. "
                f"Expected columns: {sorted(expected_keys)}, "
                f"got: {sorted(provided_keys)}. "
                f"{', '.join(error_parts)}"
            )

        # Construct error record
        error_record = {
            **id_column_values,  # Spread id columns
            "error_message": message,
            "error_type": error_type,
        }

        # Append to shared errors list
        self.errors.append(error_record)


class MetadataStore(ABC):
    """
    Abstract base class for metadata storage backends.
    """

    # Subclasses can override this to disable auto_create_tables warning
    # Set to False for stores where table creation is not applicable (e.g., InMemoryMetadataStore)
    _should_warn_auto_create_tables: bool = True

    def __init__(
        self,
        *,
        versioning_engine_cls: type[VersioningEngineT],
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
        self.versioning_engine_cls = versioning_engine_cls

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
        self._collected_errors: dict[FeatureKey, list[dict[str, Any]]] = {}

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

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[CoercibleToFeatureKey, Sequence[nw.Expr]] | None = None,
        global_filters: Sequence[nw.Expr] | None = None,
        lazy: Literal[False] = False,
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        skip_comparison: bool = False,
        exclude_errors: bool = True,
        **kwargs: Any,
    ) -> Increment: ...

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[CoercibleToFeatureKey, Sequence[nw.Expr]] | None = None,
        global_filters: Sequence[nw.Expr] | None = None,
        lazy: Literal[True],
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        skip_comparison: bool = False,
        exclude_errors: bool = True,
        **kwargs: Any,
    ) -> LazyIncrement: ...

    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[CoercibleToFeatureKey, Sequence[nw.Expr]] | None = None,
        global_filters: Sequence[nw.Expr] | None = None,
        lazy: bool = False,
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        skip_comparison: bool = False,
        exclude_errors: bool = True,
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
            exclude_errors: If True (default), exclude samples that have recorded errors
                for the current feature_version from the `added` frame. Changed and removed
                frames are not affected. Has no effect when skip_comparison=True.

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

        # Normalize filter keys to FeatureKey
        normalized_filters: dict[FeatureKey, list[nw.Expr]] = {}
        if filters:
            for key, exprs in filters.items():
                feature_key = self._resolve_feature_key(key)
                normalized_filters[feature_key] = list(exprs)

        # Convert global_filters to a list for easy concatenation
        global_filter_list = list(global_filters) if global_filters else []

        graph = current_graph()
        plan = graph.get_feature_plan(feature.spec().key)

        # Root features without samples: error (samples required)
        if not plan.deps and samples is None:
            raise ValueError(
                f"Feature {feature.spec().key} has no upstream dependencies (root feature). "
                f"Must provide 'samples' parameter with sample_uid and {METAXY_PROVENANCE_BY_FIELD} columns. "
                f"Root features require manual {METAXY_PROVENANCE_BY_FIELD} computation."
            )

        # Combine feature-specific filters with global filters
        current_feature_filters = [
            *normalized_filters.get(feature.spec().key, []),
            *global_filter_list,
        ]

        current_metadata = self.read_metadata_in_store(
            feature,
            filters=[
                nw.col(METAXY_FEATURE_VERSION)
                == graph.get_feature_version(feature.spec().key),
                *current_feature_filters,
            ],
        )

        upstream_by_key: dict[FeatureKey, nw.LazyFrame[Any]] = {}
        filters_by_key: dict[FeatureKey, list[nw.Expr]] = {}

        # if samples are provided, use them as source of truth for upstream data
        if samples is not None:
            # Apply filters to samples if any
            filtered_samples = samples
            if current_feature_filters:
                filtered_samples = samples.filter(current_feature_filters)

            # fill in METAXY_PROVENANCE column if it's missing (e.g. for root features)
            samples = self.hash_struct_version_column(
                plan,
                df=filtered_samples,
                struct_column=METAXY_PROVENANCE_BY_FIELD,
                hash_column=METAXY_PROVENANCE,
            )

            # For root features, add data_version columns if they don't exist
            # (root features have no computation, so data_version equals provenance)
            if METAXY_DATA_VERSION_BY_FIELD not in samples.columns:
                samples = samples.with_columns(
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
                samples is not None
                and samples.implementation != self.native_implementation()
            ):
                if not switched_to_polars:
                    if engine_mode == "native":
                        # Always raise error for samples with wrong implementation, regardless
                        # of fallback stores, because samples come from user argument, not from fallback
                        raise VersioningEngineMismatchError(
                            f"versioning_engine='native' but provided `samples` have implementation {samples.implementation}, "
                            f"expected {self.native_implementation()}"
                        )
                    elif engine_mode == "auto":
                        PolarsMaterializationWarning.warn_on_implementation_mismatch(
                            expected=self.native_implementation(),
                            actual=samples.implementation,
                            message=f"Provided `samples` have implementation {samples.implementation}. Using Polars for resolving the increment instead.",
                        )
                implementation = nw.Implementation.POLARS
                switched_to_polars = True

        if switched_to_polars:
            if current_metadata:
                current_metadata = switch_implementation_to_polars(current_metadata)
            if samples:
                samples = switch_implementation_to_polars(samples)
            for upstream_key, df in upstream_by_key.items():
                upstream_by_key[upstream_key] = switch_implementation_to_polars(df)

        with self.create_versioning_engine(
            plan=plan, implementation=implementation
        ) as engine:
            if skip_comparison:
                # Skip comparison: return all upstream samples as added
                if samples is not None:
                    # Root features or user-provided samples: use samples directly
                    # Note: samples already has metaxy_provenance computed
                    added = samples.lazy()
                else:
                    # Non-root features: load all upstream with provenance
                    added = engine.load_upstream_with_provenance(
                        upstream=upstream_by_key,
                        hash_algo=self.hash_algorithm,
                        filters=filters_by_key,
                    )
                changed = None
                removed = None
            else:
                added, changed, removed = engine.resolve_increment_with_provenance(
                    current=current_metadata,
                    upstream=upstream_by_key,
                    hash_algorithm=self.hash_algorithm,
                    filters=filters_by_key,
                    sample=samples.lazy() if samples is not None else None,
                )

        # Convert None to empty DataFrames
        if changed is None:
            changed = empty_frame_like(added)
        if removed is None:
            removed = empty_frame_like(added)

        # Exclude samples with known errors (if enabled)
        if exclude_errors and not skip_comparison:
            import logging

            logger = logging.getLogger(__name__)

            try:
                # Read errors for current feature version
                current_feature_version = graph.get_feature_version(feature.spec().key)
                feature_errors = self.read_errors(
                    feature,
                    feature_version=current_feature_version,
                    latest_only=True,
                )

                if feature_errors is not None:
                    # Count errors before exclusion
                    try:
                        error_count = len(feature_errors.collect())
                    except Exception:
                        error_count = None

                    if error_count and error_count > 0:
                        # Get id_columns for anti-join
                        id_cols = list(feature.spec().id_columns)

                        # Get unique sample IDs with errors and collect to Polars
                        # We need both frames to be the same implementation for anti-join
                        from metaxy._utils import collect_to_polars

                        error_sample_ids_lazy = feature_errors.select(id_cols).unique()
                        error_sample_ids_polars = collect_to_polars(
                            error_sample_ids_lazy
                        )

                        # Convert added to Polars if it isn't already
                        was_lazy = isinstance(added, nw.LazyFrame)
                        added_polars = collect_to_polars(added)

                        # Do anti-join in Polars
                        added_polars = added_polars.join(
                            error_sample_ids_polars, on=id_cols, how="anti"
                        )

                        # Convert back to appropriate form
                        if was_lazy:
                            added = nw.from_native(added_polars, eager_only=True).lazy()
                        else:
                            added = nw.from_native(added_polars, eager_only=True)

                        logger.info(
                            f"Excluded {error_count} sample(s) with known errors "
                            f"from {feature.spec().key.to_string()}"
                        )
            except Exception as e:
                # If error exclusion fails, log warning but don't fail the operation
                logger.warning(
                    f"Failed to exclude errors for {feature.spec().key.to_string()}: {e}. "
                    f"Proceeding without error exclusion."
                )

        if lazy:
            return LazyIncrement(
                added=added
                if isinstance(added, nw.LazyFrame)
                else nw.from_native(nw.to_native(added), eager_only=True).lazy(),
                changed=changed
                if isinstance(changed, nw.LazyFrame)
                else nw.from_native(nw.to_native(changed), eager_only=True).lazy(),
                removed=removed
                if isinstance(removed, nw.LazyFrame)
                else nw.from_native(nw.to_native(removed), eager_only=True).lazy(),
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

        # Try fallback stores
        if allow_fallback:
            for store in self.fallback_stores:
                try:
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

        assert isinstance(df_nw, nw.DataFrame), "df must be a Narwhal DataFrame"

        # For system tables, write directly without feature_version tracking
        if is_system_table:
            self._validate_schema_system_table(df_nw)
            self.write_metadata_to_store(feature_key, df_nw)
            return

        if METAXY_PROVENANCE_BY_FIELD not in df_nw.columns:
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

        # Auto-clear errors for successfully written samples
        # This implements the "automatic on success" invalidation strategy
        try:
            # Get feature spec to know which columns are id_columns
            feature_spec = self._resolve_feature_plan(feature_key).feature
            id_cols = list(feature_spec.id_columns)

            # Extract sample IDs from written DataFrame
            # Convert to list of dicts (one dict per sample)
            sample_ids_df = df_nw.select(id_cols).unique()

            # Convert to list of dicts for clear_errors API
            # Narwhals doesn't have to_dicts, so convert to native Polars first
            import polars as pl

            sample_ids_native = sample_ids_df.to_native()
            if isinstance(sample_ids_native, pl.DataFrame):
                sample_uids = sample_ids_native.to_dicts()
            elif isinstance(sample_ids_native, pl.LazyFrame):
                sample_uids = sample_ids_native.collect().to_dicts()
            else:
                # Fallback for other implementations - try to convert to Polars
                try:
                    sample_uids = pl.from_arrow(sample_ids_native.to_arrow()).to_dicts()  # pyright: ignore[reportAttributeAccessIssue]
                except Exception:
                    # If conversion fails, skip error clearing for this write
                    sample_uids = []

            if sample_uids:
                # Clear errors for these successfully written samples
                # This is safe even if no errors exist (clear_errors is a no-op then)
                self.clear_errors(feature_key, sample_uids=sample_uids)
        except Exception as e:
            # If error clearing fails, log warning but don't fail the write
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to auto-clear errors after successful write for "
                f"{feature_key.to_string()}: {e}. "
                f"Error records may still exist for successfully processed samples."
            )

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
    ) -> Iterator[VersioningEngineT]:
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
                    df,  # pyright: ignore[reportArgumentType]
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
        self._open_cm = self.open(mode)
        self._open_cm.__enter__()

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

    # ========== Error Table Helpers ==========

    def _get_error_table_name(self, feature_key: FeatureKey) -> str:
        """Get error table name for a feature.

        Error tables use the naming convention: {feature_table}__errors

        Args:
            feature_key: Feature key to get error table name for

        Returns:
            Error table name string

        Example:
            ```py
            feature_key = FeatureKey(["video", "processing"])
            error_table = store._get_error_table_name(feature_key)
            # Returns: "video__processing__errors"
            ```
        """
        base_table_name = feature_key.table_name
        return f"{base_table_name}__errors"

    def _is_error_table(self, table_name: str) -> bool:
        """Check if a table name represents an error table.

        Error tables are identified by the __errors suffix.

        Args:
            table_name: Table name to check

        Returns:
            True if table is an error table, False otherwise

        Example:
            ```py
            store._is_error_table("video__processing__errors")  # True
            store._is_error_table("video__processing")          # False
            ```
        """
        return table_name.endswith("__errors")

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
        feature_project = feature_cls.project  # type: ignore[attr-defined]

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

        # Check if feature_version and snapshot_version already exist in DataFrame
        has_feature_version = METAXY_FEATURE_VERSION in df.columns
        has_snapshot_version = METAXY_SNAPSHOT_VERSION in df.columns

        # In suppression mode (migrations), use existing values as-is
        if (
            _suppress_feature_version_warning.get()
            and has_feature_version
            and has_snapshot_version
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
            if columns_to_drop:
                df = df.drop(*columns_to_drop)

            # Get current feature version and snapshot_version from code and add them
            # Use duck typing to avoid Ray serialization issues with issubclass
            if (
                isinstance(feature, type)
                and hasattr(feature, "feature_version")
                and callable(feature.feature_version)
            ):
                current_feature_version = feature.feature_version()
            else:
                from metaxy import get_feature_by_key

                feature_cls = get_feature_by_key(feature_key)
                current_feature_version = feature_cls.feature_version()

            # Get snapshot_version from active graph
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
            current_snapshot_version = graph.snapshot_version

            df = df.with_columns(
                [
                    nw.lit(current_feature_version).alias(METAXY_FEATURE_VERSION),
                    nw.lit(current_snapshot_version).alias(METAXY_SNAPSHOT_VERSION),
                ]
            )

        # These should normally be added by the provenance engine during resolve_update
        from metaxy.models.constants import (
            METAXY_CREATED_AT,
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
        )

        if METAXY_PROVENANCE_BY_FIELD not in df.columns:
            raise ValueError(
                f"Metadata is missing a required column `{METAXY_PROVENANCE_BY_FIELD}`. It should have been created by a prior `MetadataStore.resolve_update` call. Did you drop it on the way?"
            )

        if METAXY_PROVENANCE not in df.columns:
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

        if METAXY_CREATED_AT not in df.columns:
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

        if METAXY_DATA_VERSION_BY_FIELD not in df.columns:
            df = df.with_columns(
                nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD)
            )
            df = df.with_columns(nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION))
        elif METAXY_DATA_VERSION not in df.columns:
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

    def get_store_metadata(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        """Arbitrary key-value pairs with useful metadata like path in storage.

        Useful for logging purposes. This method should not expose sensitive information.
        """
        return {}

    def copy_metadata(
        self,
        from_store: MetadataStore,
        features: list[CoercibleToFeatureKey] | None = None,
        *,
        from_snapshot: str | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        incremental: bool = True,
    ) -> dict[str, int]:
        """Copy metadata from another store with fine-grained filtering.

        This is a reusable method that can be called programmatically or from CLI/migrations.
        Copies metadata for specified features, preserving the original snapshot_version.

        Args:
            from_store: Source metadata store to copy from (must be opened)
            features: List of features to copy. Can be:
                - None: copies all features from source store
                - List of FeatureKey or Feature classes: copies specified features
            from_snapshot: Snapshot version to filter source data by. If None, uses latest snapshot
                from source store. Only rows with this snapshot_version will be copied.
                The snapshot_version is preserved in the destination store.
            filters: Dict mapping feature keys (as strings) to sequences of Narwhals filter expressions.
                These filters are applied when reading from the source store.
                Example: {"feature/key": [nw.col("x") > 10], "other/feature": [...]}
            incremental: If True (default), filter out rows that already exist in the destination
                store by performing an anti-join on sample_uid for the same snapshot_version.

                The implementation uses an anti-join: source LEFT ANTI JOIN destination ON sample_uid
                filtered by snapshot_version.

                Disabling incremental (incremental=False) may improve performance when:
                - You know the destination is empty or has no overlap with source
                - The destination store uses deduplication

                When incremental=False, it's the user's responsibility to avoid duplicates or
                configure deduplication at the storage layer.

        Returns:
            Dict with statistics: {"features_copied": int, "rows_copied": int}

        Raises:
            ValueError: If from_store or self (destination) is not open
            FeatureNotFoundError: If a specified feature doesn't exist in source store

        Examples:
            ```py
            # Simple: copy all features from latest snapshot
            stats = dest_store.copy_metadata(from_store=source_store)
            ```

            ```py
            # Copy specific features from a specific snapshot
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=[FeatureKey(["my_feature"])],
                from_snapshot="abc123",
            )
            ```

            ```py
            # Copy with filters
            stats = dest_store.copy_metadata(
                from_store=source_store,
                filters={"my/feature": [nw.col("sample_uid").is_in(["s1", "s2"])]},
            )
            ```

            ```py
            # Copy specific features with filters
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=[
                    FeatureKey(["feature_a"]),
                    FeatureKey(["feature_b"]),
                ],
                filters={
                    "feature_a": [nw.col("field_a") > 10, nw.col("sample_uid").is_in(["s1", "s2"])],
                    "feature_b": [nw.col("field_b") < 30],
                },
            )
            ```
        """
        import logging

        logger = logging.getLogger(__name__)

        # Validate destination store is open
        if not self._is_open:
            raise ValueError(
                'Destination store must be opened with store.open("write") before use'
            )

        # Auto-open source store if not already open
        if not from_store._is_open:
            with from_store.open("read"):
                return self._copy_metadata_impl(
                    from_store=from_store,
                    features=features,
                    from_snapshot=from_snapshot,
                    filters=filters,
                    incremental=incremental,
                    logger=logger,
                )
        else:
            return self._copy_metadata_impl(
                from_store=from_store,
                features=features,
                from_snapshot=from_snapshot,
                filters=filters,
                incremental=incremental,
                logger=logger,
            )

    def _copy_metadata_impl(
        self,
        from_store: MetadataStore,
        features: list[CoercibleToFeatureKey] | None,
        from_snapshot: str | None,
        filters: Mapping[str, Sequence[nw.Expr]] | None,
        incremental: bool,
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

        # Log snapshot usage
        if from_snapshot is not None:
            logger.info(f"Filtering by snapshot: {from_snapshot}")
        else:
            logger.info("Copying all data (no snapshot filter)")

        # Copy metadata for each feature
        total_rows = 0
        features_copied = 0

        with allow_feature_version_override():
            for feature_key in features_to_copy:
                try:
                    # Read metadata from source, filtering by from_snapshot
                    # Use current_only=False to avoid filtering by feature_version
                    source_lazy = from_store.read_metadata(
                        feature_key,
                        allow_fallback=False,
                        current_only=False,
                    )

                    # Filter by from_snapshot if specified
                    import narwhals as nw

                    if from_snapshot is not None:
                        source_filtered = source_lazy.filter(
                            nw.col(METAXY_SNAPSHOT_VERSION) == from_snapshot
                        )
                    else:
                        source_filtered = source_lazy

                    # Apply filters for this feature (if any)
                    if filters:
                        feature_key_str = feature_key.to_string()
                        if feature_key_str in filters:
                            for filter_expr in filters[feature_key_str]:
                                source_filtered = source_filtered.filter(filter_expr)

                    # Apply incremental filtering if enabled
                    if incremental:
                        try:
                            # Read existing sample_uids from destination for the same snapshot
                            # This is much cheaper than comparing metaxy_provenance_by_field structs
                            dest_lazy = self.read_metadata(
                                feature_key,
                                allow_fallback=False,
                                current_only=False,
                            )
                            # Filter destination to same snapshot_version (if specified)
                            if from_snapshot is not None:
                                dest_for_snapshot = dest_lazy.filter(
                                    nw.col(METAXY_SNAPSHOT_VERSION) == from_snapshot
                                )
                            else:
                                dest_for_snapshot = dest_lazy

                            # Materialize destination sample_uids to avoid cross-backend join issues
                            # When copying between different stores (e.g., different DuckDB files),
                            # Ibis can't join tables from different backends
                            dest_sample_uids = (
                                dest_for_snapshot.select("sample_uid")
                                .collect()
                                .to_polars()
                            )

                            # Convert to Polars LazyFrame and wrap in Narwhals
                            dest_sample_uids_lazy = nw.from_native(
                                dest_sample_uids.lazy(), eager_only=False
                            )

                            # Collect source to Polars for anti-join
                            source_df = source_filtered.collect().to_polars()
                            source_lazy = nw.from_native(
                                source_df.lazy(), eager_only=False
                            )

                            # Anti-join: keep only source rows with sample_uid not in destination
                            source_filtered = source_lazy.join(
                                dest_sample_uids_lazy,
                                on="sample_uid",
                                how="anti",
                            )

                            # Collect after filtering
                            source_df = source_filtered.collect().to_polars()

                            logger.info(
                                f"Incremental: copying only new sample_uids for {feature_key.to_string()}"
                            )
                        except FeatureNotFoundError:
                            # Feature doesn't exist in destination yet - copy all rows
                            logger.debug(
                                f"Feature {feature_key.to_string()} not in destination, copying all rows"
                            )
                            source_df = source_filtered.collect().to_polars()
                        except Exception as e:
                            # If incremental check fails, log warning but continue with full copy
                            logger.warning(
                                f"Incremental check failed for {feature_key.to_string()}: {e}. Copying all rows."
                            )
                            source_df = source_filtered.collect().to_polars()
                    else:
                        # Non-incremental: collect all filtered rows
                        source_df = source_filtered.collect().to_polars()

                    if source_df.height == 0:
                        logger.warning(
                            f"No rows found for {feature_key.to_string()} with snapshot {from_snapshot}, skipping"
                        )
                        continue

                    # Write to destination (preserving snapshot_version and feature_version)
                    self.write_metadata(feature_key, source_df)

                    features_copied += 1
                    total_rows += source_df.height
                    logger.info(
                        f"Copied {source_df.height} rows for {feature_key.to_string()}"
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

    # ========== Error Tracking (Abstract Methods) ==========

    @abstractmethod
    def write_errors_to_store(
        self,
        feature_key: FeatureKey,
        errors_df: Frame,
    ) -> None:
        """Backend-specific implementation for writing errors.

        Subclasses must implement this to write error records to their storage backend.
        Error tables use the naming convention: {feature_table}__errors

        Args:
            feature_key: Feature key to write errors for
            errors_df: Narwhals DataFrame with error records containing:
                - Feature's id_columns (e.g., sample_uid, timestamp)
                - error_message: str
                - error_type: str
                - metaxy_feature_version: str
                - metaxy_snapshot_version: str
                - metaxy_created_at: datetime

        Note:
            - Errors are append-only (like regular metadata)
            - Subclasses handle table creation if needed
            - Should support schema evolution (new columns added gracefully)
        """
        pass

    @abstractmethod
    def read_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Backend-specific implementation for reading errors.

        Args:
            feature_key: Feature key to read errors for
            filters: Optional Narwhals filter expressions to apply

        Returns:
            Narwhals LazyFrame with error records, or None if error table doesn't exist

        Note:
            - Returns None if error table doesn't exist (not an error condition)
            - Filters are applied at read time for efficiency
        """
        pass

    @abstractmethod
    def clear_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        sample_uids: Sequence[dict[str, Any]] | None = None,
        feature_version: str | None = None,
    ) -> None:
        """Backend-specific implementation for clearing errors.

        Args:
            feature_key: Feature key to clear errors for
            sample_uids: Optional list of sample ID dicts to clear errors for.
                Each dict should contain values for all id_columns.
                If None, clears based on feature_version or all errors.
            feature_version: Optional specific feature version to clear errors for.
                If None and sample_uids is None, clears all errors for feature.

        Examples:
            ```py
            # Clear all errors
            store.clear_errors_from_store(feature_key)

            # Clear specific samples (single id_column)
            store.clear_errors_from_store(
                feature_key,
                sample_uids=[{"sample_uid": "s1"}, {"sample_uid": "s2"}]
            )

            # Clear specific samples (multi id_columns)
            store.clear_errors_from_store(
                feature_key,
                sample_uids=[
                    {"sample_uid": "s1", "timestamp": "2024-01-01"},
                    {"sample_uid": "s2", "timestamp": "2024-01-02"}
                ]
            )

            # Clear errors for specific feature version
            store.clear_errors_from_store(feature_key, feature_version="abc123")
            ```

        Note:
            - If error table doesn't exist, should be a no-op (no error raised)
            - For SQL backends, use DELETE with WHERE clause
            - For file-based backends, filter and overwrite
        """
        pass

    # ========== Error Tracking (Public API) ==========

    @property
    def collected_errors(self) -> dict[str, Frame]:
        """Access collected errors as DataFrames by feature key string.

        Returns errors collected by catch_errors() with autoflush=False.
        Each feature's errors are returned as a Narwhals DataFrame.

        Returns:
            Dict mapping feature key strings to DataFrames with error records

        Example:
            ```py
            with store.catch_errors(MyFeature, autoflush=False):
                # ... processing that logs errors ...
                pass

            # Access collected errors
            errors_by_feature = store.collected_errors
            my_errors_df = errors_by_feature["my/feature"]
            ```
        """
        import polars as pl

        return {
            key.to_string(): nw.from_native(pl.DataFrame(errors))
            for key, errors in self._collected_errors.items()
        }

    def write_errors(
        self,
        feature: CoercibleToFeatureKey,
        errors_df: IntoFrame,
    ) -> None:
        """Write error records for a feature.

        Args:
            feature: Feature to write errors for
            errors_df: DataFrame with error records containing:
                - All id_columns from feature spec (to identify samples)
                - error_message: str (required)
                - error_type: str (required)
                Additional columns will be preserved but not validated.

        Raises:
            ValueError: If required columns are missing or invalid
            StoreNotOpenError: If store is not open

        Note:
            - Automatically adds system columns (metaxy_feature_version, etc.)
            - Errors are append-only (never updated or deleted by this method)
            - Use clear_errors() to remove error records

        Example:
            ```py
            import polars as pl

            errors = pl.DataFrame({
                "sample_uid": ["s1", "s2"],
                "error_message": ["Division by zero", "Invalid input"],
                "error_type": ["ZeroDivisionError", "ValueError"],
            })

            with store.open(mode="write"):
                store.write_errors(MyFeature, errors)
            ```
        """
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        feature_spec = self._resolve_feature_plan(feature_key).feature

        # Convert to Narwhals
        df_nw = nw.from_native(errors_df)

        # Validate schema
        required_cols = set(feature_spec.id_columns) | {"error_message", "error_type"}
        actual_cols = set(df_nw.columns)
        missing_cols = required_cols - actual_cols

        if missing_cols:
            raise ValueError(
                f"Error DataFrame missing required columns: {sorted(missing_cols)}. "
                f"Required: {sorted(required_cols)}, got: {sorted(actual_cols)}"
            )

        # Add system columns
        from datetime import datetime, timezone

        from metaxy.models.constants import (
            METAXY_CREATED_AT,
            METAXY_FEATURE_VERSION,
            METAXY_SNAPSHOT_VERSION,
        )

        current_feature_version = current_graph().get_feature_version(feature_key)
        current_snapshot_version = current_graph().snapshot_version

        df_nw = df_nw.with_columns(
            [
                nw.lit(current_feature_version).alias(METAXY_FEATURE_VERSION),
                nw.lit(current_snapshot_version).alias(METAXY_SNAPSHOT_VERSION),
                nw.lit(datetime.now(timezone.utc)).alias(METAXY_CREATED_AT),
            ]
        )

        # Write to backend
        self.write_errors_to_store(feature_key, df_nw)

    def read_errors(
        self,
        feature: CoercibleToFeatureKey,
        *,
        sample_uids: Sequence[dict[str, Any]] | None = None,
        feature_version: str | None = None,
        latest_only: bool = True,
    ) -> nw.LazyFrame[Any] | None:
        """Read error records for a feature.

        Args:
            feature: Feature to read errors for
            sample_uids: Optional list of sample ID dicts to filter errors.
                Each dict should contain values for all id_columns.
            feature_version: Optional feature version to filter by.
                If None, uses current feature version.
            latest_only: If True, keeps only the most recent error per sample
                (by id_columns, ordered by metaxy_created_at DESC)

        Returns:
            Narwhals LazyFrame with error records, or None if no errors exist

        Example:
            ```py
            # Read all errors for current feature version
            errors = store.read_errors(MyFeature)

            # Read errors for specific samples
            errors = store.read_errors(
                MyFeature,
                sample_uids=[{"sample_uid": "s1"}, {"sample_uid": "s2"}]
            )

            # Read errors for specific feature version
            errors = store.read_errors(MyFeature, feature_version="abc123")
            ```
        """
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        feature_spec = self._resolve_feature_plan(feature_key).feature

        # Build filters
        filters = []

        # Filter by feature_version (default to current)
        if feature_version is None:
            feature_version = current_graph().get_feature_version(feature_key)

        from metaxy.models.constants import METAXY_FEATURE_VERSION

        filters.append(nw.col(METAXY_FEATURE_VERSION) == feature_version)

        # Filter by sample_uids if provided
        if sample_uids is not None:
            # Build filter for each sample_uid
            # For single id_column: sample_uid.is_in([...])
            # For multi id_columns: (col1 == v1 & col2 == v2) | (col1 == v3 & col2 == v4) | ...

            id_cols = list(feature_spec.id_columns)

            if len(id_cols) == 1:
                # Simple case: single id_column
                col_name = id_cols[0]
                values = [uid_dict[col_name] for uid_dict in sample_uids]
                filters.append(nw.col(col_name).is_in(values))
            else:
                # Complex case: multiple id_columns
                # Build OR of AND conditions
                sample_filters = []
                for uid_dict in sample_uids:
                    # Build AND condition for this sample
                    and_conditions = [
                        nw.col(col_name) == uid_dict[col_name] for col_name in id_cols
                    ]
                    # Combine with &
                    sample_filter = and_conditions[0]
                    for cond in and_conditions[1:]:
                        sample_filter = sample_filter & cond
                    sample_filters.append(sample_filter)

                # Combine with |
                if sample_filters:
                    combined_filter = sample_filters[0]
                    for f in sample_filters[1:]:
                        combined_filter = combined_filter | f
                    filters.append(combined_filter)

        # Read from backend
        lazy_frame = self.read_errors_from_store(feature_key, filters=filters)

        if lazy_frame is None:
            return None

        # Apply deduplication if requested
        if latest_only:
            from metaxy.models.constants import METAXY_CREATED_AT

            lazy_frame = self.versioning_engine_cls.keep_latest_by_group(
                df=lazy_frame,
                group_columns=list(feature_spec.id_columns),
                timestamp_column=METAXY_CREATED_AT,
            )

        return lazy_frame

    def clear_errors(
        self,
        feature: CoercibleToFeatureKey,
        *,
        sample_uids: Sequence[dict[str, Any]] | None = None,
        feature_version: str | None = None,
    ) -> None:
        """Clear error records for a feature.

        Args:
            feature: Feature to clear errors for
            sample_uids: Optional list of sample ID dicts to clear errors for.
                If provided, only clears errors for these specific samples.
            feature_version: Optional feature version to clear errors for.
                If provided, only clears errors for this version.
                If None and sample_uids is None, clears ALL errors.

        Note:
            - If error table doesn't exist, this is a no-op (no error raised)
            - Use carefully when clearing all errors (no filtering)

        Example:
            ```py
            # Clear all errors
            store.clear_errors(MyFeature)

            # Clear specific samples
            store.clear_errors(
                MyFeature,
                sample_uids=[{"sample_uid": "s1"}, {"sample_uid": "s2"}]
            )

            # Clear errors for old version
            store.clear_errors(MyFeature, feature_version="old_version")
            ```
        """
        self._check_open()

        feature_key = self._resolve_feature_key(feature)

        # Delegate to backend
        self.clear_errors_from_store(
            feature_key,
            sample_uids=sample_uids,
            feature_version=feature_version,
        )

    def has_errors(
        self,
        feature: CoercibleToFeatureKey,
        *,
        sample_uid: dict[str, Any] | None = None,
    ) -> bool:
        """Check if feature has any error records.

        Args:
            feature: Feature to check for errors
            sample_uid: Optional sample ID dict to check for specific sample.
                If None, checks if ANY errors exist for the feature.

        Returns:
            True if errors exist, False otherwise

        Example:
            ```py
            # Check if any errors exist
            if store.has_errors(MyFeature):
                print("Feature has errors")

            # Check specific sample
            if store.has_errors(MyFeature, sample_uid={"sample_uid": "s1"}):
                print("Sample s1 has errors")
            ```
        """
        self._check_open()

        # Try to read errors
        if sample_uid is None:
            errors = self.read_errors(feature, latest_only=False)
        else:
            errors = self.read_errors(
                feature, sample_uids=[sample_uid], latest_only=False
            )

        if errors is None:
            return False

        # Check if any rows exist
        # We need to collect to check length
        try:
            return len(errors.head(1).collect()) > 0
        except Exception:
            # If collection fails (e.g., empty frame), assume no errors
            return False

    @contextmanager
    def catch_errors(
        self,
        feature: CoercibleToFeatureKey,
        *,
        autoflush: bool = True,
        exception_types: tuple[type[Exception], ...] | None = None,
    ) -> Iterator[ErrorContext]:
        """Context manager to catch and record errors during data processing.

        Supports both automatic exception catching and manual error logging.
        Errors can be automatically written (autoflush=True) or collected
        for later processing (autoflush=False).

        Args:
            feature: Feature to associate errors with
            autoflush: If True, automatically write errors on context exit.
                If False, collect errors in collected_errors for manual writing.
            exception_types: Tuple of exception types to catch automatically.
                If None, catches all exceptions.
                Only applies to exceptions raised within the context.

        Yields:
            ErrorContext: Object with log_error() method for manual error logging

        Example - Automatic exception catching:
            ```py
            with store.catch_errors(MyFeature, autoflush=True):
                result = compute_feature(data)  # Exceptions caught and written
            ```

        Example - Manual error logging:
            ```py
            with store.catch_errors(MyFeature, autoflush=False) as ctx:
                for sample in samples:
                    try:
                        process(sample)
                    except ValueError as e:
                        ctx.log_error(
                            message=str(e),
                            error_type="ValueError",
                            sample_uid=sample['id']
                        )

            # Errors collected but not written yet
            errors_df = store.collected_errors["my/feature"]
            store.write_errors(MyFeature, errors_df)
            ```

        Example - Both approaches:
            ```py
            with store.catch_errors(MyFeature) as ctx:
                for sample in samples:
                    try:
                        result = risky_operation(sample)
                    except ValueError as e:
                        # Log specific error
                        ctx.log_error(
                            message=f"Validation failed: {e}",
                            error_type="ValueError",
                            sample_uid=sample['id']
                        )
                    except Exception:
                        # Let context manager catch unexpected errors
                        raise
            ```

        Note:
            - When autoflush=True, requires store to be open in write mode
            - When autoflush=False, errors accumulate in collected_errors
            - Exception catching only applies to exceptions that propagate out
              of the context (not caught exceptions)
            - For per-sample error logging, use ctx.log_error() explicitly
        """
        import logging

        logger = logging.getLogger(__name__)

        feature_key = self._resolve_feature_key(feature)
        feature_spec = self._resolve_feature_plan(feature_key).feature

        # Initialize error collection for this feature
        errors: list[dict[str, Any]] = []
        error_ctx = ErrorContext(
            feature_key=feature_key,
            id_columns=feature_spec.id_columns,
            errors=errors,
        )

        try:
            yield error_ctx
        except Exception as e:
            # Check if we should catch this exception type
            should_catch = exception_types is None or isinstance(e, exception_types)

            if should_catch:
                # Log a warning about the caught exception
                logger.warning(
                    f"Caught exception in catch_errors() for {feature_key.to_string()}: "
                    f"{type(e).__name__}: {e}"
                )

                # Record the exception as an error
                # Since we don't have sample context, id_columns will be None
                error_record = {
                    # Set all id_columns to None (we don't know which sample failed)
                    **{col: None for col in feature_spec.id_columns},
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                }
                errors.append(error_record)

                logger.info(
                    "Exception caught and recorded without sample ID context. "
                    "Use ctx.log_error() for sample-specific error tracking."
                )

                # Don't re-raise - error is recorded
            else:
                # Not in our exception_types filter, re-raise
                raise
        finally:
            # Process collected errors
            if errors:
                if autoflush:
                    # Convert to DataFrame and write immediately
                    try:
                        import polars as pl

                        errors_df = pl.DataFrame(errors)

                        # Validate that all required columns are present
                        id_cols_set = set(feature_spec.id_columns)
                        errors_cols_set = set(errors_df.columns)

                        if not id_cols_set.issubset(errors_cols_set):
                            missing = id_cols_set - errors_cols_set
                            logger.error(
                                f"Cannot flush errors: missing id_columns {missing}. "
                                f"Errors will be available in collected_errors instead."
                            )
                            # Store for manual retrieval
                            self._collected_errors[feature_key] = errors
                        else:
                            # Write errors
                            self.write_errors(feature_key, errors_df)
                            logger.info(
                                f"Flushed {len(errors)} error(s) for {feature_key.to_string()}"
                            )
                    except Exception as write_err:
                        logger.error(
                            f"Failed to write errors for {feature_key.to_string()}: {write_err}. "
                            f"Errors will be available in collected_errors instead."
                        )
                        # Store for manual retrieval even if autoflush failed
                        self._collected_errors[feature_key] = errors
                else:
                    # Store for later retrieval
                    self._collected_errors[feature_key] = errors
                    logger.debug(
                        f"Collected {len(errors)} error(s) for {feature_key.to_string()}. "
                        f"Access via collected_errors property or write manually."
                    )
