"""Abstract base class for metadata storage backends."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import narwhals as nw
from narwhals.typing import Frame, IntoFrame
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
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.feature import BaseFeature, FeatureGraph, current_graph
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.versioning import VersioningEngine
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm, Increment, LazyIncrement

if TYPE_CHECKING:
    pass


VersioningEngineT = TypeVar("VersioningEngineT", bound=VersioningEngine)
VersioningEngineOptions = Literal["auto", "native", "polars"]


class MetadataStore(ABC):
    """
    Abstract base class for metadata storage backends.

    Supports:
    - Append-only metadata storage patterns

    - Composable fallback store chains for development and testing purposes

    - Backend-specific computation optimizations

    Context Manager:
        Stores must be used as context managers for resource management.
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
    ):
        """
        Initialize the metadata store.

        Args:
            hash_algorithm: Hash algorithm to use for field provenance.
                Default: None (uses default algorithm for this store type)
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
                WARNING: Auto-create is intended for development/testing only. Do not use in production.
                Use proper database migration tools like Alembic for production deployments.
                Default: None (reads from global config, falls back to False for safety)

        Raises:
            ValueError: If fallback stores use different hash algorithms or truncation lengths
            VersioningEngineMismatchError: If versioning_engine="native" and the user-provided dataframe has a wrong implementation
        """
        # Initialize state early so properties can check it
        self._is_open = False
        self._context_depth = 0
        self._versioning_engine = versioning_engine
        self._allow_cross_project_writes = False
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

    @property
    def hash_truncation_length(self) -> int:
        return MetaxyConfig.get().hash_truncation_length or 64

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: Literal[False] = False,
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        **kwargs: Any,
    ) -> Increment: ...

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: Literal[True],
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        **kwargs: Any,
    ) -> LazyIncrement: ...

    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: bool = False,
        versioning_engine: Literal["auto", "native", "polars"] | None = None,
        **kwargs: Any,
    ) -> Increment | LazyIncrement:
        """Calculate an incremental update for a feature.

        Args:
            feature: Feature class to resolve updates for
            samples: Pre-computed DataFrame with ID columns
                and `PROVENANCE_BY_FIELD_COL` column. When provided, `MetadataStore` skips upstream loading, joining,
                and field provenance calculation.

                **Required for root features** (features with no upstream dependencies).
                Root features don't have upstream to calculate `PROVENANCE_BY_FIELD_COL` from, so users
                must provide samples with manually computed `PROVENANCE_BY_FIELD_COL` column.

                For non-root features, use this when you
                want to bypass the automatic upstream loading and field provenance calculation.

                Examples:

                - Loading upstream from custom sources

                - Pre-computing field provenances with custom logic

                - Testing specific scenarios

                Setting this parameter during normal operations is not required.

            filters: Dict mapping feature keys (as strings) to lists of Narwhals filter expressions.
                Applied at read-time. May filter the current feature,
                in this case it will also be applied to `samples` (if provided).
                Example: {"upstream/feature": [nw.col("x") > 10], ...}
            lazy: If `True`, return [metaxy.versioning.types.LazyIncrement][] with lazy Narwhals LazyFrames.
                If `False`, return [metaxy.versioning.types.Increment][] with eager Narwhals DataFrames.
            versioning_engine: Override the store's versioning engine for this operation.
            **kwargs: Backend-specific parameters

        Raises:
            ValueError: If no `samples` DataFrame has been provided when resolving an update for a root feature.
            VersioningEngineMismatchError: If versioning_engine="native" and data has wrong implementation

        Examples:
            ```py
            # Root feature - samples required
            samples = pl.DataFrame({
                "sample_uid": [1, 2, 3],
                PROVENANCE_BY_FIELD_COL: [{"field": "h1"}, {"field": "h2"}, {"field": "h3"}],
            })
            result = store.resolve_update(RootFeature, samples=nw.from_native(samples))
            ```

            ```py
            # Non-root feature - automatic (normal usage)
            result = store.resolve_update(DownstreamFeature)
            ```

            ```py
            # Non-root feature - with escape hatch (advanced)
            custom_samples = compute_custom_field_provenance(...)
            result = store.resolve_update(DownstreamFeature, samples=custom_samples)
            ```

        Note:
            Users can then process only added/changed and call write_metadata().
        """
        import narwhals as nw

        filters = filters or defaultdict(list)

        graph = current_graph()
        plan = graph.get_feature_plan(feature.spec().key)

        # Root features without samples: error (samples required)
        if not plan.deps and samples is None:
            raise ValueError(
                f"Feature {feature.spec().key} has no upstream dependencies (root feature). "
                f"Must provide 'samples' parameter with sample_uid and {METAXY_PROVENANCE_BY_FIELD} columns. "
                f"Root features require manual {METAXY_PROVENANCE_BY_FIELD} computation."
            )

        current_feature_filters = [*filters.get(feature.spec().key.to_string(), [])]

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
        else:
            for upstream_spec in plan.deps or []:
                upstream_feature_metadata = self.read_metadata(
                    upstream_spec.key,
                    filters=filters.get(upstream_spec.key.to_string(), []),
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
    def open(self, mode: AccessMode = AccessMode.READ) -> Iterator[Self]:
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

        For explicit mode control, use `with store.open(mode):` instead.

        Returns:
            Self: The opened store instance
        """
        # Determine mode based on auto_create_tables
        mode = AccessMode.WRITE if self.auto_create_tables else AccessMode.READ

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
                "Use it as a context manager: `with store: ...` or `with store.open(mode=AccessMode.WRITE): ...`"
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

    def _resolve_feature_key(
        self, feature: FeatureKey | type[BaseFeature]
    ) -> FeatureKey:
        """Resolve a Feature class or FeatureKey to FeatureKey."""
        if isinstance(feature, FeatureKey):
            return feature
        else:
            return feature.spec().key

    def _resolve_feature_plan(
        self, feature: FeatureKey | type[BaseFeature]
    ) -> FeaturePlan:
        """Resolve to FeaturePlan for dependency resolution."""
        graph = current_graph()
        if isinstance(feature, FeatureKey):
            # When given a FeatureKey, get the graph from the active context
            return graph.get_feature_plan(feature)
        else:
            # When given a Feature class, use its bound graph
            return graph.get_feature_plan(feature.spec().key)

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

    def _validate_project_write(self, feature: FeatureKey | type[BaseFeature]) -> None:
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
    ) -> None:
        """
        Internal write implementation (backend-specific).

        Backends may convert to their specific type if needed (e.g., Polars, Ibis).

        Args:
            feature_key: Feature key to write to
            df: [Narwhals](https://narwhals-dev.github.io/narwhals/)-compatible DataFrame with metadata to write

        Note: Subclasses implement this for their storage backend.
        """
        pass

    def write_metadata(
        self,
        feature: FeatureKey | type[BaseFeature],
        df: IntoFrame,
    ) -> None:
        """
        Write metadata for a feature (immutable, append-only).

        Automatically adds the canonical system columns (`metaxy_feature_version`,
        `metaxy_snapshot_version`) unless they already exist in the DataFrame
        (useful for migrations).

        Args:
            feature: Feature to write metadata for
            df: Metadata DataFrame of any type supported by [Narwhals](https://narwhals-dev.github.io/narwhals/).
                Must have `metaxy_provenance_by_field` column of type Struct with fields matching feature's fields.
                Optionally, may also contain `metaxy_data_version_by_field`.

        Raises:
            MetadataSchemaError: If DataFrame schema is invalid
            StoreNotOpenError: If store is not open
            ValueError: If writing to a feature from a different project than expected

        Note:
            - Never writes to fallback stores.

            - Project validation is performed unless disabled via `allow_cross_project_writes()` context manager.

            - Must be called within `store.open(mode=AccessMode.WRITE)` context manager.
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
        # warning: for dataframes that do not match the native MetadatStore implementation
        # and are missing the METAXY_DATA_VERSION column, this call will lead to materializing the equivalent Polars DataFrame
        # while calculating the missing METAXY_DATA_VERSION column
        df_nw = self._add_system_columns(df_nw, feature)

        self._validate_schema(df_nw)
        self.write_metadata_to_store(feature_key, df_nw)

    def _add_system_columns(
        self,
        df: Frame,
        feature: FeatureKey | type[BaseFeature],
    ) -> Frame:
        """Add all required system columns to the DataFrame.

        Args:
            df: Narwhals DataFrame/LazyFrame
            feature: Feature class or key

        Returns:
            DataFrame with all system columns added
        """
        feature_key = self._resolve_feature_key(feature)

        # Check if feature_version and snapshot_version already exist in DataFrame
        if (
            METAXY_FEATURE_VERSION in df.columns
            and METAXY_SNAPSHOT_VERSION in df.columns
        ):
            # DataFrame already has feature_version and snapshot_version - use as-is
            # This is intended for migrations writing historical versions
            # Issue a warning unless we're in a suppression context
            if not _suppress_feature_version_warning.get():
                warnings.warn(
                    f"Writing metadata for {feature_key.to_string()} with existing "
                    f"{METAXY_FEATURE_VERSION} and {METAXY_SNAPSHOT_VERSION} columns. This is intended for migrations only. "
                    "Normal code should let write_metadata() add the current versions automatically.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Get current feature version and snapshot_version from code and add them
            if isinstance(feature, type) and issubclass(feature, BaseFeature):
                current_feature_version = feature.feature_version()  # type: ignore[attr-defined]
            else:
                from metaxy.models.feature import FeatureGraph

                graph = FeatureGraph.get_active()
                feature_cls = graph.features_by_key[feature_key]
                current_feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]

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
            MetaxyColumnMissingWarning.warn_on_missing_column(
                expected=METAXY_PROVENANCE,
                df=df,
                message=f"It should have been created by a prior `MetadataStore.resolve_update` call. Re-crearing it from `{METAXY_PROVENANCE_BY_FIELD}` Did you drop it on the way?",
            )

            df = self.hash_struct_version_column(
                plan=self._resolve_feature_plan(feature_key),
                df=df,
                struct_column=METAXY_PROVENANCE_BY_FIELD,
                hash_column=METAXY_PROVENANCE,
            )

        if METAXY_CREATED_AT not in df.columns:
            from datetime import datetime, timezone

            df = df.with_columns(
                nw.lit(datetime.now(timezone.utc)).alias(METAXY_CREATED_AT)
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

    def drop_feature_metadata(self, feature: FeatureKey | type[BaseFeature]) -> None:
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
        feature: FeatureKey | type[BaseFeature],
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """
        Read metadata from THIS store only without using any fallbacks stores.

        Args:
            feature: Feature to read metadata for
            filters: List of Narwhals filter expressions for this specific feature.
            columns: Subset of columns to return

        Returns:
            Narwhals LazyFrame with metadata, or None if feature not found in the store
        """
        pass

    def read_metadata(
        self,
        feature: FeatureKey | type[BaseFeature],
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
                Example: [nw.col("x") > 10, nw.col("y") < 5]
            columns: Subset of columns to include. Metaxy's system columns are always included.
            allow_fallback: If True, check fallback stores on local miss
            current_only: If True, only return rows with current feature_version
                (default: True for safety)
            latest_only: Whether to deduplicate samples within `id_columns` groups ordered by `metaxy_created_at`.

        Returns:
            Narwhals LazyFrame with metadata

        Raises:
            FeatureNotFoundError: If feature not found in any store
            SystemDataNotFoundError: When attempting to read non-existant Metaxy system data
            ValueError: If both feature_version and current_only=True are provided
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

    # ========== Feature Existence ==========

    def has_feature(
        self,
        feature: FeatureKey | type[BaseFeature],
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
        # Check local
        if self.read_metadata_in_store(feature) is not None:
            return True

        # Check fallback stores
        if check_fallback:
            for store in self.fallback_stores:
                if store.has_feature(feature, check_fallback=True):
                    return True

        return False

    @abstractmethod
    def display(self) -> str:
        """Return a human-readable display string for this store.

        Used in warnings, logs, and CLI output to identify the store.

        Returns:
            Display string (e.g., "DuckDBMetadataStore(database=/path/to/db.duckdb)")
        """
        pass

    def copy_metadata(
        self,
        from_store: MetadataStore,
        features: list[FeatureKey | type[BaseFeature]] | None = None,
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
                "Destination store must be opened with store.open(AccessMode.WRITE) before use"
            )

        # Auto-open source store if not already open
        if not from_store._is_open:
            with from_store.open(AccessMode.READ):
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
        features: list[FeatureKey | type[BaseFeature]] | None,
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
            # Convert all to FeatureKey
            features_to_copy = []
            for item in features:
                if isinstance(item, FeatureKey):
                    features_to_copy.append(item)
                else:
                    # Must be Feature class
                    features_to_copy.append(item.spec().key)
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
