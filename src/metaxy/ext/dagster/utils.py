from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

import dagster as dg
import narwhals as nw

import metaxy as mx
from metaxy._decorators import public
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_CODE_VERSION_TAG_KEY,
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_FEATURE_VERSION_TAG_KEY,
    DAGSTER_METAXY_PARTITION_KEY,
    DAGSTER_METAXY_PARTITION_METADATA_KEY,
    METAXY_DAGSTER_METADATA_KEY,
)
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.models.constants import METAXY_CREATED_AT, METAXY_MATERIALIZATION_ID


@public
class FeatureStats(NamedTuple):
    """Statistics about a feature's metadata for Dagster events."""

    row_count: int
    data_version: dg.DataVersion


def _is_all_partitions_subset(context: dg.InputContext) -> bool:
    """Check if the input context represents all partitions (e.g., via AllPartitionMapping).

    When AllPartitionMapping is used, the asset_partitions_subset will be an
    AllPartitionsSubset. In this case, we should not add partition filters since
    filtering by all partitions would be expensive and equivalent to no filter.
    """
    # Import here to avoid potential issues with private API
    from dagster._core.definitions.partitions.subset.all import AllPartitionsSubset

    # Access the private _asset_partitions_subset to check its type
    # This is more efficient than iterating all partition keys
    subset = context._asset_partitions_subset  # noqa: SLF001
    return isinstance(subset, AllPartitionsSubset)


def build_partition_filter_from_input_context(
    context: dg.InputContext,
) -> list[nw.Expr]:
    """Build partition filter expressions from an InputContext.

    Extracts partition information from the upstream asset's metadata and the
    current partition context to build appropriate filter expressions.

    Handles:
    - `partition_by` metadata: filters by the specified column using partition key(s)
    - `metaxy/partition` metadata: additional static filters as {column: value} dict
    - `AllPartitionMapping`: skips partition filters (loading all partitions anyway)

    Args:
        context: Dagster InputContext for loading upstream data.

    Returns:
        List of filter expressions to apply when reading upstream data.
    """
    filters: list[nw.Expr] = []

    # Get upstream asset's metadata
    upstream_metadata = context.upstream_output.definition_metadata if context.upstream_output else None

    if upstream_metadata is None:
        return filters

    # Handle partition_by: filter by Dagster partition key(s)
    partition_col = upstream_metadata.get(DAGSTER_METAXY_PARTITION_KEY)
    if partition_col and context.has_asset_partitions:
        # Skip partition filter for AllPartitionMapping - it would include all partitions
        # anyway, and enumerating them all is expensive
        if _is_all_partitions_subset(context):
            pass  # No filter needed - we want all partitions
        else:
            # Get partition keys for this input
            partition_keys = list(context.asset_partition_keys)
            if len(partition_keys) == 1:
                filters.append(nw.col(partition_col) == partition_keys[0])
            elif len(partition_keys) > 1:
                filters.append(nw.col(partition_col).is_in(partition_keys))

    # Handle metaxy/partition: additional static filters
    metaxy_partition = upstream_metadata.get(DAGSTER_METAXY_PARTITION_METADATA_KEY)
    if isinstance(metaxy_partition, dict):
        for col, value in metaxy_partition.items():
            if isinstance(value, list):
                filters.append(nw.col(col).is_in(value))
            else:
                filters.append(nw.col(col) == value)

    return filters


def build_partition_filter(
    partition_col: str | None,
    partition_key: str | None,
) -> list[nw.Expr]:
    """Build partition filter expressions from column name and partition key.

    Args:
        partition_col: The column to filter by (from `partition_by` metadata).
        partition_key: The partition key value to filter for.

    Returns:
        List with a single filter expression, or empty list if either arg is None.
    """
    if partition_col is None or partition_key is None:
        return []
    return [nw.col(partition_col) == partition_key]


def build_metaxy_partition_filter(
    partition_metadata: dict[str, str] | None,
) -> list[nw.Expr]:
    """Build filter expressions from metaxy/partition metadata.

    Args:
        partition_metadata: Dict mapping column name to value, e.g., {"region": "us"}

    Returns:
        List of filter expressions (one per column/value pair).
    """
    if partition_metadata is None:
        return []
    return [nw.col(col) == value for col, value in partition_metadata.items()]


_DAGSTER_TAG_VALUE_MAX_LENGTH = 63


def build_feature_event_tags(feature: mx.CoercibleToFeatureKey) -> dict[str, str]:
    """Build tags for Dagster events (MaterializeResult, ObserveResult, AssetObservation).

    Creates a dictionary with version tags for the given Metaxy feature.
    These tags can be used to filter and search for events in the Dagster UI.

    Args:
        feature: The Metaxy feature (class, key, or string).

    Returns:
        A dictionary with the following tags:

        - `metaxy/feature`: The feature key as table_name format (uses `__` separator
          since Dagster tags don't allow `/` in values).
        - `metaxy/feature_version`: The feature version (combines spec + dependencies),
          truncated to 63 characters (Dagster limit).
        - `metaxy/feature_code_version`: The feature spec code version,
          truncated to 63 characters (Dagster limit).
    """
    feature_key = mx.coerce_to_feature_key(feature)
    feature_def = mx.get_feature_by_key(feature_key)
    feature_version = mx.current_graph().get_feature_version(feature_key)
    return {
        # Use table_name format since Dagster tags don't allow '/' in values
        DAGSTER_METAXY_FEATURE_METADATA_KEY: feature_key.table_name,
        # Truncate version hashes to 63 chars (Dagster tag value limit)
        DAGSTER_METAXY_FEATURE_VERSION_TAG_KEY: feature_version[:_DAGSTER_TAG_VALUE_MAX_LENGTH],
        DAGSTER_METAXY_FEATURE_CODE_VERSION_TAG_KEY: feature_def.spec.code_version[:_DAGSTER_TAG_VALUE_MAX_LENGTH],
    }


def get_partition_filter(
    context: dg.AssetExecutionContext,
    spec: dg.AssetSpec,
) -> list[nw.Expr]:
    """Get partition filter expressions for a partitioned asset.

    Args:
        context: The Dagster asset execution context.
        spec: The AssetSpec containing `partition_by` metadata.

    Returns:
        List of filter expressions. Empty if not partitioned or no partition_by metadata.
    """
    if not context.has_partition_key:
        return []

    partition_col = spec.metadata.get(DAGSTER_METAXY_PARTITION_KEY)
    if not isinstance(partition_col, str):
        return []

    return build_partition_filter(partition_col, context.partition_key)


def compute_row_count(lazy_df: nw.LazyFrame) -> int:
    """Compute row count from a narwhals LazyFrame.

    Args:
        lazy_df: A narwhals LazyFrame.

    Returns:
        The number of rows in the frame.
    """
    return lazy_df.select(nw.len()).collect().item(0, 0)


def compute_stats_from_lazy_frame(lazy_df: nw.LazyFrame) -> FeatureStats:
    """Compute statistics from a narwhals LazyFrame.

    Computes row count and data version from the frame.
    The data version is based on mean(metaxy_created_at) to detect both
    additions and deletions.

    Args:
        lazy_df: A narwhals LazyFrame with metaxy metadata.

    Returns:
        FeatureStats with row_count and data_version.
    """
    stats = lazy_df.select(
        nw.len().alias("__count"),
        nw.col(METAXY_CREATED_AT).cast(nw.Float64).mean().alias("__mean_ts"),
    ).collect()

    row_count: int = stats.item(0, "__count")
    if row_count == 0:
        return FeatureStats(row_count=0, data_version=dg.DataVersion("empty"))

    mean_ts = stats.item(0, "__mean_ts")
    return FeatureStats(row_count=row_count, data_version=dg.DataVersion(str(mean_ts)))


def compute_feature_stats(
    store: mx.MetadataStore,
    feature: mx.CoercibleToFeatureKey,
) -> FeatureStats:
    """Compute statistics for a feature's metadata.

    Reads the feature metadata and computes row count and data version.
    The data version is based on mean(metaxy_created_at) to detect both
    additions and deletions.

    Args:
        store: The Metaxy metadata store to read from.
        feature: The feature to compute stats for.

    Returns:
        FeatureStats with row_count and data_version.
    """
    with store:
        lazy_df = store.read_metadata(feature)
        return compute_stats_from_lazy_frame(lazy_df)


def get_asset_key_for_metaxy_feature_spec(
    feature_spec: mx.FeatureSpec,
) -> dg.AssetKey:
    """Get the Dagster asset key for a Metaxy feature spec.

    Args:
        feature_spec: The Metaxy feature spec.

    Returns:
        The Dagster asset key, determined as follows:

        1. If feature spec has `dagster/attributes.asset_key` set, that value is used.

        2. Otherwise, the feature key is used.
    """
    # If dagster/attributes.asset_key is set, use it as-is
    dagster_attrs = feature_spec.metadata.get(METAXY_DAGSTER_METADATA_KEY)
    if isinstance(dagster_attrs, dict) and (custom_asset_key := dagster_attrs.get("asset_key")):
        return dg.AssetKey(custom_asset_key)

    # Use the feature key as the asset key
    return dg.AssetKey(list(feature_spec.key.parts))


@public
def generate_materialize_results(
    context: dg.AssetExecutionContext | dg.OpExecutionContext,
    store: mx.MetadataStore | MetaxyStoreFromConfigResource,
    specs: Iterable[dg.AssetSpec] | None = None,
) -> Iterator[dg.MaterializeResult[None]]:
    """Generate `dagster.MaterializeResult` events for assets in topological order.

    Yields a `MaterializeResult` for each asset spec, sorted by their associated
    Metaxy features in topological order (dependencies before dependents).
    Each result includes the row count as `"dagster/row_count"` metadata.

    Args:
        context: The Dagster execution context.
        store: The Metaxy metadata store to read from.
        specs: Concrete Dagster asset specs. Required when using `OpExecutionContext`.
            Optional for `AssetExecutionContext` (defaults to `context.assets_def.specs`).

    Yields:
        Materialization result for each asset in topological order.

    Example:
        Using with `@multi_asset`:
        ```python
        specs = [
            dg.AssetSpec("output_a", metadata={"metaxy/feature": "my/feature/a"}),
            dg.AssetSpec("output_b", metadata={"metaxy/feature": "my/feature/b"}),
        ]


        @metaxify
        @dg.multi_asset(specs=specs)
        def my_multi_asset(context: dg.AssetExecutionContext, store: mx.MetadataStore):
            # ... compute and write data ...
            yield from generate_materialize_results(context, store)
        ```

        Using with `@op`:
        ```python
        specs = [
            dg.AssetSpec("output_a", metadata={"metaxy/feature": "my/feature/a"}),
        ]


        @dg.op
        def my_op(context: dg.OpExecutionContext, store: mx.MetadataStore):
            # ... compute and write data ...
            yield from generate_materialize_results(context, store, specs=specs)
        ```
    """
    # Build mapping from feature key to asset spec
    spec_by_feature_key: dict[mx.FeatureKey, dg.AssetSpec] = {}
    if specs is None:
        if not isinstance(context, dg.AssetExecutionContext):
            raise ValueError("specs must be provided when using OpExecutionContext")
        specs = context.assets_def.specs
    for spec in specs:
        if feature_key_raw := spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY):
            feature_key = mx.coerce_to_feature_key(feature_key_raw)
            spec_by_feature_key[feature_key] = spec

    # Sort by topological order of feature keys
    graph = mx.FeatureGraph.get_active()
    sorted_keys = graph.topological_sort_features(list(spec_by_feature_key.keys()))

    for key in sorted_keys:
        asset_spec = spec_by_feature_key[key]
        partition_col = asset_spec.metadata.get(DAGSTER_METAXY_PARTITION_KEY)
        metaxy_partition = asset_spec.metadata.get(DAGSTER_METAXY_PARTITION_METADATA_KEY)

        with store:  # ty: ignore[invalid-context-manager]
            try:
                # Build runtime metadata (handles reading, filtering, and stats internally)
                metadata, stats = build_runtime_feature_metadata(
                    key,
                    store,
                    context,
                    partition_col=partition_col,
                    metaxy_partition=metaxy_partition,
                )
            except FeatureNotFoundError:
                context.log.exception(f"Feature {key.to_string()} not found in store, skipping materialization result")
                continue

            # Get materialized-in-run count if materialization_id is set
            if store.materialization_id is not None:  # ty: ignore[possibly-missing-attribute]
                mat_df = store.read_metadata(  # ty: ignore[possibly-missing-attribute]
                    key,
                    filters=[
                        nw.col(METAXY_MATERIALIZATION_ID) == store.materialization_id  # ty: ignore[possibly-missing-attribute]
                    ],
                )
                metadata["metaxy/materialized_in_run"] = mat_df.select(nw.len()).collect().item(0, 0)

        yield dg.MaterializeResult(
            value=None,
            asset_key=asset_spec.key,
            metadata=metadata,
            data_version=stats.data_version,
            tags=build_feature_event_tags(key),
        )


@public
def build_feature_info_metadata(
    feature: mx.CoercibleToFeatureKey,
) -> dict[str, Any]:
    """Build feature info metadata dict for Dagster assets.

    Creates a dictionary with information about the Metaxy feature that can be
    used as Dagster asset metadata under the `"metaxy/feature_info"` key.

    Args:
        feature: The Metaxy feature (class, key, or string).

    Returns:
        A nested dictionary containing:

        - `feature`: Feature information
            - `project`: The project name
            - `spec`: The full feature spec as a dict (via `model_dump()`)
            - `version`: The feature version string
            - `type`: The feature class module path
        - `metaxy`: Metaxy library information
            - `version`: The metaxy library version

    !!! tip
        This is automatically injected by [`@metaxify`][metaxy.ext.dagster.metaxify.metaxify]

    Example:
        ```python
        from metaxy.ext.dagster.utils import build_feature_info_metadata

        info = build_feature_info_metadata(MyFeature)
        # {
        #     "feature": {
        #         "project": "my_project",
        #         "spec": {...},  # Full FeatureSpec model_dump()
        #         "version": "my__feature@abc123",
        #         "type": "myproject.features",
        #     },
        #     "metaxy": {
        #         "version": "0.1.0",
        #     },
        # }
        ```
    """
    feature_key = mx.coerce_to_feature_key(feature)
    feature_def = mx.get_feature_by_key(feature_key)
    feature_version = mx.current_graph().get_feature_version(feature_key)

    return {
        "feature": {
            "project": feature_def.project,
            "spec": feature_def.spec.model_dump(mode="json"),
            "version": feature_version,
            "type": feature_def.feature_class_path,
        },
        "metaxy": {
            "version": mx.__version__,
            "plugins": mx.MetaxyConfig.get().plugins,
        },
    }


def build_runtime_feature_metadata(
    feature_key: mx.FeatureKey,
    store: mx.MetadataStore | MetaxyStoreFromConfigResource,
    context: dg.AssetExecutionContext | dg.OpExecutionContext | dg.OutputContext,
    *,
    partition_col: str | None = None,
    metaxy_partition: dict[str, str] | None = None,
) -> tuple[dict[str, Any], FeatureStats]:
    """Build runtime metadata for a Metaxy feature in Dagster.

    This function consolidates all runtime metadata construction for Dagster events.
    It is used by the IOManager, generate_materialize_results, and generate_observe_results.

    Handles reading data from the store and applying partition filters automatically
    based on the context's partition key.

    Args:
        feature_key: The Metaxy feature key.
        store: The metadata store (used for store-specific metadata like URI, table_name).
        context: Dagster context for determining partition state and logging errors.
        partition_col: Optional column name to filter by for Dagster partitioned assets.
            If provided and context has a partition key, data will be filtered.
        metaxy_partition: Optional dict mapping column names to values.
            Is expected to be set on Dagster assets that contribute to the same Metaxy feature.
            Allows computing individual metadata for these assets.

    Returns:
        A tuple of (metadata_dict, feature_stats) where:

        metadata_dict contains:
        - `metaxy/feature`: Feature key as string
        - `metaxy/info`: Feature and metaxy library information (from `build_feature_info_metadata`)
        - `metaxy/store`: Store type and configuration
        - `dagster/row_count`: Total row count (across all partitions)
        - `dagster/partition_row_count`: Row count for current partition (only if partitioned)
        - `dagster/table_name`: Table name from store (if available)
        - `dagster/uri`: URI from store (if available)
        - `dagster/table`: Table preview

        feature_stats contains row_count and data_version for the partition-filtered data.

        Returns (empty dict, empty stats) if an error occurs during metadata collection.

    Raises:
        FeatureNotFoundError: If the feature is not found in the store.

    Example:
        ```python
        with store:
            metadata, stats = build_runtime_feature_metadata(feature_key, store, context, partition_col="date")
            context.add_output_metadata(metadata)
        ```
    """
    # Import here to avoid circular import
    from metaxy.ext.dagster.table_metadata import (
        build_column_schema,
        build_table_preview_metadata,
    )

    # Build Dagster partition filter if applicable (for time/date partitions etc.)
    partition_key = context.partition_key if context.has_partition_key else None
    dagster_partition_filters = build_partition_filter(partition_col, partition_key)

    # Build metaxy partition filter (for multi-asset logical partitions)
    metaxy_partition_filters = build_metaxy_partition_filter(metaxy_partition)

    # Combine both filter types for reading this asset's view of the data
    all_filters = dagster_partition_filters + metaxy_partition_filters

    lazy_df = store.read_metadata(feature_key, filters=all_filters)  # ty: ignore[possibly-missing-attribute]

    try:
        # Compute stats from filtered data (includes data_version for callers)
        stats = compute_stats_from_lazy_frame(lazy_df)

        # Get store metadata
        store_metadata = store.get_store_metadata(feature_key)  # ty: ignore[possibly-missing-attribute]

        # Build metadata dict with metaxy info and store info
        store_cls = store.__class__
        metadata: dict[str, Any] = {
            "metaxy/feature": feature_key.to_string(),
            "metaxy/info": build_feature_info_metadata(feature_key),
            "metaxy/store": {
                "type": f"{store_cls.__module__}.{store_cls.__qualname__}",
                "display": store.display(),  # ty: ignore[possibly-missing-attribute]
                "versioning_engine": store._versioning_engine,  # ty: ignore[possibly-missing-attribute]
                **store_metadata,
            },
        }

        # For Dagster partitioned assets, compute total row count by re-reading
        # with only Dagster partition filters (not metaxy partition)
        if context.has_partition_key:
            # Read with only dagster partition filter for total count
            full_lazy_df = store.read_metadata(  # ty: ignore[possibly-missing-attribute]
                feature_key, filters=metaxy_partition_filters
            )
            metadata["dagster/row_count"] = compute_row_count(full_lazy_df)
            metadata["dagster/partition_row_count"] = stats.row_count
        else:
            # When metaxy_partition is set, stats.row_count is the partition-filtered count
            # dagster/row_count should reflect what this asset sees
            metadata["dagster/row_count"] = stats.row_count

        # Map store metadata to dagster standard keys
        if "table_name" in store_metadata:
            metadata["dagster/table_name"] = store_metadata["table_name"]

        if "uri" in store_metadata:
            metadata["dagster/uri"] = dg.MetadataValue.path(store_metadata["uri"])

        # Build table preview (from partition-filtered data)
        # Skip schema extraction for external features (no Python class)
        feature_def = mx.get_feature_by_key(feature_key)
        if not feature_def.is_external:
            schema = build_column_schema(feature_def)
            metadata["dagster/table"] = build_table_preview_metadata(lazy_df, schema)

        return metadata, stats
    except FeatureNotFoundError:
        raise
    except Exception:
        context.log.exception(f"Failed to build runtime metadata for feature {feature_key.to_string()}")
        return {}, FeatureStats(row_count=0, data_version=dg.DataVersion("empty"))


@public
def generate_observe_results(
    context: dg.AssetExecutionContext | dg.OpExecutionContext,
    store: mx.MetadataStore | MetaxyStoreFromConfigResource,
    specs: Iterable[dg.AssetSpec] | None = None,
) -> Iterator[dg.ObserveResult]:
    """Generate `dagster.ObserveResult` events for assets in topological order.

    Yields an `ObserveResult` for each asset spec that has `"metaxy/feature"` metadata key set, sorted by their associated
    Metaxy features in topological order.
    Each result includes the row count as `"dagster/row_count"` metadata.

    Args:
        context: The Dagster execution context.
        store: The Metaxy metadata store to read from.
        specs: Concrete Dagster asset specs. Required when using `OpExecutionContext`.
            Optional for `AssetExecutionContext` (defaults to `context.assets_def.specs`).

    Yields:
        Observation result for each asset in topological order.

    Example:
        Using with `@multi_observable_source_asset`:
        ```python
        specs = [
            dg.AssetSpec("output_a", metadata={"metaxy/feature": "my/feature/a"}),
            dg.AssetSpec("output_b", metadata={"metaxy/feature": "my/feature/b"}),
        ]


        @metaxify
        @dg.multi_observable_source_asset(specs=specs)
        def my_observable_assets(context: dg.AssetExecutionContext, store: mx.MetadataStore):
            yield from generate_observe_results(context, store)
        ```

        Using with `@op`:
        ```python
        specs = [
            dg.AssetSpec("output_a", metadata={"metaxy/feature": "my/feature/a"}),
        ]


        @dg.op
        def my_op(context: dg.OpExecutionContext, store: mx.MetadataStore):
            yield from generate_observe_results(context, store, specs=specs)
        ```
    """
    # Build mapping from feature key to asset spec
    spec_by_feature_key: dict[mx.FeatureKey, dg.AssetSpec] = {}
    if specs is None:
        if not isinstance(context, dg.AssetExecutionContext):
            raise ValueError("specs must be provided when using OpExecutionContext")
        specs = context.assets_def.specs

    for spec in specs:
        if feature_key_raw := spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY):
            feature_key = mx.coerce_to_feature_key(feature_key_raw)
            spec_by_feature_key[feature_key] = spec

    # Sort by topological order of feature keys
    graph = mx.FeatureGraph.get_active()
    sorted_keys = graph.topological_sort_features(list(spec_by_feature_key.keys()))

    for key in sorted_keys:
        asset_spec = spec_by_feature_key[key]
        partition_col = asset_spec.metadata.get(DAGSTER_METAXY_PARTITION_KEY)
        metaxy_partition = asset_spec.metadata.get(DAGSTER_METAXY_PARTITION_METADATA_KEY)

        with store:  # ty: ignore[invalid-context-manager]
            try:
                # Build runtime metadata (handles reading, filtering, and stats internally)
                # For observers with no metaxy_partition, this reads all data
                metadata, stats = build_runtime_feature_metadata(
                    key,
                    store,
                    context,
                    partition_col=partition_col,
                    metaxy_partition=metaxy_partition,
                )
            except FeatureNotFoundError:
                context.log.exception(f"Feature {key.to_string()} not found in store, skipping observation result")
                continue

        yield dg.ObserveResult(
            asset_key=asset_spec.key,
            metadata=metadata,
            data_version=stats.data_version,
            tags=build_feature_event_tags(key),
        )
