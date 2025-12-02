from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

import dagster as dg
import narwhals as nw

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PARTITION_KEY,
    METAXY_DAGSTER_METADATA_KEY,
)
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.models.constants import METAXY_CREATED_AT, METAXY_MATERIALIZATION_ID


class FeatureStats(NamedTuple):
    """Statistics about a feature's metadata for Dagster events."""

    row_count: int
    data_version: dg.DataVersion


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


def compute_stats_from_lazy_frame(lazy_df: nw.LazyFrame) -> FeatureStats:  # pyright: ignore[reportMissingTypeArgument]
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
        nw.col(METAXY_CREATED_AT).mean().alias("__mean_ts"),
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
    inherit_feature_key_as_asset_key: bool = False,
    dagster_key: dg.AssetKey | None = None,
) -> dg.AssetKey:
    """Get the Dagster asset key for a Metaxy feature spec.

    Args:
        feature_spec: The Metaxy feature spec.
        inherit_feature_key_as_asset_key: If `True`, use the feature key as the asset key
            when `dagster/attributes.asset_key` is not set on the feature spec.
        dagster_key: Optional existing Dagster asset key. Used by `metaxify` to preserve
            original asset keys when `inherit_feature_key_as_asset_key` is False.

    Returns:
        The Dagster asset key, determined as follows:

        1. If feature spec has `dagster/attributes.asset_key` set, that value is used as-is.

        2. Otherwise, if `inherit_feature_key_as_asset_key` is True, the feature key is used.

        3. Otherwise, if `dagster_key` is provided, it's returned as-is.

        4. Otherwise, the feature key is used.
    """
    # If dagster/attributes.asset_key is set, use it as-is
    dagster_attrs = feature_spec.metadata.get(METAXY_DAGSTER_METADATA_KEY)
    if isinstance(dagster_attrs, dict) and (
        custom_asset_key := dagster_attrs.get("asset_key")
    ):
        return dg.AssetKey(custom_asset_key)  # pyright: ignore[reportArgumentType]

    # If inherit_feature_key_as_asset_key is set, use the feature key
    if inherit_feature_key_as_asset_key:
        return dg.AssetKey(list(feature_spec.key.parts))

    # Otherwise, use dagster_key if provided, else fall back to feature key
    if dagster_key is not None:
        return dagster_key

    return dg.AssetKey(list(feature_spec.key.parts))


def generate_materialize_results(
    context: dg.AssetExecutionContext,
    store: mx.MetadataStore | MetaxyStoreFromConfigResource,
    specs: Iterable[dg.AssetSpec] | None = None,
) -> Iterator[dg.MaterializeResult[None]]:
    """Generate `dagster.MaterializeResult` events for assets in topological order.

    Yields a `MaterializeResult` for each asset spec, sorted by their associated
    Metaxy features in topological order (dependencies before dependents).
    Each result includes the row count as `"dagster/row_count"` metadata.

    Args:
        context: The Dagster asset execution context.
        store: The Metaxy metadata store to read from.
        specs: Optional, concrete Dagster asset specs.
            If missing, specs will be taken from the context.

    Yields:
        Materialization result for each asset in topological order.

    Example:
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
    """
    # Build mapping from feature key to asset spec
    spec_by_feature_key: dict[mx.FeatureKey, dg.AssetSpec] = {}
    specs = specs or context.assets_def.specs
    for spec in specs:
        if feature_key_raw := spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY):
            feature_key = mx.coerce_to_feature_key(feature_key_raw)
            spec_by_feature_key[feature_key] = spec

    # Sort by topological order of feature keys
    graph = mx.FeatureGraph.get_active()
    sorted_keys = graph.topological_sort_features(list(spec_by_feature_key.keys()))

    for key in sorted_keys:
        asset_spec = spec_by_feature_key[key]
        partition_filters = get_partition_filter(context, asset_spec)

        with store:
            try:
                lazy_df = store.read_metadata(key, filters=partition_filters)
            except FeatureNotFoundError:
                context.log.exception(
                    f"Feature {key.to_string()} not found in store, skipping materialization result"
                )
                continue

            stats = compute_stats_from_lazy_frame(lazy_df)

            # Build runtime metadata using shared function
            metadata = build_runtime_feature_metadata(key, store, lazy_df, context)

            # Get materialized-in-run count if materialization_id is set
            if store.materialization_id is not None:
                mat_filters = partition_filters + [
                    nw.col(METAXY_MATERIALIZATION_ID) == store.materialization_id
                ]
                mat_df = store.read_metadata(key, filters=mat_filters)
                metadata["metaxy/materialized_in_run"] = (
                    mat_df.select(nw.len()).collect().item(0, 0)
                )

        yield dg.MaterializeResult(
            value=None,
            asset_key=asset_spec.key,
            metadata=metadata,
            data_version=stats.data_version,
        )


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
    feature_cls = mx.get_feature_by_key(feature_key)

    return {
        "feature": {
            "project": feature_cls.project,
            "spec": feature_cls.spec().model_dump(mode="json"),
            "version": feature_cls.feature_version(),
            "type": feature_cls.__module__,
        },
        "metaxy": {
            "version": mx.__version__,
            "plugins": mx.MetaxyConfig.get().plugins,
        },
    }


def build_runtime_feature_metadata(
    feature_key: mx.FeatureKey,
    store: mx.MetadataStore | MetaxyStoreFromConfigResource,
    lazy_df: nw.LazyFrame[Any],
    context: dg.AssetExecutionContext | dg.OutputContext,
) -> dict[str, Any]:
    """Build runtime metadata for a Metaxy feature in Dagster.

    This function consolidates all runtime metadata construction for Dagster events.
    It is used by the IOManager, generate_materialize_results, and generate_observe_results.

    Args:
        feature_key: The Metaxy feature key.
        store: The metadata store (used for store-specific metadata like URI, table_name).
        lazy_df: The LazyFrame containing the feature data (for stats and preview).
        context: Dagster context for determining partition state and logging errors.

    Returns:
        A dictionary containing all runtime metadata:
        - `dagster/row_count`: Total row count
        - `dagster/partition_row_count`: Row count (only if partitioned)
        - `dagster/table_name`: Table name from store (if available)
        - `dagster/uri`: URI from store (if available)
        - `dagster/table`: Table preview

        Returns empty dict if an error occurs during metadata collection.

    Example:
        ```python
        with store:
            lazy_df = store.read_metadata(feature_key)
            metadata = build_runtime_feature_metadata(feature_key, store, lazy_df, context)
            context.add_output_metadata(metadata)
        ```
    """
    # Import here to avoid circular import
    from metaxy.ext.dagster.table_metadata import (
        build_column_schema,
        build_table_preview_metadata,
    )

    try:
        # Compute stats from the lazy frame
        stats = compute_stats_from_lazy_frame(lazy_df)

        # Get store metadata
        store_metadata = store.get_store_metadata(feature_key)

        # Build metadata dict
        metadata: dict[str, Any] = {
            "dagster/row_count": stats.row_count,
        }

        # Add partition_row_count for partitioned assets
        if context is not None and context.has_partition_key:
            metadata["dagster/partition_row_count"] = stats.row_count

        # Map store metadata to dagster standard keys
        if "table_name" in store_metadata:
            metadata["dagster/table_name"] = store_metadata["table_name"]

        if "uri" in store_metadata:
            metadata["dagster/uri"] = dg.MetadataValue.path(store_metadata["uri"])

        # Build table preview
        feature_cls = mx.get_feature_by_key(feature_key)
        schema = build_column_schema(feature_cls)
        metadata["dagster/table"] = build_table_preview_metadata(lazy_df, schema)

        return metadata
    except Exception:
        context.log.exception(
            f"Failed to build runtime metadata for feature {feature_key.to_string()}"
        )
        return {}


def generate_observe_results(
    context: dg.AssetExecutionContext,
    store: mx.MetadataStore | MetaxyStoreFromConfigResource,
    specs: Iterable[dg.AssetSpec] | None = None,
) -> Iterator[dg.ObserveResult]:
    """Generate `dagster.ObserveResult` events for assets in topological order.

    Yields an `ObserveResult` for each asset spec that has `"metaxy/feature"` metadata key set, sorted by their associated
    Metaxy features in topological order.
    Each result includes the row count as `"dagster/row_count"` metadata.

    Args:
        context: The Dagster asset execution context.
        store: The Metaxy metadata store to read from.
        specs: Optional, concrete Dagster asset specs.
            If missing, this function will take the current specs from the context.

    Yields:
        Observation result for each asset in topological order.

    Example:
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
    """
    # Build mapping from feature key to asset spec
    spec_by_feature_key: dict[mx.FeatureKey, dg.AssetSpec] = {}
    specs = specs or context.assets_def.specs

    for spec in specs:
        if feature_key_raw := spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY):
            feature_key = mx.coerce_to_feature_key(feature_key_raw)
            spec_by_feature_key[feature_key] = spec

    # Sort by topological order of feature keys
    graph = mx.FeatureGraph.get_active()
    sorted_keys = graph.topological_sort_features(list(spec_by_feature_key.keys()))

    for key in sorted_keys:
        asset_spec = spec_by_feature_key[key]
        partition_filters = get_partition_filter(context, asset_spec)

        with store:
            try:
                lazy_df = store.read_metadata(key, filters=partition_filters)
            except FeatureNotFoundError:
                context.log.exception(
                    f"Feature {key.to_string()} not found in store, skipping observation result"
                )
                continue

            stats = compute_stats_from_lazy_frame(lazy_df)
            metadata = build_runtime_feature_metadata(key, store, lazy_df, context)

        yield dg.ObserveResult(
            asset_key=asset_spec.key,
            metadata=metadata,
            data_version=stats.data_version,
        )
