from collections.abc import Iterator, Sequence
from typing import NamedTuple

import dagster as dg
import narwhals as nw

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PARTITION_KEY,
    METAXY_DAGSTER_METADATA_KEY,
)
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
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
    specs: Sequence[dg.AssetSpec],
) -> Iterator[dg.MaterializeResult[None]]:
    """Generate `dagster.MaterializeResult` events for assets in topological order.

    Yields a `MaterializeResult` for each asset spec, sorted by their associated
    Metaxy features in topological order (dependencies before dependents).
    Each result includes the row count as `"dagster/row_count"` metadata.

    Args:
        context: The Dagster asset execution context.
        store: The Metaxy metadata store to read from.
        specs: Sequence of asset specs with `"metaxy/feature"` metadata set.

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
            yield from generate_materialize_results(context, store, specs)
        ```
    """
    # Build mapping from feature key to asset spec
    spec_by_feature_key: dict[mx.FeatureKey, dg.AssetSpec] = {}
    for spec in specs:
        feature_key_raw = spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY)
        if feature_key_raw is None:
            raise ValueError(
                f"AssetSpec {spec.key} missing '{DAGSTER_METAXY_FEATURE_METADATA_KEY}' metadata"
            )
        feature_key = mx.coerce_to_feature_key(feature_key_raw)
        spec_by_feature_key[feature_key] = spec

    # Sort by topological order of feature keys
    graph = mx.FeatureGraph.get_active()
    sorted_keys = graph.topological_sort_features(list(spec_by_feature_key.keys()))

    for key in sorted_keys:
        asset_spec = spec_by_feature_key[key]
        partition_filters = get_partition_filter(context, asset_spec)

        with store:
            # Get total stats (partition-filtered)
            lazy_df = store.read_metadata(key, filters=partition_filters)
            stats = compute_stats_from_lazy_frame(lazy_df)

            # Get materialized-in-run count if materialization_id is set
            materialized_in_run: int | None = None
            if store.materialization_id is not None:
                mat_filters = partition_filters + [
                    nw.col(METAXY_MATERIALIZATION_ID) == store.materialization_id
                ]
                mat_df = store.read_metadata(key, filters=mat_filters)
                materialized_in_run = mat_df.select(nw.len()).collect().item(0, 0)

        metadata: dict[str, int] = {"dagster/row_count": stats.row_count}
        if materialized_in_run is not None:
            metadata["metaxy/materialized_in_run"] = materialized_in_run

        yield dg.MaterializeResult(
            value=None,
            asset_key=asset_spec.key,
            metadata=metadata,
            data_version=stats.data_version,
        )


def generate_observe_results(
    context: dg.AssetExecutionContext,
    store: mx.MetadataStore | MetaxyStoreFromConfigResource,
    specs: Sequence[dg.AssetSpec],
) -> Iterator[dg.ObserveResult]:
    """Generate `dagster.ObserveResult` events for assets in topological order.

    Yields an `ObserveResult` for each asset spec, sorted by their associated
    Metaxy features in topological order.
    Each result includes the row count as `"dagster/row_count"` metadata.

    Args:
        context: The Dagster asset execution context.
        store: The Metaxy metadata store to read from.
        specs: Sequence of asset specs with `"metaxy/feature"` metadata set.

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
            yield from generate_observe_results(context, store, specs)
        ```
    """
    # Build mapping from feature key to asset spec
    spec_by_feature_key: dict[mx.FeatureKey, dg.AssetSpec] = {}
    for spec in specs:
        feature_key_raw = spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY)
        if feature_key_raw is None:
            raise ValueError(
                f"AssetSpec {spec.key} missing '{DAGSTER_METAXY_FEATURE_METADATA_KEY}' metadata"
            )
        feature_key = mx.coerce_to_feature_key(feature_key_raw)
        spec_by_feature_key[feature_key] = spec

    # Sort by topological order of feature keys
    graph = mx.FeatureGraph.get_active()
    sorted_keys = graph.topological_sort_features(list(spec_by_feature_key.keys()))

    for key in sorted_keys:
        asset_spec = spec_by_feature_key[key]
        filters = get_partition_filter(context, asset_spec)

        with store:
            lazy_df = store.read_metadata(key, filters=filters)
            stats = compute_stats_from_lazy_frame(lazy_df)

        yield dg.ObserveResult(
            asset_key=asset_spec.key,
            metadata={"dagster/row_count": stats.row_count},
            data_version=stats.data_version,
        )
