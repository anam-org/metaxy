from collections.abc import Iterator, Sequence

import dagster as dg
import narwhals as nw

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    METAXY_DAGSTER_METADATA_KEY,
)


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


def generate_materialization_events(
    context: dg.AssetExecutionContext,
    store: mx.MetadataStore,
    specs: Sequence[dg.AssetSpec],
) -> Iterator[dg.MaterializeResult[None]]:  # pyright: ignore[reportMissingTypeArgument]
    """Generate [dagster.MaterializeResult][] events for assets in topological order.

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
            yield from generate_materialization_events(context, store, specs)
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
        with store:
            row_count = store.read_metadata(key).select(nw.len()).collect().item(0, 0)

        yield dg.MaterializeResult(
            value=None,
            asset_key=asset_spec.key,
            metadata={"dagster/row_count": row_count},
        )


def generate_observation_events(
    context: dg.AssetExecutionContext,
    store: mx.MetadataStore,
    specs: Sequence[dg.AssetSpec],
) -> Iterator[dg.ObserveResult]:
    """Generate [dagster.ObserveResult][] events for assets in topological order.

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
        @dg.multi_asset(specs=specs)
        def my_observable_assets(context: dg.AssetExecutionContext, store: mx.MetadataStore):
            yield from generate_observation_events(context, store, specs)
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
        with store:
            row_count = store.read_metadata(key).select(nw.len()).collect().item(0, 0)

        yield dg.ObserveResult(
            asset_key=asset_spec.key,
            metadata={"dagster/row_count": row_count},
        )
