"""Observable source assets for Metaxy features."""

from collections.abc import Callable
from typing import Any

import dagster as dg

import metaxy as mx
from metaxy._decorators import public
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PARTITION_METADATA_KEY,
)
from metaxy.ext.dagster.metaxify import metaxify
from metaxy.ext.dagster.utils import (
    build_feature_event_tags,
    build_metaxy_partition_filter,
    compute_stats_from_lazy_frame,
)


@public
def observable_metaxy_asset(
    feature: mx.CoercibleToFeatureKey,
    *,
    store_resource_key: str = "store",
    # metaxify kwargs
    inject_metaxy_kind: bool = True,
    inject_code_version: bool = True,
    set_description: bool = True,
    # observable_source_asset kwargs
    **observable_kwargs: Any,
):
    """Decorator to create an observable source asset for a Metaxy feature.

    The observation reads the feature's metadata from the store, counts rows,
    and uses `mean(metaxy_created_at)` as the data version to track changes.
    Using mean ensures that both additions and deletions are detected.

    The decorated function receives `(context, store, lazy_df)` and can return
    a dict of additional metadata to include in the observation.

    Args:
        feature: The Metaxy feature to observe.
        store_resource_key: Resource key for the MetadataStore (default: `"store"`).
        inject_metaxy_kind: Whether to inject `"metaxy"` kind into asset kinds.
        inject_code_version: Whether to inject the Metaxy feature code version.
        set_description: Whether to set description from feature class docstring.
        **observable_kwargs: Passed to `@observable_source_asset`
            (key, group_name, tags, metadata, description, partitions_def, etc.)

    Example:
        ```python
        import metaxy.ext.dagster as mxd
        from myproject.features import ExternalFeature


        @mxd.observable_metaxy_asset(feature=ExternalFeature)
        def external_data(context, store, lazy_df):
            pass


        # With custom metadata - return a dict
        @mxd.observable_metaxy_asset(feature=ExternalFeature)
        def external_data_with_metrics(context, store, lazy_df):
            # Run aggregations in the database
            total = lazy_df.select(nw.col("value").sum()).collect().item(0, 0)
            return {"custom/total": total}
        ```

    Note:
        `observable_source_asset` does not support `deps`. Upstream Metaxy feature
        dependencies from the feature spec are not propagated to the SourceAsset.
    """
    feature_key = mx.coerce_to_feature_key(feature)

    def decorator(fn: Callable[..., Any]) -> dg.SourceAsset:
        # Build an AssetSpec from kwargs and enrich with metaxify
        # Merge user metadata with metaxy/feature
        user_metadata = observable_kwargs.pop("metadata", None) or {}
        spec = dg.AssetSpec(
            key=observable_kwargs.pop("key", None) or fn.__name__,  # ty: ignore[unresolved-attribute]
            group_name=observable_kwargs.pop("group_name", None),
            tags=observable_kwargs.pop("tags", None),
            metadata={
                **user_metadata,
                DAGSTER_METAXY_FEATURE_METADATA_KEY: feature_key.to_string(),
            },
            description=observable_kwargs.pop("description", None),
        )
        enriched = metaxify(
            inject_metaxy_kind=inject_metaxy_kind,
            inject_code_version=inject_code_version,
            set_description=set_description,
        )(spec)

        def _observe(context: dg.AssetExecutionContext) -> dg.ObserveResult:
            store: mx.MetadataStore = getattr(context.resources, store_resource_key)

            # Check for metaxy/partition metadata to apply filtering
            metaxy_partition = enriched.metadata.get(DAGSTER_METAXY_PARTITION_METADATA_KEY)
            filters = build_metaxy_partition_filter(metaxy_partition)

            with store:
                lazy_df = store.read_metadata(feature_key, filters=filters)
                stats = compute_stats_from_lazy_frame(lazy_df)

                # Call the user's function - it can return additional metadata
                extra_metadata = fn(context, store, lazy_df) or {}

            metadata: dict[str, Any] = {"dagster/row_count": stats.row_count}
            metadata.update(extra_metadata)

            return dg.ObserveResult(
                data_version=stats.data_version,
                metadata=metadata,
                tags=build_feature_event_tags(feature_key),
            )

        # Apply observable_source_asset decorator
        return dg.observable_source_asset(
            key=enriched.key,
            description=enriched.description,
            group_name=enriched.group_name,
            tags=dict(enriched.tags) if enriched.tags else None,
            metadata=dict(enriched.metadata) if enriched.metadata else None,
            required_resource_keys={store_resource_key},
            **observable_kwargs,
        )(_observe)

    return decorator
