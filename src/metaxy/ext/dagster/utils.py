from collections.abc import Sequence

import dagster as dg

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
    DAGSTER_METAXY_METADATA_METADATA_KEY,
)


def get_asset_key_for_metaxy_feature_spec(
    feature_spec: mx.FeatureSpec,
    key_prefix: Sequence[str] | None = None,
    dagster_key: dg.AssetKey | None = None,
) -> dg.AssetKey:
    """Get the Dagster asset key for a Metaxy feature spec.

    Args:
        feature_spec: The Metaxy feature spec.
        key_prefix: Optional prefix to prepend to the asset key (only applied when
            `metaxy/metadata` is not set on the feature spec).
        dagster_key: Optional existing Dagster asset key to use as the base key
            (instead of the feature key). Used by `metaxify` to preserve original
            asset keys.

    Returns:
        The Dagster asset key, determined as follows:

        1. If feature spec has `metaxy/metadata` set, that value is used as-is
           (no prefix applied).
        2. Otherwise, if `dagster_key` is provided, it's used with `key_prefix`
           prepended if provided.
        3. Otherwise, the feature key is used with `key_prefix` prepended if provided.
    """
    # If metaxy/metadata is set, use it as-is (no prefix)
    if metaxy_defined_dagster_asset_key := feature_spec.metadata.get(
        DAGSTER_METAXY_METADATA_METADATA_KEY
    ):
        return dg.AssetKey(metaxy_defined_dagster_asset_key)  # pyright: ignore[reportArgumentType]

    # Determine base key
    if dagster_key is not None:
        base_key_parts = list(dagster_key.path)
    else:
        base_key_parts = list(feature_spec.key.parts)

    # Apply prefix if provided
    if key_prefix:
        return dg.AssetKey([*key_prefix, *base_key_parts])
    else:
        return dg.AssetKey(base_key_parts)


def build_asset_spec(
    feature: mx.CoercibleToFeatureKey,
    *,
    key_prefix: Sequence[str] | None = None,
    include_kind: bool = True,
) -> dg.AssetSpec:
    """Build a Dagster AssetSpec from a Metaxy feature.

    This creates an AssetSpec with the correct asset key, metadata, dependencies,
    and kinds based on the Metaxy feature definition.

    Args:
        feature: A Metaxy feature key (string, list, or feature class).
        key_prefix: Optional prefix to prepend to asset keys.
        include_kind: Whether to include the "metaxy" kind.

    Returns:
        A Dagster AssetSpec configured for the Metaxy feature.

    Example:
        ```python
        import dagster as dg
        import metaxy.ext.dagster as mxd

        # Build specs for Metaxy features
        embeddings_spec = mxd.build_asset_spec("audio/embeddings")

        @dg.asset(
            ins={"data": dg.AssetIn(key=embeddings_spec.key)}
        )
        def my_asset(data):
            ...
        ```
    """
    feature_cls = mx.get_feature_by_key(feature)
    feature_spec = feature_cls.spec()

    asset_key = get_asset_key_for_metaxy_feature_spec(
        feature_spec, key_prefix=key_prefix
    )

    # Build deps from feature dependencies
    deps: set[dg.AssetDep] = set()
    for dep in feature_spec.deps:
        upstream_spec = mx.get_feature_by_key(dep.feature).spec()
        deps.add(
            dg.AssetDep(
                asset=get_asset_key_for_metaxy_feature_spec(
                    upstream_spec, key_prefix=key_prefix
                )
            )
        )

    # Build kinds
    kinds: set[str] = set()
    if include_kind:
        kinds.add(DAGSTER_METAXY_KIND)

    return dg.AssetSpec(
        key=asset_key,
        deps=deps,
        metadata={
            DAGSTER_METAXY_FEATURE_METADATA_KEY: feature_spec.key.to_string(),
            **feature_spec.metadata,
        },
        kinds=kinds,
    )
