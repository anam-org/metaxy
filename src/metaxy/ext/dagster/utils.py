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
) -> dg.AssetKey:
    """Get the Dagster asset key for a Metaxy feature spec.

    Args:
        feature_spec: The Metaxy feature spec.
        key_prefix: Optional prefix to prepend to the asset key.

    Returns:
        The Dagster asset key.
    """
    if metaxy_defined_dagster_asset_key := feature_spec.metadata.get(
        DAGSTER_METAXY_METADATA_METADATA_KEY
    ):
        return dg.AssetKey(metaxy_defined_dagster_asset_key)  # pyright: ignore[reportArgumentType]
    else:
        return dg.AssetKey([*(key_prefix or []), *feature_spec.key.parts])


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
