import dagster as dg

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
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


def build_asset_spec(
    feature: mx.CoercibleToFeatureKey,
    *,
    inherit_feature_key_as_asset_key: bool = True,
    include_kind: bool = True,
) -> dg.AssetSpec:
    """Build a Dagster AssetSpec from a Metaxy feature.

    This creates an AssetSpec with the correct asset key, metadata, dependencies,
    and kinds based on the Metaxy feature definition.

    Args:
        feature: A Metaxy feature key (string, list, or feature class).
        inherit_feature_key_as_asset_key: If True, use the feature key as the asset key
            (unless `dagster/attributes.asset_key` is set on the feature spec).
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
        feature_spec, inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key
    )

    # Build deps from feature dependencies
    deps: set[dg.AssetDep] = set()
    for dep in feature_spec.deps:
        upstream_spec = mx.get_feature_by_key(dep.feature).spec()
        deps.add(
            dg.AssetDep(
                asset=get_asset_key_for_metaxy_feature_spec(
                    upstream_spec,
                    inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
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
