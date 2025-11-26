import dagster as dg

import metaxy as mx
from metaxy.ext.dagster.constants import METAXY_DAGSTER_METADATA_KEY


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
