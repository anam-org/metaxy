from collections.abc import Callable
from typing import Any, TypeVar

import dagster as dg

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
    METAXY_DAGSTER_METADATA_KEY,
)
from metaxy.ext.dagster.utils import get_asset_key_for_metaxy_feature_spec

_T = TypeVar("_T", dg.AssetsDefinition, dg.AssetSpec)


def _metaxify_spec(
    spec: dg.AssetSpec,
    *,
    inherit_feature_key_as_asset_key: bool,
    inject_metaxy_kind: bool,
) -> dg.AssetSpec:
    """Transform a single AssetSpec with Metaxy metadata.

    Returns the spec unchanged if it doesn't have metaxy/feature metadata.
    """
    feature_key = spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY)
    if not feature_key:
        return spec

    feature_cls = mx.get_feature_by_key(feature_key)
    feature_spec = feature_cls.spec()

    # Determine the final asset key
    final_key = get_asset_key_for_metaxy_feature_spec(
        feature_spec,
        inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
        dagster_key=spec.key,
    )

    # Build deps from feature dependencies
    deps_to_add: set[dg.AssetDep] = set()
    for dep in feature_spec.deps:
        upstream_feature_spec = mx.get_feature_by_key(dep.feature).spec()
        deps_to_add.add(
            dg.AssetDep(
                asset=get_asset_key_for_metaxy_feature_spec(
                    upstream_feature_spec,
                    inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                )
            )
        )

    # Build kinds
    kinds_to_add: set[str] = set()
    if inject_metaxy_kind and len(spec.kinds) < 3:
        kinds_to_add.add(DAGSTER_METAXY_KIND)

    # Extract dagster attributes (excluding asset_key which is handled separately)
    dagster_attrs: dict[str, Any] = {}
    raw_dagster_attrs = feature_spec.metadata.get(METAXY_DAGSTER_METADATA_KEY)
    if raw_dagster_attrs is not None:
        if not isinstance(raw_dagster_attrs, dict):
            raise ValueError(
                f"Invalid metadata format for `{feature_spec.key}` "
                f"Metaxy feature metadata key {METAXY_DAGSTER_METADATA_KEY}: "
                f"expected dict, got {type(raw_dagster_attrs).__name__}"
            )
        dagster_attrs = {k: v for k, v in raw_dagster_attrs.items() if k != "asset_key"}

    # Build the replacement attributes
    replace_attrs: dict[str, Any] = {
        "key": final_key,
        "deps": {*spec.deps, *deps_to_add},
        "metadata": {**spec.metadata, **feature_spec.metadata},
        "kinds": {*spec.kinds, *kinds_to_add},
        **dagster_attrs,
    }

    return spec.replace_attributes(**replace_attrs)


def metaxify(
    inherit_feature_key_as_asset_key: bool = False,
    inject_metaxy_kind: bool = True,
) -> Callable[[_T], _T]:
    """Inject Metaxy metadata into a Dagster [`AssetsDefinition`][dg.AssetsDefinition] or [`AssetSpec`][dg.AssetSpec].

    Modifies assets that have `metaxy/feature` metadata set.

    Args:
        inherit_feature_key_as_asset_key: If True, use the Metaxy feature key as the
            Dagster asset key (unless `dagster/attributes.asset_key` is set on the feature spec).
        inject_metaxy_kind: Whether to inject `"metaxy"` kind into asset kinds.
            Currently, kinds count is limited by 3, and `metaxify` will skip kind injection
            if there are already 3 kinds on the asset.

    Returns:
        The original Dagster asset or spec, enriched with Metaxy:

            - The asset key is determined as follows:

                1. If the Metaxy feature spec has `"dagster/attributes": {"asset_key": ...}` set,
                   that value is used as-is.

                2. Otherwise, if `inherit_feature_key_as_asset_key` is True, the Metaxy feature key is used.

                3. Otherwise, the original Dagster asset key is kept as-is.

            - Asset keys for upstream Metaxy features are injected into `deps`. The dep keys follow the same logic.

            - Feature spec metadata is injected into the Dagster asset metadata.

            - Additional Dagster attributes from `"dagster/attributes"` in the feature spec metadata
              (such as `group_name`, `owners`, `tags`) are applied to the asset spec (with replacement).

            - `"metaxy"` kind is injected into asset kinds if `inject_metaxy_kind` is `True` and there are less than 3 kinds on the asset.

            In the future, `@metaxify` will also inject table schemas and column lineage Dagster metadata.

    !!! note "Multiple assets per feature"

        Multiple Dagster assets can contribute to the same Metaxy feature by setting the same
        `"metaxy/feature"` metadata. This is a perfectly valid setup since Metaxy operations are append-only.

    !!! example "Apply to `AssetsDefinition`"

        ```py
        import dagster as dg
        import metaxy as mx
        import metaxy.ext.dagster as mxd

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "my/feature/key"})
        def my_asset(store: mx.MetadataStore):
            with store:
                increment = store.resolve_update("my/feature/key")

            ...
        ```

    !!! example "Apply to `AssetSpec`"
        ```py
        import dagster as dg
        import metaxy.ext.dagster as mxd

        asset_spec = dg.AssetSpec(
            key="my_asset",
            metadata={"metaxy/feature": "my/feature/key"},
        )
        spec = mxd.metaxify()(asset_spec)
        ```
    """

    def inner(asset: _T) -> _T:
        if isinstance(asset, dg.AssetSpec):
            return _metaxify_spec(
                asset,
                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                inject_metaxy_kind=inject_metaxy_kind,
            )

        # Handle AssetsDefinition
        keys_to_replace: dict[dg.AssetKey, dg.AssetKey] = {}
        transformed_specs: dict[dg.AssetKey, dg.AssetSpec] = {}

        for key, asset_spec in asset.specs_by_key.items():
            new_spec = _metaxify_spec(
                asset_spec,
                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                inject_metaxy_kind=inject_metaxy_kind,
            )
            if new_spec.key != key:
                keys_to_replace[key] = new_spec.key
            transformed_specs[new_spec.key] = new_spec

        return asset.with_attributes(
            asset_key_replacements=keys_to_replace
        ).map_asset_specs(lambda spec: transformed_specs.get(spec.key, spec))

    return inner
