from collections import defaultdict
from collections.abc import Callable
from typing import Any

import dagster as dg

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
)
from metaxy.ext.dagster.utils import get_asset_key_for_metaxy_feature_spec


def metaxify(
    inherit_feature_key_as_asset_key: bool = False, inject_metaxy_kind: bool = True
) -> Callable[[dg.AssetsDefinition], dg.AssetsDefinition]:
    """Inject Metaxy metadata into a Dagster [`AssetsDefinition`][dg.AssetsDefinition].

    Modifies assets that have `metaxy/feature` metadata set.

    Args:
        inherit_feature_key_as_asset_key: If True, use the Metaxy feature key as the
            Dagster asset key (unless `metaxy/metadata` is set on the feature spec).
        inject_metaxy_kind: Whether to inject `"metaxy"` kind into asset kinds.
            Currently, kinds count is limited by 3, and `metaxify` will skip kind injection
            if there are already 3 kinds on the asset.

    Returns:
        The original Dagster asset, enriched with Metaxy:

            - The asset key is determined as follows:

                1. If the Metaxy feature spec has `"metaxy/metadata"` set, that value is used as-is.

                2. Otherwise, if `inherit_feature_key_as_asset_key` is True, the Metaxy feature key is used.

                3. Otherwise, the original Dagster asset key is kept as-is.

            - Asset keys for upstream Metaxy features are injected into `deps`. The dep keys follow the same logic.

            - Feature spec metadata is injected into the Dagster asset metadata.

            - `"metaxy"` kind is injected into asset kinds if `inject_metaxy_kind` is `True` and there are less than 3 kinds on the asset.

            In the future, `@metaxify` will also inject table schemas and column lineage Dagster metadata.

    !!! note "Multiple assets per feature"

        Multiple Dagster assets can contribute to the same Metaxy feature by setting the same
        `"metaxy/feature"` metadata. This is a perfectly valid setup since Metaxy operations are append-only.

    !!! example

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

        assert "metaxy/metadata" in my_asset.metadata
        ```
    """

    def inner(asset: dg.AssetsDefinition) -> dg.AssetsDefinition:
        keys_to_replace: dict[dg.AssetKey, dg.AssetKey] = {}
        deps_to_inject: defaultdict[dg.AssetKey, set[dg.AssetDep]] = defaultdict(set)
        metadata_to_inject: dict[dg.AssetKey, dict[str, Any]] = {}
        kinds_to_inject: defaultdict[dg.AssetKey, set[str]] = defaultdict(set)

        for key, asset_spec in asset.specs_by_key.items():
            if feature_key := asset.metadata_by_key[key].get(
                DAGSTER_METAXY_FEATURE_METADATA_KEY
            ):
                feature_cls = mx.get_feature_by_key(feature_key)
                feature_spec = feature_cls.spec()

                # Determine the final asset key using the utility function
                final_key = get_asset_key_for_metaxy_feature_spec(
                    feature_spec,
                    inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                    dagster_key=key,
                )

                if final_key != key:
                    keys_to_replace[key] = final_key

                # Inject deps using feature keys
                for dep in feature_spec.deps:
                    upstream_feature_spec = mx.get_feature_by_key(dep.feature).spec()

                    deps_to_inject[final_key].add(
                        dg.AssetDep(
                            asset=get_asset_key_for_metaxy_feature_spec(
                                upstream_feature_spec,
                                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                            )
                        )
                    )

                metadata_to_inject[final_key] = feature_cls.spec().metadata

                if inject_metaxy_kind and len(asset_spec.kinds) < 3:
                    kinds_to_inject[final_key].add(DAGSTER_METAXY_KIND)

        return asset.with_attributes(
            asset_key_replacements=keys_to_replace
        ).map_asset_specs(
            lambda spec: spec.replace_attributes(
                deps={*spec.deps, *deps_to_inject[spec.key]},
                metadata={**spec.metadata, **metadata_to_inject.get(spec.key, {})},
                kinds={*spec.kinds, *kinds_to_inject[spec.key]},
            )
        )

    return inner
