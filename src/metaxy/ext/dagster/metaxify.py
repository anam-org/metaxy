from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import dagster as dg

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
)
from metaxy.ext.dagster.utils import get_asset_key_for_metaxy_feature_spec


def metaxify(
    key_prefix: Sequence[str] | None = None, inject_metaxy_kind: bool = True
) -> Callable[[dg.AssetsDefinition], dg.AssetsDefinition]:
    """Inject Metaxy metadata into a Dagster [`AssetsDefinition`][dg.AssetsDefinition].

    Modifies assets that have `metaxy/feature` metadata set.

    Args:
        key_prefix: Optional prefix to prepend to Metaxy feature keys when converting them into Dagster asset keys.
        inject_metaxy_kind: Whether to inject `"metaxy"` kind into asset kinds.
            Currently, kinds count is limited by 3, and `metaxify` will skip kind injection
            if there are already 3 kinds on the asset.

    Returns:
        The original Dagster asset, enriched with Metaxy:

            - asset keys for upstream Metaxy features are injected into `deps`. The asset keys are determined the following way:

                - If the Metaxy feature spec has a `"metaxy/dagster"` key set on its spec, then `asset_key` from this dictionary is used.

                - Otherwise, the Metaxy feature key is used, optionally prepended with `key_prefix`

            - feature spec metadata is injected into the Dagster asset metadata under `metaxy/metadata`

            - `"metaxy"` kind is injected into asset kinds if `inject_metaxy_kind` is `True` and there are less than 3 kinds on the asset.

    !!! example

        ```py
        import dagster as dg
        import metaxy as mx
        import metaxy.ext.dagster as mxd

        @mxd.metaxify
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

                new_key = get_asset_key_for_metaxy_feature_spec(
                    feature_spec, key_prefix=key_prefix
                )
                keys_to_replace[key] = new_key

                for dep in feature_spec.deps:
                    upstream_feature_spec = mx.get_feature_by_key(dep.feature).spec()

                    deps_to_inject[new_key].add(
                        dg.AssetDep(
                            asset=get_asset_key_for_metaxy_feature_spec(
                                upstream_feature_spec, key_prefix=key_prefix
                            )
                        )
                    )

                metadata_to_inject[new_key] = feature_cls.spec().metadata

                if inject_metaxy_kind and len(asset_spec.kinds) < 3:
                    kinds_to_inject[new_key].add(DAGSTER_METAXY_KIND)

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
