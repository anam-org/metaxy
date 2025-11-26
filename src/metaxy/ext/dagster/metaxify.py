from typing import Any, TypeVar, overload

import dagster as dg
from typing_extensions import Self

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
    DAGSTER_METAXY_METADATA_METADATA_KEY,
    METAXY_DAGSTER_METADATA_KEY,
)
from metaxy.ext.dagster.utils import get_asset_key_for_metaxy_feature_spec

_T = TypeVar("_T", dg.AssetsDefinition, dg.AssetSpec)


class metaxify:
    """Inject Metaxy metadata into a Dagster [`AssetsDefinition`][dg.AssetsDefinition] or [`AssetSpec`][dg.AssetSpec].

    Modifies assets that have `metaxy/feature` metadata set or when `feature` argument is provided.
    Can be used with or without parentheses.

    The decorated asset or spec is enriched with Metaxy:

    - **Asset key** is determined as follows:

        1. If the Metaxy feature spec has `"dagster/attributes": {"asset_key": ...}` set,
           that value is used as-is.

        2. Otherwise, if `inherit_feature_key_as_asset_key` is True, the Metaxy feature key is used.

        3. Otherwise, the original Dagster asset key is kept as-is.

    - **Dependencies** for upstream Metaxy features are injected into `deps`. The dep keys follow the same logic.

    - **Metadata** from the feature spec is injected into the Dagster asset metadata.

    - **Dagster attributes** from `"dagster/attributes"` in the feature spec metadata
      (such as `group_name`, `owners`, `tags`) are applied to the asset spec (with replacement).

    - **Kind** `"metaxy"` is injected into asset kinds if `inject_metaxy_kind` is `True` and there are less than 3 kinds.

    - **Code version** from the feature spec is appended to the asset's code version in the format `metaxy:<version>`.

    In the future, `@metaxify` will also inject table schemas and column lineage Dagster metadata.

    Args:
        feature: The Metaxy feature to associate with the asset. If provided, this takes precedence
            over `metaxy/feature` metadata. If both are set and don't match, an error is raised.
        inherit_feature_key_as_asset_key: If True, use the Metaxy feature key as the
            Dagster asset key (unless `dagster/attributes.asset_key` is set on the feature spec).
        inject_metaxy_kind: Whether to inject `"metaxy"` kind into asset kinds.
            Currently, kinds count is limited by 3, and `metaxify` will skip kind injection
            if there are already 3 kinds on the asset.
        inject_code_version: Whether to inject the Metaxy feature code version into the asset's
            code version. The version is appended in the format `metaxy:<version>`.

    !!! note
        Multiple Dagster assets can contribute to the same Metaxy feature by setting the same
        `"metaxy/feature"` metadata. This is a perfectly valid setup since Metaxy writes are append-only.

    !!! example "Apply to `dagster.AssetDefinition`"
        ```py
        import dagster as dg
        import metaxy.ext.dagster as mxd
        from myproject.features import MyFeature

        @mxd.metaxify(feature=MyFeature)
        @dg.asset
        def my_asset():
            ...
        ```

    ??? example "Apply to `dagster.AssetSpec`"
        ```py
        import dagster as dg
        import metaxy.ext.dagster as mxd

        asset_spec = dg.AssetSpec(key="my_asset")
        asset_spec = mxd.metaxify(feature=MyFeature)(asset_spec)
        ```

    ??? example "Use `"metaxy/feature"` asset metadata key"
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
        ```
    """

    feature: mx.FeatureKey | None
    inherit_feature_key_as_asset_key: bool
    inject_metaxy_kind: bool
    inject_code_version: bool

    def __init__(
        self,
        _asset: "_T | None" = None,
        *,
        feature: mx.CoercibleToFeatureKey | None = None,
        inherit_feature_key_as_asset_key: bool = False,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
    ) -> None:
        # Actual initialization happens in __new__, but we set defaults here for type checkers
        self.feature = (
            mx.coerce_to_feature_key(feature) if feature is not None else None
        )
        self.inherit_feature_key_as_asset_key = inherit_feature_key_as_asset_key
        self.inject_metaxy_kind = inject_metaxy_kind
        self.inject_code_version = inject_code_version

    @overload
    def __new__(cls, _asset: _T) -> _T: ...

    @overload
    def __new__(
        cls,
        _asset: None = None,
        *,
        feature: mx.CoercibleToFeatureKey | None = None,
        inherit_feature_key_as_asset_key: bool = False,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
    ) -> Self: ...

    def __new__(
        cls,
        _asset: _T | None = None,
        *,
        feature: mx.CoercibleToFeatureKey | None = None,
        inherit_feature_key_as_asset_key: bool = False,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
    ) -> "Self | _T":
        coerced_feature = (
            mx.coerce_to_feature_key(feature) if feature is not None else None
        )
        if _asset is not None:
            # Called as @metaxify without parentheses
            return cls._transform(
                _asset,
                feature=coerced_feature,
                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
            )

        # Called as @metaxify() with parentheses - return instance for __call__
        instance = object.__new__(cls)
        instance.feature = coerced_feature
        instance.inherit_feature_key_as_asset_key = inherit_feature_key_as_asset_key
        instance.inject_metaxy_kind = inject_metaxy_kind
        instance.inject_code_version = inject_code_version
        return instance

    def __call__(self, asset: _T) -> _T:
        """Transform the asset when used as @metaxify() with parentheses."""
        return self._transform(
            asset,
            feature=self.feature,
            inherit_feature_key_as_asset_key=self.inherit_feature_key_as_asset_key,
            inject_metaxy_kind=self.inject_metaxy_kind,
            inject_code_version=self.inject_code_version,
        )

    @staticmethod
    def _transform(
        asset: _T,
        *,
        feature: mx.FeatureKey | None,
        inherit_feature_key_as_asset_key: bool,
        inject_metaxy_kind: bool,
        inject_code_version: bool,
    ) -> _T:
        """Transform an AssetsDefinition or AssetSpec with Metaxy metadata."""
        if isinstance(asset, dg.AssetSpec):
            return _metaxify_spec(
                asset,
                feature=feature,
                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
            )

        # Handle AssetsDefinition
        keys_to_replace: dict[dg.AssetKey, dg.AssetKey] = {}
        transformed_specs: dict[dg.AssetKey, dg.AssetSpec] = {}

        for key, asset_spec in asset.specs_by_key.items():
            new_spec = _metaxify_spec(
                asset_spec,
                feature=feature,
                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
            )
            if new_spec.key != key:
                keys_to_replace[key] = new_spec.key
            transformed_specs[new_spec.key] = new_spec

        return asset.with_attributes(
            asset_key_replacements=keys_to_replace
        ).map_asset_specs(lambda spec: transformed_specs.get(spec.key, spec))


def _metaxify_spec(
    spec: dg.AssetSpec,
    *,
    feature: mx.FeatureKey | None,
    inherit_feature_key_as_asset_key: bool,
    inject_metaxy_kind: bool,
    inject_code_version: bool,
) -> dg.AssetSpec:
    """Transform a single AssetSpec with Metaxy metadata.

    Returns the spec unchanged if neither `feature` argument nor `metaxy/feature` metadata is set.
    """
    metadata_feature_key = spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY)

    # Determine which feature key to use
    if feature is not None and metadata_feature_key is not None:
        # Both are set - verify they match
        metadata_coerced = mx.coerce_to_feature_key(metadata_feature_key)
        if feature != metadata_coerced:
            raise ValueError(
                f"Feature key mismatch for asset `{spec.key}`: "
                f"`feature` argument is `{feature}` but `metaxy/feature` metadata is `{metadata_coerced}`"
            )
        feature_key = feature
    elif feature is not None:
        feature_key = feature
    elif metadata_feature_key is not None:
        feature_key = mx.coerce_to_feature_key(metadata_feature_key)
    else:
        # Neither is set - return spec unchanged
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

    # Build code version: append metaxy version to existing code version if present
    if inject_code_version:
        metaxy_code_version = f"metaxy:{feature_spec.code_version}"
        if spec.code_version:
            final_code_version = f"{spec.code_version},{metaxy_code_version}"
        else:
            final_code_version = metaxy_code_version
    else:
        final_code_version = spec.code_version

    # Build the replacement attributes
    replace_attrs: dict[str, Any] = {
        "key": final_key,
        "deps": {*spec.deps, *deps_to_add},
        "metadata": {
            **spec.metadata,
            DAGSTER_METAXY_METADATA_METADATA_KEY: feature_spec.metadata,
        },
        "kinds": {*spec.kinds, *kinds_to_add},
        **dagster_attrs,
    }

    if final_code_version is not None:
        replace_attrs["code_version"] = final_code_version

    return spec.replace_attributes(**replace_attrs)
