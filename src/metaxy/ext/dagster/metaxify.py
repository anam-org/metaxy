from typing import Any, TypeVar, overload

import dagster as dg
from dagster._core.definitions.events import (
    CoercibleToAssetKey,
    CoercibleToAssetKeyPrefix,
)
from typing_extensions import Self

import metaxy as mx
from metaxy._decorators import public
from metaxy.ext.dagster.constants import (
    DAGSTER_COLUMN_LINEAGE_METADATA_KEY,
    DAGSTER_COLUMN_SCHEMA_METADATA_KEY,
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_INFO_METADATA_KEY,
    DAGSTER_METAXY_KIND,
    DAGSTER_METAXY_PROJECT_TAG_KEY,
    METAXY_DAGSTER_METADATA_KEY,
)
from metaxy.ext.dagster.table_metadata import (
    build_column_lineage,
    build_column_schema,
)
from metaxy.ext.dagster.utils import (
    build_feature_info_metadata,
    get_asset_key_for_metaxy_feature_spec,
)

_T = TypeVar("_T", dg.AssetsDefinition, dg.AssetSpec)


@public
class metaxify:
    """Inject Metaxy metadata into a Dagster [`AssetsDefinition`][dg.AssetsDefinition] or [`AssetSpec`][dg.AssetSpec].

    Affects assets with `metaxy/feature` metadata set.

    Learn more about `@metaxify` and see example screenshots [here](metaxify.md).

    Args:
        key: Explicit asset key that overrides all other key resolution logic. Cannot be used
            with `key_prefix` or with multi-asset definitions that produce multiple outputs.
        key_prefix: Prefix to prepend to the resolved asset key. Also applied to upstream
            dependency keys. Cannot be used with `key`.
        inject_metaxy_kind: Whether to inject `"metaxy"` kind into asset kinds.
        inject_code_version: Whether to inject the Metaxy feature code version into the asset's
            code version. The version is appended in the format `metaxy:<version>`.
        set_description: Whether to set the asset description from the feature class docstring
            if the asset doesn't already have a description.
        inject_column_schema: Whether to inject Pydantic field definitions as Dagster column schema.
            Field types are converted to strings, and field descriptions are used as column descriptions.
        inject_column_lineage: Whether to inject column-level lineage into the asset metadata under
            `dagster/column_lineage`. Uses Pydantic model fields to track
            column provenance via `FeatureDep.rename`, `FeatureDep.lineage`, and direct pass-through.

    !!! tip
        Multiple Dagster assets can contribute to the same Metaxy feature.
        This is a perfectly valid setup since Metaxy writes are append-only. In order to do this, set the following metadata keys:

            - `"metaxy/feature"` pointing to the same Metaxy feature key
            - `"metaxy/partition"` should be set to a dictionary mapping column names to values produced by the specific Dagster asset

    !!! example
        ```py  {hl_lines="8"}
        import dagster as dg
        import metaxy as mx
        import metaxy.ext.dagster as mxd


        @mxd.metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "my/feature/key"},
        )
        def my_asset(store: mx.MetadataStore):
            with store:
                increment = store.resolve_update("my/feature/key")
            ...
        ```

    ??? example "With `@multi_asset`"
        Multiple Metaxy features can be produced by the same `@multi_asset`. (1)
        { .annotate }

        1. Typically, they are produced independently of each other

        ```python
        @mxd.metaxify()
        @dg.multi_asset(
            specs=[
                dg.AssetSpec("output_a", metadata={"metaxy/feature": "feature/a"}),
                dg.AssetSpec("output_b", metadata={"metaxy/feature": "feature/b"}),
            ]
        )
        def my_multi_asset(): ...
        ```

    ??? example "With `dagster.AssetSpec`"
        ```py
        asset_spec = dg.AssetSpec(
            key="my_asset",
            metadata={"metaxy/feature": "my/feature/key"},
        )
        asset_spec = mxd.metaxify()(asset_spec)
        ```

    ??? example "Multiple Dagster assets contributing to the same Metaxy feature"
        ```py
        @dg.asset(
            metadata={
                "metaxy/feature": "my/feature/key",
                "metaxy/partition": {"dataset": "a"},
            },
        )
        def my_feature_dataset_a(): ...


        @dg.asset(
            metadata={
                "metaxy/feature": "my/feature/key",
                "metaxy/partition": {"dataset": "b"},
            },
        )
        def my_feature_dataset_b(): ...
        ```
    """

    key: dg.AssetKey | None
    key_prefix: dg.AssetKey | None
    inject_metaxy_kind: bool
    inject_code_version: bool
    set_description: bool
    inject_column_schema: bool
    inject_column_lineage: bool

    def __init__(
        self,
        _asset: "_T | None" = None,
        *,
        key: CoercibleToAssetKey | None = None,
        key_prefix: CoercibleToAssetKeyPrefix | None = None,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
        set_description: bool = True,
        inject_column_schema: bool = True,
        inject_column_lineage: bool = True,
    ) -> None:
        # Actual initialization happens in __new__, but we set defaults here for type checkers
        self.key = dg.AssetKey.from_coercible(key) if key is not None else None
        self.key_prefix = dg.AssetKey.from_coercible(key_prefix) if key_prefix is not None else None
        self.inject_metaxy_kind = inject_metaxy_kind
        self.inject_code_version = inject_code_version
        self.set_description = set_description
        self.inject_column_schema = inject_column_schema
        self.inject_column_lineage = inject_column_lineage

    @overload
    def __new__(cls, _asset: _T) -> _T: ...

    @overload
    def __new__(
        cls,
        _asset: None = None,
        *,
        key: CoercibleToAssetKey | None = None,
        key_prefix: CoercibleToAssetKeyPrefix | None = None,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
        set_description: bool = True,
        inject_column_schema: bool = True,
        inject_column_lineage: bool = True,
    ) -> Self: ...

    def __new__(
        cls,
        _asset: _T | None = None,
        *,
        key: CoercibleToAssetKey | None = None,
        key_prefix: CoercibleToAssetKeyPrefix | None = None,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
        set_description: bool = True,
        inject_column_schema: bool = True,
        inject_column_lineage: bool = True,
    ) -> "Self | _T":
        if key is not None and key_prefix is not None:
            raise ValueError("Cannot specify both `key` and `key_prefix`")

        coerced_key = dg.AssetKey.from_coercible(key) if key is not None else None
        coerced_key_prefix = dg.AssetKey.from_coercible(key_prefix) if key_prefix is not None else None

        if _asset is not None:
            # Called as @metaxify without parentheses
            return cls._transform(
                _asset,
                key=coerced_key,
                key_prefix=coerced_key_prefix,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
                set_description=set_description,
                inject_column_schema=inject_column_schema,
                inject_column_lineage=inject_column_lineage,
            )

        # Called as @metaxify() with parentheses - return instance for __call__
        instance = object.__new__(cls)
        instance.key = coerced_key
        instance.key_prefix = coerced_key_prefix
        instance.inject_metaxy_kind = inject_metaxy_kind
        instance.inject_code_version = inject_code_version
        instance.set_description = set_description
        instance.inject_column_schema = inject_column_schema
        instance.inject_column_lineage = inject_column_lineage
        return instance

    def __call__(self, asset: _T) -> _T:
        return self._transform(
            asset,  # ty: ignore[invalid-argument-type]
            key=self.key,
            key_prefix=self.key_prefix,
            inject_metaxy_kind=self.inject_metaxy_kind,
            inject_code_version=self.inject_code_version,
            set_description=self.set_description,
            inject_column_schema=self.inject_column_schema,
            inject_column_lineage=self.inject_column_lineage,
        )

    @staticmethod
    def _transform(
        asset: _T,
        *,
        key: dg.AssetKey | None,
        key_prefix: dg.AssetKey | None,
        inject_metaxy_kind: bool,
        inject_code_version: bool,
        set_description: bool,
        inject_column_schema: bool,
        inject_column_lineage: bool,
    ) -> _T:
        """Transform an AssetsDefinition or AssetSpec with Metaxy metadata."""
        if isinstance(asset, dg.AssetSpec):
            return _metaxify_spec(  # ty: ignore[invalid-return-type]
                asset,
                key=key,
                key_prefix=key_prefix,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
                set_description=set_description,
                inject_column_schema=inject_column_schema,
                inject_column_lineage=inject_column_lineage,
            )

        # Handle AssetsDefinition
        # Validate that key argument is not used with multi-asset
        if key is not None and len(asset.keys) > 1:
            raise ValueError(
                f"Cannot use `key` argument with multi-asset `{asset.node_def.name}` "
                f"that produces {len(asset.keys)} outputs. "
                f"Use `key_prefix` instead to apply a common prefix to all outputs."
            )

        keys_to_replace: dict[dg.AssetKey, dg.AssetKey] = {}
        transformed_specs: list[dg.AssetSpec] = []

        for orig_key, asset_spec in asset.specs_by_key.items():
            new_spec = _metaxify_spec(
                asset_spec,
                key=key,
                key_prefix=key_prefix,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
                set_description=set_description,
                inject_column_schema=inject_column_schema,
                inject_column_lineage=inject_column_lineage,
            )
            if new_spec.key != orig_key:
                keys_to_replace[orig_key] = new_spec.key
            transformed_specs.append(new_spec)

        return _replace_specs_on_assets_definition(  # ty: ignore[invalid-return-type]
            asset, transformed_specs, keys_to_replace
        )


def _replace_specs_on_assets_definition(
    asset: dg.AssetsDefinition,
    new_specs: list[dg.AssetSpec],
    keys_to_replace: dict[dg.AssetKey, dg.AssetKey],
) -> dg.AssetsDefinition:
    """Replace specs on an AssetsDefinition without triggering Dagster's InputDefinition bug.

    Dagster's `map_asset_specs` and `replace_specs_on_asset` have a bug where they fail
    on assets with input definitions (from `ins=` parameter with `dg.AssetIn` objects).
    The bug occurs because `OpDefinition.with_replaced_properties` creates an `ins` dict
    mixing `InputDefinition` objects with `In` objects, and then `OpDefinition.__init__`
    tries to call `to_definition()` on `InputDefinition` objects which don't have that method.

    This function works around the bug by using `dagster_internal_init` directly,
    which only updates the specs without modifying the underlying node_def.
    This means new deps added to specs won't be reflected as actual inputs to the op,
    but they will be tracked correctly by Dagster's asset graph for dependency purposes.

    Args:
        asset: The original AssetsDefinition to transform.
        new_specs: The transformed specs to use.
        keys_to_replace: A mapping of old keys to new keys for assets whose keys changed.

    Returns:
        A new AssetsDefinition with the transformed specs.
    """
    # Get the current attributes from the asset
    attrs = asset.get_attributes_dict()

    # Update the specs
    attrs["specs"] = new_specs

    # If there are key replacements, also update keys_by_output_name and selected_asset_keys
    if keys_to_replace:
        attrs["keys_by_output_name"] = {
            output_name: keys_to_replace.get(key, key) for output_name, key in attrs["keys_by_output_name"].items()
        }
        attrs["selected_asset_keys"] = {keys_to_replace.get(key, key) for key in attrs["selected_asset_keys"]}

    # Create a new AssetsDefinition with the updated attributes
    # This bypasses the buggy code path in Dagster's replace_specs_on_asset
    result = asset.__class__.dagster_internal_init(**attrs)

    # Use with_attributes to update check specs - Dagster handles this automatically
    # when asset_key_replacements is provided
    if keys_to_replace:
        result = result.with_attributes(asset_key_replacements=keys_to_replace)

    return result


def _metaxify_spec(
    spec: dg.AssetSpec,
    *,
    key: dg.AssetKey | None,
    key_prefix: dg.AssetKey | None,
    inject_metaxy_kind: bool,
    inject_code_version: bool,
    set_description: bool,
    inject_column_schema: bool,
    inject_column_lineage: bool,
) -> dg.AssetSpec:
    """Transform a single AssetSpec with Metaxy metadata.

    Returns the spec unchanged if `metaxy/feature` metadata is not set,
    unless `key_prefix` is provided (which applies to all specs).
    """
    metadata_feature_key = spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY)

    # Feature key must come from metadata
    if metadata_feature_key is None:
        # No feature key set - but still apply key_prefix if provided
        if key_prefix is not None:
            new_key = dg.AssetKey([*key_prefix.path, *spec.key.path])
            return spec.replace_attributes(key=new_key)
        return spec

    feature_key = mx.coerce_to_feature_key(metadata_feature_key)
    feature_def = mx.get_feature_by_key(feature_key)
    feature_spec = feature_def.spec

    # Determine the final asset key
    # Priority: key > key_prefix + resolved_key > resolved_key
    if key is not None:
        # Explicit key overrides everything
        final_key = key
    else:
        # Resolve key from feature spec
        resolved_key = get_asset_key_for_metaxy_feature_spec(feature_spec)
        if key_prefix is not None:
            # Prepend prefix to resolved key
            final_key = dg.AssetKey([*key_prefix.path, *resolved_key.path])
        else:
            final_key = resolved_key

    # Build deps from feature dependencies
    metaxy_deps_by_key: dict[dg.AssetKey, dg.AssetDep] = {}
    for dep in feature_spec.deps:
        upstream_feature_def = mx.get_feature_by_key(dep.feature)
        upstream_key = get_asset_key_for_metaxy_feature_spec(upstream_feature_def.spec)
        # Apply key_prefix to upstream deps as well
        if key_prefix is not None:
            upstream_key = dg.AssetKey([*key_prefix.path, *upstream_key.path])
        metaxy_deps_by_key[upstream_key] = dg.AssetDep(asset=upstream_key)

    # Merge: user-specified deps (spec.deps) take precedence over metaxy-generated deps
    # This allows users to override with custom metadata or partition_mapping
    deps_by_key = {**metaxy_deps_by_key}
    for dep in spec.deps:
        deps_by_key[dep.asset_key] = dep

    # Build kinds
    kinds_to_add: set[str] = set()
    if inject_metaxy_kind:
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

    # Use feature schema description if not set on asset spec
    final_description = spec.description
    if set_description and final_description is None:
        schema_desc = feature_def.feature_schema.get("description")
        if schema_desc:
            final_description = schema_desc

    # Build tags for project and feature
    # Note: Dagster tag values only allow alpha-numeric, '_', '-', '.'
    # so we use table_name which uses '__' separator
    tags_to_add: dict[str, str] = {
        DAGSTER_METAXY_PROJECT_TAG_KEY: feature_def.project,
        DAGSTER_METAXY_FEATURE_METADATA_KEY: feature_key.table_name,
    }

    # Build column schema from Pydantic fields (includes inherited system columns)
    # Respects existing user-defined column schema and appends Metaxy columns
    column_schema: dg.TableSchema | None = None
    if inject_column_schema:
        # Start with user-defined columns if present
        existing_schema = spec.metadata.get(DAGSTER_COLUMN_SCHEMA_METADATA_KEY)
        existing_columns: list[dg.TableColumn] = []
        existing_column_names: set[str] = set()
        if existing_schema is not None:
            existing_columns = list(existing_schema.columns)
            existing_column_names = {col.name for col in existing_columns}

        # Add Metaxy columns that aren't already defined by user
        # (user-defined columns take precedence)
        # Skip for external features (no Python class to extract schema from)
        metaxy_columns: list[dg.TableColumn] = []
        if not feature_def.is_external:
            metaxy_schema = build_column_schema(feature_def)
            metaxy_columns = [col for col in metaxy_schema.columns if col.name not in existing_column_names]

        all_columns = existing_columns + metaxy_columns
        if all_columns:
            # Sort columns alphabetically by name
            all_columns.sort(key=lambda col: col.name)
            column_schema = dg.TableSchema(columns=all_columns)

    # Build column lineage from upstream dependencies
    # Respects existing user-defined column lineage and merges with Metaxy lineage
    # Skip for external features (no Python class to extract columns from)
    column_lineage: dg.TableColumnLineage | None = None
    if inject_column_lineage and feature_spec.deps and not feature_def.is_external:
        # Start with user-defined lineage if present
        existing_lineage = spec.metadata.get(DAGSTER_COLUMN_LINEAGE_METADATA_KEY)
        existing_deps_by_column: dict[str, list[dg.TableColumnDep]] = {}
        if existing_lineage is not None:
            existing_deps_by_column = dict(existing_lineage.deps_by_column)

        metaxy_lineage = build_column_lineage(
            feature=feature_def,
            feature_spec=feature_spec,
        )

        if metaxy_lineage is not None:
            # Merge: user-defined lineage takes precedence for same columns
            merged_deps_by_column: dict[str, list[dg.TableColumnDep]] = {
                col: list(deps) for col, deps in metaxy_lineage.deps_by_column.items()
            }
            for col, deps in existing_deps_by_column.items():
                if col in merged_deps_by_column:
                    # Append user deps to metaxy deps (user can add extra lineage)
                    merged_deps_by_column[col] = merged_deps_by_column[col] + deps
                else:
                    merged_deps_by_column[col] = deps
            # Sort columns alphabetically
            sorted_deps = {k: merged_deps_by_column[k] for k in sorted(merged_deps_by_column)}
            column_lineage = dg.TableColumnLineage(deps_by_column=sorted_deps)
        elif existing_deps_by_column:
            # Sort columns alphabetically
            sorted_deps = {k: existing_deps_by_column[k] for k in sorted(existing_deps_by_column)}
            column_lineage = dg.TableColumnLineage(deps_by_column=sorted_deps)

    # Build the replacement attributes
    metadata_to_add: dict[str, Any] = {
        **spec.metadata,
        DAGSTER_METAXY_FEATURE_METADATA_KEY: feature_key.to_string(),
        DAGSTER_METAXY_INFO_METADATA_KEY: build_feature_info_metadata(feature_key),
    }
    if column_schema is not None:
        metadata_to_add[DAGSTER_COLUMN_SCHEMA_METADATA_KEY] = column_schema
    if column_lineage is not None:
        metadata_to_add[DAGSTER_COLUMN_LINEAGE_METADATA_KEY] = column_lineage

    replace_attrs: dict[str, Any] = {
        "key": final_key,
        "deps": list(deps_by_key.values()),
        "metadata": metadata_to_add,
        "kinds": {*spec.kinds, *kinds_to_add},
        "tags": {**spec.tags, **tags_to_add},
        **dagster_attrs,
    }

    if final_code_version is not None:
        replace_attrs["code_version"] = final_code_version

    if final_description is not None:
        replace_attrs["description"] = final_description

    return spec.replace_attributes(**replace_attrs)
