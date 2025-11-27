import inspect
from typing import Any, TypeVar, get_origin, overload

import dagster as dg
from typing_extensions import Self

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_COLUMN_LINEAGE_METADATA_KEY,
    DAGSTER_COLUMN_SCHEMA_METADATA_KEY,
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
    DAGSTER_METAXY_METADATA_METADATA_KEY,
    DAGSTER_METAXY_PROJECT_TAG_KEY,
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

    - **Code version** from the feature spec is appended to the asset's code version in the format `metaxy:<version>`.

    - **Metadata** from the feature spec is injected into the Dagster asset metadata.

    - **Description** from the feature class docstring is used if the asset spec doesn't have a description set.

    - **Kind** `"metaxy"` is injected into asset kinds if `inject_metaxy_kind` is `True` and there are less than 3 kinds.

    - **Tags** `metaxy/feature` and `metaxy/project` are injected into the asset tags.

    - **Arbitrary asset attributes** from `"dagster/attributes"` in the feature spec metadata
      (such as `group_name`, `owners`, `tags`) are applied to the asset spec (with replacement).

    - **Column schema** from Pydantic fields is injected into the asset metadata under `dagster/column_schema`.
      Field types are converted to strings, and field descriptions are used as column descriptions.

    !!! warning
        Pydantic feature schema may not match the corresponding table schema in the metadata store.

    - **Column lineage** is injected into the asset metadata under `dagster/column_lineage`.
      Tracks which upstream columns each downstream column depends on by analyzing:

        - **Direct pass-through**: Columns with the same name in both upstream and downstream features.

        - **`FeatureDep.rename`**: Renamed columns trace back to their original upstream column names.

        - **`FeatureSpec.lineage`**: ID column relationships based on lineage type (identity, aggregation, expansion).

      Column lineage is derived from Pydantic model fields on the feature class.

    Args:
        feature: The Metaxy feature to associate with the asset. If provided, this takes precedence
            over `metaxy/feature` metadata. If both are set and don't match, an error is raised.
        inherit_feature_key_as_asset_key: If True (default), use the Metaxy feature key as the
            Dagster asset key (unless `dagster/attributes.asset_key` is set on the feature spec).
            This ensures consistent key resolution between assets and their upstream dependencies.
        inject_metaxy_kind: Whether to inject `"metaxy"` kind into asset kinds.
            Currently, kinds count is limited by 3, and `metaxify` will skip kind injection
            if there are already 3 kinds on the asset.
        inject_code_version: Whether to inject the Metaxy feature code version into the asset's
            code version. The version is appended in the format `metaxy:<version>`.
        set_description: Whether to set the asset description from the feature class docstring
            if the asset doesn't already have a description.
        inject_column_schema: Whether to inject Pydantic field definitions as Dagster column schema.
            Field types are converted to strings, and field descriptions are used as column descriptions.
        inject_column_lineage: Whether to inject column-level lineage into the asset metadata under
            `dagster/column_lineage`. Uses Pydantic model fields to track
            column provenance via `FeatureDep.rename`, `FeatureSpec.lineage`, and direct pass-through.

    !!! note
        Multiple Dagster assets can contribute to the same Metaxy feature by setting the same
        `"metaxy/feature"` metadata. This is a perfectly valid setup since Metaxy writes are append-only.

    !!! warning "Using `@metaxify` with multi assets"
        The `feature` argument cannot be used with `@dg.multi_asset` that produces multiple outputs.
        Instead, set `"metaxy/feature"` metadata on the right output's `AssetSpec`:

        ```python
        @mxd.metaxify()
        @dg.multi_asset(
            specs=[
                dg.AssetSpec("output_a", metadata={"metaxy/feature": "feature/a"}),
                dg.AssetSpec("output_b", metadata={"metaxy/feature": "feature/b"}),
            ]
        )
        def my_multi_asset():
            ...
        ```

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
    set_description: bool
    inject_column_schema: bool
    inject_column_lineage: bool

    def __init__(
        self,
        _asset: "_T | None" = None,
        *,
        feature: mx.CoercibleToFeatureKey | None = None,
        inherit_feature_key_as_asset_key: bool = True,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
        set_description: bool = True,
        inject_column_schema: bool = True,
        inject_column_lineage: bool = True,
    ) -> None:
        # Actual initialization happens in __new__, but we set defaults here for type checkers
        self.feature = (
            mx.coerce_to_feature_key(feature) if feature is not None else None
        )
        self.inherit_feature_key_as_asset_key = inherit_feature_key_as_asset_key
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
        feature: mx.CoercibleToFeatureKey | None = None,
        inherit_feature_key_as_asset_key: bool = True,
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
        feature: mx.CoercibleToFeatureKey | None = None,
        inherit_feature_key_as_asset_key: bool = True,
        inject_metaxy_kind: bool = True,
        inject_code_version: bool = True,
        set_description: bool = True,
        inject_column_schema: bool = True,
        inject_column_lineage: bool = True,
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
                set_description=set_description,
                inject_column_schema=inject_column_schema,
                inject_column_lineage=inject_column_lineage,
            )

        # Called as @metaxify() with parentheses - return instance for __call__
        instance = object.__new__(cls)
        instance.feature = coerced_feature
        instance.inherit_feature_key_as_asset_key = inherit_feature_key_as_asset_key
        instance.inject_metaxy_kind = inject_metaxy_kind
        instance.inject_code_version = inject_code_version
        instance.set_description = set_description
        instance.inject_column_schema = inject_column_schema
        instance.inject_column_lineage = inject_column_lineage
        return instance

    def __call__(self, asset: _T) -> _T:
        """Transform the asset when used as `@metaxify()` with parentheses."""
        return self._transform(
            asset,
            feature=self.feature,
            inherit_feature_key_as_asset_key=self.inherit_feature_key_as_asset_key,
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
        feature: mx.FeatureKey | None,
        inherit_feature_key_as_asset_key: bool,
        inject_metaxy_kind: bool,
        inject_code_version: bool,
        set_description: bool,
        inject_column_schema: bool,
        inject_column_lineage: bool,
    ) -> _T:
        """Transform an AssetsDefinition or AssetSpec with Metaxy metadata."""
        if isinstance(asset, dg.AssetSpec):
            return _metaxify_spec(
                asset,
                feature=feature,
                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
                set_description=set_description,
                inject_column_schema=inject_column_schema,
                inject_column_lineage=inject_column_lineage,
            )

        # Handle AssetsDefinition
        # Validate that feature argument is not used with multi-asset
        if feature is not None and len(asset.keys) > 1:
            raise ValueError(
                f"Cannot use `feature` argument with multi-asset `{asset.node_def.name}` "
                f"that produces {len(asset.keys)} outputs. "
                f"Instead, set `metaxy/feature` metadata on each output's AssetSpec."
            )

        keys_to_replace: dict[dg.AssetKey, dg.AssetKey] = {}
        transformed_specs: list[dg.AssetSpec] = []

        for key, asset_spec in asset.specs_by_key.items():
            new_spec = _metaxify_spec(
                asset_spec,
                feature=feature,
                inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
                inject_metaxy_kind=inject_metaxy_kind,
                inject_code_version=inject_code_version,
                set_description=set_description,
                inject_column_schema=inject_column_schema,
                inject_column_lineage=inject_column_lineage,
            )
            if new_spec.key != key:
                keys_to_replace[key] = new_spec.key
            transformed_specs.append(new_spec)

        return _replace_specs_on_assets_definition(
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
            output_name: keys_to_replace.get(key, key)
            for output_name, key in attrs["keys_by_output_name"].items()
        }
        attrs["selected_asset_keys"] = {
            keys_to_replace.get(key, key) for key in attrs["selected_asset_keys"]
        }

    # Create a new AssetsDefinition with the updated attributes
    # This bypasses the buggy code path in Dagster's replace_specs_on_asset
    return asset.__class__.dagster_internal_init(**attrs)


def _metaxify_spec(
    spec: dg.AssetSpec,
    *,
    feature: mx.FeatureKey | None,
    inherit_feature_key_as_asset_key: bool,
    inject_metaxy_kind: bool,
    inject_code_version: bool,
    set_description: bool,
    inject_column_schema: bool,
    inject_column_lineage: bool,
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

    # Use feature class docstring as description if not set on asset spec
    final_description = spec.description
    if set_description and final_description is None and feature_cls.__doc__:
        final_description = inspect.cleandoc(feature_cls.__doc__)

    # Build tags for project and feature
    # Note: Dagster tag values only allow alpha-numeric, '_', '-', '.'
    # so we use table_name which uses '__' separator
    tags_to_add: dict[str, str] = {
        DAGSTER_METAXY_PROJECT_TAG_KEY: mx.MetaxyConfig.get().project,
        DAGSTER_METAXY_FEATURE_METADATA_KEY: feature_key.table_name,
    }

    # Build column schema from Pydantic fields
    column_schema: dg.TableSchema | None = None
    if inject_column_schema:
        columns: list[dg.TableColumn] = []
        for field_name, field_info in feature_cls.model_fields.items():
            columns.append(
                dg.TableColumn(
                    name=field_name,
                    type=_get_type_string(field_info.annotation),
                    description=field_info.description,
                )
            )
        if columns:
            column_schema = dg.TableSchema(columns=columns)

    # Build column lineage from upstream dependencies
    column_lineage: dg.TableColumnLineage | None = None
    if inject_column_lineage and feature_spec.deps:
        column_lineage = _build_column_lineage(
            feature_cls=feature_cls,
            feature_spec=feature_spec,
            inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
        )

    # Build the replacement attributes
    metadata_to_add: dict[str, Any] = {
        **spec.metadata,
        DAGSTER_METAXY_FEATURE_METADATA_KEY: feature_key.to_string(),
        DAGSTER_METAXY_METADATA_METADATA_KEY: feature_spec.metadata,
    }
    if column_schema is not None:
        metadata_to_add[DAGSTER_COLUMN_SCHEMA_METADATA_KEY] = column_schema
    if column_lineage is not None:
        metadata_to_add[DAGSTER_COLUMN_LINEAGE_METADATA_KEY] = column_lineage

    replace_attrs: dict[str, Any] = {
        "key": final_key,
        "deps": {*spec.deps, *deps_to_add},
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


def _build_column_lineage(
    feature_cls: type[mx.BaseFeature],
    feature_spec: mx.FeatureSpec,
    inherit_feature_key_as_asset_key: bool,
) -> dg.TableColumnLineage | None:
    """Build column-level lineage from feature dependencies.

    Tracks column provenance by analyzing:
    - `FeatureDep.rename` mappings: renamed columns trace back to their upstream source
    - `FeatureSpec.lineage`: ID column relationships between features
    - Direct pass-through: columns with same name in both upstream and downstream

    Args:
        feature_cls: The downstream feature class.
        feature_spec: The downstream feature specification.
        inherit_feature_key_as_asset_key: Whether to use feature keys as asset keys.

    Returns:
        TableColumnLineage mapping downstream columns to their upstream sources,
        or None if no column lineage can be determined.
    """
    deps_by_column: dict[str, list[dg.TableColumnDep]] = {}
    downstream_columns = set(feature_cls.model_fields.keys())

    for dep in feature_spec.deps:
        upstream_feature_cls = mx.get_feature_by_key(dep.feature)
        upstream_feature_spec = upstream_feature_cls.spec()
        upstream_asset_key = get_asset_key_for_metaxy_feature_spec(
            upstream_feature_spec,
            inherit_feature_key_as_asset_key=inherit_feature_key_as_asset_key,
        )
        upstream_columns = set(upstream_feature_cls.model_fields.keys())

        # Build reverse rename map: downstream_name -> upstream_name
        # FeatureDep.rename is {old_upstream_name: new_downstream_name}
        reverse_rename: dict[str, str] = {}
        if dep.rename:
            reverse_rename = {v: k for k, v in dep.rename.items()}

        # Track columns based on lineage relationship
        lineage = feature_spec.lineage

        # Get ID column mappings based on lineage type
        id_column_mapping = _get_id_column_mapping(
            downstream_id_columns=feature_spec.id_columns,
            upstream_id_columns=upstream_feature_spec.id_columns,
            lineage=lineage,
            rename=reverse_rename,
        )

        # Process ID columns
        for downstream_col, upstream_col in id_column_mapping.items():
            if downstream_col in downstream_columns:
                if downstream_col not in deps_by_column:
                    deps_by_column[downstream_col] = []
                deps_by_column[downstream_col].append(
                    dg.TableColumnDep(
                        asset_key=upstream_asset_key,
                        column_name=upstream_col,
                    )
                )

        # Process renamed columns (that aren't ID columns)
        for downstream_col, upstream_col in reverse_rename.items():
            if (
                downstream_col in downstream_columns
                and downstream_col not in id_column_mapping
            ):
                if upstream_col in upstream_columns:
                    if downstream_col not in deps_by_column:
                        deps_by_column[downstream_col] = []
                    deps_by_column[downstream_col].append(
                        dg.TableColumnDep(
                            asset_key=upstream_asset_key,
                            column_name=upstream_col,
                        )
                    )

        # Process direct pass-through columns (same name in both, not renamed or ID)
        handled_columns = set(id_column_mapping.keys()) | set(reverse_rename.keys())
        for col in downstream_columns - handled_columns:
            if col in upstream_columns:
                if col not in deps_by_column:
                    deps_by_column[col] = []
                deps_by_column[col].append(
                    dg.TableColumnDep(
                        asset_key=upstream_asset_key,
                        column_name=col,
                    )
                )

    if not deps_by_column:
        return None

    return dg.TableColumnLineage(deps_by_column=deps_by_column)


def _get_id_column_mapping(
    downstream_id_columns: tuple[str, ...],
    upstream_id_columns: tuple[str, ...],
    lineage: mx.LineageRelationship,
    rename: dict[str, str],
) -> dict[str, str]:
    """Get mapping of downstream ID columns to upstream ID columns.

    Args:
        downstream_id_columns: ID columns of the downstream feature.
        upstream_id_columns: ID columns of the upstream feature.
        lineage: The lineage relationship between features.
        rename: Reverse rename map (downstream_name -> upstream_name).

    Returns:
        Mapping of downstream ID column names to upstream ID column names.
    """
    from metaxy.models.lineage import (
        AggregationRelationship,
        ExpansionRelationship,
        IdentityRelationship,
    )

    mapping: dict[str, str] = {}
    rel = lineage.relationship

    if isinstance(rel, IdentityRelationship):
        # 1:1 - downstream ID columns map to same-named upstream ID columns
        # (accounting for any renames)
        for downstream_col in downstream_id_columns:
            # Check if this column was renamed from upstream
            upstream_col = rename.get(downstream_col, downstream_col)
            if upstream_col in upstream_id_columns:
                mapping[downstream_col] = upstream_col

    elif isinstance(rel, AggregationRelationship):
        # N:1 - aggregation columns map to upstream
        # Use `on` columns if specified, otherwise use all downstream ID columns
        agg_columns = rel.on if rel.on is not None else downstream_id_columns
        for downstream_col in agg_columns:
            if downstream_col in downstream_id_columns:
                upstream_col = rename.get(downstream_col, downstream_col)
                if upstream_col in upstream_id_columns:
                    mapping[downstream_col] = upstream_col

    elif isinstance(rel, ExpansionRelationship):
        # 1:N - `on` columns (parent ID columns) map to upstream ID columns
        for downstream_col in rel.on:
            if downstream_col in downstream_id_columns:
                upstream_col = rename.get(downstream_col, downstream_col)
                if upstream_col in upstream_id_columns:
                    mapping[downstream_col] = upstream_col

    return mapping


def _get_type_string(annotation: Any) -> str:
    """Get a clean string representation of a type annotation.

    For generic types (list[str], dict[str, int], etc.), str() works well.
    For simple types (str, int, etc.), use __name__ to avoid "<class 'str'>" output.
    """
    # For generic types (list[str], dict[str, int], Union, etc.), str() works well
    if get_origin(annotation) is not None:
        return str(annotation)
    # For simple types, use __name__ if available
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    # Fallback to str()
    return str(annotation)
