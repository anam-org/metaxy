"""Table metadata utilities for Dagster integration.

This module provides utilities for building Dagster table metadata
(column schema, column lineage, etc.) from Metaxy feature definitions.
"""

import types
from typing import Any, Union, get_args, get_origin

import dagster as dg

import metaxy as mx
from metaxy.ext.dagster.utils import get_asset_key_for_metaxy_feature_spec
from metaxy.models.constants import ALL_SYSTEM_COLUMNS, SYSTEM_COLUMNS_WITH_LINEAGE


def build_column_schema(feature_cls: type[mx.BaseFeature]) -> dg.TableSchema | None:
    """Build a Dagster TableSchema from a Metaxy feature class.

    Creates column definitions from Pydantic model fields, including inherited
    system columns. Field types are converted to strings and field descriptions
    are used as column descriptions.

    Args:
        feature_cls: The Metaxy feature class to extract schema from.

    Returns:
        A TableSchema with columns derived from Pydantic model fields,
        sorted alphabetically by name, or None if the feature has no fields.

    !!! tip
        This is automatically injected by [`@metaxify`][metaxy.ext.dagster.metaxify.metaxify]
    """
    columns: list[dg.TableColumn] = []
    for field_name, field_info in feature_cls.model_fields.items():
        columns.append(
            dg.TableColumn(
                name=field_name,
                type=_get_type_string(field_info.annotation),
                description=field_info.description,
            )
        )

    if not columns:
        return None

    # Sort columns alphabetically by name
    columns.sort(key=lambda col: col.name)
    return dg.TableSchema(columns=columns)


def _get_type_string(annotation: Any) -> str:
    """Get a clean string representation of a type annotation.

    For generic types (list[str], dict[str, int], etc.), str() works well.
    For simple types (str, int, etc.), use __name__ to avoid "<class 'str'>" output.

    Special handling:
    - Pydantic datetime types show cleaner representations
    - None is stripped from union types (nullability is handled separately via DB constraints)
    """
    from pydantic import AwareDatetime, NaiveDatetime

    # Map Pydantic datetime types to cleaner representations
    pydantic_type_names = {
        AwareDatetime: "datetime (UTC)",
        NaiveDatetime: "datetime (naive)",
    }

    # For generic types (list[str], dict[str, int], Union, etc.), handle args recursively
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if args:
            # Handle Union types (X | Y syntax uses types.UnionType, typing.Union is different)
            if origin is Union or isinstance(annotation, types.UnionType):
                # Filter out None - nullability is handled via DB constraints, not Pydantic types
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    # Simple optional type like `str | None` -> just return the base type
                    return _get_type_string(non_none_args[0])
                # Multiple non-None types in union
                clean_args = [_get_type_string(arg) for arg in non_none_args]
                return " | ".join(clean_args)
            # Handle other generic types
            clean_args = [_get_type_string(arg) for arg in args]
            origin_name = getattr(origin, "__name__", str(origin))
            return f"{origin_name}[{', '.join(clean_args)}]"
        return str(annotation)

    # Check for Pydantic special types
    if annotation in pydantic_type_names:
        return pydantic_type_names[annotation]

    # For simple types, use __name__ if available
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Fallback to str()
    return str(annotation)


def build_column_lineage(
    feature_cls: type[mx.BaseFeature],
    feature_spec: mx.FeatureSpec | None = None,
    inherit_feature_key_as_asset_key: bool = True,
) -> dg.TableColumnLineage | None:
    """Build column-level lineage from feature dependencies.

    Tracks column provenance by analyzing:
    - `FeatureDep.rename` mappings: renamed columns trace back to their upstream source
    - `FeatureSpec.lineage`: ID column relationships between features
    - Direct pass-through: columns with same name in both upstream and downstream
    - System columns: `metaxy_provenance_by_field` and `metaxy_provenance` have lineage
      from corresponding upstream columns

    Args:
        feature_cls: The downstream feature class.
        feature_spec: The downstream feature specification. If None, uses feature_cls.spec().
        inherit_feature_key_as_asset_key: Whether to use feature keys as asset keys.

    Returns:
        TableColumnLineage mapping downstream columns to their upstream sources,
        or None if no column lineage can be determined.

    !!! tip
        This is automatically injected by [`@metaxify`][metaxy.ext.dagster.metaxify.metaxify]
    """
    if feature_spec is None:
        feature_spec = feature_cls.spec()

    if not feature_spec.deps:
        return None

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

        # Process direct pass-through columns (same name in both, not renamed, ID, or system)
        # System columns are handled separately below since only some have lineage
        handled_columns = (
            set(id_column_mapping.keys())
            | set(reverse_rename.keys())
            | ALL_SYSTEM_COLUMNS
        )
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

        # Process system columns with lineage (metaxy_provenance_by_field, metaxy_provenance)
        # These columns are always present in both upstream and downstream features
        # and have a direct lineage relationship (downstream values are computed from upstream)
        for sys_col in SYSTEM_COLUMNS_WITH_LINEAGE:
            if sys_col not in deps_by_column:
                deps_by_column[sys_col] = []
            deps_by_column[sys_col].append(
                dg.TableColumnDep(
                    asset_key=upstream_asset_key,
                    column_name=sys_col,
                )
            )

    if not deps_by_column:
        return None

    # Sort columns alphabetically
    sorted_deps = {k: deps_by_column[k] for k in sorted(deps_by_column)}
    return dg.TableColumnLineage(deps_by_column=sorted_deps)


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
