"""DagsterType builder for Metaxy features.

This module provides utilities for creating Dagster types that validate
Metaxy feature outputs with proper metadata injection (table schema, etc.).
"""

from collections.abc import Callable, Mapping
from typing import Any

import dagster as dg
import narwhals as nw

import metaxy as mx
from metaxy._decorators import public
from metaxy.ext.dagster.constants import (
    DAGSTER_COLUMN_LINEAGE_METADATA_KEY,
    DAGSTER_COLUMN_SCHEMA_METADATA_KEY,
    DAGSTER_METAXY_INFO_METADATA_KEY,
)
from metaxy.ext.dagster.table_metadata import build_column_lineage, build_column_schema
from metaxy.ext.dagster.utils import build_feature_info_metadata


def _create_type_check_fn(
    feature_key: mx.FeatureKey,
) -> Callable[[dg.TypeCheckContext, Any], dg.TypeCheck]:
    """Create a type check function for a Metaxy feature.

    The type check function validates that the output is either:
    - None (allowed for MetaxyOutput)
    - A narwhals-compatible dataframe (IntoFrame)

    Args:
        feature_key: The Metaxy feature key for error messages.

    Returns:
        A callable type check function for DagsterType.
    """

    def type_check_fn(context: dg.TypeCheckContext, value: Any) -> dg.TypeCheck:
        # None is a valid MetaxyOutput (indicates no data to write)
        if value is None:
            return dg.TypeCheck(success=True)

        # Try to convert to narwhals frame - this validates the type
        try:
            nw.from_native(value)
            return dg.TypeCheck(success=True)
        except TypeError as e:
            return dg.TypeCheck(
                success=False,
                description=(
                    f"Expected a narwhals-compatible dataframe or None for "
                    f"Metaxy feature '{feature_key.to_string()}', "
                    f"but got {type(value).__name__}:\n{e}"
                ),
            )

    return type_check_fn


@public
def feature_to_dagster_type(
    feature: mx.CoercibleToFeatureKey,
    *,
    name: str | None = None,
    description: str | None = None,
    inject_column_schema: bool = True,
    inject_column_lineage: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dg.DagsterType:
    """Build a Dagster type from a Metaxy feature.

    Creates a `dagster.DagsterType` that validates outputs are
    [`MetaxyOutput`][metaxy.ext.dagster.MetaxyOutput] (i.e., narwhals-compatible
    dataframes or `None`) and includes metadata derived from the feature's Pydantic
    model fields.

    Args:
        feature: The Metaxy feature to create a type for. Can be a feature class,
            feature key, or string that can be coerced to a feature key.
        name: Optional custom name for the DagsterType. Defaults to the feature's
            table name (e.g., "project__feature_name").
        description: Optional custom description. Defaults to the feature class
            docstring or a generated description.
        inject_column_schema: Whether to inject the column schema as metadata.
            The schema is derived from Pydantic model fields.
        inject_column_lineage: Whether to inject column lineage as metadata.
            The lineage is derived from feature dependencies.
        metadata: Optional custom metadata to inject into the DagsterType.

    Returns:
        A DagsterType configured for the Metaxy feature with appropriate
        type checking and metadata.

    !!! tip
        This is automatically injected by [`@metaxify`][metaxy.ext.dagster.metaxify.metaxify]

    Example:
        ```python
        import dagster as dg
        import polars as pl
        import metaxy.ext.dagster as mxd
        from myproject.features import MyFeature  # Your Metaxy feature class


        @mxd.metaxify(feature=MyFeature)
        @dg.asset(dagster_type=mxd.feature_to_dagster_type(MyFeature))
        def my_asset():
            return pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        ```

    !!! info "See also"
        - [`metaxify`][metaxy.ext.dagster.metaxify.metaxify]: Decorator for injecting
          Metaxy metadata into Dagster assets.
        - [`MetaxyOutput`][metaxy.ext.dagster.MetaxyOutput]: The type alias for valid
          Metaxy outputs.
    """
    from metaxy.ext.dagster.io_manager import MetaxyOutput

    feature_key = mx.coerce_to_feature_key(feature)
    feature_def = mx.get_feature_by_key(feature_key)

    # For build_column_schema, prefer the original class if provided
    # (handles cases where class is defined inside a function and can't be imported)
    feature_for_schema: mx.FeatureDefinition | type[mx.BaseFeature]
    if isinstance(feature, type) and issubclass(feature, mx.BaseFeature):
        feature_for_schema = feature
    else:
        feature_for_schema = feature_def

    # Determine name
    type_name = name or feature_key.table_name

    # Determine description - use schema description if available, else default
    if description is None:
        schema_desc = feature_def.feature_schema.get("description")
        if schema_desc:
            description = schema_desc
        else:
            description = f"Metaxy feature '{feature_key.to_string()}'."

    # Build metadata - start with custom metadata if provided
    final_metadata: dict[str, Any] = dict(metadata) if metadata else {}
    final_metadata[DAGSTER_METAXY_INFO_METADATA_KEY] = build_feature_info_metadata(feature_key)
    # Skip column schema for external features (no Python class to extract schema from)
    if inject_column_schema and not feature_def.is_external:
        column_schema = build_column_schema(feature_for_schema)
        if column_schema is not None:
            final_metadata[DAGSTER_COLUMN_SCHEMA_METADATA_KEY] = column_schema

    # Skip column lineage for external features (no Python class to extract columns from)
    if inject_column_lineage and not feature_def.is_external:
        column_lineage = build_column_lineage(feature_for_schema)
        if column_lineage is not None:
            final_metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY] = column_lineage

    dagster_type = dg.DagsterType(
        type_check_fn=_create_type_check_fn(feature_key),
        name=type_name,
        description=description,
        typing_type=MetaxyOutput,
        metadata=final_metadata,
    )

    return dagster_type
