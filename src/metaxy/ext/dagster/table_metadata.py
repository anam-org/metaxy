"""Table metadata utilities for Dagster integration.

This module provides utilities for building Dagster table metadata
(column schema, column lineage, table previews, etc.) from Metaxy feature definitions.
"""

import types
from typing import Any, Union, get_args, get_origin

import dagster as dg
import narwhals as nw
import polars as pl

import metaxy as mx
from metaxy.ext.dagster.utils import get_asset_key_for_metaxy_feature_spec
from metaxy.models.constants import (
    ALL_SYSTEM_COLUMNS,
    METAXY_CREATED_AT,
    SYSTEM_COLUMNS_WITH_LINEAGE,
)


def build_column_schema(feature: mx.FeatureDefinition | type[mx.BaseFeature]) -> dg.TableSchema:
    """Build a Dagster TableSchema from a Metaxy feature class.

    Creates column definitions from Pydantic model fields, including inherited
    system columns. Field types are converted to strings and field descriptions
    are used as column descriptions.

    For FeatureDefinition objects, imports the feature class from its class path.

    Args:
        feature: The Metaxy feature definition or class to extract schema from.

    Returns:
        A TableSchema with columns derived from Pydantic model fields,
        sorted alphabetically by name.

    !!! tip
        This is automatically injected by [`@metaxify`][metaxy.ext.dagster.metaxify.metaxify]
    """
    # Get the feature class
    if isinstance(feature, mx.FeatureDefinition):
        feature_cls = feature._get_feature_class()
    else:
        feature_cls = feature

    columns: list[dg.TableColumn] = []
    for field_name, field_info in feature_cls.model_fields.items():
        columns.append(
            dg.TableColumn(
                name=field_name,
                type=_get_type_string(field_info.annotation),
                description=field_info.description,
            )
        )

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
    feature: mx.FeatureDefinition | type[mx.BaseFeature],
    feature_spec: mx.FeatureSpec | None = None,
) -> dg.TableColumnLineage | None:
    """Build column-level lineage from feature dependencies.

    Tracks column provenance by analyzing:
    - `FeatureDep.rename` mappings: renamed columns trace back to their upstream source
    - `FeatureDep.lineage`: ID column relationships between features
    - Direct pass-through: columns with same name in both upstream and downstream
    - System columns: `metaxy_provenance_by_field` and `metaxy_provenance` have lineage
      from corresponding upstream columns

    Args:
        feature: The downstream feature definition or class.
        feature_spec: The downstream feature specification. If None, extracted from feature.

    Returns:
        TableColumnLineage mapping downstream columns to their upstream sources,
        or None if no column lineage can be determined.

    !!! tip
        This is automatically injected by [`@metaxify`][metaxy.ext.dagster.metaxify.metaxify]
    """
    # Get spec and columns from either FeatureDefinition or feature class
    if isinstance(feature, mx.FeatureDefinition):
        if feature_spec is None:
            feature_spec = feature.spec
        downstream_columns = set(feature.columns)
    else:
        if feature_spec is None:
            feature_spec = feature.spec()
        downstream_columns = set(feature.model_fields.keys())

    assert feature_spec is not None

    if not feature_spec.deps:
        return None

    deps_by_column: dict[str, list[dg.TableColumnDep]] = {}

    for dep in feature_spec.deps:
        upstream_feature_def = mx.get_feature_by_key(dep.feature)
        upstream_feature_spec = upstream_feature_def.spec
        upstream_asset_key = get_asset_key_for_metaxy_feature_spec(upstream_feature_spec)
        upstream_columns = set(upstream_feature_def.columns)

        # Build reverse rename map: downstream_name -> upstream_name
        # FeatureDep.rename is {old_upstream_name: new_downstream_name}
        reverse_rename: dict[str, str] = {}
        if dep.rename:
            reverse_rename = {v: k for k, v in dep.rename.items()}

        # Track columns based on lineage relationship (now per-dependency)
        lineage = dep.lineage

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
            if downstream_col in downstream_columns and downstream_col not in id_column_mapping:
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
        handled_columns = set(id_column_mapping.keys()) | set(reverse_rename.keys()) | ALL_SYSTEM_COLUMNS
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


def _collect_tail(lazy_df: nw.LazyFrame[Any], n_rows: int) -> pl.DataFrame:
    """Collect the last N rows from a LazyFrame.

    Handles both Polars and Ibis backends:
    - Polars: Uses .tail() directly
    - Ibis: Uses sort + head since tail() is not available

    For backends without tail(), we sort by `metaxy_created_at` (which always
    exists in Metaxy feature tables) to get the most recent rows.

    Args:
        lazy_df: A narwhals LazyFrame to collect from.
        n_rows: Number of rows to collect from the end.

    Returns:
        A Polars DataFrame containing the last n_rows.
    """
    if hasattr(lazy_df._compliant_frame, "tail"):  # there is no better way to check whether .tail is supported :)
        # Polars and other backends that support tail()
        return lazy_df.tail(n_rows).collect().to_polars()
    else:
        # For backends without tail() (e.g., Ibis), use sort + head to simulate it
        # Sort descending by metaxy_created_at (latest first), take head(n_rows),
        # then sort ascending to restore chronological order
        return (
            lazy_df.sort(METAXY_CREATED_AT, descending=True).head(n_rows).sort(METAXY_CREATED_AT).collect().to_polars()
        )


def build_table_preview_metadata(
    lazy_df: nw.LazyFrame[Any],
    schema: dg.TableSchema,
    *,
    n_rows: int = 5,
) -> dg.TableMetadataValue:
    """Build a Dagster table preview from the last N rows of a LazyFrame.

    Collects the last `n_rows` from the LazyFrame and converts them to
    Dagster TableRecord objects suitable for display in the Dagster UI.
    Complex types (Struct, List, Array) are converted to JSON strings;
    primitive types (str, int, float, bool, None) are kept as-is.

    Args:
        lazy_df: A narwhals LazyFrame to preview.
        schema: The TableSchema for the table. Use `build_column_schema()` to
            create this from a Metaxy feature class.
        n_rows: Number of rows to include in the preview (from the end). Defaults to 5.

    Returns:
        A TableMetadataValue containing the preview rows as TableRecord objects.
        Returns an empty table if the DataFrame is empty.

    !!! tip

        This is automatically injected by [`MetaxyIOManager`][metaxy.ext.dagster.io_manager.MetaxyIOManager]
    """
    # Collect the last n_rows from the LazyFrame
    # .tail() is not implemented for Ibis lazy frames in Narwhals, since it's often undefined
    # for most SQL engines. See https://github.com/narwhals-dev/narwhals/issues/2389
    # For Ibis backends, we use with_row_index + filter to get the last N rows.
    df_polars = _collect_tail(lazy_df, n_rows)

    # Handle empty DataFrames
    if df_polars.is_empty():
        return dg.MetadataValue.table(records=[], schema=schema)

    # Convert complex types to strings, keep primitives as-is
    df_processed = _prepare_dataframe_for_table_record(df_polars)

    # Convert to TableRecord objects
    records = [dg.TableRecord(data=row) for row in df_processed.to_dicts()]

    return dg.MetadataValue.table(records=records, schema=schema)


def _prepare_dataframe_for_table_record(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare a Polars DataFrame for conversion to Dagster TableRecord objects.

    Complex types (Struct, List, Array) and temporal types are converted to strings.
    Lists/Arrays with more than 4 items are truncated to show first 2 and last 2
    with "..." in between.
    Primitive types (str, int, float, bool, None) are kept as-is since
    Dagster's TableRecord accepts them directly.

    Args:
        df: The Polars DataFrame to prepare.

    Returns:
        A DataFrame with complex/temporal types converted to strings.
    """
    exprs: list[pl.Expr] = []

    for col_name in df.columns:
        dtype = df[col_name].dtype
        if isinstance(dtype, pl.Struct):
            # Struct types: use json_encode for a clean JSON representation
            exprs.append(pl.col(col_name).struct.json_encode())
        elif isinstance(dtype, pl.List):
            # List types: truncate and convert to string
            exprs.append(_truncate_list_expr(pl.col(col_name), alias=col_name))
        elif isinstance(dtype, pl.Array):
            # Array types: convert to list first, then truncate
            exprs.append(_truncate_list_expr(pl.col(col_name).arr.to_list(), alias=col_name))
        elif dtype in (pl.Datetime, pl.Date, pl.Time, pl.Duration) or isinstance(
            dtype, (pl.Datetime, pl.Date, pl.Time, pl.Duration)
        ):
            # Temporal types: cast to string (ISO format)
            exprs.append(pl.col(col_name).cast(pl.String))
        else:
            # Primitive types: keep as-is
            exprs.append(pl.col(col_name))

    return df.select(exprs)


def _truncate_list_expr(list_expr: pl.Expr, alias: str, max_items: int = 2) -> pl.Expr:
    """Truncate a list expression and convert to string.

    Lists with more than max_items show first 1 and last 1 items with "..." between.

    Args:
        list_expr: A Polars expression that evaluates to a List type.
        alias: The output column name.
        max_items: Maximum items to show without truncation. Default 4.

    Returns:
        A Polars expression that truncates and converts the list to string.
    """
    list_len = list_expr.list.len()
    half = max_items // 2

    # For short lists: just json_encode the whole thing
    # For long lists: concat first 2 + last 2 and json_encode, then insert "..."
    truncated = pl.concat_list(
        list_expr.list.head(half),
        list_expr.list.tail(half),
    )

    # Convert to JSON string via struct wrapper
    def to_json(expr: pl.Expr) -> pl.Expr:
        return pl.struct(expr.alias("_")).struct.json_encode().str.extract(r'\{"_":(.*)\}', 1)

    short_result = to_json(list_expr)
    # For truncated: insert ".." after the first half elements
    # e.g., [1,10] -> [1,..,10]
    # Match: opening bracket, then `half` comma-separated values
    # The pattern matches values that may contain nested brackets
    value_pattern = r"[^\[\],]+(?:\[[^\]]*\])?"  # matches value or value[...]
    first_n_values = ",".join([value_pattern] * half)
    long_result = to_json(truncated).str.replace(
        r"^(\[" + first_n_values + r"),",
        "$1,..,",
    )

    return pl.when(list_len <= max_items).then(short_result).otherwise(long_result).alias(alias)
