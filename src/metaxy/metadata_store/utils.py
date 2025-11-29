from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import narwhals as nw
from narwhals.typing import Frame, FrameT
from sqlglot import exp

from metaxy.utils.constants import TEMP_TABLE_NAME

if TYPE_CHECKING:
    from collections.abc import Callable

# Context variable for suppressing feature_version warning in migrations
_suppress_feature_version_warning: ContextVar[bool] = ContextVar("_suppress_feature_version_warning", default=False)


def is_local_path(path: str) -> bool:
    """Return True when the path points to the local filesystem."""
    if path.startswith(("file://", "local://")):
        return True
    return "://" not in path


@contextmanager
def allow_feature_version_override() -> Iterator[None]:
    """Context manager to suppress warnings when writing metadata with pre-existing metaxy_feature_version.

    This should only be used in migration code where writing historical feature versions
    is intentional and necessary.

    Example:
        ```py
        with allow_feature_version_override():
            # DataFrame already has metaxy_feature_version column from migration
            store.write_metadata(MyFeature, df_with_feature_version)
        ```
    """
    token = _suppress_feature_version_warning.set(True)
    try:
        yield
    finally:
        _suppress_feature_version_warning.reset(token)


# Helper to create empty DataFrame with correct schema and backend
#
def empty_frame_like(ref_frame: FrameT) -> FrameT:
    """Create an empty LazyFrame with the same schema as ref_frame."""
    return ref_frame.head(0)  # ty: ignore[invalid-argument-type]


def sanitize_uri(uri: str) -> str:
    """Sanitize URI to mask credentials.

    Replaces passwords in URIs (both in netloc and query parameters) with `***`
    to prevent credential exposure in logs, display strings, and error messages.
    Usernames are preserved for debugging purposes.

    Examples:
        >>> sanitize_uri("s3://bucket/path")
        's3://bucket/path'
        >>> sanitize_uri("db://user:pass@host/db")
        'db://user:***@host/db'
        >>> sanitize_uri("postgresql://admin:secret@host:5432/db")
        'postgresql://admin:***@host:5432/db'
        >>> sanitize_uri("postgresql://host/db?password=secret")
        'postgresql://host/db?password=***'
        >>> sanitize_uri("db://host?user=admin&pwd=secret")
        'db://host?user=admin&pwd=***'
        >>> sanitize_uri("./local/path")
        './local/path'

    Args:
        uri: URI or path string that may contain credentials

    Returns:
        Sanitized URI with passwords masked as ***
    """
    # Try to parse as URI
    try:
        parsed = urlparse(uri)

        # If no scheme, it's likely a local path - return as-is
        if not parsed.scheme or parsed.scheme in ("file", "local"):
            return uri

        # Sanitize netloc (username:password@host:port)
        sanitized_netloc = parsed.netloc
        if parsed.password:
            # Keep username visible, only mask password
            if "@" in sanitized_netloc:
                userinfo, hostinfo = sanitized_netloc.rsplit("@", 1)
                if ":" in userinfo:
                    username, _, _ = userinfo.partition(":")
                    userinfo = f"{username}:***"
                sanitized_netloc = f"{userinfo}@{hostinfo}"

        # Sanitize query parameters
        sanitized_query = parsed.query
        if parsed.query:
            query_params = parse_qsl(parsed.query, keep_blank_values=True)
            masked_parts = []
            changed = False
            for key, val in query_params:
                if key.lower() in {"password", "pwd", "pass"}:
                    # Use *** without URL encoding for readability
                    masked_parts.append(f"{quote(key, safe='')}=***")
                    changed = True
                else:
                    masked_parts.append(f"{quote(key, safe='')}={quote(val, safe='')}")
            if changed:
                sanitized_query = "&".join(masked_parts)

        # Only reconstruct if something changed
        if sanitized_netloc != parsed.netloc or sanitized_query != parsed.query:
            return urlunparse(
                (
                    parsed.scheme,
                    sanitized_netloc,
                    parsed.path,
                    parsed.params,
                    sanitized_query,
                    parsed.fragment,
                )
            )
    except Exception:
        # If parsing fails, return as-is (likely a local path)
        pass

    return uri


def generate_sql(
    narwhals_function: Callable[[Frame], Frame],
    schema: nw.Schema,
    *,
    dialect: str,
) -> str:
    """Generate SQL for a Narwhals transformation given a schema."""
    import ibis

    ibis_table = ibis.table(ibis.schema(schema.to_arrow()), name=TEMP_TABLE_NAME)
    nw_lf = nw.from_native(ibis_table, eager_only=False)
    result_lf = narwhals_function(nw_lf)
    ibis_expr = result_lf.to_native()
    return ibis.to_sql(ibis_expr, dialect=dialect)


def narwhals_expr_to_sql_predicate(
    filters: nw.Expr | Sequence[nw.Expr],
    schema: nw.Schema,
    *,
    dialect: str,
    extra_transforms: (
        Callable[[exp.Expression], exp.Expression] | Sequence[Callable[[exp.Expression], exp.Expression]] | None
    ) = None,
) -> str:
    """Convert Narwhals filter expressions to a SQL WHERE clause predicate.

    This utility converts Narwhals filter expressions to SQL predicates by:
    1. Creating a temporary Ibis table from the provided schema
    2. Applying the Narwhals filters to generate SQL
    3. Extracting the WHERE clause predicate
    4. Stripping any table qualifiers (single-table only; not safe for joins)
    5. Applying any extra transforms provided

    Args:
        filters: Narwhals filter expression or sequence of expressions to convert
        schema: Narwhals schema to build the Ibis table
        dialect: SQL dialect to use when generating SQL
        extra_transforms: Optional sqlglot expression transformer(s) to apply after
            stripping table qualifiers. Can be a single callable or sequence of callables.
            Each callable should take an `exp.Expression` and return an `exp.Expression`.

    Returns:
        SQL WHERE clause predicate string (without the "WHERE" keyword)

    Raises:
        RuntimeError: If WHERE clause cannot be extracted from generated SQL

    Example:
        ```py
        import narwhals as nw
        import polars as pl

        df = pl.DataFrame({"status": ["active"], "age": [25]})
        filters = nw.col("status") == "active"
        narwhals_expr_to_sql_predicate(filters, df, dialect="duckdb")
        # '"status" = \'active\''
        ```

    Example: With extra transforms
        ```py
        from metaxy.metadata_store.utils import unquote_identifiers

        # Generate unquoted SQL for LanceDB
        sql = narwhals_expr_to_sql_predicate(
            filters,
            schema,
            dialect="datafusion",
            extra_transforms=unquote_identifiers(),
        )
        # 'status = \'active\''  (no quotes around column name)
        ```
    """
    filter_list = list(filters) if isinstance(filters, Sequence) and not isinstance(filters, nw.Expr) else [filters]
    if not filter_list:
        raise ValueError("narwhals_expr_to_sql_predicate expects at least one filter")
    sql = generate_sql(lambda lf: lf.filter(*filter_list), schema, dialect=dialect)

    from sqlglot.optimizer.simplify import simplify

    predicate_expr = _extract_where_expression(sql, dialect=dialect)
    if predicate_expr is None:
        raise RuntimeError(
            f"Could not extract WHERE clause from generated SQL for filters: {filters}\nGenerated SQL: {sql}"
        )

    predicate_expr = simplify(predicate_expr)

    # Apply table qualifier stripping first
    predicate_expr = predicate_expr.transform(_strip_table_qualifiers())

    # Apply extra transforms if provided
    if extra_transforms is not None:
        transform_list = [extra_transforms] if callable(extra_transforms) else list(extra_transforms)
        for transform in transform_list:
            predicate_expr = predicate_expr.transform(transform)

    return predicate_expr.sql(dialect=dialect)


def _strip_table_qualifiers() -> Callable[[exp.Expression], exp.Expression]:
    """Return a transformer function that removes table qualifiers from column references.

    Used to convert qualified column names like `table.column` to unqualified `column`
    when generating DELETE statements from SELECT queries.
    """

    def _strip(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column):
            return node

        if node.args.get("table") is None:
            return node

        cleaned = node.copy()
        cleaned.set("table", None)
        return cleaned

    return _strip


def unquote_identifiers() -> Callable[[exp.Expression], exp.Expression]:
    """Return a transformer function that removes quotes from column identifiers.

    LanceDB (and some other systems) require unquoted identifiers in SQL predicates.
    This transformer removes the `quoted` flag from identifier nodes.

    Returns:
        A transformer function that unquotes column identifiers

    Example:
        ```py
        import sqlglot
        from sqlglot import exp

        sql = '"status" = \'active\''
        parsed = sqlglot.parse_one(sql)
        transformed = parsed.transform(unquote_identifiers())
        # Result: status = 'active'
        ```
    """

    def _unquote(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier):
            unquoted = node.copy()
            unquoted.this.set("quoted", False)
            return unquoted
        return node

    return _unquote


def _extract_where_expression(
    sql: str,
    *,
    dialect: str | None = None,
) -> exp.Expression | None:
    import sqlglot

    parsed = sqlglot.parse_one(sql, read=dialect) if dialect else sqlglot.parse_one(sql)
    where_expr = parsed.args.get("where")
    if where_expr is None:
        return None
    return where_expr.this
