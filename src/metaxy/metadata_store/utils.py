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
            pass  # Warnings suppressed within this block
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

    Replaces username and password in URIs with `***` to prevent credential exposure
    in logs, display strings, and error messages.

    Examples:
        >>> sanitize_uri("s3://bucket/path")
        's3://bucket/path'
        >>> sanitize_uri("db://user:pass@host/db")
        'db://***:***@host/db'
        >>> sanitize_uri("postgresql://admin:secret@host:5432/db")
        'postgresql://***:***@host:5432/db'
        >>> sanitize_uri("./local/path")
        './local/path'

    Args:
        uri: URI or path string that may contain credentials

    Returns:
        Sanitized URI with credentials masked as ***
    """
    # Try to parse as URI
    try:
        parsed = urlparse(uri)

        # If no scheme, it's likely a local path - return as-is
        if not parsed.scheme or parsed.scheme in ("file", "local"):
            return uri

        # Check if URI contains credentials (username or password)
        if parsed.username or parsed.password:
            # Replace credentials with ***
            username = "***" if parsed.username else ""
            password = "***" if parsed.password else ""
            credentials = f"{username}:{password}@" if username or password else ""
            # Reconstruct netloc without credentials
            host_port = parsed.netloc.split("@")[-1]
            masked_netloc = f"{credentials}{host_port}"

            # Reconstruct URI with masked credentials
            return urlunparse(
                (
                    parsed.scheme,
                    masked_netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
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

        schema = nw.Schema({"status": nw.String, "age": nw.Int64})
        filters = nw.col("status") == "active"
        result = narwhals_expr_to_sql_predicate(filters, schema, dialect="duckdb")
        ```

    Example: With extra transforms
        ```py
        import narwhals as nw
        from metaxy.metadata_store.utils import unquote_identifiers

        schema = nw.Schema({"status": nw.String})
        filters = nw.col("status") == "active"
        sql = narwhals_expr_to_sql_predicate(
            filters,
            schema,
            dialect="datafusion",
            extra_transforms=unquote_identifiers(),
        )
        assert "status" in sql  # Unquoted column name
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

        sql = '''"status" = 'active' '''
        parsed = sqlglot.parse_one(sql)
        transformed = parsed.transform(unquote_identifiers())
        assert "status" in str(transformed)
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
