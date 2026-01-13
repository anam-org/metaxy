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
_suppress_feature_version_warning: ContextVar[bool] = ContextVar(
    "_suppress_feature_version_warning", default=False
)


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


def adapt_trino_query_for_datafusion(sql: str) -> str:
    """Adapt Trino SQL for Datafusion by rewriting FROM_ISO8601_TIMESTAMP calls.

    Transforms SQL like:
        FROM_ISO8601_TIMESTAMP('2024-01-01T00:00:00.000Z')

    Into:
        CAST('2024-01-01T00:00:00.000Z' AS TIMESTAMP)

    Args:
        sql: SQL string that may contain literal FROM_ISO8601_TIMESTAMP calls.

    Returns:
        SQL with FROM_ISO8601_TIMESTAMP replaced by CAST.
    """
    import sqlglot

    parsed = sqlglot.parse_one(sql, read="trino")

    def _adapt(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Cast):
            target = node.args.get("to")
            if isinstance(target, exp.DataType):
                inner = node.this
                if isinstance(inner, exp.FromISO8601Timestamp) or (
                    isinstance(inner, exp.Anonymous)
                    and inner.name.upper() == "FROM_ISO8601_TIMESTAMP"
                ):
                    arg = inner.this
                    if not isinstance(arg, exp.Literal):
                        raise ValueError(
                            "FROM_ISO8601_TIMESTAMP expects a literal string for Datafusion"
                        )

                    iso_str = str(arg.this)
                    if iso_str.endswith("Z"):
                        iso_str = iso_str[:-1]
                    elif (
                        len(iso_str) >= 6
                        and iso_str[-6] in {"+", "-"}
                        and iso_str[-3] == ":"
                    ):
                        iso_str = iso_str[:-6]

                    return exp.Cast(
                        this=exp.Literal.string(iso_str),
                        to=exp.DataType.build("TIMESTAMP"),
                    )

                if target.this == exp.DataType.Type.TIMESTAMPTZ:
                    return exp.Cast(this=node.this, to=exp.DataType.build("TIMESTAMP"))
        elif isinstance(node, exp.FromISO8601Timestamp) or (
            isinstance(node, exp.Anonymous)
            and node.name.upper() == "FROM_ISO8601_TIMESTAMP"
        ):
            arg = node.this
            if not isinstance(arg, exp.Literal):
                raise ValueError(
                    "FROM_ISO8601_TIMESTAMP expects a literal string for Datafusion"
                )

            iso_str = str(arg.this)
            if iso_str.endswith("Z"):
                iso_str = iso_str[:-1]
            elif len(iso_str) >= 6 and iso_str[-6] in {"+", "-"} and iso_str[-3] == ":":
                iso_str = iso_str[:-6]

            return exp.Cast(
                this=exp.Literal.string(iso_str),
                to=exp.DataType.build("TIMESTAMP"),
            )

        return node

    transformed = parsed.transform(_adapt)
    return transformed.sql(dialect="trino")


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
) -> str:
    """Convert Narwhals filter expressions to a SQL WHERE clause predicate.

    This utility converts Narwhals filter expressions to SQL predicates by:
    1. Creating a temporary Ibis table from the provided schema
    2. Applying the Narwhals filters to generate SQL
    3. Extracting the WHERE clause predicate
    4. Stripping any table qualifiers (single-table only; not safe for joins)

    Args:
        filters: Narwhals filter expression or sequence of expressions to convert
        schema: Narwhals schema to build the Ibis table

    Returns:
        SQL WHERE clause predicate string (without the "WHERE" keyword)

    Raises:
        RuntimeError: If WHERE clause cannot be extracted from generated SQL

    Example:
        >>> import narwhals as nw
        >>> import polars as pl
        >>> df = pl.DataFrame({"status": ["active"], "age": [25]})
        >>> filters = nw.col("status") == "active"
        >>> narwhals_expr_to_sql_predicate(filters, df, dialect="duckdb")
        '"status" = \'active\''
    """
    filter_list = (
        list(filters)
        if isinstance(filters, Sequence) and not isinstance(filters, nw.Expr)
        else [filters]
    )
    if not filter_list:
        raise ValueError("narwhals_expr_to_sql_predicate expects at least one filter")
    sql = generate_sql(lambda lf: lf.filter(*filter_list), schema, dialect=dialect)

    import sqlglot
    from sqlglot.optimizer.simplify import simplify

    parsed = sqlglot.parse_one(sql, read=dialect)
    where_expr = parsed.args.get("where")
    if not where_expr:
        raise RuntimeError(
            f"Could not extract WHERE clause from generated SQL for filters: {filters}\n"
            f"Generated SQL: {sql}"
        )

    predicate_expr = simplify(where_expr.this)

    predicate_expr = predicate_expr.transform(_strip_table_qualifiers())

    return predicate_expr.sql(dialect=dialect)


def _strip_table_qualifiers() -> Callable[[exp.Expression], exp.Expression]:
    def _strip(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column):
            return node

        table_arg = node.args.get("table")
        if table_arg is None:
            return node

        cleaned = node.copy()
        cleaned.set("table", None)
        return cleaned

    return _strip
