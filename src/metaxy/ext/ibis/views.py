"""SQL views that expose current feature rows and feature statistics to readers without the Python library.

Three kinds of views are derived from the active feature graph:

- one *current* view per internal feature, returning the same rows as
  [`MetadataStore.read`][metaxy.MetadataStore.read] with default arguments
  (current feature version, latest row per sample, no soft-deleted rows),
- one *statistics* view with the row, missing, stale and orphaned counts that
  `mx metadata status` reports, computed live by the database,
- one *manifest* view that describes every other view (names, feature versions, id columns,
  column types and a hash of the view definition).

Every view is pinned to the feature versions of the graph it was created from.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import narwhals as nw
import sqlglot
import sqlglot.expressions as sge

from metaxy.ext.ibis.versioning import IbisVersioningEngine
from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_UPDATED_AT,
    is_system_column,
)
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types as ir

    from metaxy.ext.ibis.metadata_store import IbisMetadataStore
    from metaxy.models.feature import FeatureGraph

CURRENT_VIEW_SUFFIX = "__current"
STATISTICS_VIEW_NAME = "metaxy_feature_statistics"
MANIFEST_VIEW_NAME = "metaxy_views_manifest"

ViewKind = Literal["current", "statistics"]


@dataclass(frozen=True)
class ViewSpec:
    """A view that reads from the metadata store, described by its `SELECT` statement."""

    kind: ViewKind
    name: str
    select_sql: str
    columns: tuple[tuple[str, str], ...]
    feature_key: FeatureKey | None = None
    feature_version: str | None = None
    id_columns: tuple[str, ...] = ()

    @property
    def definition_hash(self) -> str:
        return hashlib.sha256(self.select_sql.encode()).hexdigest()[:16]

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "view_name": self.name,
            "feature_key": self.feature_key.to_string() if self.feature_key is not None else None,
            "feature_version": self.feature_version,
            "id_columns": list(self.id_columns),
            "columns": [{"name": name, "type": dtype} for name, dtype in self.columns],
            "definition_hash": self.definition_hash,
        }


@dataclass(frozen=True)
class ViewPlan:
    """All views to synchronize, plus the manifest view that describes them."""

    schema: str | None
    dialect: str
    views: tuple[ViewSpec, ...]
    skipped_features: tuple[FeatureKey, ...]

    @property
    def manifest_entries(self) -> list[dict[str, Any]]:
        return [view.to_manifest_entry() for view in self.views]

    @property
    def manifest_select_sql(self) -> str:
        return _manifest_select_sql(self.manifest_entries, self.dialect)

    def create_statements(self, views: Sequence[ViewSpec] | None = None) -> list[str]:
        """`CREATE OR REPLACE VIEW` statements for the given views (all views by default) and the manifest."""
        selected = self.views if views is None else views
        return [
            *(_create_view_sql(view.name, view.select_sql, self.schema, self.dialect) for view in selected),
            _create_view_sql(MANIFEST_VIEW_NAME, self.manifest_select_sql, self.schema, self.dialect),
        ]


def current_view_name(store: IbisMetadataStore, key: FeatureKey) -> str:
    return f"{store.get_table_name(key)}{CURRENT_VIEW_SUFFIX}"


def supports_views(store: IbisMetadataStore) -> bool:
    """Views can be derived only when the store computes increments natively with Ibis."""
    return issubclass(store.versioning_engine_cls, IbisVersioningEngine)


def plan_views(
    store: IbisMetadataStore,
    graph: FeatureGraph,
    feature_keys: Sequence[FeatureKey],
    *,
    schema: str | None = None,
) -> ViewPlan:
    """Build the current views and the statistics view for the given internal features.

    Features whose table does not exist in the store yet are skipped. The store must be open
    and `graph` must be the active graph.
    """
    existing_tables = set(store.conn.list_tables())
    included = [key for key in feature_keys if store.get_table_name(key) in existing_tables]
    skipped = tuple(key for key in feature_keys if key not in included)
    dialect = store.conn.name

    current_views = [_current_view_spec(store, graph, key, dialect) for key in included]
    views: list[ViewSpec] = list(current_views)
    if included:
        views.append(_statistics_view_spec(store, graph, included, dialect))

    return ViewPlan(schema=schema, dialect=dialect, views=tuple(views), skipped_features=skipped)


def sync_views(store: IbisMetadataStore, plan: ViewPlan, views: Sequence[ViewSpec] | None = None) -> None:
    """Create or replace the given views (all views by default) and the manifest view."""
    for statement in plan.create_statements(views):
        store.conn.raw_sql(statement)  # ty: ignore[unresolved-attribute]


def check_views(store: IbisMetadataStore, plan: ViewPlan, views: Sequence[ViewSpec] | None = None) -> list[str]:
    """Return the names of views that are missing or whose definition differs from the plan."""
    selected = plan.views if views is None else views
    existing = set(store.conn.list_tables(database=plan.schema))
    if MANIFEST_VIEW_NAME not in existing:
        return sorted({*(view.name for view in selected if view.name not in existing), MANIFEST_VIEW_NAME})

    stored_hashes = {
        row["view_name"]: row["definition_hash"]
        for row in store.conn.table(MANIFEST_VIEW_NAME, database=plan.schema)
        .select("view_name", "definition_hash")
        .to_pyarrow()
        .to_pylist()
    }
    outdated = {
        view.name
        for view in selected
        if view.name not in existing or stored_hashes.get(view.name) != view.definition_hash
    }
    if set(stored_hashes) != {view.name for view in plan.views}:
        outdated.add(MANIFEST_VIEW_NAME)
    return sorted(outdated)


def _current_rows(store: IbisMetadataStore, graph: FeatureGraph, key: FeatureKey) -> ir.Table:
    """Rows returned by `read()` with default arguments, restricted to user columns and the data version."""
    table = store.conn.table(store.get_table_name(key))
    id_columns = list(graph.get_feature_definition(key).id_columns)
    value_columns = [
        column
        for column in table.columns
        if column in id_columns or column == METAXY_DATA_VERSION or not is_system_column(column)
    ]
    versioned = table.filter(table[METAXY_FEATURE_VERSION] == graph.get_feature_version(key)).select(
        *value_columns, METAXY_DELETED_AT, METAXY_UPDATED_AT
    )
    latest = cast(
        "ir.Table",
        store.versioning_engine_cls.keep_latest_by_group(
            nw.from_native(versioned),
            group_columns=id_columns,
            timestamp_columns=[METAXY_DELETED_AT, METAXY_UPDATED_AT],
        ).to_native(),
    )
    return latest.filter(latest[METAXY_DELETED_AT].isnull()).select(*value_columns)


def _current_view_spec(store: IbisMetadataStore, graph: FeatureGraph, key: FeatureKey, dialect: str) -> ViewSpec:
    expr = _current_rows(store, graph, key)
    return ViewSpec(
        kind="current",
        name=current_view_name(store, key),
        select_sql=_to_view_sql(store, expr, dialect),
        columns=_columns(expr),
        feature_key=key,
        feature_version=graph.get_feature_version(key),
        id_columns=tuple(graph.get_feature_definition(key).id_columns),
    )


def _statistics_view_spec(
    store: IbisMetadataStore, graph: FeatureGraph, keys: Sequence[FeatureKey], dialect: str
) -> ViewSpec:
    import ibis

    rows = [_statistics_row(store, graph, key) for key in keys]
    expr = functools.reduce(lambda left, right: left.union(right, distinct=False), rows)
    return ViewSpec(
        kind="statistics",
        name=STATISTICS_VIEW_NAME,
        select_sql=_to_view_sql(store, expr.order_by(ibis._.feature_key), dialect),
        columns=_columns(expr),
    )


def _statistics_row(store: IbisMetadataStore, graph: FeatureGraph, key: FeatureKey) -> ir.Table:
    """One row with the counts of `mx metadata status` for a feature, evaluated by the database."""
    import ibis
    import ibis.expr.datatypes as dt

    feature_version = graph.get_feature_version(key)
    table = store.conn.table(store.get_table_name(key))
    versioned = table.filter(table[METAXY_FEATURE_VERSION] == feature_version)
    is_root_feature = not graph.get_feature_plan(key).deps

    if is_root_feature:
        missing = stale = orphaned = ibis.literal(0, type=dt.int64)
    else:
        increment = store.resolve_update(key, lazy=True, versioning_engine="native")
        missing, stale, orphaned = (
            cast("ir.Table", frame.to_native()).count().as_scalar().cast(dt.int64)
            for frame in (increment.new, increment.stale, increment.orphaned)
        )

    return versioned.aggregate(
        last_write_at=versioned[METAXY_UPDATED_AT].max().cast(dt.Timestamp(timezone="UTC")),
    ).select(
        feature_key=ibis.literal(key.to_string(), type=dt.string),
        feature_version=ibis.literal(feature_version, type=dt.string),
        is_root_feature=ibis.literal(is_root_feature, type=dt.boolean),
        row_count=_current_rows(store, graph, key).count().as_scalar().cast(dt.int64),
        missing_count=missing,
        stale_count=stale,
        orphaned_count=orphaned,
        last_write_at=ibis._.last_write_at,
    )


def _columns(expr: ir.Table) -> tuple[tuple[str, str], ...]:
    return tuple((name, str(dtype)) for name, dtype in expr.schema().items())


def _to_view_sql(store: IbisMetadataStore, expr: ibis.Table, dialect: str) -> str:
    """Compile an expression to SQL whose table references are qualified with the store's database.

    Qualification keeps the view valid when it lives in a different schema than the feature tables.
    """
    import ibis

    tree = sqlglot.parse_one(ibis.to_sql(expr, dialect=dialect), dialect=dialect)
    cte_names = {cte.alias_or_name for cte in tree.find_all(sge.CTE)}
    database = store.conn.current_database  # ty: ignore[unresolved-attribute]
    for table in tree.find_all(sge.Table):
        if not table.args.get("db") and table.name not in cte_names:
            table.set("db", sge.to_identifier(database, quoted=True))
    return tree.sql(dialect=dialect)


def _manifest_select_sql(entries: Sequence[dict[str, Any]], dialect: str) -> str:
    columns = ("kind", "view_name", "feature_key", "feature_version", "id_columns", "columns", "definition_hash")

    def literal(value: Any) -> sge.Expression:
        if isinstance(value, list):
            value = json.dumps(value)
        return sge.cast(sge.Literal.string(value) if value is not None else sge.null(), "TEXT")

    selects = [sge.select(*(literal(entry[column]).as_(column) for column in columns)) for entry in entries]
    if not selects:
        return sge.select(*(literal(None).as_(column) for column in columns)).where(sge.false()).sql(dialect=dialect)
    return functools.reduce(lambda left, right: sge.union(left, right, distinct=False), selects).sql(dialect=dialect)


def _create_view_sql(name: str, select_sql: str, schema: str | None, dialect: str) -> str:
    return sge.Create(
        this=sge.table_(name, db=schema, quoted=True),
        kind="VIEW",
        replace=True,
        expression=sqlglot.parse_one(select_sql, dialect=dialect),
    ).sql(dialect=dialect)
