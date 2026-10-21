"""Tests for SQL views of current rows and feature statistics on Ibis stores with the native versioning engine."""

from __future__ import annotations

from typing import Any

import ibis
import narwhals as nw
import polars as pl
import polars.testing as pl_testing
from metaxy import FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy.ext.ibis.metadata_store import IbisMetadataStore
from metaxy.ext.ibis.views import (
    MANIFEST_VIEW_NAME,
    STATISTICS_VIEW_NAME,
    check_views,
    current_view_name,
    plan_views,
    sync_views,
)
from metaxy.graph.status import get_feature_metadata_status
from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import METAXY_DATA_VERSION, METAXY_PROVENANCE_BY_FIELD, is_system_column
from metaxy.models.feature import FeatureGraph
from metaxy.utils import collect_to_polars
from metaxy_testing.models import SampleFeature, SampleFeatureSpec

from metaxy_testing import add_metaxy_provenance_column

UPSTREAM_KEY = FeatureKey(["views_upstream"])
DOWNSTREAM_KEY = FeatureKey(["views_downstream"])


def _make_graph(downstream_code_version: str = "1") -> tuple[FeatureGraph, Any, Any]:
    graph = FeatureGraph()
    with graph.use():

        class Upstream(
            SampleFeature,
            spec=SampleFeatureSpec(key=UPSTREAM_KEY, fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")]),
        ):
            pass

        class Downstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=DOWNSTREAM_KEY,
                deps=[FeatureDep(feature=Upstream)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version=downstream_code_version)],
            ),
        ):
            pass

    return graph, Upstream, Downstream


def _write_metadata(store: MetadataStore, upstream: Any, downstream: Any) -> None:
    """Write upstream samples 1-4 and downstream samples 1-2.

    Upstream sample 1 is written twice (the second write wins) and sample 4 is soft-deleted,
    so the current upstream rows are samples 1-3. Downstream misses sample 3.
    """
    rows = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3, 4],
            METAXY_PROVENANCE_BY_FIELD: [{"value": f"v{i}"} for i in range(1, 5)],
            "payload": ["old", "b", "c", "d"],
        }
    )
    store.write(upstream, add_metaxy_provenance_column(rows, upstream))
    store.write(
        upstream,
        add_metaxy_provenance_column(
            pl.DataFrame({"sample_uid": [1], METAXY_PROVENANCE_BY_FIELD: [{"value": "v1"}], "payload": ["new"]}),
            upstream,
        ),
    )
    store.delete(upstream, filters=nw.col("sample_uid") == 4)

    store.write(downstream, _missing_downstream_rows(store, downstream).filter(pl.col("sample_uid") <= 2))


def _missing_downstream_rows(store: MetadataStore, downstream: Any) -> pl.DataFrame:
    """Missing downstream samples with their system columns only."""
    new = store.resolve_update(downstream).new.to_polars()
    return new.select("sample_uid", *[column for column in new.columns if is_system_column(column)])


class ViewsTests:
    """Views must return what the Python library reports for the same store and graph."""

    def test_current_view_matches_read(self, store: IbisMetadataStore) -> None:
        graph, upstream, downstream = _make_graph()

        with graph.use(), store.open("w"):
            _write_metadata(store, upstream, downstream)
            plan = plan_views(store, graph, [UPSTREAM_KEY, DOWNSTREAM_KEY])
            sync_views(store, plan)

            view = store.conn.table(current_view_name(store, UPSTREAM_KEY)).to_polars().sort("sample_uid")
            expected = (
                collect_to_polars(store.read(UPSTREAM_KEY))
                .select("sample_uid", "payload", METAXY_DATA_VERSION)
                .sort("sample_uid")
            )

            assert view.columns == ["sample_uid", "payload", METAXY_DATA_VERSION]
            assert view["payload"].to_list() == ["new", "b", "c"]
            pl_testing.assert_frame_equal(view, expected, check_dtypes=False)

    def test_statistics_view_matches_status(self, store: IbisMetadataStore) -> None:
        graph, upstream, downstream = _make_graph()

        with graph.use(), store.open("w"):
            _write_metadata(store, upstream, downstream)
            sync_views(store, plan_views(store, graph, [UPSTREAM_KEY, DOWNSTREAM_KEY]))

            statistics = {
                row["feature_key"]: row for row in store.conn.table(STATISTICS_VIEW_NAME).to_pyarrow().to_pylist()
            }

            assert set(statistics) == {UPSTREAM_KEY.to_string(), DOWNSTREAM_KEY.to_string()}
            for key in (UPSTREAM_KEY, DOWNSTREAM_KEY):
                status = get_feature_metadata_status(key, store).status
                row = statistics[key.to_string()]
                assert row["feature_version"] == graph.get_feature_version(key)
                assert bool(row["is_root_feature"]) == (key == UPSTREAM_KEY)
                assert (row["row_count"], row["missing_count"], row["stale_count"], row["orphaned_count"]) == (
                    status.store_row_count,
                    status.missing_count,
                    status.stale_count,
                    status.orphaned_count,
                )
                assert row["last_write_at"] is not None

            assert statistics[DOWNSTREAM_KEY.to_string()]["missing_count"] == 1

    def test_statistics_view_is_live(self, store: IbisMetadataStore) -> None:
        """Counts reflect writes made after the views were created."""
        graph, upstream, downstream = _make_graph()

        with graph.use(), store.open("w"):
            _write_metadata(store, upstream, downstream)
            sync_views(store, plan_views(store, graph, [UPSTREAM_KEY, DOWNSTREAM_KEY]))
            store.write(downstream, _missing_downstream_rows(store, downstream))

            row = (
                store.conn.table(STATISTICS_VIEW_NAME)
                .filter(ibis._.feature_key == DOWNSTREAM_KEY.to_string())
                .to_pyarrow()
                .to_pylist()
            )
            assert (row[0]["row_count"], row[0]["missing_count"]) == (3, 0)

    def test_check_views_detects_missing_and_outdated_views(self, store: IbisMetadataStore) -> None:
        graph, upstream, downstream = _make_graph()
        keys = [UPSTREAM_KEY, DOWNSTREAM_KEY]

        with graph.use(), store.open("w"):
            _write_metadata(store, upstream, downstream)
            plan = plan_views(store, graph, keys)
            all_views = sorted([*(view.name for view in plan.views), MANIFEST_VIEW_NAME])

            assert check_views(store, plan) == all_views
            sync_views(store, plan)
            assert check_views(store, plan) == []

        changed_graph, _, _ = _make_graph(downstream_code_version="2")
        with changed_graph.use(), store.open("r"):
            changed_plan = plan_views(store, changed_graph, keys)
            assert check_views(store, changed_plan) == sorted(
                [current_view_name(store, DOWNSTREAM_KEY), STATISTICS_VIEW_NAME]
            )

    def test_features_without_table_are_skipped(self, store: IbisMetadataStore) -> None:
        graph, upstream, _ = _make_graph()

        with graph.use(), store.open("w"):
            store.write(
                upstream,
                add_metaxy_provenance_column(
                    pl.DataFrame({"sample_uid": [1], METAXY_PROVENANCE_BY_FIELD: [{"value": "v1"}]}), upstream
                ),
            )
            plan = plan_views(store, graph, [UPSTREAM_KEY, DOWNSTREAM_KEY])

            assert plan.skipped_features == (DOWNSTREAM_KEY,)
            assert [view.name for view in plan.views] == [current_view_name(store, UPSTREAM_KEY), STATISTICS_VIEW_NAME]

    def test_views_in_separate_schema(self, store: IbisMetadataStore) -> None:
        graph, upstream, downstream = _make_graph()

        with graph.use(), store.open("w"):
            _write_metadata(store, upstream, downstream)
            store.conn.create_database("metaxy_views", force=True)  # ty: ignore[unresolved-attribute]
            plan = plan_views(store, graph, [UPSTREAM_KEY, DOWNSTREAM_KEY], schema="metaxy_views")
            sync_views(store, plan)

            assert check_views(store, plan) == []
            assert {*store.conn.list_tables(database="metaxy_views")} >= {
                current_view_name(store, UPSTREAM_KEY),
                STATISTICS_VIEW_NAME,
                MANIFEST_VIEW_NAME,
            }
            view = store.conn.table(current_view_name(store, UPSTREAM_KEY), database="metaxy_views").to_polars()
            assert sorted(view["sample_uid"].to_list()) == [1, 2, 3]

    def test_manifest_view_describes_views(self, store: IbisMetadataStore) -> None:
        graph, upstream, downstream = _make_graph()

        with graph.use(), store.open("w"):
            _write_metadata(store, upstream, downstream)
            plan = plan_views(store, graph, [UPSTREAM_KEY, DOWNSTREAM_KEY])
            sync_views(store, plan)

            rows = {row["view_name"]: row for row in store.conn.table(MANIFEST_VIEW_NAME).to_pyarrow().to_pylist()}
            entries = {entry["view_name"]: entry for entry in plan.manifest_entries}

            assert set(rows) == set(entries)
            upstream_row = rows[current_view_name(store, UPSTREAM_KEY)]
            assert upstream_row["kind"] == "current"
            assert upstream_row["feature_key"] == UPSTREAM_KEY.to_string()
            assert upstream_row["feature_version"] == graph.get_feature_version(UPSTREAM_KEY)
            assert upstream_row["id_columns"] == '["sample_uid"]'
            assert rows[STATISTICS_VIEW_NAME]["feature_key"] is None
