"""Test resolve_update with staleness_predicates parameter.

Records matching any staleness predicate are treated as stale by resolve_update,
regardless of version. Predicates are OR'd and applied against existing store records.

skip_comparison=True takes precedence (no predicate evaluation needed).

Tests use downstream features with a resolve-then-write pattern to establish
properly computed provenance before testing predicate behavior.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import polars as pl
from metaxy_testing import add_metaxy_provenance_column
from metaxy_testing.models import SampleFeature, SampleFeatureSpec

from metaxy import FeatureDep, FeatureGraph
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.types import LazyIncrement


def _setup_upstream_and_downstream(
    store: MetadataStore,
    graph: FeatureGraph,
    downstream_cls: Any,
    upstream_cls: Any,
    upstream_provenance: list[dict[str, str]],
    downstream_extra_columns: dict[str, list[Any]] | None = None,
) -> None:
    """Write upstream, resolve downstream, augment with extra columns, and write downstream."""
    sample_uids = list(range(1, len(upstream_provenance) + 1))

    upstream_data = pl.DataFrame(
        {
            "sample_uid": sample_uids,
            METAXY_PROVENANCE_BY_FIELD: upstream_provenance,
        }
    )
    upstream_data = add_metaxy_provenance_column(upstream_data, upstream_cls)
    store.write(upstream_cls, upstream_data)

    # Resolve downstream to get correctly computed provenance
    inc = store.resolve_update(downstream_cls)
    downstream_df = inc.new.to_polars()

    # Add extra user columns by joining on sample_uid (order-independent)
    if downstream_extra_columns:
        schema_overrides = {
            col: pl.Utf8 if all(v is None for v in values) else None for col, values in downstream_extra_columns.items()
        }
        extra_df = pl.DataFrame(
            {"sample_uid": sample_uids, **downstream_extra_columns},
            schema_overrides={k: v for k, v in schema_overrides.items() if v is not None},
        )
        downstream_df = downstream_df.join(extra_df, on="sample_uid", how="left")

    store.write(downstream_cls, downstream_df)


def _make_features(graph: FeatureGraph) -> tuple[Any, Any]:
    """Define a simple upstream/downstream feature pair within the given graph."""
    with graph.use():

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

    return UpstreamFeature, DownstreamFeature


def test_predicate_marks_matching_records_as_stale(any_store: MetadataStore) -> None:
    """Records matching a staleness predicate should appear in inc.stale."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        _setup_upstream_and_downstream(
            any_store,
            graph,
            DownstreamFeature,
            UpstreamFeature,
            upstream_provenance=[{"value": "up1"}, {"value": "up2"}, {"value": "up3"}],
            downstream_extra_columns={
                "dataset": ["mead", "mead", "voxceleb"],
                "extra": [None, "has_extra", None],
            },
        )

        result = any_store.resolve_update(
            DownstreamFeature,
            staleness_predicates=[
                (nw.col("dataset") == "mead") & nw.col("extra").is_null(),
            ],
        )

        # sample_uid=1: dataset='mead' AND extra IS NULL -> stale
        # sample_uid=2: dataset='mead' but extra='has_extra' -> not stale
        # sample_uid=3: extra IS NULL but dataset='voxceleb' -> not stale
        assert len(result.new) == 0
        assert len(result.orphaned) == 0
        assert sorted(result.stale["sample_uid"].to_list()) == [1]


def test_multiple_predicates_are_ored(any_store: MetadataStore) -> None:
    """Multiple staleness predicates should be OR'd together."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        _setup_upstream_and_downstream(
            any_store,
            graph,
            DownstreamFeature,
            UpstreamFeature,
            upstream_provenance=[{"value": "up1"}, {"value": "up2"}, {"value": "up3"}, {"value": "up4"}],
            downstream_extra_columns={
                "dataset": ["mead", "voxceleb", "mead", "voxceleb"],
                "extra": ["has", None, None, "has"],
            },
        )

        result = any_store.resolve_update(
            DownstreamFeature,
            staleness_predicates=[
                nw.col("dataset") == "mead",
                nw.col("extra").is_null(),
            ],
        )

        # 1: dataset='mead' -> match
        # 2: extra IS NULL -> match
        # 3: dataset='mead' AND extra IS NULL -> match (both)
        # 4: neither -> no match
        assert sorted(result.stale["sample_uid"].to_list()) == [1, 2, 3]


def test_predicates_dont_duplicate_version_stale(any_store: MetadataStore) -> None:
    """Records already stale from version diff shouldn't be duplicated."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        _setup_upstream_and_downstream(
            any_store,
            graph,
            DownstreamFeature,
            UpstreamFeature,
            upstream_provenance=[{"value": "up1"}, {"value": "up2"}],
            downstream_extra_columns={
                "dataset": ["mead", "mead"],
            },
        )

        # Change upstream provenance for sample 1 (version-based stale)
        upstream_data_v2 = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"value": "up1_changed"},
                    {"value": "up2"},
                ],
            }
        )
        upstream_data_v2 = add_metaxy_provenance_column(upstream_data_v2, UpstreamFeature)
        any_store.write(UpstreamFeature, upstream_data_v2)

        result = any_store.resolve_update(
            DownstreamFeature,
            staleness_predicates=[nw.col("dataset") == "mead"],
        )

        # sample 1: stale from version diff AND matches predicate -> appears once
        # sample 2: matches predicate only -> stale from predicate
        assert sorted(result.stale["sample_uid"].to_list()) == [1, 2]


def test_predicates_dont_duplicate_new(any_store: MetadataStore) -> None:
    """Records already in inc.new shouldn't be added to inc.stale by predicates."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        # Write upstream for sample 1 only
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"value": "up1"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamFeature)
        any_store.write(UpstreamFeature, upstream_data)

        # Resolve and write downstream for sample 1 with extra column
        inc = any_store.resolve_update(DownstreamFeature)
        downstream_df = inc.new.to_polars().with_columns(pl.Series("dataset", ["mead"]))
        any_store.write(DownstreamFeature, downstream_df)

        # Now add sample 2 to upstream
        upstream_data_v2 = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [{"value": "up1"}, {"value": "up2"}],
            }
        )
        upstream_data_v2 = add_metaxy_provenance_column(upstream_data_v2, UpstreamFeature)
        any_store.write(UpstreamFeature, upstream_data_v2)

        result = any_store.resolve_update(
            DownstreamFeature,
            staleness_predicates=[nw.col("dataset") == "mead"],
        )

        # sample 2: new (not in downstream store) - should NOT also be in stale
        # sample 1: matches predicate -> stale
        assert result.new["sample_uid"].to_list() == [2]
        assert result.stale["sample_uid"].to_list() == [1]


def test_skip_comparison_bypasses_predicates(any_store: MetadataStore) -> None:
    """skip_comparison=True should take precedence over staleness predicates."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"value": "up1"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamFeature)
        any_store.write(UpstreamFeature, upstream_data)

        inc = any_store.resolve_update(DownstreamFeature)
        downstream_df = inc.new.to_polars().with_columns(pl.Series("needs_reprocess", [1]))
        any_store.write(DownstreamFeature, downstream_df)

        result = any_store.resolve_update(
            DownstreamFeature,
            skip_comparison=True,
            staleness_predicates=[nw.col("needs_reprocess") == 1],
        )

        # Everything in new, nothing in stale (skip_comparison bypasses predicates)
        assert len(result.new) == 1
        assert len(result.stale) == 0


def test_first_run_no_current_metadata(any_store: MetadataStore) -> None:
    """When no current metadata exists, predicates have nothing to match against."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"value": "up1"},
                    {"value": "up2"},
                ],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamFeature)
        any_store.write(UpstreamFeature, upstream_data)

        # No downstream data written - first run
        result = any_store.resolve_update(
            DownstreamFeature,
            staleness_predicates=[nw.col("dataset") == "mead"],
        )

        # First run: all samples are new, no stale
        assert len(result.new) == 2
        assert len(result.stale) == 0


def test_lazy_mode(any_store: MetadataStore) -> None:
    """Test staleness predicates work with lazy=True."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        _setup_upstream_and_downstream(
            any_store,
            graph,
            DownstreamFeature,
            UpstreamFeature,
            upstream_provenance=[{"value": "up1"}, {"value": "up2"}],
            downstream_extra_columns={
                "dataset": ["mead", "voxceleb"],
                "extra": [None, None],
            },
        )

        lazy_result = any_store.resolve_update(
            DownstreamFeature,
            lazy=True,
            staleness_predicates=[
                (nw.col("dataset") == "mead") & nw.col("extra").is_null(),
            ],
        )

        assert isinstance(lazy_result, LazyIncrement)
        result = lazy_result.collect()
        # Only sample 1 matches: dataset='mead' AND extra IS NULL
        assert result.stale["sample_uid"].to_list() == [1]
        assert len(result.new) == 0


def test_global_filters_scope_predicates(any_store: MetadataStore) -> None:
    """Test that global_filters scope the records predicates are evaluated against."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        _setup_upstream_and_downstream(
            any_store,
            graph,
            DownstreamFeature,
            UpstreamFeature,
            upstream_provenance=[{"value": "up1"}, {"value": "up2"}, {"value": "up3"}],
            downstream_extra_columns={
                "dataset": ["mead", "mead", "voxceleb"],
            },
        )

        # Only resolve for sample_uid 1 and 3 via global_filters
        result = any_store.resolve_update(
            DownstreamFeature,
            global_filters=[nw.col("sample_uid").is_in([1, 3])],
            staleness_predicates=[nw.col("dataset") == "mead"],
        )

        # sample 1: matches predicate (dataset='mead') and in filter scope -> stale
        # sample 2: matches predicate but excluded by global_filters
        # sample 3: in filter scope but dataset='voxceleb' -> not stale
        assert result.stale["sample_uid"].to_list() == [1]
        assert len(result.new) == 0


def test_no_predicates_default_behavior(any_store: MetadataStore) -> None:
    """Test that no staleness_predicates (default) doesn't affect behavior."""
    graph = FeatureGraph()
    UpstreamFeature, DownstreamFeature = _make_features(graph)

    with graph.use(), any_store.open("w"):
        _setup_upstream_and_downstream(
            any_store,
            graph,
            DownstreamFeature,
            UpstreamFeature,
            upstream_provenance=[{"value": "up1"}, {"value": "up2"}],
        )

        result = any_store.resolve_update(DownstreamFeature)

        # No changes, no predicates -> nothing stale
        assert len(result.new) == 0
        assert len(result.stale) == 0
        assert len(result.orphaned) == 0
