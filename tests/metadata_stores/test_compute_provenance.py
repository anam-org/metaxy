"""Test compute_provenance method for computing provenance from pre-joined DataFrames.

This file tests the compute_provenance method which allows users to perform
custom joins outside of Metaxy's auto-join system and then compute provenance.
"""

from __future__ import annotations

import narwhals as nw
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from metaxy import BaseFeature, FeatureDep, FeatureGraph
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)

from .conftest import AllStoresCases


@pytest.fixture
def graph() -> FeatureGraph:
    """Create a fresh feature graph for each test."""
    return FeatureGraph()


class TestComputeProvenance:
    """Test compute_provenance method."""

    def test_compute_provenance_single_upstream(self, graph: FeatureGraph):
        """Test computing provenance from a single upstream feature."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["value"],
            ),
        ):
            sample_uid: str

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=["result"],
            ),
        ):
            sample_uid: str

        store = InMemoryMetadataStore()

        # Create upstream metadata with provenance columns
        upstream_df = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "value": [10, 20],
                METAXY_DATA_VERSION_BY_FIELD: [
                    {"value": "hash1"},
                    {"value": "hash2"},
                ],
            }
        )

        # Rename to expected convention for downstream feature
        upstream_key = UpstreamFeature.spec().key
        renamed_col = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream_key.table_name}"
        joined_df = upstream_df.rename({METAXY_DATA_VERSION_BY_FIELD: renamed_col})

        with store:
            result = store.compute_provenance(
                DownstreamFeature, nw.from_native(joined_df)
            )

        # Verify provenance columns were added
        result_pl = result.to_polars()
        assert METAXY_PROVENANCE_BY_FIELD in result_pl.columns
        assert METAXY_PROVENANCE in result_pl.columns
        assert METAXY_DATA_VERSION_BY_FIELD in result_pl.columns
        assert METAXY_DATA_VERSION in result_pl.columns

        # Verify upstream column was dropped
        assert renamed_col not in result_pl.columns

        # Verify data is preserved
        assert result_pl["sample_uid"].to_list() == ["s1", "s2"]
        assert result_pl["value"].to_list() == [10, 20]

    def test_compute_provenance_multiple_upstreams(self, graph: FeatureGraph):
        """Test computing provenance from multiple upstream features."""

        class Upstream1Feature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="upstream1",
                fields=["a"],
            ),
        ):
            sample_uid: str

        class Upstream2Feature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="upstream2",
                fields=["b"],
            ),
        ):
            sample_uid: str

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[
                    FeatureDep(feature=Upstream1Feature),
                    FeatureDep(feature=Upstream2Feature),
                ],
                fields=["result"],
            ),
        ):
            sample_uid: str

        store = InMemoryMetadataStore()

        # Create joined DataFrame with both upstream columns renamed
        upstream1_key = Upstream1Feature.spec().key
        upstream2_key = Upstream2Feature.spec().key
        col1 = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream1_key.table_name}"
        col2 = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream2_key.table_name}"

        joined_df = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "a": [1, 2],
                "b": [10, 20],
                col1: [{"a": "hash_a1"}, {"a": "hash_a2"}],
                col2: [{"b": "hash_b1"}, {"b": "hash_b2"}],
            }
        )

        with store:
            result = store.compute_provenance(
                DownstreamFeature, nw.from_native(joined_df)
            )

        result_pl = result.to_polars()

        # Verify provenance columns were added
        assert METAXY_PROVENANCE_BY_FIELD in result_pl.columns
        assert METAXY_PROVENANCE in result_pl.columns

        # Verify upstream columns were dropped
        assert col1 not in result_pl.columns
        assert col2 not in result_pl.columns

        # Verify data is preserved
        assert result_pl["sample_uid"].to_list() == ["s1", "s2"]
        assert result_pl["a"].to_list() == [1, 2]
        assert result_pl["b"].to_list() == [10, 20]

    def test_compute_provenance_missing_column_raises_error(self, graph: FeatureGraph):
        """Test that missing upstream columns raise a clear error."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["value"],
            ),
        ):
            sample_uid: str

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=["result"],
            ),
        ):
            sample_uid: str

        store = InMemoryMetadataStore()

        # Create DataFrame WITHOUT the required renamed column
        df = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "value": [10, 20],
            }
        )

        with store:
            with pytest.raises(ValueError, match="missing required upstream columns"):
                store.compute_provenance(DownstreamFeature, nw.from_native(df))

    def test_compute_provenance_returns_eager_for_eager_input(
        self, graph: FeatureGraph
    ):
        """Test that compute_provenance returns eager DataFrame for eager input."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["value"],
            ),
        ):
            sample_uid: str

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=["result"],
            ),
        ):
            sample_uid: str

        store = InMemoryMetadataStore()

        upstream_key = UpstreamFeature.spec().key
        renamed_col = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream_key.table_name}"

        # Eager input
        df = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "value": [10],
                renamed_col: [{"value": "hash1"}],
            }
        )

        with store:
            result = store.compute_provenance(DownstreamFeature, nw.from_native(df))

        # Should return eager (not lazy)
        assert isinstance(result, nw.DataFrame)
        result_pl = result.to_polars()
        assert isinstance(result_pl, pl.DataFrame)

    def test_compute_provenance_returns_lazy_for_lazy_input(self, graph: FeatureGraph):
        """Test that compute_provenance returns LazyFrame for lazy input."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["value"],
            ),
        ):
            sample_uid: str

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=["result"],
            ),
        ):
            sample_uid: str

        store = InMemoryMetadataStore()

        upstream_key = UpstreamFeature.spec().key
        renamed_col = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream_key.table_name}"

        # Lazy input
        df = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "value": [10],
                renamed_col: [{"value": "hash1"}],
            }
        ).lazy()

        with store:
            result = store.compute_provenance(DownstreamFeature, nw.from_native(df))

        # Should return lazy
        assert isinstance(result, nw.LazyFrame)

    def test_compute_provenance_integration_with_resolve_update(
        self, graph: FeatureGraph
    ):
        """Test that compute_provenance output works with resolve_update."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["value"],
            ),
        ):
            sample_uid: str

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=["result"],
            ),
        ):
            sample_uid: str

        store = InMemoryMetadataStore()

        upstream_key = UpstreamFeature.spec().key
        renamed_col = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream_key.table_name}"

        # Create pre-joined data with renamed upstream column
        joined_df = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "value": [10, 20],
                renamed_col: [{"value": "hash1"}, {"value": "hash2"}],
            }
        )

        with store:
            # Compute provenance
            with_provenance = store.compute_provenance(
                DownstreamFeature, nw.from_native(joined_df)
            )

            # Use with resolve_update
            increment = store.resolve_update(
                DownstreamFeature,
                samples=with_provenance,
            )

        # Both samples should be added (no existing data)
        assert increment.added.to_polars().height == 2
        assert increment.changed.to_polars().height == 0
        assert increment.removed.to_polars().height == 0

    def test_compute_provenance_root_feature(self, graph: FeatureGraph):
        """Test compute_provenance with a root feature (no upstream dependencies)."""

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["value"],
            ),
        ):
            sample_uid: str

        store = InMemoryMetadataStore()

        # Root features have no upstream deps, so no renamed columns needed
        df = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "value": [10, 20],
            }
        )

        with store:
            result = store.compute_provenance(RootFeature, nw.from_native(df))

        result_pl = result.to_polars()

        # Verify provenance columns were added
        assert METAXY_PROVENANCE_BY_FIELD in result_pl.columns
        assert METAXY_PROVENANCE in result_pl.columns
        assert METAXY_DATA_VERSION_BY_FIELD in result_pl.columns
        assert METAXY_DATA_VERSION in result_pl.columns

        # Data should be preserved
        assert result_pl["sample_uid"].to_list() == ["s1", "s2"]
        assert result_pl["value"].to_list() == [10, 20]


@parametrize_with_cases("store", cases=AllStoresCases)
def test_compute_provenance_eager_frame_computes_correctly(store: MetadataStore):
    """Test that compute_provenance with eager DataFrame computes provenance correctly.

    Verifies:
    1. Return type is DataFrame (preserves eager evaluation)
    2. Provenance columns are correctly computed
    3. Data is preserved
    4. Upstream renamed columns are dropped
    """

    class UpstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="upstream",
            fields=["value"],
        ),
    ):
        sample_uid: str

    class DownstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="downstream",
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=["result"],
        ),
    ):
        sample_uid: str

    upstream_key = UpstreamFeature.spec().key
    renamed_col = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream_key.table_name}"

    # Eager input with multiple rows to verify full computation
    df = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2"],
            "value": [10, 20],
            renamed_col: [{"value": "hash1"}, {"value": "hash2"}],
        }
    )

    with store:
        result = store.compute_provenance(DownstreamFeature, nw.from_native(df))

    # Should return eager
    assert isinstance(result, nw.DataFrame)

    # Verify provenance columns were added
    result_pl = result.to_polars()
    assert METAXY_PROVENANCE_BY_FIELD in result_pl.columns
    assert METAXY_PROVENANCE in result_pl.columns
    assert METAXY_DATA_VERSION_BY_FIELD in result_pl.columns
    assert METAXY_DATA_VERSION in result_pl.columns

    # Verify upstream column was dropped
    assert renamed_col not in result_pl.columns

    # Verify data is preserved
    assert result_pl["sample_uid"].to_list() == ["s1", "s2"]
    assert result_pl["value"].to_list() == [10, 20]

    # Verify provenance was computed for each row (not empty/null)
    provenance_values = result_pl[METAXY_PROVENANCE].to_list()
    assert len(provenance_values) == 2
    assert all(v is not None and v != "" for v in provenance_values)

    # Different upstream hashes should produce different provenance
    assert provenance_values[0] != provenance_values[1]


@parametrize_with_cases("store", cases=AllStoresCases)
def test_compute_provenance_lazy_frame_computes_correctly(store: MetadataStore):
    """Test that compute_provenance with LazyFrame computes provenance correctly.

    Verifies:
    1. Return type is LazyFrame (preserves lazy evaluation)
    2. Provenance columns are correctly computed after collecting
    3. Data is preserved after collecting
    4. Upstream renamed columns are dropped
    """

    class UpstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="upstream",
            fields=["value"],
        ),
    ):
        sample_uid: str

    class DownstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="downstream",
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=["result"],
        ),
    ):
        sample_uid: str

    upstream_key = UpstreamFeature.spec().key
    renamed_col = f"{METAXY_DATA_VERSION_BY_FIELD}__{upstream_key.table_name}"

    # Lazy input with multiple rows to verify full computation
    df = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2"],
            "value": [10, 20],
            renamed_col: [{"value": "hash1"}, {"value": "hash2"}],
        }
    ).lazy()

    with store:
        result = store.compute_provenance(DownstreamFeature, nw.from_native(df))

    # Should return lazy
    assert isinstance(result, nw.LazyFrame)

    # Collect and verify provenance columns were added
    result_pl = result.collect().to_polars()
    assert METAXY_PROVENANCE_BY_FIELD in result_pl.columns
    assert METAXY_PROVENANCE in result_pl.columns
    assert METAXY_DATA_VERSION_BY_FIELD in result_pl.columns
    assert METAXY_DATA_VERSION in result_pl.columns

    # Verify upstream column was dropped
    assert renamed_col not in result_pl.columns

    # Verify data is preserved
    assert result_pl["sample_uid"].to_list() == ["s1", "s2"]
    assert result_pl["value"].to_list() == [10, 20]

    # Verify provenance was computed for each row (not empty/null)
    provenance_values = result_pl[METAXY_PROVENANCE].to_list()
    assert len(provenance_values) == 2
    assert all(v is not None and v != "" for v in provenance_values)

    # Different upstream hashes should produce different provenance
    assert provenance_values[0] != provenance_values[1]
