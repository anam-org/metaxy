"""Test resolve_update behavior across stores and feature graph configurations.

This file tests the BEHAVIOR of resolve_update:
- Root feature handling (requires samples)
- Downstream feature computation (joins, provenance)
- Incremental updates (detecting added/changed/removed samples)
- Lazy execution
- Idempotency

Hash algorithm testing is handled in test_hash_algorithms.py.
Provenance correctness is handled in test_provenance_golden_reference.py.

The goal is to verify resolve_update logic works correctly, not to test
every combination of hash algorithm × store × truncation × feature graph.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import narwhals as nw
import polars as pl
import polars.testing as pl_testing
import pytest
from hypothesis.errors import NonInteractiveExampleWarning
from pytest_cases import parametrize_with_cases

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
)
from metaxy._testing.duckdb_json_compat_store import DuckDBJsonCompatStore
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.parametric import (
    downstream_metadata_strategy,
    feature_metadata_strategy,
)
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.models.plan import FeaturePlan
from metaxy.versioning.types import Increment

if TYPE_CHECKING:
    pass

# Type alias for feature plan output
FeaturePlanOutput = tuple[FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan]


# ============= FEATURE GRAPH CONFIGURATIONS =============


class FeatureGraphCases:
    """Different feature graph topologies for testing."""

    def case_simple_chain(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Simple two-feature chain: Root -> Leaf."""

        class RootFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["value"],
            ),
        ):
            pass

        class LeafFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="leaf",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["result"],
            ),
        ):
            pass

        leaf_plan = graph.get_feature_plan(LeafFeature.spec().key)
        upstream_features = {RootFeature.spec().key: RootFeature}

        return graph, upstream_features, leaf_plan

    def case_diamond_graph(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Diamond dependency graph: Root -> BranchA,BranchB -> Leaf."""

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["value"],
            ),
        ):
            pass

        class BranchAFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="branch_a",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["a_result"],
            ),
        ):
            pass

        class BranchBFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="branch_b",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["b_result"],
            ),
        ):
            pass

        class LeafFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="leaf",
                deps=[
                    FeatureDep(feature=BranchAFeature),
                    FeatureDep(feature=BranchBFeature),
                ],
                fields=["final_result"],
            ),
        ):
            pass

        leaf_plan = graph.get_feature_plan(LeafFeature.spec().key)
        upstream_features = {
            RootFeature.spec().key: RootFeature,
            BranchAFeature.spec().key: BranchAFeature,
            BranchBFeature.spec().key: BranchBFeature,
        }

        return graph, upstream_features, leaf_plan

    def case_multi_field(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Features with multiple fields to test field-level provenance."""

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["field_a", "field_b", "field_c"],
            ),
        ):
            pass

        class LeafFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="leaf",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["result_x", "result_y"],
            ),
        ):
            pass

        leaf_plan = graph.get_feature_plan(LeafFeature.spec().key)
        upstream_features = {RootFeature.spec().key: RootFeature}

        return graph, upstream_features, leaf_plan

    def case_optional_dependency(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Feature with optional dependency (tests left join behavior)."""

        class RequiredParent(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="required_parent",
                fields=["req_field"],
            ),
        ):
            sample_uid: str

        class OptionalParent(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="optional_parent",
                fields=["opt_field"],
            ),
        ):
            sample_uid: str

        class ChildFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="child",
                deps=[
                    FeatureDep(feature=RequiredParent, optional=False),
                    FeatureDep(feature=OptionalParent, optional=True),
                ],
                fields=["result"],
            ),
        ):
            pass

        child_plan = graph.get_feature_plan(ChildFeature.spec().key)
        upstream_features = {
            RequiredParent.spec().key: RequiredParent,
            OptionalParent.spec().key: OptionalParent,
        }

        return graph, upstream_features, child_plan


class RootFeatureCases:
    """Root features (no upstream dependencies) for testing sample-based resolve_update."""

    def case_simple_root(self, graph: FeatureGraph) -> type[BaseFeature]:
        class SimpleRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="simple_root",
                fields=["value"],
            ),
        ):
            pass

        return SimpleRoot

    def case_multi_field_root(self, graph: FeatureGraph) -> type[BaseFeature]:
        class MultiFieldRoot(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="multi_root",
                fields=["field_a", "field_b", "field_c"],
            ),
        ):
            pass

        return MultiFieldRoot


# Removed: StoreCases with hash algorithm parametrization
# Removed: TruncationCases and metaxy_config fixture
# Hash algorithm and truncation testing is handled in test_hash_algorithms.py


# ============= TEST: ROOT FEATURES (NO UPSTREAM) =============


@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def test_resolve_update_root_feature_requires_samples(
    default_store: MetadataStore,
    root_feature: type[BaseFeature],
):
    """Test that resolve_update raises ValueError for root features without samples."""
    store = default_store
    with store:
        with pytest.raises(ValueError, match="root feature"):
            store.resolve_update(root_feature, lazy=True)


@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def test_resolve_update_root_feature_with_samples(
    any_store: MetadataStore,
    root_feature: type[BaseFeature],
    graph: FeatureGraph,
):
    """Test resolve_update for root features with provided samples."""
    store = any_store
    # Generate sample data using the parametric strategy
    feature_spec = root_feature.spec()
    feature_version = root_feature.feature_version()
    snapshot_version = graph.snapshot_version

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        samples_df = feature_metadata_strategy(
            feature_spec=feature_spec,
            feature_version=feature_version,
            snapshot_version=snapshot_version,
            min_rows=5,
            max_rows=10,
        ).example()

    # Select only required columns for samples (drop system columns that will be added)
    id_columns = list(feature_spec.id_columns)
    samples_df = samples_df.select(id_columns + ["metaxy_provenance_by_field"])

    # Call resolve_update with samples - pass native Polars LazyFrame, not Narwhals-wrapped
    with store, graph.use():
        try:
            increment = store.resolve_update(root_feature, samples=samples_df.lazy(), lazy=True).collect()
        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")

        # Verify all samples are added (first write)
        assert len(increment.added) == len(samples_df)
        assert len(increment.changed) == 0
        assert len(increment.removed) == 0

        # Verify provenance_by_field structure matches input
        added_df = increment.added.lazy().collect().to_polars().sort(id_columns)
        samples_sorted = samples_df.sort(id_columns)

        pl_testing.assert_series_equal(
            added_df["metaxy_provenance_by_field"],
            samples_sorted["metaxy_provenance_by_field"],
            check_names=False,
        )


def test_polars_fallback_reconstructs_structs_for_json_compat_samples(
    tmp_path: Path,
    graph: FeatureGraph,
) -> None:
    class JsonCompatRoot(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="json_compat_root",
            fields=["value"],
        ),
    ):
        pass

    store = DuckDBJsonCompatStore(database=str(tmp_path / "json_compat.duckdb"))
    samples = pl.DataFrame(
        {
            "sample_uid": [1, 2],
            "metaxy_provenance_by_field__value": ["hash_1", "hash_2"],
        }
    )

    with store, graph.use():
        increment = store.resolve_update(
            JsonCompatRoot,
            samples=samples,
            versioning_engine="polars",
        )

    added_df = increment.added.to_native()
    assert isinstance(added_df, pl.DataFrame)
    assert isinstance(added_df.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
    pl_testing.assert_series_equal(
        added_df[METAXY_PROVENANCE_BY_FIELD].struct.field("value"),
        samples["metaxy_provenance_by_field__value"],
        check_names=False,
    )


# ============= TEST: DOWNSTREAM FEATURES (WITH UPSTREAM) =============


@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_downstream_feature(
    any_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
):
    """Test resolve_update for downstream features with upstream dependencies."""
    store = any_store
    graph, upstream_features, child_plan = feature_plan_config

    # Get feature versions
    child_key = child_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    feature_versions = {
        feat_key.to_string(): feat_class.feature_version() for feat_key, feat_class in upstream_features.items()
    }
    feature_versions[child_key.to_string()] = child_version

    # Generate test data using golden reference strategy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=5,
            max_rows=15,
        ).example()

    # Write upstream metadata to store
    with store, graph.use():
        try:
            # Write all upstream data (includes transitive dependencies)
            for feat_key_str, upstream_df in upstream_data.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Call resolve_update to compute child metadata
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )
        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")

        # Get computed metadata
        added_df = increment.added.lazy().collect().to_polars()

        # Sort both for comparison
        id_columns = list(child_plan.feature.id_columns)
        added_sorted = added_df.sort(id_columns)
        golden_sorted = golden_downstream.sort(id_columns)

        # Compare provenance columns (exclude metaxy_created_at since it's a timestamp)
        from metaxy.models.constants import METAXY_CREATED_AT

        common_columns = [
            col for col in added_sorted.columns if col in golden_sorted.columns and col != METAXY_CREATED_AT
        ]
        added_selected = added_sorted.select(common_columns)
        golden_selected = golden_sorted.select(common_columns)

        pl_testing.assert_frame_equal(
            added_selected,
            golden_selected,
            check_row_order=True,
            check_column_order=False,
        )


# ============= TEST: INCREMENTAL UPDATES =============


@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_detects_changes(
    any_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
):
    """Test that resolve_update correctly detects added/changed/removed samples."""
    store = any_store
    graph, upstream_features, child_plan = feature_plan_config

    # Get feature versions
    child_key = child_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    feature_versions = {
        feat_key.to_string(): feat_class.feature_version() for feat_key, feat_class in upstream_features.items()
    }
    feature_versions[child_key.to_string()] = child_version

    # Generate initial test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        initial_upstream, initial_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=10,
            max_rows=10,
        ).example()

    # Write initial data
    with store, graph.use():
        try:
            # Write all upstream data (includes transitive dependencies)
            for feat_key_str, upstream_df in initial_upstream.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Write initial child metadata
            store.write_metadata(ChildFeature, initial_downstream)

            # Generate new upstream data (simulating changes)
            # The easiest way to test change detection is to generate completely new data
            modified_upstream, modified_downstream = downstream_metadata_strategy(
                child_plan,
                feature_versions=feature_versions,
                snapshot_version=graph.snapshot_version,
                hash_algorithm=store.hash_algorithm,
                min_rows=8,  # Different number of rows
                max_rows=8,
            ).example()

            # Write modified upstream data
            for feat_key_str, upstream_df in modified_upstream.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Call resolve_update - should detect all changes
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")

        # Verify changes were detected
        # With completely new data (8 samples vs 10 initial), we expect:
        # - Some samples added (new ones)
        # - Some samples removed (old ones not in new data)
        # - Possibly some changed (if IDs overlap but provenance differs)

        total_changes = len(increment.added) + len(increment.changed) + len(increment.removed)
        assert total_changes > 0, "Expected resolve_update to detect some changes when upstream data changes"


# ============= TEST: LAZY EXECUTION =============


def test_resolve_update_lazy_execution(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test resolve_update with lazy=True returns lazy frames with correct implementation."""
    store = any_store

    # Create a feature graph with multiple parents (realistic scenario)
    class Parent1(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="parent1",
            fields=["field_a", "field_b"],
        ),
    ):
        pass

    class Parent2(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="parent2",
            fields=["field_c"],
        ),
    ):
        pass

    class Child(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[
                FeatureDep(feature=Parent1),
                FeatureDep(feature=Parent2),
            ],
            fields=["result_x", "result_y"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(Child.spec().key)

    # Get feature versions
    feature_versions = {
        "parent1": Parent1.feature_version(),
        "parent2": Parent2.feature_version(),
        "child": Child.feature_version(),
    }

    # Generate test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=10,
            max_rows=10,
        ).example()

    with store:
        try:
            # Write upstream metadata
            for feat_key_str, upstream_df in upstream_data.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Call resolve_update with lazy=True
            lazy_increment = store.resolve_update(
                Child,
                target_version=Child.feature_version(),
                snapshot_version=graph.snapshot_version,
                lazy=True,
            )

            # Verify we got a LazyIncrement
            from metaxy.versioning.types import LazyIncrement

            assert isinstance(lazy_increment, LazyIncrement), (
                f"Expected LazyIncrement with lazy=True, got {type(lazy_increment)}"
            )

            # Verify lazy frames have correct implementation
            expected_impl = store.native_implementation()

            for frame_name in ["added", "changed", "removed"]:
                lazy_frame = getattr(lazy_increment, frame_name)
                actual_impl = lazy_frame.implementation

                assert actual_impl == expected_impl, (
                    f"Expected {frame_name} to have implementation {expected_impl}, "
                    f"but got {actual_impl} for store type {type(store).__name__}"
                )

            # Collect lazy result
            eager_increment = lazy_increment.collect()

            # Also get eager result for comparison
            eager_increment_direct = store.resolve_update(
                Child,
                target_version=Child.feature_version(),
                snapshot_version=graph.snapshot_version,
                lazy=False,
            )

            # Verify both approaches produce same results
            id_columns = list(child_plan.feature.id_columns)

            # Compare added frames
            lazy_added = eager_increment.added.lazy().collect().to_polars().sort(id_columns)
            eager_added = eager_increment_direct.added.lazy().collect().to_polars().sort(id_columns)

            pl_testing.assert_frame_equal(lazy_added, eager_added)

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")


# ============= TEST: IDEMPOTENCY =============


@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_idempotency(
    any_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
):
    """Test that calling resolve_update multiple times is idempotent."""
    store = any_store
    graph, upstream_features, child_plan = feature_plan_config

    child_key = child_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    feature_versions = {
        feat_key.to_string(): feat_class.feature_version() for feat_key, feat_class in upstream_features.items()
    }
    feature_versions[child_key.to_string()] = child_version

    # Generate test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=5,
            max_rows=10,
        ).example()

    with store, graph.use():
        try:
            # Write all upstream data (includes transitive dependencies)
            for feat_key_str, upstream_df in upstream_data.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # First resolve_update
            increment1 = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

            assert len(increment1.added) > 0, "Expected resolve_update to detect added samples"

            # Write the increment
            added_df = increment1.added.lazy().collect().to_polars()
            store.write_metadata(ChildFeature, added_df)

            # Second resolve_update - should be empty
            increment2 = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")

        # Verify second call is idempotent
        assert len(increment2.added) == 0, "Second resolve_update should have no added samples"
        assert len(increment2.changed) == 0, "Second resolve_update should have no changed samples"
        assert len(increment2.removed) == 0, "Second resolve_update should have no removed samples"


# ============= TEST: FILTER KEY TYPES =============


def test_resolve_update_filters_with_feature_class_key(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that resolve_update accepts feature classes as filter keys."""
    import polars as pl

    store = default_store

    class UpstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="upstream",
            fields=["value"],
        ),
    ):
        pass

    class DownstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="downstream",
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=["result"],
        ),
    ):
        pass

    # Create upstream metadata with different values
    upstream_df = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2", "s3"],
            "value": [10, 20, 30],
            "metaxy_provenance_by_field": [
                {"value": "h1"},
                {"value": "h2"},
                {"value": "h3"},
            ],
        }
    )

    with store, graph.use():
        store.write_metadata(UpstreamFeature, upstream_df)

        # Use feature class as filter key - should filter upstream to only value > 15
        increment = store.resolve_update(
            DownstreamFeature,
            filters={UpstreamFeature: [nw.col("value") > 15]},
        )

        # Should only get 2 samples (s2 and s3 where value > 15)
        assert len(increment.added) == 2
        added_df = increment.added.lazy().collect().to_polars()
        assert set(added_df["sample_uid"].to_list()) == {"s2", "s3"}


def test_resolve_update_filters_with_feature_key_object(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that resolve_update accepts FeatureKey objects as filter keys."""
    import polars as pl

    store = default_store

    class UpstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="upstream_fk",
            fields=["value"],
        ),
    ):
        pass

    class DownstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="downstream_fk",
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=["result"],
        ),
    ):
        pass

    # Create upstream metadata
    upstream_df = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2", "s3", "s4"],
            "value": [5, 15, 25, 35],
            "metaxy_provenance_by_field": [
                {"value": "h1"},
                {"value": "h2"},
                {"value": "h3"},
                {"value": "h4"},
            ],
        }
    )

    with store, graph.use():
        store.write_metadata(UpstreamFeature, upstream_df)

        # Use FeatureKey as filter key
        upstream_key = FeatureKey(["upstream_fk"])
        increment = store.resolve_update(
            DownstreamFeature,
            filters={upstream_key: [nw.col("value") >= 15, nw.col("value") < 30]},
        )

        # Should get 2 samples (s2 and s3 where 15 <= value < 30)
        assert len(increment.added) == 2
        added_df = increment.added.lazy().collect().to_polars()
        assert set(added_df["sample_uid"].to_list()) == {"s2", "s3"}


def test_resolve_update_global_filters(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that resolve_update applies global_filters to all features."""
    import polars as pl

    store = default_store

    class UpstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="upstream_gf",
            fields=["value"],
        ),
    ):
        pass

    class DownstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="downstream_gf",
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=["result"],
        ),
    ):
        pass

    # Create upstream metadata
    upstream_df = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2", "s3", "s4"],
            "value": [10, 20, 30, 40],
            "metaxy_provenance_by_field": [
                {"value": "h1"},
                {"value": "h2"},
                {"value": "h3"},
                {"value": "h4"},
            ],
        }
    )

    with store, graph.use():
        store.write_metadata(UpstreamFeature, upstream_df)

        # Use global_filters to filter by sample_uid across all features
        increment = store.resolve_update(
            DownstreamFeature,
            global_filters=[nw.col("sample_uid").is_in(["s2", "s3"])],
        )

        # Should only get 2 samples (s2 and s3)
        assert len(increment.added) == 2
        added_df = increment.added.lazy().collect().to_polars()
        assert set(added_df["sample_uid"].to_list()) == {"s2", "s3"}


def test_resolve_update_global_filters_combined_with_filters(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that global_filters are combined with feature-specific filters."""
    import polars as pl

    store = default_store

    class UpstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="upstream_gfc",
            fields=["value"],
        ),
    ):
        pass

    class DownstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="downstream_gfc",
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=["result"],
        ),
    ):
        pass

    # Create upstream metadata
    upstream_df = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2", "s3", "s4", "s5"],
            "value": [10, 20, 30, 40, 50],
            "metaxy_provenance_by_field": [
                {"value": "h1"},
                {"value": "h2"},
                {"value": "h3"},
                {"value": "h4"},
                {"value": "h5"},
            ],
        }
    )

    with store, graph.use():
        store.write_metadata(UpstreamFeature, upstream_df)

        # Use both global_filters and feature-specific filters
        # global_filters: sample_uid in [s2, s3, s4, s5]
        # feature filter: value < 40
        # Combined result should be s2, s3 (in both sets AND value < 40)
        increment = store.resolve_update(
            DownstreamFeature,
            filters={UpstreamFeature: [nw.col("value") < 40]},
            global_filters=[nw.col("sample_uid").is_in(["s2", "s3", "s4", "s5"])],
        )

        # Should get 2 samples: s2 (value=20) and s3 (value=30)
        # s4 has value=40 which fails "value < 40"
        # s5 has value=50 which fails "value < 40"
        assert len(increment.added) == 2
        added_df = increment.added.lazy().collect().to_polars()
        assert set(added_df["sample_uid"].to_list()) == {"s2", "s3"}


# ============= TEST: EXPANSION LINEAGE WITH MULTIPLE MATERIALIZATIONS =============


def test_expansion_lineage_multiple_writes_with_resolve_update(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that resolve_update orphaned count is correct after multiple writes with expansion lineage.

    This test simulates a realistic scenario where:
    1. A user writes frames multiple times using resolve_update -> write_metadata
    2. Each write might add different frames for the same parent videos
    3. When checking status, orphaned should only count PARENT-level removals

    The user reported: 151152 materialized, 755755 orphaned
    This was likely caused by expansion lineage not properly grouping current
    metadata by parent columns when calculating orphaned.
    """
    import polars as pl

    from metaxy import FieldKey, FieldSpec
    from metaxy.models.lineage import LineageRelationship

    class Video(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["resolution"]), code_version="1"),
            ],
        ),
    ):
        pass

    class VideoFrames(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video_frames"]),
            id_columns=("video_id", "frame_id"),
            deps=[
                FeatureDep(
                    feature=Video,
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                )
            ],
            fields=[
                FieldSpec(key=FieldKey(["embedding"]), code_version="1"),
            ],
        ),
    ):
        pass

    store = default_store

    with store, graph.use():
        # Write upstream video metadata for 2 videos
        video_data = pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "resolution": ["1080p", "720p"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_hash_1"},
                    {"resolution": "res_hash_2"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
            }
        )
        store.write_metadata(Video, video_data)

        import narwhals as nw

        # First write: manually create frames for the first batch (since we need to compute them)
        # In reality, user would compute frames and then write them
        first_batch = pl.DataFrame(
            {
                "video_id": ["v1", "v1", "v2", "v2"],
                "frame_id": [0, 1, 0, 1],
                "embedding": ["emb_v1_0", "emb_v1_1", "emb_v2_0", "emb_v2_1"],
            }
        )
        # Join with upstream to get proper data_version columns
        upstream_df = store.read_metadata(Video).collect().to_polars()
        first_batch_joined = first_batch.join(
            upstream_df.select(["video_id", "metaxy_data_version_by_field", "metaxy_provenance"]).rename(
                {
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__video",
                    "metaxy_provenance": "__upstream_provenance",
                }
            ),
            on="video_id",
        )
        # Use compute_provenance to get proper provenance from upstream
        first_batch_with_prov = store.compute_provenance(VideoFrames, nw.from_native(first_batch_joined))
        store.write_metadata(VideoFrames, first_batch_with_prov)

        # Second write: add more frames (simulating re-running the pipeline)
        second_batch = pl.DataFrame(
            {
                "video_id": ["v1", "v1", "v2"],
                "frame_id": [2, 3, 2],  # New frame IDs
                "embedding": ["emb_v1_2", "emb_v1_3", "emb_v2_2"],
            }
        )
        second_batch_joined = second_batch.join(
            upstream_df.select(["video_id", "metaxy_data_version_by_field", "metaxy_provenance"]).rename(
                {
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__video",
                    "metaxy_provenance": "__upstream_provenance",
                }
            ),
            on="video_id",
        )
        second_batch_with_prov = store.compute_provenance(VideoFrames, nw.from_native(second_batch_joined))
        store.write_metadata(VideoFrames, second_batch_with_prov)

        # Now we have 7 total frames in the store:
        # v1: frames 0, 1, 2, 3 (4 frames)
        # v2: frames 0, 1, 2 (3 frames)

        # Call resolve_update - upstream hasn't changed, so there should be:
        # - 0 added (no new videos, but we'll see "added" for frames to process)
        # - Changed frames might exist if provenance differs
        # - 0 orphaned at the VIDEO level (both v1 and v2 still exist upstream)
        increment = store.resolve_update(VideoFrames, lazy=False)

        # With expansion lineage:
        # - The comparison happens at the parent (video_id) level
        # - Both v1 and v2 exist in upstream
        # - So orphaned (removed) should be 0 at the parent level
        removed_count = len(increment.removed)

        # If this fails with a large removed_count, it means expansion lineage
        # is not properly grouping current metadata before comparison
        assert removed_count == 0, (
            f"Expected 0 orphaned (no videos removed from upstream), "
            f"but got {removed_count}. This suggests expansion lineage is counting "
            f"at the frame level instead of the parent video level."
        )


def test_expansion_lineage_orphaned_when_upstream_removed(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test orphaned count when upstream parents are removed.

    This test verifies that when upstream videos are removed (or filtered out),
    the orphaned count reflects the number of PARENTS removed, not the number
    of child frames for those parents.

    The user reported: 151152 materialized, 755755 orphaned
    If the user had 5x more parent videos in their current frames than in the
    filtered upstream, and each parent had ~5 frames on average, this could
    explain the discrepancy.
    """
    import polars as pl

    from metaxy import FieldKey, FieldSpec
    from metaxy.models.lineage import LineageRelationship

    class Video(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["resolution"]), code_version="1"),
            ],
        ),
    ):
        pass

    class VideoFrames(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video_frames"]),
            id_columns=("video_id", "frame_id"),
            deps=[
                FeatureDep(
                    feature=Video,
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                )
            ],
            fields=[
                FieldSpec(key=FieldKey(["embedding"]), code_version="1"),
            ],
        ),
    ):
        pass

    store = default_store

    with store, graph.use():
        import narwhals as nw

        # Write upstream video metadata for 5 videos
        video_data = pl.DataFrame(
            {
                "video_id": ["v1", "v2", "v3", "v4", "v5"],
                "resolution": ["1080p", "720p", "4K", "1080p", "720p"],
                "metaxy_provenance_by_field": [{"resolution": f"res_hash_{i}"} for i in range(1, 6)],
                "metaxy_provenance": [f"video_prov_{i}" for i in range(1, 6)],
            }
        )
        store.write_metadata(Video, video_data)

        # Write frames for ALL 5 videos (10 frames each = 50 total frames)
        frame_rows = []
        for vid in ["v1", "v2", "v3", "v4", "v5"]:
            for fid in range(10):
                frame_rows.append(
                    {
                        "video_id": vid,
                        "frame_id": fid,
                        "embedding": f"emb_{vid}_{fid}",
                    }
                )
        frames_df = pl.DataFrame(frame_rows)

        # Join with upstream to compute proper provenance
        upstream_df = store.read_metadata(Video).collect().to_polars()
        frames_joined = frames_df.join(
            upstream_df.select(["video_id", "metaxy_data_version_by_field", "metaxy_provenance"]).rename(
                {
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__video",
                    "metaxy_provenance": "__upstream_provenance",
                }
            ),
            on="video_id",
        )
        frames_with_prov = store.compute_provenance(VideoFrames, nw.from_native(frames_joined))
        store.write_metadata(VideoFrames, frames_with_prov)

        # Now REMOVE 3 videos from upstream (simulating upstream data change)
        # This is like the user's scenario where upstream was filtered/changed
        store.drop_feature_metadata(Video)
        reduced_video_data = pl.DataFrame(
            {
                "video_id": ["v1", "v2"],  # Only keep v1 and v2
                "resolution": ["1080p", "720p"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_hash_1"},
                    {"resolution": "res_hash_2"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
            }
        )
        store.write_metadata(Video, reduced_video_data)

        # Check status - current has frames for 5 videos, upstream only has 2
        # Expected: 3 orphaned PARENTS (v3, v4, v5)
        # Bug: might show 30 orphaned (all 30 frames for v3, v4, v5)
        increment = store.resolve_update(VideoFrames, lazy=False)

        removed_count = len(increment.removed)

        # With expansion lineage, orphaned should be at PARENT level
        # 3 videos were removed from upstream (v3, v4, v5)
        # So orphaned should be 3, not 30
        assert removed_count == 3, (
            f"Expected 3 orphaned (3 parent videos removed from upstream: v3, v4, v5), "
            f"but got {removed_count}. "
            f"{'This is correct!' if removed_count == 3 else ''}"
            f"{'Bug: counting at frame level instead of parent level.' if removed_count > 3 else ''}"
        )


def test_expansion_lineage_orphaned_with_duplicate_writes(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test orphaned count when there are duplicate writes with expansion lineage.

    This test reproduces the user's scenario where they wrote to the feature
    multiple times (without deduplication) and see incorrect orphaned counts.

    The issue: resolve_update uses read_metadata_in_store (no deduplication)
    while get_feature_metadata_status.store_row_count uses read_metadata
    (which has latest_only=True deduplication).

    This can cause orphaned_count to be larger than store_row_count if there
    are duplicate writes with the same id_columns.
    """
    import polars as pl

    from metaxy import FieldKey, FieldSpec
    from metaxy.models.lineage import LineageRelationship

    class Video(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["resolution"]), code_version="1"),
            ],
        ),
    ):
        pass

    class VideoFrames(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video_frames"]),
            id_columns=("video_id", "frame_id"),
            deps=[
                FeatureDep(
                    feature=Video,
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                )
            ],
            fields=[
                FieldSpec(key=FieldKey(["embedding"]), code_version="1"),
            ],
        ),
    ):
        pass

    store = default_store

    with store, graph.use():
        import narwhals as nw

        # Write upstream video metadata for 2 videos
        video_data = pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "resolution": ["1080p", "720p"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_hash_1"},
                    {"resolution": "res_hash_2"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
            }
        )
        store.write_metadata(Video, video_data)

        # Write frames for both videos - FIRST WRITE
        first_batch = pl.DataFrame(
            {
                "video_id": ["v1", "v1", "v2", "v2"],
                "frame_id": [0, 1, 0, 1],
                "embedding": ["emb_v1_0", "emb_v1_1", "emb_v2_0", "emb_v2_1"],
            }
        )
        upstream_df = store.read_metadata(Video).collect().to_polars()
        first_batch_joined = first_batch.join(
            upstream_df.select(["video_id", "metaxy_data_version_by_field", "metaxy_provenance"]).rename(
                {
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__video",
                    "metaxy_provenance": "__upstream_provenance",
                }
            ),
            on="video_id",
        )
        first_batch_with_prov = store.compute_provenance(VideoFrames, nw.from_native(first_batch_joined))
        store.write_metadata(VideoFrames, first_batch_with_prov)

        # Write SAME frames again - SECOND WRITE (duplicate rows)
        # This simulates user writing multiple times without deduplication
        store.write_metadata(VideoFrames, first_batch_with_prov)

        # Write SAME frames a THIRD time
        store.write_metadata(VideoFrames, first_batch_with_prov)

        # Now the store has 12 rows (4 frames x 3 writes)
        # But read_metadata with latest_only=True should show 4 rows
        # Let's verify this
        current_deduplicated = store.read_metadata(VideoFrames).collect()
        current_all = store.read_metadata(VideoFrames, latest_only=False).collect()

        assert len(current_deduplicated) == 4, f"Expected 4 deduplicated rows, got {len(current_deduplicated)}"
        assert len(current_all) == 12, f"Expected 12 total rows (with duplicates), got {len(current_all)}"

        # Now check resolve_update - it uses read_metadata_in_store (no dedup)
        # So it might see more rows than expected
        increment = store.resolve_update(VideoFrames, lazy=False)

        # With expansion lineage and proper grouping:
        # - current is grouped by video_id, so we should only see 2 "groups" (v1, v2)
        # - expected also has 2 videos (v1, v2)
        # - orphaned should be 0 (both videos exist in both)
        removed_count = len(increment.removed)

        # BUG HYPOTHESIS: If resolve_update doesn't deduplicate, the grouping
        # might still work (first distinct per video_id), but let's verify
        assert removed_count == 0, (
            f"Expected 0 orphaned (all videos exist in upstream), "
            f"but got {removed_count}. This might indicate that resolve_update "
            f"is not handling duplicate rows correctly."
        )


def test_identity_lineage_orphaned_with_multiple_writes_no_dedup(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test orphaned count with identity lineage when writing multiple times without deduplication.

    This reproduces the user's scenario where:
    1. A2MRecords feature has identity lineage (1:1 with chunk_id)
    2. User materializes multiple times, writing same chunks repeatedly
    3. Store uses append mode (no deduplication)
    4. When checking status, orphaned count is inflated

    The bug: resolve_update uses read_metadata_in_store (no deduplication by default)
    while metadata status store_row_count uses read_metadata with latest_only=True.

    User reported: 151152 materialized, 755755 orphaned
    This suggests ~5x more rows in the store than unique samples.
    """
    import polars as pl

    from metaxy import FieldKey, FieldSpec

    class Upstream(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
            id_columns=("chunk_id",),
            fields=[
                FieldSpec(key=FieldKey(["video_path"]), code_version="1"),
            ],
        ),
    ):
        pass

    class Downstream(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("chunk_id",),
            deps=[FeatureDep(feature=Upstream)],
            fields=[
                FieldSpec(key=FieldKey(["result"]), code_version="1"),
            ],
            # Identity lineage (default) - 1:1 relationship
        ),
    ):
        pass

    store = default_store

    with store, graph.use():
        import narwhals as nw

        # Write upstream metadata for 10 chunks
        upstream_data = pl.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(10)],
                "video_path": [f"/path/to/video_{i}.mp4" for i in range(10)],
                "metaxy_provenance_by_field": [{"video_path": f"path_hash_{i}"} for i in range(10)],
                "metaxy_provenance": [f"upstream_prov_{i}" for i in range(10)],
            }
        )
        store.write_metadata(Upstream, upstream_data)

        # First materialization: write downstream for all 10 chunks
        downstream_batch = pl.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(10)],
                "result": [f"result_{i}" for i in range(10)],
            }
        )
        upstream_df = store.read_metadata(Upstream).collect().to_polars()
        downstream_joined = downstream_batch.join(
            upstream_df.select(["chunk_id", "metaxy_data_version_by_field", "metaxy_provenance"]).rename(
                {
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__upstream",
                    "metaxy_provenance": "__upstream_provenance",
                }
            ),
            on="chunk_id",
        )
        downstream_with_prov = store.compute_provenance(Downstream, nw.from_native(downstream_joined))
        store.write_metadata(Downstream, downstream_with_prov)

        # Write SAME data 4 more times (5 total writes, like the user scenario)
        for _ in range(4):
            store.write_metadata(Downstream, downstream_with_prov)

        # Now we have 50 rows (10 chunks x 5 writes) in the store
        # But read_metadata with latest_only=True should show 10 rows
        current_deduplicated = store.read_metadata(Downstream).collect()
        current_all = store.read_metadata(Downstream, latest_only=False).collect()

        assert len(current_deduplicated) == 10, f"Expected 10 deduplicated rows, got {len(current_deduplicated)}"
        assert len(current_all) == 50, f"Expected 50 total rows (with duplicates), got {len(current_all)}"

        # Now check resolve_update
        increment = store.resolve_update(Downstream, lazy=False)

        # With identity lineage:
        # - Each chunk should match 1:1 with upstream
        # - All 10 chunks exist in both current and upstream
        # - orphaned should be 0
        removed_count = len(increment.removed)

        # BUG: If resolve_update sees 50 rows (not deduplicated) and compares
        # with 10 expected rows, it might show 40 orphaned (50-10=40)
        # OR if provenance doesn't match, it could show even more
        assert removed_count == 0, (
            f"Expected 0 orphaned (all chunks exist in upstream), "
            f"but got {removed_count}. This indicates resolve_update is seeing "
            f"duplicate rows from multiple writes and not deduplicating them."
        )


def test_resolve_update_deduplicates_current_metadata_delta_store(
    tmp_path,
    graph: FeatureGraph,
):
    """Regression test: resolve_update must deduplicate current metadata when using append-mode stores.

    This test reproduces the exact bug reported by the user:
    - User materialized training/a2m feature 5+ times
    - DeltaMetadataStore uses append mode (no automatic deduplication)
    - 151152 materialized rows became 755755+ total rows (with duplicates)
    - resolve_update saw all rows (not deduplicated) and reported 755755 orphaned

    The fix: resolve_update now uses read_metadata() with latest_only=True
    instead of read_metadata_in_store() which doesn't deduplicate.

    This test MUST use DeltaMetadataStore because:
    - DeltaMetadataStore uses append mode, preserving duplicates
    """
    import polars as pl

    from metaxy import FieldKey, FieldSpec
    from metaxy.metadata_store.delta import DeltaMetadataStore

    class Upstream(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["delta_dedup_test", "upstream"]),
            id_columns=("chunk_id",),
            fields=[
                FieldSpec(key=FieldKey(["video_path"]), code_version="1"),
            ],
        ),
    ):
        pass

    class Downstream(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["delta_dedup_test", "downstream"]),
            id_columns=("chunk_id",),
            deps=[FeatureDep(feature=Upstream)],
            fields=[
                FieldSpec(key=FieldKey(["result"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Use DeltaMetadataStore which has append mode by default
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")

    with store.open(mode="write"), graph.use():
        import narwhals as nw

        # Write upstream metadata for 10 chunks
        upstream_data = pl.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(10)],
                "video_path": [f"/path/to/video_{i}.mp4" for i in range(10)],
                "metaxy_provenance_by_field": [{"video_path": f"path_hash_{i}"} for i in range(10)],
                "metaxy_provenance": [f"upstream_prov_{i}" for i in range(10)],
            }
        )
        store.write_metadata(Upstream, upstream_data)

        # First materialization: compute downstream for all 10 chunks
        downstream_batch = pl.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(10)],
                "result": [f"result_{i}" for i in range(10)],
            }
        )
        upstream_df = store.read_metadata(Upstream).collect().to_polars()
        downstream_joined = downstream_batch.join(
            upstream_df.select(["chunk_id", "metaxy_data_version_by_field", "metaxy_provenance"]).rename(
                {
                    # table_name uses underscores, not slashes
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__delta_dedup_test_upstream",
                    "metaxy_provenance": "__upstream_provenance",
                }
            ),
            on="chunk_id",
        )
        downstream_with_prov = store.compute_provenance(Downstream, nw.from_native(downstream_joined))
        store.write_metadata(Downstream, downstream_with_prov)

        # Write SAME data 4 more times (5 total writes, like the user scenario)
        # This simulates the user materializing the same feature multiple times
        for _ in range(4):
            store.write_metadata(Downstream, downstream_with_prov)

        # Verify we have 50 total rows but only 10 unique after deduplication
        current_all = store.read_metadata(Downstream, latest_only=False).collect()
        current_dedup = store.read_metadata(Downstream, latest_only=True).collect()

        # Debug: check what read_metadata_in_store returns
        from metaxy.models.constants import METAXY_FEATURE_VERSION

        raw_metadata = store.read_metadata_in_store(
            Downstream,
            filters=[nw.col(METAXY_FEATURE_VERSION) == Downstream.feature_version()],
        )
        raw_count = len(raw_metadata.collect()) if raw_metadata else 0
        print(f"\nDEBUG: Total rows (latest_only=False): {len(current_all)}")
        print(f"DEBUG: Deduplicated rows (latest_only=True): {len(current_dedup)}")
        print(f"DEBUG: read_metadata_in_store rows: {raw_count}")

        assert len(current_all) == 50, f"Expected 50 total rows, got {len(current_all)}"
        assert len(current_dedup) == 10, f"Expected 10 deduplicated rows, got {len(current_dedup)}"

        # THE KEY TEST: resolve_update should see only 10 rows (deduplicated)
        # and report 0 orphaned (all chunks exist in upstream)
        increment = store.resolve_update(Downstream, lazy=False)

        removed_count = len(increment.removed)
        assert removed_count == 0, (
            f"REGRESSION: Expected 0 orphaned (all chunks exist in upstream), "
            f"but got {removed_count}. This indicates resolve_update is NOT "
            f"deduplicating current metadata rows. "
            f"With 50 total rows and 10 expected, the old buggy behavior would show 40 orphaned."
        )


def test_identity_lineage_orphaned_with_stale_upstream_from_fallback(
    graph: FeatureGraph,
    tmp_path,
):
    """Test orphaned count when upstream data differs between fallback and local store.

    This reproduces the user's actual scenario where:
    1. User pulls upstream metadata from a FALLBACK store (production)
    2. Upstream in fallback has MORE samples than local upstream
    3. User materializes downstream based on fallback upstream
    4. When checking status, local upstream has fewer samples
    5. Orphaned count shows samples that exist in downstream but NOT in local upstream

    The user reported: 151152 materialized, 755755 orphaned
    This could happen if:
    - Fallback upstream had 906907 samples (151152 + 755755)
    - User materialized all of them
    - Local upstream only has 151152 samples
    - So 755755 samples in downstream have no matching upstream in local store
    """
    import polars as pl

    from metaxy import FieldKey, FieldSpec
    from metaxy.metadata_store.delta import DeltaMetadataStore

    class Upstream(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
            id_columns=("chunk_id",),
            fields=[
                FieldSpec(key=FieldKey(["video_path"]), code_version="1"),
            ],
        ),
    ):
        pass

    class Downstream(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("chunk_id",),
            deps=[FeatureDep(feature=Upstream)],
            fields=[
                FieldSpec(key=FieldKey(["result"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Create fallback store with MORE upstream data
    fallback_store = DeltaMetadataStore(root_path=tmp_path / "delta_fallback")
    # Create local store that reads upstream from fallback
    local_store = DeltaMetadataStore(root_path=tmp_path / "delta_local", fallback_stores=[fallback_store])

    with fallback_store, local_store, graph.use():
        import narwhals as nw

        # Write upstream metadata to FALLBACK store for 100 chunks
        fallback_upstream_data = pl.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(100)],
                "video_path": [f"/path/to/video_{i}.mp4" for i in range(100)],
                "metaxy_provenance_by_field": [{"video_path": f"path_hash_{i}"} for i in range(100)],
                "metaxy_provenance": [f"upstream_prov_{i}" for i in range(100)],
            }
        )
        fallback_store.write_metadata(Upstream, fallback_upstream_data)

        # User reads upstream from fallback (via local_store) and materializes downstream
        # This should read 100 chunks from fallback
        upstream_df = local_store.read_metadata(Upstream).collect().to_polars()
        assert len(upstream_df) == 100, f"Expected 100 upstream rows from fallback, got {len(upstream_df)}"

        # User computes and writes downstream for all 100 chunks
        downstream_batch = pl.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(100)],
                "result": [f"result_{i}" for i in range(100)],
            }
        )
        downstream_joined = downstream_batch.join(
            upstream_df.select(["chunk_id", "metaxy_data_version_by_field", "metaxy_provenance"]).rename(
                {
                    "metaxy_data_version_by_field": "metaxy_data_version_by_field__upstream",
                    "metaxy_provenance": "__upstream_provenance",
                }
            ),
            on="chunk_id",
        )
        downstream_with_prov = local_store.compute_provenance(Downstream, nw.from_native(downstream_joined))
        local_store.write_metadata(Downstream, downstream_with_prov)

        # Now write LOCAL upstream with FEWER samples (only 20 chunks)
        # This simulates the scenario where local upstream is out of sync with fallback
        local_upstream_data = pl.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(20)],
                "video_path": [f"/path/to/video_{i}.mp4" for i in range(20)],
                "metaxy_provenance_by_field": [{"video_path": f"path_hash_{i}"} for i in range(20)],
                "metaxy_provenance": [f"upstream_prov_{i}" for i in range(20)],
            }
        )
        local_store.write_metadata(Upstream, local_upstream_data)

        # Now check status WITHOUT fallback
        # Create a new store without fallback to simulate checking local-only status
        local_only_store = DeltaMetadataStore(root_path=tmp_path / "delta_local_only")
        with local_only_store:
            # Copy upstream and downstream from local_store to local_only_store
            upstream_data = local_store.read_metadata_in_store(Upstream)
            downstream_data = local_store.read_metadata_in_store(Downstream)
            assert upstream_data is not None
            assert downstream_data is not None
            local_only_store.write_metadata(Upstream, upstream_data.collect())
            local_only_store.write_metadata(Downstream, downstream_data.collect())

            # Check status:
            # - Downstream has 100 chunks
            # - Local upstream only has 20 chunks
            # - So 80 chunks in downstream have no upstream -> orphaned = 80
            increment = local_only_store.resolve_update(Downstream, lazy=False)

            removed_count = len(increment.removed)
            added_count = len(increment.added)

            # Expected: 80 orphaned (chunks 20-99 have no upstream)
            assert removed_count == 80, f"Expected 80 orphaned (100 downstream - 20 upstream), but got {removed_count}."

            # Also check added: 0 because all local upstream (20 chunks) are in downstream
            assert added_count == 0, (
                f"Expected 0 added (all 20 local upstream chunks are in downstream), but got {added_count}."
            )


# ============= OPTIONAL DEPENDENCY TESTS =============


class OptionalDependencyCases:
    """Feature graph configurations specific to optional dependency behavior."""

    def case_required_plus_optional(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Required + optional dependency (left join behavior)."""
        return FeatureGraphCases().case_optional_dependency(graph)

    def case_all_optional(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """All optional dependencies (full outer join behavior)."""

        class OptionalA(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="optional_a",
                fields=["data_a"],
            ),
        ):
            pass

        class OptionalB(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="optional_b",
                fields=["data_b"],
            ),
        ):
            pass

        class ChildFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="child",
                deps=[
                    FeatureDep(feature=OptionalA, optional=True),
                    FeatureDep(feature=OptionalB, optional=True),
                ],
                fields=["result"],
            ),
        ):
            pass

        child_plan = graph.get_feature_plan(ChildFeature.spec().key)
        upstream_features = {
            OptionalA.spec().key: OptionalA,
            OptionalB.spec().key: OptionalB,
        }
        return graph, upstream_features, child_plan


def _compute_golden_increment_for_optional_deps(
    child_plan: FeaturePlan,
    upstream_data: Mapping[str, pl.DataFrame],
    hash_algorithm,
) -> Increment:
    """Compute golden increment using PolarsVersioningEngine for optional dependency tests.

    This is the reference implementation - store implementations should produce
    the same result.
    """
    import narwhals as nw

    from metaxy.versioning.polars import PolarsVersioningEngine

    engine = PolarsVersioningEngine(plan=child_plan)

    # Convert upstream data to Narwhals LazyFrames with FeatureKey keys
    upstream_nw = {FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()}

    # Compute increment with provenance (no current downstream for first run)
    added, changed, removed, _ = engine.resolve_increment_with_provenance(
        current=None,
        upstream=upstream_nw,
        hash_algorithm=hash_algorithm,
        filters={},
        sample=None,
    )

    # Collect lazy frames and convert to Increment
    added_collected = added.collect()
    changed_collected = changed.collect() if changed is not None else added_collected.head(0)
    removed_collected = removed.collect() if removed is not None else added_collected.head(0)

    return Increment(
        added=added_collected,
        changed=changed_collected,
        removed=removed_collected,
    )


@parametrize_with_cases("feature_plan_config", cases=OptionalDependencyCases)
def test_resolve_update_optional_dependencies(
    graph: FeatureGraph,
    any_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
) -> None:
    """Test resolve_update optional dependency join behavior against golden Polars implementation.

    Uses the PolarsVersioningEngine as the reference implementation and verifies that
    the store's resolve_update produces the same result.
    """
    from metaxy.models.constants import METAXY_CREATED_AT

    store = any_store
    graph, upstream_features, child_plan = feature_plan_config
    child_key = child_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]

    feature_versions = {
        feat_key.to_string(): feat_class.feature_version() for feat_key, feat_class in upstream_features.items()
    }
    feature_versions[child_key.to_string()] = ChildFeature.feature_version()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, _ = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=4,
            max_rows=8,
        ).example()

    try:
        with store, graph.use():
            # Write upstream data to store
            for feature_key_str, upstream_df in upstream_data.items():
                feat_key = FeatureKey([feature_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Get store's result
            increment = store.resolve_update(ChildFeature, lazy=False)

            # Compute golden increment using PolarsVersioningEngine
            golden_increment = _compute_golden_increment_for_optional_deps(
                child_plan, upstream_data, store.hash_algorithm
            )

            # Compare added frames (excluding timestamp which varies)
            actual_added = increment.added.lazy().collect().to_polars()
            golden_added = golden_increment.added.lazy().collect().to_polars()

            # Get common columns for comparison
            common_columns = [
                col for col in actual_added.columns if col in golden_added.columns and col != METAXY_CREATED_AT
            ]

            # Sort by all comparable columns for deterministic comparison
            actual_sorted = actual_added.sort(common_columns).select(common_columns)
            golden_sorted = golden_added.sort(common_columns).select(common_columns)

            pl_testing.assert_frame_equal(
                actual_sorted,
                golden_sorted,
                check_row_order=True,
                check_column_order=False,
            )

            # Verify provenance columns are present
            assert "metaxy_provenance" in actual_added.columns
            assert "metaxy_provenance_by_field" in actual_added.columns

    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")


def test_optional_dependency_null_handling_and_provenance_stability(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that optional dependencies produce NULLs for missing rows and provenance is stable.

    This test verifies critical properties of optional dependencies:

    1. **NULL handling**: When a sample exists in required parent but not in optional parent,
       the downstream should still include that sample with NULL values for optional fields.

    2. **Provenance stability for missing optional deps**: When an optional dependency's
       field version changes, downstream samples that have NULL for that optional dep
       should NOT have their provenance changed (since they don't depend on it).

    3. **Provenance changes for present optional deps**: When an optional dependency's
       field version changes, downstream samples that DO have values from the optional
       dep SHOULD have their provenance changed.

    Scenario:
    - RequiredParent: samples [s1, s2, s3] with field 'req_value'
    - OptionalParent: samples [s1, s3] only (s2 is missing) with field 'opt_value'
    - Child: depends on both, with 'computed' field depending on ALL upstream fields

    After initial resolve:
    - s1, s3: have both req_value and opt_value
    - s2: has req_value but NULL for opt_value

    When OptionalParent's opt_value version changes:
    - s1, s3: provenance SHOULD change (they have actual optional dep values)
    - s2: provenance should NOT change (optional dep is NULL, so not dependent)
    """
    import polars as pl

    from metaxy.models.field import FieldSpec, SpecialFieldDep
    from metaxy.models.types import FieldKey

    class RequiredParent(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="required_parent_null_test",
            fields=[FieldSpec(key=FieldKey(["req_value"]), code_version="1")],
        ),
    ):
        pass

    class OptionalParent(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="optional_parent_null_test",
            fields=[FieldSpec(key=FieldKey(["opt_value"]), code_version="1")],
        ),
    ):
        pass

    class ChildFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="child_null_test",
            deps=[
                FeatureDep(feature=RequiredParent, optional=False),
                FeatureDep(feature=OptionalParent, optional=True),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["computed"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,  # Depends on ALL upstream fields
                ),
            ],
        ),
    ):
        pass

    try:
        with any_store:
            # === INITIAL DATA ===
            # Required parent has samples s1, s2, s3
            required_df = pl.DataFrame(
                {
                    "sample_uid": ["s1", "s2", "s3"],
                    "metaxy_provenance_by_field": [
                        {"req_value": "req_s1_v1"},
                        {"req_value": "req_s2_v1"},
                        {"req_value": "req_s3_v1"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"req_value": "req_s1_v1"},
                        {"req_value": "req_s2_v1"},
                        {"req_value": "req_s3_v1"},
                    ],
                }
            )
            any_store.write_metadata(RequiredParent, required_df)

            # Optional parent has only s1 and s3 (s2 is MISSING)
            optional_df = pl.DataFrame(
                {
                    "sample_uid": ["s1", "s3"],
                    "metaxy_provenance_by_field": [
                        {"opt_value": "opt_s1_v1"},
                        {"opt_value": "opt_s3_v1"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"opt_value": "opt_s1_v1"},
                        {"opt_value": "opt_s3_v1"},
                    ],
                }
            )
            any_store.write_metadata(OptionalParent, optional_df)

            # Initial resolve
            increment_v1 = any_store.resolve_update(ChildFeature)
            initial_downstream = increment_v1.added.lazy().collect().to_polars()

            # === VERIFY NULL HANDLING ===
            # Should have all 3 samples (left join keeps s2 even though optional parent is missing)
            assert len(initial_downstream) == 3, (
                f"Expected 3 samples (s1, s2, s3) from left join, got {len(initial_downstream)}"
            )

            # Sort for consistent comparison
            initial_downstream = initial_downstream.sort("sample_uid")
            sample_uids = initial_downstream["sample_uid"].to_list()
            assert sample_uids == ["s1", "s2", "s3"], f"Expected samples ['s1', 's2', 's3'], got {sample_uids}"

            # Store initial provenance for comparison
            initial_provenance = {
                row["sample_uid"]: row["metaxy_provenance_by_field"] for row in initial_downstream.iter_rows(named=True)
            }

            # Write initial downstream
            any_store.write_metadata(ChildFeature, initial_downstream)

            # === UPDATE OPTIONAL PARENT (version change) ===
            # Only change the optional parent's provenance (simulates code_version bump)
            optional_df_v2 = pl.DataFrame(
                {
                    "sample_uid": ["s1", "s3"],
                    "metaxy_provenance_by_field": [
                        {"opt_value": "opt_s1_v2"},  # Changed!
                        {"opt_value": "opt_s3_v2"},  # Changed!
                    ],
                    "metaxy_data_version_by_field": [
                        {"opt_value": "opt_s1_v2"},
                        {"opt_value": "opt_s3_v2"},
                    ],
                }
            )
            any_store.write_metadata(OptionalParent, optional_df_v2)

            # Resolve after optional parent change
            increment_v2 = any_store.resolve_update(ChildFeature)

            # === VERIFY PROVENANCE STABILITY FOR MISSING OPTIONAL DEP ===
            changed_df = increment_v2.changed
            assert changed_df is not None, "Expected changed to not be None"
            changed_downstream = changed_df.lazy().collect().to_polars()

            # s1 and s3 should be in changed (they had actual optional dep values)
            changed_uids = set(changed_downstream["sample_uid"].to_list())

            # s2 should NOT be in changed - its optional dep was NULL, so it doesn't
            # depend on the optional parent's field version
            assert "s2" not in changed_uids, (
                "s2 should NOT be in changed because its optional dep is NULL. "
                "Changing the optional parent's version should not affect samples "
                "where the optional dep is missing."
            )

            # s1 and s3 SHOULD be in changed (they have actual optional dep values)
            assert "s1" in changed_uids, "s1 should be in changed because it has actual optional dep values"
            assert "s3" in changed_uids, "s3 should be in changed because it has actual optional dep values"

            # === VERIFY PROVENANCE ACTUALLY CHANGED FOR s1, s3 ===
            changed_provenance = {
                row["sample_uid"]: row["metaxy_provenance_by_field"] for row in changed_downstream.iter_rows(named=True)
            }

            # s1's computed field provenance should have changed
            assert changed_provenance["s1"]["computed"] != initial_provenance["s1"]["computed"], (
                "s1's computed field provenance should change when optional dep changes"
            )

            # s3's computed field provenance should have changed
            assert changed_provenance["s3"]["computed"] != initial_provenance["s3"]["computed"], (
                "s3's computed field provenance should change when optional dep changes"
            )

            # Added should be empty (no new samples)
            added_df = increment_v2.added.lazy().collect().to_polars()
            assert len(added_df) == 0, f"Expected 0 added rows, got {len(added_df)}"

    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}")


def test_all_optional_deps_outer_join_behavior(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that all-optional dependencies use full outer join.

    When ALL dependencies are optional (no required deps), the engine should use
    full outer join to include samples that exist in ANY parent, even if they
    don't exist in all parents.

    Scenario:
    - OptionalA: samples [s1, s2]
    - OptionalB: samples [s2, s3]
    - Child (all optional deps): should include [s1, s2, s3]
      - s1: has OptionalA data, NULL for OptionalB
      - s2: has both
      - s3: NULL for OptionalA, has OptionalB data
    """
    import polars as pl

    from metaxy.models.field import FieldSpec, SpecialFieldDep
    from metaxy.models.types import FieldKey

    class OptionalA(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="optional_a_outer_test",
            fields=[FieldSpec(key=FieldKey(["value_a"]), code_version="1")],
        ),
    ):
        pass

    class OptionalB(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="optional_b_outer_test",
            fields=[FieldSpec(key=FieldKey(["value_b"]), code_version="1")],
        ),
    ):
        pass

    class ChildFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="child_outer_test",
            deps=[
                FeatureDep(feature=OptionalA, optional=True),  # ALL optional
                FeatureDep(feature=OptionalB, optional=True),  # ALL optional
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["result"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        pass

    try:
        with any_store:
            # OptionalA has s1, s2
            optional_a_df = pl.DataFrame(
                {
                    "sample_uid": ["s1", "s2"],
                    "metaxy_provenance_by_field": [
                        {"value_a": "a_s1"},
                        {"value_a": "a_s2"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"value_a": "a_s1"},
                        {"value_a": "a_s2"},
                    ],
                }
            )
            any_store.write_metadata(OptionalA, optional_a_df)

            # OptionalB has s2, s3
            optional_b_df = pl.DataFrame(
                {
                    "sample_uid": ["s2", "s3"],
                    "metaxy_provenance_by_field": [
                        {"value_b": "b_s2"},
                        {"value_b": "b_s3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"value_b": "b_s2"},
                        {"value_b": "b_s3"},
                    ],
                }
            )
            any_store.write_metadata(OptionalB, optional_b_df)

            # Resolve
            increment = any_store.resolve_update(ChildFeature)
            downstream = increment.added.lazy().collect().to_polars()

            # === VERIFY FULL OUTER JOIN BEHAVIOR ===
            # Should have all 3 samples from the union of both parents
            assert len(downstream) == 3, (
                f"Expected 3 samples from full outer join, got {len(downstream)}. "
                f"All-optional deps should use full outer join to include samples "
                f"from any parent."
            )

            downstream = downstream.sort("sample_uid")
            sample_uids = downstream["sample_uid"].to_list()
            assert sample_uids == ["s1", "s2", "s3"], (
                f"Expected samples ['s1', 's2', 's3'] from full outer join, got {sample_uids}"
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}")


# ============= TEST: AGGREGATION LINEAGE FIELD-LEVEL ISOLATION =============


def test_aggregation_lineage_field_level_provenance_isolation(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that field-level dependencies are preserved through aggregation lineage.

    This tests a critical property of aggregation: changing an upstream field
    should only affect downstream fields that depend on it, not unrelated fields.

    Scenario:
    - Upstream: SensorReadings with temperature and humidity fields
    - Downstream: HourlyStats with avg_temp (depends on temperature) and
      avg_humidity (depends on humidity)
    - When temperature upstream changes, only avg_temp downstream should change
    """
    import polars as pl

    from metaxy.models.field import FieldDep, FieldSpec
    from metaxy.models.lineage import LineageRelationship
    from metaxy.models.types import FieldKey

    class SensorReadings(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="sensor_readings_store_test",
            id_columns=("sensor_id", "reading_id"),
            fields=[
                FieldSpec(key=FieldKey(["temperature"]), code_version="1"),
                FieldSpec(key=FieldKey(["humidity"]), code_version="1"),
            ],
        ),
    ):
        pass

    class HourlyStats(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="hourly_stats_store_test",
            id_columns=("sensor_id",),
            deps=[
                FeatureDep(
                    feature=SensorReadings,
                    lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                )
            ],
            fields=[
                # avg_temp depends ONLY on temperature
                FieldSpec(
                    key=FieldKey(["avg_temp"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["sensor_readings_store_test"]),
                            fields=[FieldKey(["temperature"])],
                        )
                    ],
                ),
                # avg_humidity depends ONLY on humidity
                FieldSpec(
                    key=FieldKey(["avg_humidity"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["sensor_readings_store_test"]),
                            fields=[FieldKey(["humidity"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    try:
        with any_store:
            # Initial upstream data - using minimal required columns
            # write_metadata fills in missing metaxy columns
            upstream_v1 = pl.DataFrame(
                {
                    "sensor_id": ["s1", "s1"],
                    "reading_id": ["r1", "r2"],
                    "metaxy_provenance_by_field": [
                        {"temperature": "temp_v1_r1", "humidity": "hum_v1_r1"},
                        {"temperature": "temp_v1_r2", "humidity": "hum_v1_r2"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"temperature": "temp_v1_r1", "humidity": "hum_v1_r1"},
                        {"temperature": "temp_v1_r2", "humidity": "hum_v1_r2"},
                    ],
                }
            )
            any_store.write_metadata(SensorReadings, upstream_v1)

            # Initial resolve - creates downstream
            increment_v1 = any_store.resolve_update(HourlyStats)
            initial_downstream = increment_v1.added.lazy().collect().to_polars()
            any_store.write_metadata(HourlyStats, initial_downstream)

            # Get initial provenance
            initial_prov_by_field = initial_downstream["metaxy_provenance_by_field"][0]
            initial_avg_temp = initial_prov_by_field["avg_temp"]
            initial_avg_humidity = initial_prov_by_field["avg_humidity"]

            # Update upstream - ONLY temperature changes
            upstream_v2 = pl.DataFrame(
                {
                    "sensor_id": ["s1", "s1"],
                    "reading_id": ["r1", "r2"],
                    "metaxy_provenance_by_field": [
                        # Only temperature field version changed
                        {"temperature": "temp_v2_r1", "humidity": "hum_v1_r1"},
                        {"temperature": "temp_v2_r2", "humidity": "hum_v1_r2"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"temperature": "temp_v2_r1", "humidity": "hum_v1_r1"},
                        {"temperature": "temp_v2_r2", "humidity": "hum_v1_r2"},
                    ],
                }
            )
            any_store.write_metadata(SensorReadings, upstream_v2)

            # Resolve after upstream change
            increment_v2 = any_store.resolve_update(HourlyStats)

            # Should have changed (not added, since sensor_id exists)
            # With aggregation lineage using window functions, we get N rows per
            # aggregation group (one per upstream reading), all with identical provenance.
            # Here sensor s1 has 2 readings (r1, r2), so we expect 2 changed rows.
            changed_df = increment_v2.changed
            assert changed_df is not None, "Expected changed to not be None"
            changed_downstream = changed_df.lazy().collect().to_polars()
            assert len(changed_downstream) == 2, "Expected 2 changed rows (one per reading in aggregation group)"

            # All rows in the aggregation group have identical provenance
            # Pick the first row to check the provenance values
            updated_prov_by_field = changed_downstream["metaxy_provenance_by_field"][0]
            updated_avg_temp = updated_prov_by_field["avg_temp"]
            updated_avg_humidity = updated_prov_by_field["avg_humidity"]

            # Verify all rows in the group have the same provenance (window function behavior)
            for i in range(len(changed_downstream)):
                row_prov = changed_downstream["metaxy_provenance_by_field"][i]
                assert row_prov["avg_temp"] == updated_avg_temp, (
                    f"Row {i} should have same avg_temp provenance as row 0"
                )
                assert row_prov["avg_humidity"] == updated_avg_humidity, (
                    f"Row {i} should have same avg_humidity provenance as row 0"
                )

            # CRITICAL: avg_temp should change (temperature upstream changed)
            assert updated_avg_temp != initial_avg_temp, (
                "avg_temp provenance should change when upstream temperature changes"
            )

            # CRITICAL: avg_humidity should NOT change (humidity upstream unchanged)
            assert updated_avg_humidity == initial_avg_humidity, (
                "avg_humidity provenance should NOT change when only temperature changes"
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}")


def test_aggregation_lineage_field_level_provenance_definition_change(
    any_store: MetadataStore,
):
    """Test field-level provenance isolation when upstream field definition changes.

    When upstream field code_version changes, the upstream data would be recomputed
    with new data_version_by_field. This test verifies that only dependent downstream
    fields are affected.

    Uses separate FeatureGraphs to simulate definition changes.
    """
    import polars as pl

    from metaxy.models.field import FieldDep, FieldSpec
    from metaxy.models.lineage import LineageRelationship
    from metaxy.models.types import FieldKey

    # Variables to hold v1 results for comparison with v2
    v1_avg_temp = None
    v1_avg_humidity = None

    # === Version 1: Initial definitions ===
    with FeatureGraph().use():

        class SensorReadingsV1(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="sensor_readings_defn_store",
                id_columns=("sensor_id", "reading_id"),
                fields=[
                    FieldSpec(key=FieldKey(["temperature"]), code_version="1"),
                    FieldSpec(key=FieldKey(["humidity"]), code_version="1"),
                ],
            ),
        ):
            pass

        class HourlyStatsV1(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="hourly_stats_defn_store",
                id_columns=("sensor_id",),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["sensor_readings_defn_store"]),
                        lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                    )
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["avg_temp"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn_store"]),
                                fields=[FieldKey(["temperature"])],
                            )
                        ],
                    ),
                    FieldSpec(
                        key=FieldKey(["avg_humidity"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn_store"]),
                                fields=[FieldKey(["humidity"])],
                            )
                        ],
                    ),
                ],
            ),
        ):
            pass

        # V1 upstream data - inside context!
        upstream_v1 = pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "reading_id": ["r1", "r2"],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_cv1_r1", "humidity": "hum_cv1_r1"},
                    {"temperature": "temp_cv1_r2", "humidity": "hum_cv1_r2"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_cv1_r1", "humidity": "hum_cv1_r1"},
                    {"temperature": "temp_cv1_r2", "humidity": "hum_cv1_r2"},
                ],
            }
        )

        try:
            with any_store:  # Store ops inside FeatureGraph context!
                any_store.write_metadata(SensorReadingsV1, upstream_v1)

                increment_v1 = any_store.resolve_update(HourlyStatsV1)

                result_v1 = increment_v1.added.lazy().collect().to_polars()
                v1_prov_by_field = result_v1["metaxy_provenance_by_field"][0]
                v1_avg_temp = v1_prov_by_field["avg_temp"]
                v1_avg_humidity = v1_prov_by_field["avg_humidity"]

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}")

    # === Version 2: Temperature code_version changes to "2" ===
    with FeatureGraph().use():

        class SensorReadingsV2(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="sensor_readings_defn_store",
                id_columns=("sensor_id", "reading_id"),
                fields=[
                    FieldSpec(key=FieldKey(["temperature"]), code_version="2"),  # Changed!
                    FieldSpec(key=FieldKey(["humidity"]), code_version="1"),  # Unchanged
                ],
            ),
        ):
            pass

        class HourlyStatsV2(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="hourly_stats_defn_store",
                id_columns=("sensor_id",),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["sensor_readings_defn_store"]),
                        lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                    )
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["avg_temp"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn_store"]),
                                fields=[FieldKey(["temperature"])],
                            )
                        ],
                    ),
                    FieldSpec(
                        key=FieldKey(["avg_humidity"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn_store"]),
                                fields=[FieldKey(["humidity"])],
                            )
                        ],
                    ),
                ],
            ),
        ):
            pass

        # V2 upstream data - inside context!
        # Temperature data_version changed (simulates recompute after code_version bump)
        upstream_v2 = pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "reading_id": ["r1", "r2"],
                "metaxy_provenance_by_field": [
                    {
                        "temperature": "temp_cv2_r1",
                        "humidity": "hum_cv1_r1",
                    },  # temp changed
                    {
                        "temperature": "temp_cv2_r2",
                        "humidity": "hum_cv1_r2",
                    },  # temp changed
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_cv2_r1", "humidity": "hum_cv1_r1"},
                    {"temperature": "temp_cv2_r2", "humidity": "hum_cv1_r2"},
                ],
            }
        )

        try:
            with any_store:  # Store ops inside FeatureGraph context!
                any_store.write_metadata(SensorReadingsV2, upstream_v2)

                increment_v2 = any_store.resolve_update(HourlyStatsV2)

                result_v2 = increment_v2.added.lazy().collect().to_polars()
                v2_prov_by_field = result_v2["metaxy_provenance_by_field"][0]
                v2_avg_temp = v2_prov_by_field["avg_temp"]
                v2_avg_humidity = v2_prov_by_field["avg_humidity"]

                # CRITICAL: avg_temp should change (temperature upstream data changed)
                assert v2_avg_temp != v1_avg_temp, (
                    "avg_temp provenance should change when upstream temperature "
                    "data_version changes (due to code_version bump)"
                )

                # CRITICAL: avg_humidity should NOT change (humidity unchanged)
                assert v2_avg_humidity == v1_avg_humidity, (
                    "avg_humidity provenance should NOT change when only temperature changes"
                )

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}")


def test_aggregation_lineage_preserves_user_columns(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that aggregation lineage preserves user-specified columns in FeatureDep.columns.

    Regression test for a bug where columns specified in FeatureDep.columns were dropped
    during aggregation because aggregate_strings only kept group_by and aggregated columns.

    Scenario:
    - DirectorNotesTexts has columns like 'dataset', 'director_notes_type', 'path', 'size'
    - DirectorNotesAggregatedByChunk aggregates by chunk_id with columns=("...", "dataset")
    - After aggregation, 'dataset' column should still be present in the result

    Bug symptoms:
    - User filters by global_filters=[nw.col("dataset") == "some_value"]
    - resolve_update works but 'dataset' column is missing from result
    - User code fails with ColumnNotFoundError when trying to access 'dataset'
    """
    import polars as pl

    from metaxy import FieldKey, FieldSpec
    from metaxy.models.lineage import LineageRelationship

    class DirectorNotesTexts(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["director_notes_texts_test"]),
            id_columns=("director_notes_id",),
            fields=[
                FieldSpec(key=FieldKey(["text"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DirectorNotesAggregated(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["director_notes_aggregated_test"]),
            id_columns=("chunk_id",),
            deps=[
                FeatureDep(
                    feature=DirectorNotesTexts,
                    # User explicitly requests dataset and director_notes_type columns
                    columns=(
                        "dataset",
                        "director_notes_type",
                        "path",
                        "size",
                        "chunk_id",
                    ),
                    rename={"path": "text_path", "size": "text_size"},
                    lineage=LineageRelationship.aggregation(on=["chunk_id"]),
                ),
            ],
            fields=[
                FieldSpec(key=FieldKey(["aggregated"]), code_version="1"),
            ],
        ),
    ):
        pass

    store = default_store

    with store, graph.use():
        # Write upstream data - multiple notes per chunk
        upstream_df = pl.DataFrame(
            {
                "director_notes_id": ["note1", "note2", "note3", "note4"],
                "dataset": ["dataset_a", "dataset_a", "dataset_b", "dataset_b"],
                "director_notes_type": [
                    "descriptive",
                    "imperative",
                    "descriptive",
                    "imperative",
                ],
                "path": ["/path/1.txt", "/path/2.txt", "/path/3.txt", "/path/4.txt"],
                "size": [100, 200, 300, 400],
                "chunk_id": ["chunk_1", "chunk_1", "chunk_2", "chunk_2"],
                "metaxy_provenance_by_field": [
                    {"text": "hash1"},
                    {"text": "hash2"},
                    {"text": "hash3"},
                    {"text": "hash4"},
                ],
            }
        )
        store.write_metadata(DirectorNotesTexts, upstream_df)

        # Call resolve_update - should work and include 'dataset' column
        increment = store.resolve_update(DirectorNotesAggregated, lazy=False)

        assert len(increment.added) > 0, "Expected added rows from resolve_update"

        added_df = increment.added.lazy().collect().to_polars()

        # CRITICAL: 'dataset' should be present in the result
        # This was the bug - aggregate_strings dropped all non-aggregated columns
        assert "dataset" in added_df.columns, (
            f"Expected 'dataset' column in aggregation result. "
            f"Got columns: {added_df.columns}. "
            f"This indicates that FeatureDep.columns are not being preserved through aggregation lineage."
        )

        # Also verify other requested columns are present
        expected_columns = {
            "chunk_id",
            "dataset",
            "director_notes_type",
            "text_path",
            "text_size",
        }
        actual_columns = set(added_df.columns)
        missing_columns = expected_columns - actual_columns
        assert not missing_columns, (
            f"Missing columns in aggregation result: {missing_columns}. Got columns: {actual_columns}"
        )


def test_aggregation_lineage_preserves_columns_with_global_filter(
    default_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test that global_filters work with aggregation lineage when filtering on user columns.

    This reproduces the exact bug scenario:
    - User has aggregation lineage feature with columns=(..., "dataset")
    - User calls resolve_update with global_filters=[nw.col("dataset") == "some_value"]
    - Filter should work AND 'dataset' column should be in the result

    The bug was: filter worked but 'dataset' was dropped during aggregation, causing
    ColumnNotFoundError when user code tried to access it in the result.
    """
    import narwhals as nw
    import polars as pl

    from metaxy import FieldKey, FieldSpec
    from metaxy.models.lineage import LineageRelationship

    class UpstreamWithDataset(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_with_dataset_test"]),
            id_columns=("item_id",),
            fields=[
                FieldSpec(key=FieldKey(["value"]), code_version="1"),
            ],
        ),
    ):
        pass

    class AggregatedByGroup(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["aggregated_by_group_test"]),
            id_columns=("group_id",),
            deps=[
                FeatureDep(
                    feature=UpstreamWithDataset,
                    # Explicitly request 'dataset' column
                    columns=("dataset", "group_id"),
                    lineage=LineageRelationship.aggregation(on=["group_id"]),
                ),
            ],
            fields=[
                FieldSpec(key=FieldKey(["result"]), code_version="1"),
            ],
        ),
    ):
        pass

    store = default_store

    with store, graph.use():
        # Write upstream data with different datasets
        upstream_df = pl.DataFrame(
            {
                "item_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
                "dataset": ["ds_a", "ds_a", "ds_a", "ds_b", "ds_b", "ds_b"],
                "group_id": ["g1", "g1", "g2", "g1", "g2", "g2"],
                "metaxy_provenance_by_field": [{"value": f"hash{i}"} for i in range(1, 7)],
            }
        )
        store.write_metadata(UpstreamWithDataset, upstream_df)

        # Call resolve_update with global_filters filtering by dataset
        increment = store.resolve_update(
            AggregatedByGroup,
            global_filters=[nw.col("dataset") == "ds_a"],
            lazy=False,
        )

        assert len(increment.added) > 0, "Expected added rows from resolve_update"

        added_df = increment.added.lazy().collect().to_polars()

        # CRITICAL: 'dataset' should be in the result
        assert "dataset" in added_df.columns, (
            f"Expected 'dataset' column in result after global_filters. "
            f"Got columns: {added_df.columns}. "
            f"global_filters should not cause columns to be dropped."
        )

        # Verify only ds_a rows were included (filter worked)
        datasets = added_df["dataset"].unique().to_list()
        assert datasets == ["ds_a"], f"Expected only 'ds_a' dataset after filtering. Got: {datasets}"

        # Verify correct groups were returned
        groups = set(added_df["group_id"].to_list())
        # ds_a has items in g1 (i1, i2) and g2 (i3), so both groups should be present
        assert groups == {"g1", "g2"}, f"Expected groups {{'g1', 'g2'}}, got {groups}"
