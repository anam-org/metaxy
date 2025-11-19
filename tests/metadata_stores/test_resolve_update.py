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
from typing import TYPE_CHECKING

import polars.testing as pl_testing
import pytest
from hypothesis.errors import NonInteractiveExampleWarning
from pytest_cases import parametrize_with_cases

from metaxy import (
    BaseFeature,
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    SampleFeatureSpec,
)
from metaxy._testing.parametric import (
    downstream_metadata_strategy,
    feature_metadata_strategy,
)
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from metaxy.models.plan import FeaturePlan

if TYPE_CHECKING:
    pass

# Type alias for feature plan output
FeaturePlanOutput = tuple[
    FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan
]


# ============= FEATURE GRAPH CONFIGURATIONS =============


class FeatureGraphCases:
    """Different feature graph topologies for testing."""

    def case_simple_chain(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Simple two-feature chain: Root -> Leaf."""

        class RootFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["value"],
            ),
        ):
            pass

        class LeafFeature(
            Feature,
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
            Feature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["value"],
            ),
        ):
            pass

        class BranchAFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="branch_a",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["a_result"],
            ),
        ):
            pass

        class BranchBFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="branch_b",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["b_result"],
            ),
        ):
            pass

        class LeafFeature(
            Feature,
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
            Feature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["field_a", "field_b", "field_c"],
            ),
        ):
            pass

        class LeafFeature(
            Feature,
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


class RootFeatureCases:
    """Root features (no upstream dependencies) for testing sample-based resolve_update."""

    def case_simple_root(self, graph: FeatureGraph) -> type[BaseFeature]:
        class SimpleRoot(
            Feature,
            spec=SampleFeatureSpec(
                key="simple_root",
                fields=["value"],
            ),
        ):
            pass

        return SimpleRoot

    def case_multi_field_root(self, graph: FeatureGraph) -> type[BaseFeature]:
        class MultiFieldRoot(
            Feature,
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

    # Convert to Narwhals
    import narwhals as nw

    samples_nw = nw.from_native(samples_df.lazy())

    # Call resolve_update with samples
    with store, graph.use():
        try:
            increment = store.resolve_update(
                root_feature, samples=samples_nw, lazy=True
            ).collect()
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

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
        feat_key.to_string(): feat_class.feature_version()
        for feat_key, feat_class in upstream_features.items()
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
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        # Get computed metadata
        added_df = increment.added.lazy().collect().to_polars()

        # Sort both for comparison
        id_columns = list(child_plan.feature.id_columns)
        added_sorted = added_df.sort(id_columns)
        golden_sorted = golden_downstream.sort(id_columns)

        # Compare provenance columns (exclude metaxy_created_at since it's a timestamp)
        from metaxy.models.constants import METAXY_CREATED_AT

        common_columns = [
            col
            for col in added_sorted.columns
            if col in golden_sorted.columns and col != METAXY_CREATED_AT
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
        feat_key.to_string(): feat_class.feature_version()
        for feat_key, feat_class in upstream_features.items()
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
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        # Verify changes were detected
        # With completely new data (8 samples vs 10 initial), we expect:
        # - Some samples added (new ones)
        # - Some samples removed (old ones not in new data)
        # - Possibly some changed (if IDs overlap but provenance differs)

        total_changes = (
            len(increment.added) + len(increment.changed) + len(increment.removed)
        )
        assert total_changes > 0, (
            "Expected resolve_update to detect some changes when upstream data changes"
        )


# ============= TEST: LAZY EXECUTION =============


def test_resolve_update_lazy_execution(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test resolve_update with lazy=True returns lazy frames with correct implementation."""
    store = any_store

    # Create a feature graph with multiple parents (realistic scenario)
    class Parent1(
        Feature,
        spec=SampleFeatureSpec(
            key="parent1",
            fields=["field_a", "field_b"],
        ),
    ):
        pass

    class Parent2(
        Feature,
        spec=SampleFeatureSpec(
            key="parent2",
            fields=["field_c"],
        ),
    ):
        pass

    class Child(
        Feature,
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
            lazy_added = (
                eager_increment.added.lazy().collect().to_polars().sort(id_columns)
            )
            eager_added = (
                eager_increment_direct.added.lazy()
                .collect()
                .to_polars()
                .sort(id_columns)
            )

            pl_testing.assert_frame_equal(lazy_added, eager_added)

        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )


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
        feat_key.to_string(): feat_class.feature_version()
        for feat_key, feat_class in upstream_features.items()
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

            assert len(increment1.added) > 0, (
                "Expected resolve_update to detect added samples"
            )

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
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        # Verify second call is idempotent
        assert len(increment2.added) == 0, (
            "Second resolve_update should have no added samples"
        )
        assert len(increment2.changed) == 0, (
            "Second resolve_update should have no changed samples"
        )
        assert len(increment2.removed) == 0, (
            "Second resolve_update should have no removed samples"
        )
