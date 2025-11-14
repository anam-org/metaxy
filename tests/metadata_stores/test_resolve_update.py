"""Test resolve_update across different stores and feature graph configurations.

This test suite verifies that resolve_update produces correct and consistent results
across different metadata store backends, hash algorithms, and feature graph topologies.

The tests use the parametric metadata generation utilities from metaxy._testing.parametric
to avoid manual data construction and ensure correctness.
"""

import warnings
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import polars.testing as pl_testing
import pytest
import pytest_cases
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
from metaxy._testing import HashAlgorithmCases
from metaxy._testing.parametric import (
    downstream_metadata_strategy,
    feature_metadata_strategy,
)
from metaxy.config import MetaxyConfig
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    InMemoryMetadataStore,
    MetadataStore,
)
from metaxy.metadata_store.clickhouse import ClickHouseMetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.plan import FeaturePlan
from metaxy.provenance.types import HashAlgorithm

# Type alias for feature plan output
FeaturePlanOutput = tuple[
    FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan
]


# ============= FEATURE GRAPH CONFIGURATIONS =============


class FeatureGraphCases:
    """Different feature graph topologies for testing."""

    def case_simple_chain(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Simple two-feature chain: Root -> Leaf.

        Returns:
            (graph, upstream_features, leaf_plan)
        """

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
        """Diamond dependency graph: Root -> BranchA,BranchB -> Leaf.

        Returns:
            (graph, upstream_features, leaf_plan)
        """

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
        """Features with multiple fields to test field-level provenance.

        Returns:
            (graph, upstream_features, leaf_plan)
        """

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
        """Single-field root feature."""

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
        """Multi-field root feature."""

        class MultiFieldRoot(
            Feature,
            spec=SampleFeatureSpec(
                key="multi_root",
                fields=["field_a", "field_b", "field_c"],
            ),
        ):
            pass

        return MultiFieldRoot


# ============= STORE CONFIGURATIONS =============


class StoreCases:
    """Different metadata store backend configurations."""

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_inmemory(self, hash_algorithm: HashAlgorithm) -> MetadataStore:
        """In-memory Polars-based store."""
        try:
            return InMemoryMetadataStore(hash_algorithm=hash_algorithm)
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by InMemoryMetadataStore"
            )

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_duckdb(self, hash_algorithm: HashAlgorithm, tmp_path) -> MetadataStore:
        """DuckDB SQL-based store."""
        store = DuckDBMetadataStore(
            tmp_path / "test.duckdb",
            hash_algorithm=hash_algorithm,
            extensions=["hashfuncs"],
            prefer_native=True,
        )
        # Check if hash algorithm is supported before opening
        try:
            with store:
                pass  # Just validate
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by DuckDBMetadataStore"
            )
        return store

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_clickhouse(
        self, hash_algorithm: HashAlgorithm, clickhouse_db: str
    ) -> MetadataStore:
        """ClickHouse SQL-based store."""
        store = ClickHouseMetadataStore(
            connection_string=clickhouse_db,
            hash_algorithm=hash_algorithm,
            prefer_native=True,
        )
        # Check if hash algorithm is supported before opening
        try:
            with store:
                pass  # Just validate
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by ClickHouseMetadataStore"
            )
        return store


class TruncationCases:
    """Hash truncation length configurations."""

    def case_none(self):
        """No truncation."""
        return None

    def case_16(self):
        """Truncate to 16 characters."""
        return 16


# ============= FIXTURES =============


@pytest_cases.fixture
@parametrize_with_cases("hash_truncation_length", cases=TruncationCases)
def metaxy_config(hash_truncation_length: int | None):
    """Configure hash truncation length for tests."""
    old = MetaxyConfig.get()
    cfg_struct = old.model_dump()
    cfg_struct["hash_truncation_length"] = hash_truncation_length
    new = MetaxyConfig.model_validate(cfg_struct)
    MetaxyConfig.set(new)
    yield new
    MetaxyConfig.set(old)


# ============= TEST: ROOT FEATURES (NO UPSTREAM) =============


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def test_resolve_update_root_feature_requires_samples(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    root_feature: type[BaseFeature],
):
    """Test that resolve_update raises ValueError for root features without samples.

    Root features have no upstream dependencies, so provenance cannot be computed
    from upstream metadata. Users must provide samples with manually computed
    provenance_by_field.
    """
    with store:
        with pytest.raises(ValueError, match="root feature"):
            store.resolve_update(root_feature, lazy=True)


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def test_resolve_update_root_feature_with_samples(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    root_feature: type[BaseFeature],
    graph: FeatureGraph,
):
    """Test resolve_update for root features with provided samples.

    When samples are provided, resolve_update should:
    - Accept the samples with user-provided provenance_by_field
    - Return all samples as "added" (first write)
    - Compute correct metaxy_provenance from the field provenances
    """
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


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_downstream_feature(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test resolve_update for downstream features with upstream dependencies.

    This test verifies that:
    - Upstream metadata is correctly joined
    - Field provenance is computed from upstream dependencies
    - Results match the golden reference implementation
    - Behavior is consistent across all store backends
    """
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


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_detects_changes(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test that resolve_update correctly detects added/changed/removed samples.

    This test:
    1. Writes initial upstream metadata
    2. Computes and writes child metadata
    3. Modifies upstream metadata (add/change/remove samples)
    4. Verifies resolve_update detects all changes correctly
    """
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


@parametrize_with_cases("store", cases=StoreCases)
def test_resolve_update_lazy_execution(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    graph: FeatureGraph,
):
    """Test resolve_update with lazy=True returns lazy frames with correct implementation.

    This test verifies that:
    - lazy=True returns LazyIncrement with lazy frames
    - Lazy frames have correct implementation (POLARS for InMemory, IBIS for SQL stores)
    - Lazy frames can be collected to produce results
    - Results match eager execution
    """

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
            from metaxy.provenance.types import LazyIncrement

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


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("simple_chain", cases=FeatureGraphCases, glob="simple_chain")
def test_resolve_update_with_manual_data_version_override(
    simple_chain: FeaturePlanOutput,
    store: MetadataStore,
    tmp_path: Path,
) -> None:
    """Test that users can manually override data_version columns.

    Demonstrates that:
    - Users can provide custom data_version_by_field (e.g., content hashes)
    - Downstream features recompute only when data_version changes
    - metaxy_provenance columns are tracked separately for audit
    - read_metadata(current_only=True, latest_data_only=True) deduplicates by timestamp
    """
    graph, upstream_features, leaf_plan = simple_chain
    root_feature = list(upstream_features.values())[0]
    leaf_feature = graph.features_by_key[leaf_plan.feature.key]

    with graph.use(), store:
        # Write root feature with custom data_version (e.g., content hash)
        root_data = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "value": ["data1", "data2", "data3"],  # User data column
                "metaxy_provenance_by_field": [
                    {"value": "prov_hash_1"},
                    {"value": "prov_hash_2"},
                    {"value": "prov_hash_3"},
                ],
                "metaxy_data_version_by_field": [
                    {"value": "content_hash_v1"},  # Custom version from content
                    {"value": "content_hash_v2"},
                    {"value": "content_hash_v3"},
                ],
            }
        )
        store.write_metadata(root_feature, root_data)

        # Compute child feature
        increment = store.resolve_update(leaf_feature)
        child_data = increment.added.to_polars()
        assert len(child_data) == 3

        # Write child metadata
        store.write_metadata(leaf_feature, increment.added)

        # Now update root's data_version for sample s1 only (simulating content change)
        # Keep provenance same but change data_version
        root_update = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "value": ["data1"],  # Same user data
                "metaxy_provenance_by_field": [
                    {"value": "prov_hash_1"},  # Same provenance
                ],
                "metaxy_data_version_by_field": [
                    {"value": "content_hash_v1_updated"},  # Changed data version
                ],
                "metaxy_feature_version": [root_feature.feature_version()],
                "metaxy_snapshot_version": [graph.snapshot_version],
            }
        )
        # Sleep to ensure different timestamp (important for deduplication)
        # Using 100ms to be safe on fast systems
        import time

        time.sleep(0.1)
        store.write_metadata(root_feature, root_update)

        # Debug: Check what's in the root feature after update
        all_root = (
            store.read_metadata(root_feature, current_only=False).collect().to_polars()
        )
        s1_records = all_root.filter(pl.col("sample_uid") == "s1")

        # Should have 2 records for s1 with different created_at timestamps
        assert len(s1_records) == 2, f"Expected 2 records for s1, got {len(s1_records)}"

        # Check current_only reads - should get latest record for s1
        current_root = (
            store.read_metadata(root_feature, current_only=True).collect().to_polars()
        )
        print(f"\nCurrent root records (should be 3): {len(current_root)}")
        s1_current = current_root.filter(pl.col("sample_uid") == "s1")
        print(
            f"S1 current record data_version: {s1_current['metaxy_data_version_by_field'][0]}"
        )
        assert len(s1_current) == 1, (
            f"Should have 1 current record for s1, got {len(s1_current)}"
        )

        # Resolve update for child - should detect s1 as changed (data_version changed)
        increment2 = store.resolve_update(leaf_feature)

        # Only s1 should be in changed (data_version changed for s1 in root)
        changed_samples = set(increment2.changed.to_polars()["sample_uid"].to_list())
        assert changed_samples == {"s1"}, f"Expected {{'s1'}}, got {changed_samples}"

        # s2 and s3 should not appear (their data_version didn't change)
        assert len(increment2.added.to_polars()) == 0
        assert len(increment2.removed.to_polars()) == 0


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_idempotency(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test that calling resolve_update multiple times is idempotent.

    After writing the increment from first resolve_update, calling it again
    should return empty increments (no changes).
    """
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
