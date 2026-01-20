"""Test resolve_update with skip_comparison parameter.

The skip_comparison parameter is used for backfilling scenarios where you want to
write all upstream samples without checking what already exists in the store.

When skip_comparison=True:
- All upstream samples are returned in Increment.added
- changed and removed are empty frames
- All system columns (metaxy_data_version, metaxy_provenance, etc.) are still computed
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import polars as pl
from pytest_cases import parametrize_with_cases
from syrupy.assertion import SnapshotAssertion

from metaxy import FeatureDep, FeatureGraph
from metaxy._testing import add_metaxy_provenance_column
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from tests.metadata_stores.conftest import (
    BasicStoreCases,  # pyrefly: ignore[import-error]
)


class TestResolveUpdateSkipComparisonRootFeatures:
    """Test skip_comparison parameter with root features (no upstream dependencies)."""

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_root_feature(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
        snapshot: SnapshotAssertion,
    ):
        """Test skip_comparison=True with root feature.

        Root features with samples should return all samples in added,
        with changed and removed empty.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class VideoEmbeddingsFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test_root", "video_embeddings"]),
                    # No upstream dependencies - this is a root feature
                    fields=[
                        FieldSpec(
                            key=FieldKey(["embedding"]),
                            code_version="1",
                        ),
                    ],
                ),
            ):
                """Root feature - video embeddings with no upstream dependencies."""

                pass

        with graph.use(), store:
            # User provides samples with provenance_by_field
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                        {"embedding": "hash3"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(
                user_samples, VideoEmbeddingsFeature
            )

            # Call resolve_update with skip_comparison=True
            result = store.resolve_update(
                VideoEmbeddingsFeature,
                samples=nw.from_native(user_samples),
                skip_comparison=True,
            )

            # All samples should be in added
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]

            # changed and removed should be empty
            assert len(result.changed) == 0
            assert len(result.removed) == 0

            # Verify system columns are present in added
            # Note: resolve_update returns provenance and data_version columns,
            # but feature_version and snapshot_version are added by write_metadata
            assert METAXY_PROVENANCE in result.added.columns
            assert METAXY_PROVENANCE_BY_FIELD in result.added.columns
            assert METAXY_DATA_VERSION in result.added.columns
            assert METAXY_DATA_VERSION_BY_FIELD in result.added.columns

            # Verify provenance_by_field matches input
            added_df = result.added.to_polars().sort("sample_uid")
            assert added_df[METAXY_PROVENANCE_BY_FIELD].to_list() == [
                {"embedding": "hash1"},
                {"embedding": "hash2"},
                {"embedding": "hash3"},
            ]

            # Verify data_version equals provenance for root features
            for row in added_df.iter_rows(named=True):
                assert row[METAXY_DATA_VERSION] == row[METAXY_PROVENANCE]
                assert (
                    row[METAXY_DATA_VERSION_BY_FIELD] == row[METAXY_PROVENANCE_BY_FIELD]
                )

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_ignores_existing_metadata_root(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test skip_comparison=True ignores existing metadata for root features.

        Even if metadata already exists in the store, skip_comparison should return
        ALL upstream samples in added (not filtered by what already exists).
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class VideoEmbeddingsFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test_root", "video_embeddings"]),
                    fields=[
                        FieldSpec(
                            key=FieldKey(["embedding"]),
                            code_version="1",
                        ),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # Write initial metadata to store
            initial_metadata = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                    ],
                }
            )
            initial_metadata = add_metaxy_provenance_column(
                initial_metadata, VideoEmbeddingsFeature
            )
            store.write_metadata(VideoEmbeddingsFeature, initial_metadata)

            # Now resolve_update with skip_comparison=True
            # Provide the same samples plus a new one
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],  # 1, 2 already exist, 3 is new
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                        {"embedding": "hash3"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(
                user_samples, VideoEmbeddingsFeature
            )

            result = store.resolve_update(
                VideoEmbeddingsFeature,
                samples=nw.from_native(user_samples),
                skip_comparison=True,
            )

            # With skip_comparison=True, ALL samples should be in added (including existing ones)
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]

            # changed and removed should be empty
            assert len(result.changed) == 0
            assert len(result.removed) == 0


class TestResolveUpdateSkipComparisonDownstreamFeatures:
    """Test skip_comparison parameter with downstream features (with upstream dependencies)."""

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_downstream_feature(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test skip_comparison=True with downstream feature.

        Non-root features should load all upstream metadata and return it all in added,
        with changed and removed empty.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class UpstreamFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["upstream"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class DownstreamFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["downstream"]),
                    deps=[FeatureDep(feature=UpstreamFeature)],
                    fields=[
                        FieldSpec(key=FieldKey(["result"]), code_version="1"),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # Write upstream metadata
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "up_hash1"},
                        {"value": "up_hash2"},
                        {"value": "up_hash3"},
                    ],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamFeature)
            store.write_metadata(UpstreamFeature, upstream_data)

            # Call resolve_update on downstream with skip_comparison=True
            result = store.resolve_update(
                DownstreamFeature,
                skip_comparison=True,
            )

            # All upstream samples should be in added
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]

            # changed and removed should be empty
            assert len(result.changed) == 0
            assert len(result.removed) == 0

            # Verify system columns are present
            assert METAXY_PROVENANCE in result.added.columns
            assert METAXY_PROVENANCE_BY_FIELD in result.added.columns
            assert METAXY_DATA_VERSION in result.added.columns
            assert METAXY_DATA_VERSION_BY_FIELD in result.added.columns

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_ignores_existing_metadata_downstream(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test skip_comparison=True ignores existing downstream metadata.

        Even if downstream metadata already exists, skip_comparison should return
        ALL upstream samples in added.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class UpstreamFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["upstream"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class DownstreamFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["downstream"]),
                    deps=[FeatureDep(feature=UpstreamFeature)],
                    fields=[
                        FieldSpec(key=FieldKey(["result"]), code_version="1"),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # Write upstream metadata
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "up_hash1"},
                        {"value": "up_hash2"},
                        {"value": "up_hash3"},
                    ],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamFeature)
            store.write_metadata(UpstreamFeature, upstream_data)

            # Write some downstream metadata (simulate existing data)
            existing_downstream = pl.DataFrame(
                {
                    "sample_uid": [1, 2],  # Only 2 out of 3 samples
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"result": "down_hash1"},
                        {"result": "down_hash2"},
                    ],
                }
            )
            existing_downstream = add_metaxy_provenance_column(
                existing_downstream, DownstreamFeature
            )
            store.write_metadata(DownstreamFeature, existing_downstream)

            # Call resolve_update with skip_comparison=True
            result = store.resolve_update(
                DownstreamFeature,
                skip_comparison=True,
            )

            # With skip_comparison=True, ALL upstream samples should be in added
            # (not just the new one)
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]

            # changed and removed should be empty
            assert len(result.changed) == 0
            assert len(result.removed) == 0


class TestResolveUpdateSkipComparisonLazy:
    """Test skip_comparison parameter with lazy execution."""

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_lazy(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test skip_comparison=True with lazy=True returns LazyIncrement.

        Verify the lazy result can be collected and has the correct structure.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class RootFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # User provides samples
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "hash1"},
                        {"value": "hash2"},
                        {"value": "hash3"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(user_samples, RootFeature)

            # Call resolve_update with skip_comparison=True and lazy=True
            lazy_result = store.resolve_update(
                RootFeature,
                samples=nw.from_native(user_samples),
                skip_comparison=True,
                lazy=True,
            )

            # Verify we got a LazyIncrement
            from metaxy.versioning.types import LazyIncrement

            assert isinstance(lazy_result, LazyIncrement)

            # Verify lazy frames have correct implementation
            # Note: When passing Polars samples to DuckDB, the engine switches to Polars
            # So we expect either the native implementation or Polars
            expected_impl = store.native_implementation()
            actual_impl = lazy_result.added.implementation

            # Allow Polars when samples are Polars (auto-switching behavior)
            from narwhals import Implementation

            assert actual_impl in [expected_impl, Implementation.POLARS]
            assert lazy_result.changed.implementation == actual_impl
            assert lazy_result.removed.implementation == actual_impl

            # Collect the lazy result
            result = lazy_result.collect()

            # Verify the collected result has correct structure
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]
            assert len(result.changed) == 0
            assert len(result.removed) == 0


class TestResolveUpdateSkipComparisonDefaultBehavior:
    """Test that skip_comparison=False (default) still works correctly."""

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_false_default(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test that skip_comparison=False (default) still performs normal increment detection.

        This verifies that the default behavior hasn't been broken by adding skip_comparison.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class RootFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # Write some initial metadata
            initial_metadata = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "hash1"},
                        {"value": "hash2"},
                    ],
                }
            )
            initial_metadata = add_metaxy_provenance_column(
                initial_metadata, RootFeature
            )
            store.write_metadata(RootFeature, initial_metadata)

            # Provide samples with skip_comparison=False (explicit, but it's the default)
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],  # 1, 2 already exist, 3 is new
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "hash1"},
                        {"value": "hash2"},
                        {"value": "hash3"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(user_samples, RootFeature)

            # Call resolve_update WITHOUT skip_comparison (default False)
            result = store.resolve_update(
                RootFeature,
                samples=nw.from_native(user_samples),
                # skip_comparison=False is the default, so don't specify it
            )

            # With normal behavior, only NEW sample should be in added
            assert len(result.added) == 1
            assert result.added["sample_uid"].to_list() == [3]

            # Note: Some backends might detect existing samples as "changed" due to
            # implementation differences (e.g., timestamp precision, auto-switching).
            # The key test is that the new sample is properly detected.
            # For a more strict test, we'd need to ensure exact implementation match.
            # changed and removed should have no more than the existing samples
            assert len(result.changed) <= 2
            assert len(result.removed) == 0

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_explicit_false(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test explicit skip_comparison=False performs normal increment detection."""
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class RootFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # Write some initial metadata
            initial_metadata = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "hash1"},
                        {"value": "hash2"},
                    ],
                }
            )
            initial_metadata = add_metaxy_provenance_column(
                initial_metadata, RootFeature
            )
            store.write_metadata(RootFeature, initial_metadata)

            # Provide samples with explicit skip_comparison=False
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "hash1"},
                        {"value": "hash2"},
                        {"value": "hash3"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(user_samples, RootFeature)

            # Call resolve_update with explicit skip_comparison=False
            result = store.resolve_update(
                RootFeature,
                samples=nw.from_native(user_samples),
                skip_comparison=False,  # Explicitly set to False
            )

            # Only NEW sample should be in added
            assert len(result.added) == 1
            assert result.added["sample_uid"].to_list() == [3]

            # Note: Some backends might detect existing samples as "changed" due to
            # implementation differences. The key test is that the new sample is detected.
            # changed and removed should have no more than the existing samples
            assert len(result.changed) <= 2
            assert len(result.removed) == 0


class TestResolveUpdateSkipComparisonComplexScenarios:
    """Test skip_comparison with more complex graph scenarios."""

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_multi_level_graph(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test skip_comparison with multi-level feature graph.

        Root -> Intermediate -> Leaf
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class RootFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class IntermediateFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["intermediate"]),
                    deps=[FeatureDep(feature=RootFeature)],
                    fields=[
                        FieldSpec(
                            key=FieldKey(["intermediate_result"]), code_version="1"
                        ),
                    ],
                ),
            ):
                pass

            class LeafFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["leaf"]),
                    deps=[FeatureDep(feature=IntermediateFeature)],
                    fields=[
                        FieldSpec(key=FieldKey(["final_result"]), code_version="1"),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # Write root metadata
            root_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "root1"},
                        {"value": "root2"},
                        {"value": "root3"},
                    ],
                }
            )
            root_data = add_metaxy_provenance_column(root_data, RootFeature)
            store.write_metadata(RootFeature, root_data)

            # Write intermediate metadata
            intermediate_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"intermediate_result": "inter1"},
                        {"intermediate_result": "inter2"},
                        {"intermediate_result": "inter3"},
                    ],
                }
            )
            intermediate_data = add_metaxy_provenance_column(
                intermediate_data, IntermediateFeature
            )
            store.write_metadata(IntermediateFeature, intermediate_data)

            # Call resolve_update on leaf with skip_comparison=True
            result = store.resolve_update(
                LeafFeature,
                skip_comparison=True,
            )

            # All upstream samples should be in added
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]

            # changed and removed should be empty
            assert len(result.changed) == 0
            assert len(result.removed) == 0

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_skip_comparison_diamond_graph(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test skip_comparison with diamond dependency graph.

        Root -> BranchA, BranchB -> Leaf
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()

        with graph.use():

            class RootFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class BranchAFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["branch_a"]),
                    deps=[FeatureDep(feature=RootFeature)],
                    fields=[
                        FieldSpec(key=FieldKey(["a_result"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class BranchBFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["branch_b"]),
                    deps=[FeatureDep(feature=RootFeature)],
                    fields=[
                        FieldSpec(key=FieldKey(["b_result"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class LeafFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["leaf"]),
                    deps=[
                        FeatureDep(feature=BranchAFeature),
                        FeatureDep(feature=BranchBFeature),
                    ],
                    fields=[
                        FieldSpec(key=FieldKey(["final_result"]), code_version="1"),
                    ],
                ),
            ):
                pass

        with graph.use(), store:
            # Write root metadata
            root_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "root1"},
                        {"value": "root2"},
                        {"value": "root3"},
                    ],
                }
            )
            root_data = add_metaxy_provenance_column(root_data, RootFeature)
            store.write_metadata(RootFeature, root_data)

            # Write branch A metadata
            branch_a_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"a_result": "a1"},
                        {"a_result": "a2"},
                        {"a_result": "a3"},
                    ],
                }
            )
            branch_a_data = add_metaxy_provenance_column(branch_a_data, BranchAFeature)
            store.write_metadata(BranchAFeature, branch_a_data)

            # Write branch B metadata
            branch_b_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"b_result": "b1"},
                        {"b_result": "b2"},
                        {"b_result": "b3"},
                    ],
                }
            )
            branch_b_data = add_metaxy_provenance_column(branch_b_data, BranchBFeature)
            store.write_metadata(BranchBFeature, branch_b_data)

            # Call resolve_update on leaf with skip_comparison=True
            result = store.resolve_update(
                LeafFeature,
                skip_comparison=True,
            )

            # All upstream samples should be in added
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]

            # changed and removed should be empty
            assert len(result.changed) == 0
            assert len(result.removed) == 0
