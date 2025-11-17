"""Test metadata store provenance calculation against golden reference.

Uses parametric metadata strategies to generate random upstream data, then compares
the provenance calculation from resolve_update against the golden reference
implementation used in the strategies.
"""

import warnings
from collections.abc import Mapping
from typing import TypeAlias

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
from metaxy._testing.parametric import downstream_metadata_strategy
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

FeaturePlanOutput: TypeAlias = tuple[
    FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan
]


class FeaturePlanCases:
    """Test cases for different feature plan configurations."""

    def case_single_upstream(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Simple parent->child feature plan with single upstream.

        Returns:
            Tuple of (graph, upstream_features_dict, child_plan)
        """

        class ParentFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="parent",
                fields=["foo"],
            ),
        ):
            pass

        class ChildFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="child",
                deps=[FeatureDep(feature=ParentFeature)],
                fields=["foo"],
            ),
        ):
            pass

        # Get the feature plan for child
        child_plan = graph.get_feature_plan(ChildFeature.spec().key)

        upstream_features = {
            ParentFeature.spec().key: ParentFeature,
        }

        return graph, upstream_features, child_plan

    def case_two_upstreams(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Feature plan with two upstream dependencies.

        Returns:
            Tuple of (graph, upstream_features_dict, child_plan)
        """

        class Parent1Feature(
            Feature,
            spec=SampleFeatureSpec(
                key="parent1",
                fields=["foo"],
            ),
        ):
            pass

        class Parent2Feature(
            Feature,
            spec=SampleFeatureSpec(
                key="parent2",
                fields=["foo"],
            ),
        ):
            pass

        class ChildFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="child",
                deps=[
                    FeatureDep(feature=Parent1Feature),
                    FeatureDep(feature=Parent2Feature),
                ],
                fields=["foo"],
            ),
        ):
            pass

        # Get the feature plan for child
        child_plan = graph.get_feature_plan(ChildFeature.spec().key)

        upstream_features = {
            Parent1Feature.spec().key: Parent1Feature,
            Parent2Feature.spec().key: Parent2Feature,
        }

        return graph, upstream_features, child_plan


class TruncationCases:
    def case_none(self):
        return None

    def case_8(self):
        return 8


@pytest_cases.fixture
@parametrize_with_cases(
    "hash_truncation_length",
    cases=TruncationCases,
)
def metaxy_config(hash_truncation_length: int | None):
    old = MetaxyConfig.get()
    cfg_struct = old.model_dump()
    cfg_struct["hash_truncation_length"] = hash_truncation_length
    new = MetaxyConfig.model_validate(cfg_struct)
    MetaxyConfig.set(new)
    yield new
    MetaxyConfig.set(old)


def setup_store_with_data(
    empty_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
) -> tuple[MetadataStore, FeaturePlanOutput, pl.DataFrame]:
    """Internal helper that does the actual setup work."""
    # Unpack feature plan configuration
    graph, upstream_features, child_feature_plan = feature_plan_config

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]

    # Feature versions for strategy
    child_version = ChildFeature.feature_version()

    feature_versions = {}
    for feature_key, upstream_feature in upstream_features.items():
        feature_versions[feature_key.to_string()] = upstream_feature.feature_version()
    feature_versions[child_key.to_string()] = child_version

    # Generate test data using the golden strategy
    # Note: Using .example() in test infrastructure is appropriate for generating
    # deterministic test data with pytest-cases parametrization. We suppress the
    # NonInteractiveExampleWarning since this is not interactive exploration.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        example_data = downstream_metadata_strategy(
            child_feature_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=empty_store.hash_algorithm,
            min_rows=5,
            max_rows=20,
        ).example()

    upstream_data, golden_downstream = example_data

    # Write upstream metadata to store
    try:
        with empty_store:
            for feature_key, upstream_feature in upstream_features.items():
                upstream_df = upstream_data[feature_key.to_string()]
                empty_store.write_metadata(upstream_feature, upstream_df)
    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {empty_store.hash_algorithm} not supported by {empty_store}"
        )

    return empty_store, feature_plan_config, golden_downstream


class EmptyStoreCases:
    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_duckdb(self, hash_algorithm: HashAlgorithm, tmp_path):
        try:
            return DuckDBMetadataStore(
                tmp_path / "db.duckdb",
                hash_algorithm=hash_algorithm,
                extensions=["hashfuncs"],
                prefer_native=True,
            )
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by {DuckDBMetadataStore}"
            )

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_clickhouse(self, hash_algorithm: HashAlgorithm, clickhouse_db: str):
        try:
            return ClickHouseMetadataStore(
                connection_string=clickhouse_db,
                hash_algorithm=hash_algorithm,
                prefer_native=True,
            )
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by {ClickHouseMetadataStore}"
            )

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_inmemory(
        self,
        hash_algorithm: HashAlgorithm,
    ) -> InMemoryMetadataStore:
        """InMemory store case."""
        try:
            return InMemoryMetadataStore(
                hash_algorithm=hash_algorithm,
            )
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by {InMemoryMetadataStore}"
            )


@parametrize_with_cases("empty_store", cases=EmptyStoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeaturePlanCases)
def test_store_resolve_update_matches_golden_provenance(
    empty_store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test metadata store provenance calculation matches golden reference.

    This test verifies that resolve_update computes the same provenance hashes
    as the reference Polars calculator implementation.

    The test is parametrized over:
    - Hash algorithms (xxhash64, xxhash32, wyhash, sha256, md5)
    - Hash truncation lengths (None, 8)
    - Feature plans (single_upstream, two_upstreams)

    Args:
        hash_algorithm: Hash algorithm to test
        hash_truncation_length: Optional hash truncation length
        feature_plan_config: Feature plan configuration from cases
        tmp_path: Pytest fixture for temporary directory
    """
    # Setup store with upstream data and get golden reference
    store, (graph, upstream_features, child_feature_plan), golden_downstream = (
        setup_store_with_data(
            empty_store,
            feature_plan_config,
        )
    )

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]

    # Get child version
    child_version = ChildFeature.feature_version()

    # Call resolve_update to compute provenance
    with store:
        try:
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        added_df = increment.added.lazy().collect().to_polars()
        # Get ID columns from the feature plan
        id_columns = list(child_feature_plan.feature.id_columns)
        # Sort both DataFrames by ID columns for comparison
        added_sorted = added_df.sort(id_columns)
        golden_sorted = golden_downstream.sort(id_columns)

        # Select only the columns that exist in both (resolve_update may not return all metadata columns)
        # Exclude metaxy_created_at since it's a timestamp that differs between generation times
        from metaxy.models.constants import METAXY_CREATED_AT

        common_columns = [
            col
            for col in added_sorted.columns
            if col in golden_sorted.columns and col != METAXY_CREATED_AT
        ]
        added_selected = added_sorted.select(common_columns)
        golden_selected = golden_sorted.select(common_columns)

        # Use Polars testing utility to compare DataFrames
        # This will check all columns including the provenance_by_field struct
        pl_testing.assert_frame_equal(
            added_selected,
            golden_selected,
            check_row_order=True,
            check_column_order=False,
        )


# ============= TEST: DEDUPLICATION WITH DUPLICATES =============


@parametrize_with_cases("empty_store", cases=EmptyStoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeaturePlanCases)
def test_golden_reference_with_duplicate_timestamps(
    empty_store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test golden reference comparison with upstream metadata containing duplicate timestamps.

    This test verifies that:
    1. When upstream metadata contains multiple versions of the same sample (duplicates)
    2. The resolve_update correctly deduplicates using keep_latest_by_group
    3. The computed provenance still matches the golden reference (which has no duplicates)

    This proves that the deduplication logic correctly filters older versions before
    computing provenance.
    """
    # Setup store with upstream data and get golden reference
    store, (graph, upstream_features, child_feature_plan), golden_downstream = (
        setup_store_with_data(
            empty_store,
            feature_plan_config,
        )
    )

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    with store:
        try:
            from datetime import timedelta

            import polars as pl

            from metaxy.models.constants import METAXY_CREATED_AT

            # Add older duplicates to upstream metadata
            for feature_key, upstream_feature in upstream_features.items():
                # Read existing upstream data
                existing_df = (
                    store.read_metadata(upstream_feature).lazy().collect().to_polars()
                )

                # Create older duplicates (same IDs, older timestamps)
                older_df = existing_df.clone()
                older_df = older_df.with_columns(
                    (pl.col(METAXY_CREATED_AT) - timedelta(hours=2)).alias(
                        METAXY_CREATED_AT
                    )
                )

                # Modify a field value to ensure different provenance
                # This tests that older version is NOT used
                user_fields = [
                    col
                    for col in older_df.columns
                    if not col.startswith("metaxy_") and col != "sample_uid"
                ]
                if user_fields:
                    field = user_fields[0]
                    older_df = older_df.with_columns(
                        pl.when(pl.col(field).is_not_null())
                        .then(pl.col(field).cast(pl.Utf8) + "_DUPLICATE_OLD")
                        .otherwise(pl.col(field))
                        .alias(field)
                    )

                # Write the older duplicates
                store.write_metadata(upstream_feature, older_df)

                # Now store has 2 versions per sample:
                # - Original (newer) - should be used
                # - Duplicate (older, modified) - should be ignored

            # Call resolve_update - should use only latest versions
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        added_df = increment.added.lazy().collect().to_polars()

        # Get ID columns from the feature plan
        id_columns = list(child_feature_plan.feature.id_columns)

        # Sort both DataFrames by ID columns for comparison
        added_sorted = added_df.sort(id_columns)
        golden_sorted = golden_downstream.sort(id_columns)

        # Exclude metaxy_created_at since it's a timestamp
        common_columns = [
            col
            for col in added_sorted.columns
            if col in golden_sorted.columns and col != METAXY_CREATED_AT
        ]
        added_selected = added_sorted.select(common_columns)
        golden_selected = golden_sorted.select(common_columns)

        # Verify that computed provenance matches golden reference
        # This proves deduplication worked correctly - only latest versions were used
        pl_testing.assert_frame_equal(
            added_selected,
            golden_selected,
            check_row_order=True,
            check_column_order=False,
        )


@parametrize_with_cases("empty_store", cases=EmptyStoreCases)
def test_golden_reference_with_all_duplicates_same_timestamp(
    empty_store: MetadataStore,
    metaxy_config: MetaxyConfig,
    graph: FeatureGraph,
):
    """Test golden reference when all upstream samples have duplicate entries with same timestamp.

    This is an edge case where every sample has multiple versions with identical timestamps.
    The deduplication should still work deterministically.
    """

    # Create simple feature graph
    class ParentFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["value"],
        ),
    ):
        pass

    class ChildFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature=ParentFeature)],
            fields=["computed"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)

    # Generate golden reference data
    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=empty_store.hash_algorithm,
            min_rows=5,
            max_rows=10,
        ).example()

    parent_df = upstream_data["parent"]

    try:
        with empty_store:
            from datetime import datetime

            import polars as pl

            from metaxy.models.constants import METAXY_CREATED_AT

            # Create duplicates with SAME timestamp for ALL samples
            same_timestamp = datetime.now()

            # Write first version
            version1 = parent_df.with_columns(
                pl.lit(same_timestamp).alias(METAXY_CREATED_AT)
            )
            empty_store.write_metadata(ParentFeature, version1)

            # Create second version with same timestamp but different provenance
            # We modify the provenance columns to simulate different data
            version2 = parent_df.with_columns(
                pl.lit(same_timestamp).alias(METAXY_CREATED_AT),
                # Modify provenance to make it different (simulate different underlying data)
                (pl.col("metaxy_provenance").cast(pl.Utf8) + "_DUP").alias(
                    "metaxy_provenance"
                ),
            )

            # Write duplicates with same timestamp
            empty_store.write_metadata(ParentFeature, version2)

            # Now every sample has 2 versions with same timestamp
            # Call resolve_update - should pick one deterministically
            increment = empty_store.resolve_update(
                ChildFeature,
                target_version=ChildFeature.feature_version(),
                snapshot_version=graph.snapshot_version,
            )

            # Verify we got results (deterministic even with same timestamps)
            added_df = increment.added.lazy().collect().to_polars()
            assert len(added_df) > 0, (
                "Expected at least some samples after deduplication"
            )

            # With duplicates at same timestamp, we should still get the original count
            # (deduplication picks one version per sample)
            assert len(added_df) == len(parent_df), (
                f"Expected {len(parent_df)} deduplicated samples, got {len(added_df)}"
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {empty_store.hash_algorithm} not supported by {empty_store}"
        )


@parametrize_with_cases("empty_store", cases=EmptyStoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeaturePlanCases)
def test_golden_reference_partial_duplicates(
    empty_store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test golden reference with only some upstream samples having duplicates.

    This test ensures that:
    - Samples with duplicates are correctly deduplicated
    - Samples without duplicates pass through unchanged
    - The final provenance matches golden reference
    """
    # Setup store with upstream data
    store, (graph, upstream_features, child_feature_plan), golden_downstream = (
        setup_store_with_data(
            empty_store,
            feature_plan_config,
        )
    )

    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    try:
        with store:
            from datetime import timedelta

            import polars as pl

            from metaxy.models.constants import METAXY_CREATED_AT

            # Add older duplicates for only HALF of the samples in each upstream
            for feature_key, upstream_feature in upstream_features.items():
                existing_df = (
                    store.read_metadata(upstream_feature).lazy().collect().to_polars()
                )

                # Get half of samples
                num_samples = len(existing_df)
                half_count = num_samples // 2

                if half_count > 0:
                    # Take first half
                    samples_to_duplicate = existing_df.head(half_count)

                    # Create older version
                    older_df = samples_to_duplicate.with_columns(
                        (pl.col(METAXY_CREATED_AT) - timedelta(hours=1)).alias(
                            METAXY_CREATED_AT
                        )
                    )

                    # Write older duplicates
                    store.write_metadata(upstream_feature, older_df)

                # Now store has:
                # - First half of samples: 2 versions each (newer and older)
                # - Second half of samples: 1 version each (original only)

            # Call resolve_update
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

            added_df = increment.added.lazy().collect().to_polars()
            id_columns = list(child_feature_plan.feature.id_columns)

            # Sort both for comparison
            added_sorted = added_df.sort(id_columns)
            golden_sorted = golden_downstream.sort(id_columns)

            # Exclude timestamp
            common_columns = [
                col
                for col in added_sorted.columns
                if col in golden_sorted.columns and col != METAXY_CREATED_AT
            ]
            added_selected = added_sorted.select(common_columns)
            golden_selected = golden_sorted.select(common_columns)

            # Verify provenance matches golden reference
            pl_testing.assert_frame_equal(
                added_selected,
                golden_selected,
                check_row_order=True,
                check_column_order=False,
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")
