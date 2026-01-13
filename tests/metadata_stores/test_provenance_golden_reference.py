"""Test metadata store provenance calculation against golden reference.

This file tests the CORRECTNESS of provenance calculations by comparing store
implementations against a golden reference. It focuses on:
- Verifying stores produce correct provenance (matches golden reference)
- Testing deduplication logic (keep_latest_by_group)
- Testing edge cases (duplicates, partial duplicates, etc.)

Hash algorithm and truncation testing is handled in test_hash_algorithms.py.
Store-specific behavior testing is handled in test_resolve_update.py.

The goal here is to verify that the core provenance calculation is correct,
not to test every possible combination of hash algorithm × store × truncation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, TypeAlias

import polars as pl
import polars.testing as pl_testing
import pytest
import pytest_cases
from hypothesis.errors import NonInteractiveExampleWarning
from pytest_cases import parametrize_with_cases
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldSpec,
    LineageRelationship,
)
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.parametric import downstream_metadata_strategy
from metaxy._utils import collect_to_polars
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DELETED_AT,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.field import SpecialFieldDep
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FieldKey
from metaxy.versioning.types import HashAlgorithm, Increment

if TYPE_CHECKING:
    pass

FeaturePlanOutput: TypeAlias = tuple[
    FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan
]

# A sequence of plans to test in order (for definition change scenarios)
FeaturePlanSequence: TypeAlias = list[FeaturePlanOutput]


class FeaturePlanCases:
    """Test cases for different feature plan configurations.

    Each case returns a list of FeaturePlanOutput tuples. Most cases return a single
    plan, but cases testing definition changes (e.g., code_version bumps) return
    multiple plans that should be tested sequentially.
    """

    def case_single_upstream(self, graph: FeatureGraph) -> FeaturePlanSequence:
        """Simple parent->child feature plan with single upstream."""

        class ParentFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="parent",
                fields=["foo"],
            ),
        ):
            sample_uid: str

        class ChildFeature(
            BaseFeature,
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

        return [(graph, upstream_features, child_plan)]

    def case_two_upstreams(self, graph: FeatureGraph) -> FeaturePlanSequence:
        """Feature plan with two upstream dependencies."""

        class Parent1Feature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="parent1",
                fields=["foo"],
            ),
        ):
            sample_uid: str

        class Parent2Feature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="parent2",
                fields=["foo"],
            ),
        ):
            sample_uid: str

        class ChildFeature(
            BaseFeature,
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

        return [(graph, upstream_features, child_plan)]

    def case_aggregation_plus_identity(
        self, graph: FeatureGraph
    ) -> FeaturePlanSequence:
        """Feature plan with aggregation + identity dependencies.

        Scenario:
        - SensorReadings: N:1 aggregation to aggregate by sensor_id
        - SensorInfo: 1:1 identity join with sensor metadata
        """

        class SensorInfo(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="sensor_info",
                id_columns=("sensor_id",),
                fields=[FieldSpec(key=FieldKey(["location"]), code_version="1")],
            ),
        ):
            sensor_id: str

        class SensorReadings(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="sensor_readings",
                id_columns=("sensor_id", "reading_id"),
                fields=[FieldSpec(key=FieldKey(["temperature"]), code_version="1")],
            ),
        ):
            sensor_id: str
            reading_id: str

        class AggregatedSensorStats(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="aggregated_sensor_stats",
                id_columns=("sensor_id",),
                deps=[
                    FeatureDep(
                        feature=SensorReadings,
                        lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                    ),
                    FeatureDep(
                        feature=SensorInfo,
                        lineage=LineageRelationship.identity(),
                    ),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["enriched_stat"]),
                        code_version="1",
                        deps=SpecialFieldDep.ALL,
                    ),
                ],
            ),
        ):
            sensor_id: str

        child_plan = graph.get_feature_plan(AggregatedSensorStats.spec().key)

        upstream_features = {
            SensorInfo.spec().key: SensorInfo,
            SensorReadings.spec().key: SensorReadings,
        }

        return [(graph, upstream_features, child_plan)]

    def case_expansion_plus_identity(self, graph: FeatureGraph) -> FeaturePlanSequence:
        """Feature plan with expansion + identity dependencies.

        Scenario:
        - Video: 1:N expansion (one video → many frames)
        - VideoMetadata: 1:1 identity join with video metadata
        """

        class Video(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="video",
                id_columns=("video_id",),
                fields=[FieldSpec(key=FieldKey(["content"]), code_version="1")],
            ),
        ):
            video_id: str

        class VideoMetadata(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="video_metadata",
                id_columns=("video_id",),
                fields=[FieldSpec(key=FieldKey(["category"]), code_version="1")],
            ),
        ):
            video_id: str

        class EnrichedFrames(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="enriched_frames",
                id_columns=("video_id",),  # At parent level for golden reference
                deps=[
                    FeatureDep(
                        feature=Video,
                        lineage=LineageRelationship.expansion(on=["video_id"]),
                    ),
                    FeatureDep(
                        feature=VideoMetadata,
                        lineage=LineageRelationship.identity(),
                    ),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frame_feature"]),
                        code_version="1",
                        deps=SpecialFieldDep.ALL,
                    ),
                ],
            ),
        ):
            video_id: str

        child_plan = graph.get_feature_plan(EnrichedFrames.spec().key)

        upstream_features = {
            Video.spec().key: Video,
            VideoMetadata.spec().key: VideoMetadata,
        }

        return [(graph, upstream_features, child_plan)]

    def case_aggregation_plus_expansion(
        self, graph: FeatureGraph
    ) -> FeaturePlanSequence:
        """Feature plan with aggregation + expansion dependencies.

        Scenario:
        - SensorReadings: N:1 aggregation by config_id
        - VideoSource: 1:N expansion by config_id
        """

        class SensorReadings(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="sensor_readings_v2",
                id_columns=("config_id", "reading_id"),
                fields=[FieldSpec(key=FieldKey(["measurement"]), code_version="1")],
            ),
        ):
            config_id: str
            reading_id: str

        class VideoSource(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="video_source",
                id_columns=("config_id",),
                fields=[FieldSpec(key=FieldKey(["video_data"]), code_version="1")],
            ),
        ):
            config_id: str

        class CombinedFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="combined_feature",
                id_columns=("config_id",),  # At parent level for golden reference
                deps=[
                    FeatureDep(
                        feature=SensorReadings,
                        lineage=LineageRelationship.aggregation(on=["config_id"]),
                    ),
                    FeatureDep(
                        feature=VideoSource,
                        lineage=LineageRelationship.expansion(on=["config_id"]),
                    ),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["combined_output"]),
                        code_version="1",
                        deps=SpecialFieldDep.ALL,
                    ),
                ],
            ),
        ):
            config_id: str

        child_plan = graph.get_feature_plan(CombinedFeature.spec().key)

        upstream_features = {
            SensorReadings.spec().key: SensorReadings,
            VideoSource.spec().key: VideoSource,
        }

        return [(graph, upstream_features, child_plan)]

    def case_upstream_field_definition_change(self) -> FeaturePlanSequence:
        """Test upstream field definition change (code_version bump).

        This case returns TWO plans to test sequentially:
        1. V1: Parent feature with field code_version="1"
        2. V2: Same parent feature with field code_version="2"

        The store accumulates data - V2 upstream data is written on top of V1,
        and keep_latest_by_group should pick the newer V2 data.
        """
        # === V1: Initial definition ===
        graph_v1 = FeatureGraph()
        with graph_v1.use():

            class ParentV1(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="parent_defn_change",
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                sample_uid: str

            class ChildV1(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="child_defn_change",
                    deps=[FeatureDep(feature=ParentV1)],
                    fields=[FieldSpec(key=FieldKey(["computed"]), code_version="1")],
                ),
            ):
                pass

            child_plan_v1 = graph_v1.get_feature_plan(ChildV1.spec().key)
            upstream_features_v1 = {ParentV1.spec().key: ParentV1}

        # === V2: Field definition changes (code_version bump) ===
        graph_v2 = FeatureGraph()
        with graph_v2.use():

            class ParentV2(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="parent_defn_change",
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="2")
                    ],  # Changed!
                ),
            ):
                sample_uid: str

            class ChildV2(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="child_defn_change",
                    deps=[FeatureDep(feature=ParentV2)],
                    fields=[FieldSpec(key=FieldKey(["computed"]), code_version="1")],
                ),
            ):
                pass

            child_plan_v2 = graph_v2.get_feature_plan(ChildV2.spec().key)
            upstream_features_v2 = {ParentV2.spec().key: ParentV2}

        return [
            (graph_v1, upstream_features_v1, child_plan_v1),
            (graph_v2, upstream_features_v2, child_plan_v2),
        ]

    def case_optional_dependency(self, graph: FeatureGraph) -> FeaturePlanSequence:
        """Feature plan with optional dependency (tests left join behavior and NULL handling)."""

        class RequiredParent(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="required_parent",
                fields=["foo"],
            ),
        ):
            sample_uid: str

        class OptionalParent(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="optional_parent",
                fields=["bar"],
            ),
        ):
            sample_uid: str

        class ChildFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="child",
                deps=[
                    FeatureDep(feature=RequiredParent, optional=False),
                    FeatureDep(feature=OptionalParent, optional=True),
                ],
                fields=["computed"],
            ),
        ):
            pass

        # Get the feature plan for child
        child_plan = graph.get_feature_plan(ChildFeature.spec().key)

        upstream_features = {
            RequiredParent.spec().key: RequiredParent,
            OptionalParent.spec().key: OptionalParent,
        }

        return [(graph, upstream_features, child_plan)]


# Removed: TruncationCases and metaxy_config fixture
# Hash truncation is now tested in test_hash_algorithms.py


def generate_plan_data(
    store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
    base_upstream_data: dict[str, pl.DataFrame] | None = None,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Generate upstream data and golden downstream for a single plan.

    Args:
        store: The metadata store (used for hash algorithm)
        feature_plan_config: The feature plan configuration
        base_upstream_data: Optional base upstream data to transform. If provided,
            the upstream data will be derived from this (same IDs, updated provenance)
            rather than generating new random data. This is useful for multi-plan
            sequences where we need consistent IDs across iterations.

    Returns:
        Tuple of (upstream_data dict, golden_downstream DataFrame)
    """
    import narwhals as nw

    from metaxy.models.constants import (
        METAXY_CREATED_AT,
        METAXY_FEATURE_VERSION,
    )
    from metaxy.versioning.polars import PolarsVersioningEngine

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

    if base_upstream_data is None:
        # Generate fresh test data using the golden strategy
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
            example_data = downstream_metadata_strategy(
                child_feature_plan,
                feature_versions=feature_versions,
                snapshot_version=graph.snapshot_version,
                hash_algorithm=store.hash_algorithm,
                min_rows=5,
                max_rows=20,
            ).example()
        return example_data
    else:
        # Transform base upstream data for this plan's feature versions
        # Update provenance columns to reflect new feature versions
        from datetime import datetime, timedelta, timezone

        upstream_data: dict[str, pl.DataFrame] = {}

        for feature_key, upstream_feature in upstream_features.items():
            key_str = feature_key.to_string()
            if key_str not in base_upstream_data:
                raise ValueError(f"Base upstream data missing for {key_str}")

            base_df = base_upstream_data[key_str]
            new_feature_version = upstream_feature.feature_version()

            # Update metaxy columns to reflect new feature version
            # The provenance will change because feature_version changes
            updated_df = base_df.with_columns(
                pl.lit(new_feature_version).alias(METAXY_FEATURE_VERSION),
                # Update timestamp to be newer so keep_latest_by_group picks this version
                (pl.col(METAXY_CREATED_AT) + timedelta(hours=1)).alias(
                    METAXY_CREATED_AT
                ),
            )

            upstream_data[key_str] = updated_df

        # Calculate golden downstream using PolarsVersioningEngine
        engine = PolarsVersioningEngine(plan=child_feature_plan)
        upstream_dict = {
            FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()
        }

        downstream_nw = engine.load_upstream_with_provenance(
            upstream=upstream_dict,
            hash_algo=store.hash_algorithm,
            filters=None,
        ).collect()

        # Add downstream feature version and snapshot version
        downstream_df = downstream_nw.with_columns(
            nw.lit(child_version).alias(METAXY_FEATURE_VERSION),
            nw.lit(graph.snapshot_version).alias("metaxy_snapshot_version"),
            nw.lit(child_feature_plan.feature.feature_spec_version).alias(
                "metaxy_feature_spec_version"
            ),
            nw.lit(datetime.now(timezone.utc)).alias(METAXY_CREATED_AT),
        )

        return upstream_data, downstream_df.to_native()


def write_upstream_to_store(
    store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
    upstream_data: dict[str, pl.DataFrame],
) -> None:
    """Write upstream data to store for a single plan."""
    graph, upstream_features, _ = feature_plan_config

    for feature_key, upstream_feature in upstream_features.items():
        upstream_df = upstream_data[feature_key.to_string()]
        store.write_metadata(upstream_feature, upstream_df)


def setup_store_with_data(
    empty_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
) -> tuple[MetadataStore, FeaturePlanOutput, pl.DataFrame]:
    """Legacy helper for single-plan setup. Wraps the new helpers."""
    graph, upstream_features, child_feature_plan = feature_plan_config
    upstream_data, golden_downstream = generate_plan_data(
        empty_store, feature_plan_config
    )

    try:
        with empty_store:
            # Use graph.use() to make it the current graph (needed for write_metadata)
            with graph.use():
                write_upstream_to_store(empty_store, feature_plan_config, upstream_data)
    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {empty_store.hash_algorithm} not supported by {empty_store}"
        )

    return empty_store, feature_plan_config, golden_downstream


# Removed: EmptyStoreCases with hash algorithm parametrization
# Now using simplified fixtures from conftest.py
# Hash algorithm × store combinations are tested in test_hash_algorithms.py


def assert_increment_matches_golden(
    actual: Increment,
    golden: Increment,
    id_columns: list[str],
) -> None:
    """Assert that actual increment matches golden increment.

    Compares all parts of the increment: added, changed, and removed.
    All Increment fields are required (not Optional), so we compare them directly.
    """
    from metaxy.models.constants import METAXY_CREATED_AT

    def compare_frames(
        actual_df: pl.DataFrame,
        golden_df: pl.DataFrame,
        frame_name: str,
    ) -> None:
        """Compare two DataFrames, excluding timestamp columns."""
        # Sort by all comparable columns to ensure deterministic ordering
        # (id_columns alone may not be unique for aggregation lineage)
        sort_columns = [
            col
            for col in actual_df.columns
            if col in golden_df.columns and col != METAXY_CREATED_AT
        ]
        actual_sorted = actual_df.sort(sort_columns)
        golden_sorted = golden_df.sort(sort_columns)

        # Select only common columns, excluding timestamps
        common_columns = [
            col
            for col in actual_sorted.columns
            if col in golden_sorted.columns and col != METAXY_CREATED_AT
        ]
        actual_selected = actual_sorted.select(common_columns)
        golden_selected = golden_sorted.select(common_columns)

        pl_testing.assert_frame_equal(
            actual_selected,
            golden_selected,
            check_row_order=True,
            check_column_order=False,
        )

    # Compare added
    actual_added = actual.added.lazy().collect().to_polars()
    golden_added = golden.added.lazy().collect().to_polars()
    compare_frames(actual_added, golden_added, "added")

    # Compare changed
    actual_changed = actual.changed.lazy().collect().to_polars()
    golden_changed = golden.changed.lazy().collect().to_polars()
    compare_frames(actual_changed, golden_changed, "changed")

    # Compare removed
    actual_removed = actual.removed.lazy().collect().to_polars()
    golden_removed = golden.removed.lazy().collect().to_polars()
    compare_frames(actual_removed, golden_removed, "removed")


def assert_resolve_update_matches_golden(
    store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
    golden_increment: Increment,
    snapshot: SnapshotAssertion | None = None,
    snapshot_suffix: str = "",
) -> None:
    """Assert that resolve_update result matches golden increment for a single plan.

    Args:
        store: The metadata store to test
        feature_plan_config: Feature plan configuration tuple
        golden_increment: The expected increment from golden reference
        snapshot: Optional syrupy snapshot fixture for recording provenance values
        snapshot_suffix: Optional suffix for snapshot names (e.g., iteration number)
    """
    from metaxy.models.constants import (
        METAXY_DATA_VERSION,
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_PROVENANCE,
        METAXY_PROVENANCE_BY_FIELD,
    )

    graph, upstream_features, child_feature_plan = feature_plan_config

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    # Call resolve_update to compute provenance (uses default/native engine)
    # Use graph.use() to make it the current graph (needed for resolve_update)
    with graph.use():
        actual_increment = store.resolve_update(
            ChildFeature,
            target_version=child_version,
            snapshot_version=graph.snapshot_version,
        )

    id_columns = list(child_feature_plan.feature.id_columns)
    assert_increment_matches_golden(actual_increment, golden_increment, id_columns)

    # Additional safeguard: also verify with polars engine explicitly
    # This ensures both native and polars engines produce the same result
    with graph.use():
        polars_increment = store.resolve_update(
            ChildFeature,
            target_version=child_version,
            snapshot_version=graph.snapshot_version,
            versioning_engine="polars",
        )
    assert_increment_matches_golden(polars_increment, golden_increment, id_columns)

    # Record snapshot of provenance values for regression testing
    if snapshot is not None:
        # Extract provenance-related columns from golden increment
        provenance_cols = [
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE,
            METAXY_PROVENANCE_BY_FIELD,
        ]

        added_df = golden_increment.added.lazy().collect().to_polars()

        # Select ID columns and provenance columns that exist
        available_cols = [
            c for c in id_columns + provenance_cols if c in added_df.columns
        ]
        if available_cols:
            # Sort by all columns for deterministic output
            provenance_df = added_df.select(available_cols).sort(available_cols)

            # Convert to a serializable format for snapshot
            # Use list of dicts for readable snapshot
            snapshot_data = provenance_df.to_dicts()

            snapshot_name = f"provenance{snapshot_suffix}"
            assert snapshot_data == snapshot(name=snapshot_name)


def compute_golden_increment(
    child_feature_plan: FeaturePlan,
    upstream_data: dict[str, pl.DataFrame],
    current_downstream: pl.DataFrame | None,
    hash_algorithm: HashAlgorithm,
) -> Increment:
    """Compute golden increment using PolarsVersioningEngine.

    This is the reference implementation - all store implementations should
    produce the same result.

    Args:
        child_feature_plan: The feature plan for the downstream feature
        upstream_data: Dict mapping upstream feature key strings to DataFrames
        current_downstream: Existing downstream data (None if first iteration)
        hash_algorithm: Hash algorithm to use

    Returns:
        Golden Increment computed by PolarsVersioningEngine
    """
    import narwhals as nw

    from metaxy.versioning.polars import PolarsVersioningEngine

    engine = PolarsVersioningEngine(plan=child_feature_plan)

    # Convert upstream data to Narwhals LazyFrames with FeatureKey keys
    upstream_nw = {
        FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()
    }

    # Convert current downstream to Narwhals if present
    current_nw = (
        nw.from_native(current_downstream.lazy())
        if current_downstream is not None
        else None
    )

    # Use the engine to compute the increment
    added, changed, removed, _ = engine.resolve_increment_with_provenance(
        current=current_nw,
        upstream=upstream_nw,
        hash_algorithm=hash_algorithm,
        filters={},
        sample=None,
    )

    # Collect lazy frames and convert to Increment
    # When changed/removed are None (first increment), create empty DataFrames with same schema
    added_collected = added.collect()
    if changed is not None:
        changed_collected = changed.collect()
    else:
        # Create empty DataFrame with same schema as added
        changed_collected = added_collected.head(0)
    if removed is not None:
        removed_collected = removed.collect()
    else:
        # Create empty DataFrame with same schema as added
        removed_collected = added_collected.head(0)

    return Increment(
        added=added_collected,
        changed=changed_collected,
        removed=removed_collected,
    )


@parametrize_with_cases("feature_plan_sequence", cases=FeaturePlanCases)
def test_store_resolve_update_matches_golden_provenance(
    any_store: MetadataStore,
    feature_plan_sequence: FeaturePlanSequence,
):
    """Test metadata store provenance calculation matches golden reference.

    The golden reference is computed using PolarsVersioningEngine.resolve_increment_with_provenance.
    All store implementations should produce the same result.

    For multi-plan sequences (e.g., definition changes), iterates through each plan
    without clearing the store between iterations - simulating real-world usage where
    new data is written on top of existing data.
    """
    try:
        with any_store:
            # Track current downstream data for golden increment calculation
            # Key: feature_version -> downstream DataFrame
            # This mirrors what the store does: filter by feature_version
            current_downstream_by_version: dict[str, pl.DataFrame] = {}
            # Track base upstream data for multi-plan sequences (to keep same IDs)
            base_upstream_data: dict[str, pl.DataFrame] | None = None

            for i, plan_config in enumerate(feature_plan_sequence):
                graph, upstream_features, child_feature_plan = plan_config
                child_key = child_feature_plan.feature.key
                ChildFeature = graph.features_by_key[child_key]
                child_version = ChildFeature.feature_version()

                # Generate upstream data for this plan
                # For first plan, generate fresh data; for subsequent plans, derive from base
                if i == 0:
                    upstream_data, _ = generate_plan_data(any_store, plan_config)
                    # Store as base for subsequent plans
                    base_upstream_data = upstream_data
                else:
                    # Use base upstream data to ensure same IDs
                    upstream_data, _ = generate_plan_data(
                        any_store, plan_config, base_upstream_data=base_upstream_data
                    )

                # Get current downstream for THIS feature version (like the store does)
                current_downstream = current_downstream_by_version.get(child_version)

                # Compute golden increment using PolarsVersioningEngine
                golden_increment = compute_golden_increment(
                    child_feature_plan,
                    upstream_data,
                    current_downstream,
                    any_store.hash_algorithm,
                )

                # Write upstream data to store (accumulates)
                with graph.use():
                    write_upstream_to_store(any_store, plan_config, upstream_data)

                # Assert store's resolve_update matches golden increment
                assert_resolve_update_matches_golden(
                    any_store, plan_config, golden_increment
                )

                # Update current downstream for this feature version
                # Use the golden added/changed to build current state
                added_df = golden_increment.added.lazy().collect().to_polars()
                if golden_increment.changed is not None:
                    changed_df = golden_increment.changed.lazy().collect().to_polars()
                    new_downstream = pl.concat([added_df, changed_df])
                else:
                    new_downstream = added_df
                current_downstream_by_version[child_version] = new_downstream

                # Also write downstream to store so store can detect changes in next iteration
                with graph.use():
                    any_store.write_metadata(ChildFeature, new_downstream)

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}"
        )


def test_soft_deleted_rows_filtered_by_default(any_store: MetadataStore):
    """Soft-deleted metadata rows should be hidden by default and opt-in with include_deleted."""

    class SoftDeleteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="soft_delete",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    initial_df = pl.DataFrame(
        {
            "sample_uid": ["a", "b"],
            "value": [1, 2],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p1"}, {"value": "p2"}],
            METAXY_CREATED_AT: [base_time, base_time],
        }
    )

    soft_deletes_df = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [1],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_del"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=1)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=1)],
        }
    )

    with any_store:
        any_store.write_metadata(SoftDeleteFeature, initial_df)
        any_store.write_metadata(SoftDeleteFeature, soft_deletes_df)

        active = any_store.read_metadata(SoftDeleteFeature).collect().to_polars()
        assert active.filter(pl.col("sample_uid") == "a").is_empty()
        assert active[METAXY_DELETED_AT].is_null().all()

        with_deleted = (
            any_store.read_metadata(SoftDeleteFeature, include_soft_deleted=True)
            .collect()
            .to_polars()
        )
        assert set(with_deleted["sample_uid"]) == {"a", "b"}
        deleted_row = with_deleted.filter(pl.col("sample_uid") == "a")
        assert deleted_row[METAXY_DELETED_AT].is_null().any() is False
        active_row = with_deleted.filter(pl.col("sample_uid") == "b")
        assert active_row[METAXY_DELETED_AT].is_null().all()


def test_write_delete_write_sequence(any_store: MetadataStore):
    """Test write->delete->write sequence: latest write is visible, soft deletes is hidden by default."""

    class WriteDeleteWriteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="write_delete_write",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # First write
    write1_df = pl.DataFrame(
        {
            "sample_uid": ["x"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )

    # Delete (soft delete)
    delete_df = pl.DataFrame(
        {
            "sample_uid": ["x"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_delete"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=1)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=1)],
        }
    )

    # Second write (resurrection)
    write2_df = pl.DataFrame(
        {
            "sample_uid": ["x"],
            "value": [200],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write2"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=2)],
        }
    )

    with any_store:
        any_store.write_metadata(WriteDeleteWriteFeature, write1_df)
        any_store.write_metadata(WriteDeleteWriteFeature, delete_df)
        any_store.write_metadata(WriteDeleteWriteFeature, write2_df)

        # Default: should see only the latest write (value=200)
        active = any_store.read_metadata(WriteDeleteWriteFeature).collect().to_polars()
        assert len(active) == 1
        assert active["sample_uid"][0] == "x"
        assert active["value"][0] == 200
        assert active[METAXY_DELETED_AT].is_null().all()

        # With deleted: should see latest (write2) because deduplication picks it over soft deletes
        with_deleted = (
            any_store.read_metadata(WriteDeleteWriteFeature, include_soft_deleted=True)
            .collect()
            .to_polars()
            .sort(METAXY_CREATED_AT)
        )
        assert len(with_deleted) == 1
        assert with_deleted["sample_uid"][0] == "x"
        assert with_deleted["value"][0] == 200
        assert with_deleted[METAXY_DELETED_AT].is_null().all()


def test_write_delete_write_delete_sequence(any_store: MetadataStore):
    """Test write->delete->write->delete: no active rows by default, latest write visible with filter."""

    class WriteDeleteWriteDeleteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="write_delete_write_delete",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # First write
    write1_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )

    # First delete
    delete1_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_delete1"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=1)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=1)],
        }
    )

    # Second write
    write2_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [200],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write2"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=2)],
        }
    )

    # Second delete
    delete2_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [200],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_delete2"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=3)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=3)],
        }
    )

    with any_store:
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, write1_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, delete1_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, write2_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, delete2_df)

        # Default: no active rows (latest is a soft delete)
        active = (
            any_store.read_metadata(WriteDeleteWriteDeleteFeature).collect().to_polars()
        )
        assert active.is_empty()

        # With deleted: should see the latest soft delete (value=200 deleted)
        with_deleted = (
            any_store.read_metadata(
                WriteDeleteWriteDeleteFeature, include_soft_deleted=True
            )
            .collect()
            .to_polars()
        )
        assert len(with_deleted) == 1
        assert with_deleted["sample_uid"][0] == "y"
        assert with_deleted["value"][0] == 200
        assert with_deleted[METAXY_DELETED_AT].is_not_null().all()


# ============= TEST: DEDUPLICATION WITH DUPLICATES =============


@parametrize_with_cases("feature_plan_sequence", cases=FeaturePlanCases)
def test_golden_reference_with_duplicate_timestamps(
    any_store: MetadataStore,
    feature_plan_sequence: FeaturePlanSequence,
):
    """Test deduplication logic correctly filters older versions before computing provenance.

    Uses only the first plan from the sequence (deduplication testing doesn't need
    multi-plan sequences).
    """
    empty_store = any_store
    feature_plan_config = feature_plan_sequence[0]  # Use first plan only
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

    with store, graph.use():
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
                upstream_id_cols = set(upstream_feature.spec().id_columns)
                user_fields = [
                    col
                    for col in older_df.columns
                    if not col.startswith("metaxy_") and col not in upstream_id_cols
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

        # Exclude metaxy_created_at since it's a timestamp
        common_columns = [
            col
            for col in added_df.columns
            if col in golden_downstream.columns and col != METAXY_CREATED_AT
        ]

        # Sort both DataFrames by all comparable columns for deterministic comparison
        # (id_columns alone may not be unique for aggregation lineage)
        added_sorted = added_df.sort(common_columns)
        golden_sorted = golden_downstream.sort(common_columns)

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


def test_golden_reference_with_all_duplicates_same_timestamp(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test deduplication with all samples having duplicate entries at same timestamp."""
    empty_store = any_store

    # Create simple feature graph
    class ParentFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["value"],
        ),
    ):
        pass

    class ChildFeature(
        BaseFeature,
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


@parametrize_with_cases("feature_plan_sequence", cases=FeaturePlanCases)
def test_golden_reference_partial_duplicates(
    any_store: MetadataStore,
    feature_plan_sequence: FeaturePlanSequence,
):
    """Test golden reference with only some upstream samples having duplicates.

    Uses only the first plan from the sequence (deduplication testing doesn't need
    multi-plan sequences).
    """
    empty_store = any_store
    feature_plan_config = feature_plan_sequence[0]  # Use first plan only
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
        with store, graph.use():
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

            # Exclude timestamp
            common_columns = [
                col
                for col in added_df.columns
                if col in golden_downstream.columns and col != METAXY_CREATED_AT
            ]

            # Sort both by all comparable columns for deterministic comparison
            # (id_columns alone may not be unique for aggregation lineage)
            added_sorted = added_df.sort(common_columns)
            golden_sorted = golden_downstream.sort(common_columns)

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


# ============= UNIT TEST: keep_latest_by_group =============


class KeepLatestTestDataCases:
    """Test data cases for keep_latest_by_group tests."""

    def case_polars(self):
        import narwhals as nw

        from metaxy.versioning.polars import PolarsVersioningEngine

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        def create_data_fn(pl_df):
            return nw.from_native(pl_df)

        return (PolarsVersioningEngine, create_data_fn, base_time)

    def case_ibis(self, tmp_path):
        import ibis
        import narwhals as nw

        from metaxy.versioning.ibis import IbisVersioningEngine

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Create a persistent connection for this test case
        con = ibis.duckdb.connect(tmp_path / "test.duckdb")
        table_counter = [0]  # Mutable counter for unique table names

        def create_data_fn(pl_df):
            # Create a unique table name for this invocation
            table_counter[0] += 1
            table_name = f"test_data_{table_counter[0]}"

            # Write to DuckDB and return as Narwhals-wrapped Ibis table
            con.create_table(table_name, pl_df.to_pandas(), overwrite=True)
            ibis_table = con.table(table_name)
            return nw.from_native(ibis_table)

        return (IbisVersioningEngine, create_data_fn, base_time)


@pytest_cases.fixture
@parametrize_with_cases("test_data", cases=KeepLatestTestDataCases)
def keep_latest_test_data(test_data):
    return test_data


def test_keep_latest_by_group(keep_latest_test_data):
    from datetime import timedelta

    import polars as pl

    # Get fixture data
    engine_class, create_data_fn, base_time = keep_latest_test_data

    # Create test data with 5 versions of the same sample
    data = pl.DataFrame(
        {
            "sample_uid": ["sample1"] * 5,
            "value": [10, 20, 30, 40, 50],  # Different values per version
            "timestamp": [
                base_time + timedelta(hours=i) for i in range(5)
            ],  # Increasing timestamps
        }
    )

    # Shuffle to ensure order doesn't matter
    shuffled_data = data.sample(fraction=1.0, shuffle=True, seed=42)

    # Convert to Narwhals using the case-specific function
    nw_df = create_data_fn(shuffled_data)

    # Call keep_latest_by_group directly (staticmethod)
    result_nw = engine_class.keep_latest_by_group(
        nw_df,
        group_columns=["sample_uid"],
        timestamp_column="timestamp",
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 1 row (only the latest)
    assert len(result) == 1, f"Expected 1 row, got {len(result)}"

    # Verify it's the latest version (value=50)
    assert result["value"][0] == 50, (
        f"Expected value=50 (latest), got {result['value'][0]}"
    )

    # Verify the timestamp is the latest
    expected_timestamp = base_time + timedelta(hours=4)
    assert result["timestamp"][0] == expected_timestamp, (
        f"Expected timestamp={expected_timestamp}, got {result['timestamp'][0]}"
    )


def test_keep_latest_by_group_aggregation_n_to_1(keep_latest_test_data):
    """Test keep_latest_by_group with N:1 aggregation (sensor readings to hourly stats)."""
    from datetime import timedelta

    import polars as pl

    engine_class, create_data_fn, base_time = keep_latest_test_data

    # Create sensor readings with duplicates (multiple versions of same reading)
    # reading_id identifies individual readings, but we have 2 versions of each
    data = pl.DataFrame(
        {
            "sensor_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
            "hour": ["10h"] * 8,
            "reading_id": [
                "r1",
                "r1",
                "r2",
                "r2",
                "r3",
                "r3",
                "r4",
                "r4",
            ],  # Duplicates
            "temperature": [
                20.0,
                20.5,
                21.0,
                21.5,
                19.0,
                19.5,
                22.0,
                22.5,
            ],  # Different values
            "timestamp": [
                base_time + timedelta(hours=i) for i in range(8)
            ],  # Increasing timestamps
        }
    )

    # Shuffle to ensure order doesn't matter
    shuffled_data = data.sample(fraction=1.0, shuffle=True, seed=42)

    # Convert to Narwhals using the case-specific function
    nw_df = create_data_fn(shuffled_data)

    # Call keep_latest_by_group
    result_nw = engine_class.keep_latest_by_group(
        nw_df,
        group_columns=["sensor_id", "hour", "reading_id"],
        timestamp_column="timestamp",
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 4 rows (one per reading_id: r1, r2, r3, r4)
    assert len(result) == 4, f"Expected 4 rows (one per reading), got {len(result)}"

    # Verify only latest versions kept (the ones with higher temperature values)
    result_sorted = result.sort(["sensor_id", "reading_id"])
    assert result_sorted["temperature"].to_list() == [20.5, 21.5, 19.5, 22.5], (
        "Expected latest versions with higher temperatures"
    )


def test_keep_latest_by_group_expansion_1_to_n(keep_latest_test_data):
    """Test keep_latest_by_group with 1:N expansion (video to video frames)."""
    from datetime import timedelta

    import polars as pl

    engine_class, create_data_fn, base_time = keep_latest_test_data

    # Create video metadata with duplicates (old and new versions)
    # Same video_id but different metadata versions
    data = pl.DataFrame(
        {
            "video_id": ["v1", "v1", "v1", "v2", "v2"],  # Duplicates for each video
            "resolution": ["720p", "1080p", "4K", "720p", "1080p"],  # Different values
            "fps": [30, 30, 60, 30, 60],  # Different values
            "timestamp": [
                base_time + timedelta(hours=i) for i in range(5)
            ],  # Increasing timestamps
        }
    )

    # Shuffle to ensure order doesn't matter
    shuffled_data = data.sample(fraction=1.0, shuffle=True, seed=42)

    # Convert to Narwhals using the case-specific function
    nw_df = create_data_fn(shuffled_data)

    # Call keep_latest_by_group
    result_nw = engine_class.keep_latest_by_group(
        nw_df,
        group_columns=["video_id"],
        timestamp_column="timestamp",
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 2 rows (one per video_id: v1, v2)
    assert len(result) == 2, f"Expected 2 rows (one per video), got {len(result)}"

    # Verify only latest versions kept
    result_sorted = result.sort("video_id")

    # v1's latest version is "4K" (3rd occurrence, timestamp +2 hours)
    # v2's latest version is "1080p" (2nd occurrence, timestamp +4 hours)
    assert result_sorted["resolution"].to_list() == ["4K", "1080p"], (
        "Expected latest versions: v1=4K, v2=1080p"
    )
    assert result_sorted["fps"].to_list() == [60, 60], "Expected latest fps values"


# ============= REGRESSION TESTS =============


def test_expansion_changed_rows_not_duplicated(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Regression test: expansion lineage should not duplicate changed rows.

    When upstream changes for expansion lineage (1:N), resolve_update should return
    one "changed" row per parent, not one per expanded child. Without proper deduplication,
    the join produces N copies of each changed parent (one per child row).

    This test verifies:
    1. Initial resolve_update returns correct parent-level rows
    2. After upstream changes, "changed" contains exactly one row per parent (not duplicated)
    """

    class Video(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="video_dedup_test",
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["content"]), code_version="1"),
            ],
        ),
    ):
        video_id: str

    class VideoFrames(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="video_frames_dedup_test",
            id_columns=("video_id", "frame_id"),
            deps=[
                FeatureDep(
                    feature=Video,
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["frame_data"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        video_id: str
        frame_id: str

    try:
        with any_store:
            # === INITIAL DATA ===
            video_df = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "metaxy_provenance_by_field": [
                        {"content": "content_v1_original"},
                        {"content": "content_v2_original"},
                    ],
                }
            )
            any_store.write_metadata(Video, video_df)

            # First resolve_update - get parent-level rows
            increment = any_store.resolve_update(
                VideoFrames,
                target_version=VideoFrames.feature_version(),
                snapshot_version=graph.snapshot_version,
            )
            added_df = increment.added.lazy().collect().to_polars()
            assert len(added_df) == 2, f"Expected 2 parent rows, got {len(added_df)}"

            # Expand each video to 3 frames
            expanded_rows = []
            for row in added_df.iter_rows(named=True):
                video_id = row["video_id"]
                provenance = row["metaxy_provenance"]
                provenance_by_field = row["metaxy_provenance_by_field"]
                for frame_idx in range(3):
                    expanded_rows.append(
                        {
                            "video_id": video_id,
                            "frame_id": f"{video_id}_frame_{frame_idx}",
                            "metaxy_provenance": provenance,
                            "metaxy_provenance_by_field": provenance_by_field,
                        }
                    )

            expanded_df = pl.DataFrame(expanded_rows)
            any_store.write_metadata(VideoFrames, expanded_df)

            # Verify 6 rows stored (2 videos × 3 frames)
            stored_df = (
                any_store.read_metadata(VideoFrames).lazy().collect().to_polars()
            )
            assert len(stored_df) == 6, (
                f"Expected 6 expanded rows, got {len(stored_df)}"
            )

            # === CHANGE UPSTREAM ===
            # Update video v1's content (change provenance)
            updated_video_df = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "metaxy_provenance_by_field": [
                        {"content": "content_v1_CHANGED"},  # Changed!
                        {"content": "content_v2_original"},  # Unchanged
                    ],
                }
            )
            any_store.write_metadata(Video, updated_video_df)

            # Resolve update after upstream change
            increment_after_change = any_store.resolve_update(
                VideoFrames,
                target_version=VideoFrames.feature_version(),
                snapshot_version=graph.snapshot_version,
            )

            changed_df = increment_after_change.changed
            assert changed_df is not None, "Expected changed to not be None"
            changed_df = changed_df.lazy().collect().to_polars()

            # CRITICAL: Should be exactly 1 changed row (v1), NOT 3 (one per frame)
            # Without proper deduplication, this would return 3 rows
            assert len(changed_df) == 1, (
                f"Expected exactly 1 changed row (one per parent), got {len(changed_df)}. "
                f"This indicates expansion deduplication is broken - getting duplicate rows per child."
            )

            # Verify it's v1 that changed
            assert changed_df["video_id"].to_list() == ["v1"], (
                f"Expected v1 to be changed, got {changed_df['video_id'].to_list()}"
            )

            # Verify added is empty (no new videos)
            added_after_change = (
                increment_after_change.added.lazy().collect().to_polars()
            )
            assert len(added_after_change) == 0, (
                f"Expected 0 added rows, got {len(added_after_change)}"
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}"
        )


# ============= SNAPSHOT TESTS FOR PROVENANCE VALUES =============


@pytest.mark.parametrize(
    "lineage_case",
    ["identity", "aggregation", "expansion"],
    ids=["identity", "aggregation", "expansion"],
)
def test_provenance_snapshot(
    lineage_case: str,
    snapshot: SnapshotAssertion,
):
    """Snapshot test for provenance computation with deterministic data.

    Records the exact provenance values computed for each lineage type,
    enabling regression detection when the versioning engine changes.

    Uses fixed input data and InMemoryMetadataStore with xxhash64 for determinism.
    """
    from datetime import datetime

    import narwhals as nw

    from metaxy._testing.models import SampleFeatureSpec
    from metaxy.models.constants import (
        METAXY_CREATED_AT,
        METAXY_DATA_VERSION,
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_FEATURE_VERSION,
        METAXY_PROVENANCE,
        METAXY_PROVENANCE_BY_FIELD,
        METAXY_SNAPSHOT_VERSION,
    )
    from metaxy.versioning.polars import PolarsVersioningEngine
    from metaxy.versioning.types import HashAlgorithm

    graph = FeatureGraph()

    # Fixed timestamp for deterministic output
    fixed_timestamp = datetime(2024, 6, 15, 12, 0, 0)

    with graph.use():
        if lineage_case == "identity":
            # Simple identity lineage: Parent -> Child
            class ParentFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="parent",
                    id_columns=("id",),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                id: str

            class ChildFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="child",
                    id_columns=("id",),
                    deps=[
                        FeatureDep(
                            feature=ParentFeature,
                            lineage=LineageRelationship.identity(),
                        ),
                    ],
                    fields=[
                        FieldSpec(
                            key=FieldKey(["derived"]),
                            code_version="1",
                            deps=SpecialFieldDep.ALL,
                        ),
                    ],
                ),
            ):
                id: str

            # Fixed upstream data
            parent_data = pl.DataFrame(
                {
                    "id": ["p1", "p2", "p3"],
                    METAXY_DATA_VERSION: ["dv_p1", "dv_p2", "dv_p3"],
                    METAXY_DATA_VERSION_BY_FIELD: [
                        {"value": "fv_p1"},
                        {"value": "fv_p2"},
                        {"value": "fv_p3"},
                    ],
                    METAXY_PROVENANCE: ["prov_p1", "prov_p2", "prov_p3"],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "prov_p1"},
                        {"value": "prov_p2"},
                        {"value": "prov_p3"},
                    ],
                    METAXY_FEATURE_VERSION: ["v1", "v1", "v1"],
                    METAXY_SNAPSHOT_VERSION: ["snap1", "snap1", "snap1"],
                    METAXY_CREATED_AT: [fixed_timestamp] * 3,
                }
            )

            upstream_data = {"parent": parent_data}
            child_plan = graph.get_feature_plan(ChildFeature.spec().key)

        elif lineage_case == "aggregation":
            # Aggregation lineage: Multiple readings per sensor -> one aggregated row
            class SensorReadings(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="sensor_readings",
                    id_columns=("sensor_id", "reading_id"),
                    fields=[FieldSpec(key=FieldKey(["temperature"]), code_version="1")],
                ),
            ):
                sensor_id: str
                reading_id: str

            class AggregatedStats(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="aggregated_stats",
                    id_columns=("sensor_id",),
                    deps=[
                        FeatureDep(
                            feature=SensorReadings,
                            lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                        ),
                    ],
                    fields=[
                        FieldSpec(
                            key=FieldKey(["avg_temp"]),
                            code_version="1",
                            deps=SpecialFieldDep.ALL,
                        ),
                    ],
                ),
            ):
                sensor_id: str

            # Fixed upstream data: 2 sensors with 2-3 readings each
            readings_data = pl.DataFrame(
                {
                    "sensor_id": ["s1", "s1", "s2", "s2", "s2"],
                    "reading_id": ["r1", "r2", "r3", "r4", "r5"],
                    METAXY_DATA_VERSION: ["dv_r1", "dv_r2", "dv_r3", "dv_r4", "dv_r5"],
                    METAXY_DATA_VERSION_BY_FIELD: [
                        {"temperature": "fv_r1"},
                        {"temperature": "fv_r2"},
                        {"temperature": "fv_r3"},
                        {"temperature": "fv_r4"},
                        {"temperature": "fv_r5"},
                    ],
                    METAXY_PROVENANCE: [
                        "prov_r1",
                        "prov_r2",
                        "prov_r3",
                        "prov_r4",
                        "prov_r5",
                    ],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"temperature": "prov_r1"},
                        {"temperature": "prov_r2"},
                        {"temperature": "prov_r3"},
                        {"temperature": "prov_r4"},
                        {"temperature": "prov_r5"},
                    ],
                    METAXY_FEATURE_VERSION: ["v1"] * 5,
                    METAXY_SNAPSHOT_VERSION: ["snap1"] * 5,
                    METAXY_CREATED_AT: [fixed_timestamp] * 5,
                }
            )

            upstream_data = {"sensor_readings": readings_data}
            child_plan = graph.get_feature_plan(AggregatedStats.spec().key)

        elif lineage_case == "expansion":
            # Expansion lineage: One video -> multiple frames
            class Video(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="video",
                    id_columns=("video_id",),
                    fields=[FieldSpec(key=FieldKey(["content"]), code_version="1")],
                ),
            ):
                video_id: str

            class VideoFrames(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="video_frames",
                    id_columns=("video_id",),  # At parent level for golden reference
                    deps=[
                        FeatureDep(
                            feature=Video,
                            lineage=LineageRelationship.expansion(on=["video_id"]),
                        ),
                    ],
                    fields=[
                        FieldSpec(
                            key=FieldKey(["frames"]),
                            code_version="1",
                            deps=SpecialFieldDep.ALL,
                        ),
                    ],
                ),
            ):
                video_id: str

            # Fixed upstream data
            video_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    METAXY_DATA_VERSION: ["dv_v1", "dv_v2"],
                    METAXY_DATA_VERSION_BY_FIELD: [
                        {"content": "fv_v1"},
                        {"content": "fv_v2"},
                    ],
                    METAXY_PROVENANCE: ["prov_v1", "prov_v2"],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"content": "prov_v1"},
                        {"content": "prov_v2"},
                    ],
                    METAXY_FEATURE_VERSION: ["v1", "v1"],
                    METAXY_SNAPSHOT_VERSION: ["snap1", "snap1"],
                    METAXY_CREATED_AT: [fixed_timestamp] * 2,
                }
            )

            upstream_data = {"video": video_data}
            child_plan = graph.get_feature_plan(VideoFrames.spec().key)

        else:
            raise ValueError(f"Unknown lineage case: {lineage_case}")

        # Use PolarsVersioningEngine directly for deterministic computation
        engine = PolarsVersioningEngine(plan=child_plan)

        # Convert upstream data to Narwhals LazyFrames
        upstream_nw = {
            FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()
        }

        # Compute provenance
        added, changed, removed, _ = engine.resolve_increment_with_provenance(
            current=None,
            upstream=upstream_nw,
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )

        # Collect results
        added_df = added.collect()

        # Extract provenance columns for snapshot
        provenance_cols = [
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE,
            METAXY_PROVENANCE_BY_FIELD,
        ]
        id_columns = list(child_plan.feature.id_columns)
        available_cols = [
            c for c in id_columns + provenance_cols if c in added_df.columns
        ]

        # Sort for deterministic output and convert to dicts
        result_df = added_df.select(available_cols).sort(available_cols)
        # Convert to native Polars for to_dicts()
        result_pl = result_df.to_native()
        snapshot_data = result_pl.to_dicts()

        # Assert against snapshot
        assert snapshot_data == snapshot
