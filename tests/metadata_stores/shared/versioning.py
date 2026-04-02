"""Versioning test pack for metadata stores.

Tests provenance golden reference correctness and hash truncation across all store
backends. The golden reference is computed using PolarsVersioningEngine, and all store
implementations are expected to produce identical results.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias

import polars as pl
import polars.testing as pl_testing
import pytest
from hypothesis.errors import NonInteractiveExampleWarning
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from metaxy_testing.parametric import downstream_metadata_strategy
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldSpec,
    LineageRelationship,
)
from metaxy.config import MetaxyConfig
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from metaxy.models.field import SpecialFieldDep
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FieldKey
from metaxy.utils import collect_to_polars
from metaxy.versioning.types import HashAlgorithm, Increment

if TYPE_CHECKING:
    pass

FeaturePlanOutput: TypeAlias = tuple[FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan]

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

    def case_aggregation_plus_identity(self, graph: FeatureGraph) -> FeaturePlanSequence:
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
        - Video: 1:N expansion (one video -> many frames)
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

    def case_aggregation_plus_expansion(self, graph: FeatureGraph) -> FeaturePlanSequence:
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
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="2")],  # Changed!
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

    from metaxy.ext.polars.versioning import PolarsVersioningEngine
    from metaxy.models.constants import (
        METAXY_CREATED_AT,
        METAXY_FEATURE_VERSION,
    )

    graph, upstream_features, child_feature_plan = feature_plan_config

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key

    # Feature versions for strategy
    child_version = graph.get_feature_version(child_key)

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
                project_version=graph.project_version,
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
                (pl.col(METAXY_CREATED_AT) + timedelta(hours=1)).alias(METAXY_CREATED_AT),
            )

            upstream_data[key_str] = updated_df

        # Calculate golden downstream using PolarsVersioningEngine
        engine = PolarsVersioningEngine(plan=child_feature_plan)
        upstream_dict = {FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()}

        downstream_nw = engine.load_upstream_with_provenance(
            upstream=upstream_dict,
            hash_algo=store.hash_algorithm,
            filters=None,
        ).collect()

        # Add downstream feature version and snapshot version
        downstream_df = downstream_nw.with_columns(
            nw.lit(child_version).alias(METAXY_FEATURE_VERSION),
            nw.lit(graph.project_version).alias("metaxy_project_version"),
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
        store.write(upstream_feature, upstream_df)


def setup_store_with_data(
    empty_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
) -> tuple[MetadataStore, FeaturePlanOutput, pl.DataFrame]:
    """Legacy helper for single-plan setup. Wraps the new helpers."""
    graph, upstream_features, child_feature_plan = feature_plan_config
    upstream_data, golden_downstream = generate_plan_data(empty_store, feature_plan_config)

    try:
        with empty_store.open("w"):
            # Use graph.use() to make it the current graph (needed for write)
            with graph.use():
                write_upstream_to_store(empty_store, feature_plan_config, upstream_data)
    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {empty_store.hash_algorithm} not supported by {empty_store}")

    return empty_store, feature_plan_config, golden_downstream


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
        sort_columns = [col for col in actual_df.columns if col in golden_df.columns and col != METAXY_CREATED_AT]
        actual_sorted = actual_df.sort(sort_columns)
        golden_sorted = golden_df.sort(sort_columns)

        # Select only common columns, excluding timestamps
        common_columns = [
            col for col in actual_sorted.columns if col in golden_sorted.columns and col != METAXY_CREATED_AT
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
    actual_added = actual.new.lazy().collect().to_polars()
    golden_added = golden.new.lazy().collect().to_polars()
    compare_frames(actual_added, golden_added, "added")

    # Compare changed
    actual_changed = actual.stale.lazy().collect().to_polars()
    golden_changed = golden.stale.lazy().collect().to_polars()
    compare_frames(actual_changed, golden_changed, "changed")

    # Compare removed
    actual_removed = actual.orphaned.lazy().collect().to_polars()
    golden_removed = golden.orphaned.lazy().collect().to_polars()
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
    child_version = graph.get_feature_version(child_key)

    # Call resolve_update to compute provenance (uses default/native engine)
    # Use graph.use() to make it the current graph (needed for resolve_update)
    with graph.use():
        actual_increment = store.resolve_update(
            child_key,
            target_version=child_version,
            project_version=graph.project_version,
        )

    id_columns = list(child_feature_plan.feature.id_columns)
    assert_increment_matches_golden(actual_increment, golden_increment, id_columns)

    # Additional safeguard: also verify with polars engine explicitly
    # This ensures both native and polars engines produce the same result
    with graph.use():
        polars_increment = store.resolve_update(
            child_key,
            target_version=child_version,
            project_version=graph.project_version,
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

        added_df = golden_increment.new.lazy().collect().to_polars()

        # Select ID columns and provenance columns that exist
        available_cols = [c for c in id_columns + provenance_cols if c in added_df.columns]
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

    from metaxy.ext.polars.versioning import PolarsVersioningEngine

    engine = PolarsVersioningEngine(plan=child_feature_plan)

    # Convert upstream data to Narwhals LazyFrames with FeatureKey keys
    upstream_nw = {FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()}

    # Convert current downstream to Narwhals if present
    current_nw = nw.from_native(current_downstream.lazy()) if current_downstream is not None else None

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
        new=added_collected,
        stale=changed_collected,
        orphaned=removed_collected,
    )


class VersioningTests:
    """Tests for provenance golden reference and hash truncation across stores."""

    def test_store_resolve_update_matches_golden_provenance(
        self,
        store: MetadataStore,
        feature_plan_sequence: FeaturePlanSequence,
    ) -> None:
        """Test metadata store provenance calculation matches golden reference.

        The golden reference is computed using PolarsVersioningEngine.resolve_increment_with_provenance.
        All store implementations should produce the same result.

        For multi-plan sequences (e.g., definition changes), iterates through each plan
        without clearing the store between iterations - simulating real-world usage where
        new data is written on top of existing data.
        """
        try:
            with store.open("w"):
                # Track current downstream data for golden increment calculation
                # Key: feature_version -> downstream DataFrame
                # This mirrors what the store does: filter by feature_version
                current_downstream_by_version: dict[str, pl.DataFrame] = {}
                # Track base upstream data for multi-plan sequences (to keep same IDs)
                base_upstream_data: dict[str, pl.DataFrame] | None = None

                for i, plan_config in enumerate(feature_plan_sequence):
                    graph, upstream_features, child_feature_plan = plan_config
                    child_key = child_feature_plan.feature.key
                    child_version = graph.get_feature_version(child_key)

                    # Generate upstream data for this plan
                    # For first plan, generate fresh data; for subsequent plans, derive from base
                    if i == 0:
                        upstream_data, _ = generate_plan_data(store, plan_config)
                        # Store as base for subsequent plans
                        base_upstream_data = upstream_data
                    else:
                        # Use base upstream data to ensure same IDs
                        upstream_data, _ = generate_plan_data(store, plan_config, base_upstream_data=base_upstream_data)

                    # Get current downstream for THIS feature version (like the store does)
                    current_downstream = current_downstream_by_version.get(child_version)

                    # Compute golden increment using PolarsVersioningEngine
                    golden_increment = compute_golden_increment(
                        child_feature_plan,
                        upstream_data,
                        current_downstream,
                        store.hash_algorithm,
                    )

                    # Write upstream data to store (accumulates)
                    with graph.use():
                        write_upstream_to_store(store, plan_config, upstream_data)

                    # Assert store's resolve_update matches golden increment
                    assert_resolve_update_matches_golden(store, plan_config, golden_increment)

                    # Update current downstream for this feature version
                    # Use the golden added/changed to build current state
                    added_df = golden_increment.new.lazy().collect().to_polars()
                    if golden_increment.stale is not None:
                        changed_df = golden_increment.stale.lazy().collect().to_polars()
                        new_downstream = pl.concat([added_df, changed_df])
                    else:
                        new_downstream = added_df
                    current_downstream_by_version[child_version] = new_downstream

                    # Also write downstream to store so store can detect changes in next iteration
                    with graph.use():
                        store.write(child_key, new_downstream)

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")

    def test_golden_reference_with_duplicate_timestamps(
        self,
        store: MetadataStore,
        feature_plan_sequence: FeaturePlanSequence,
    ) -> None:
        """Test deduplication logic correctly filters older versions before computing provenance.

        Uses only the first plan from the sequence (deduplication testing doesn't need
        multi-plan sequences).
        """
        empty_store = store
        feature_plan_config = feature_plan_sequence[0]  # Use first plan only
        # Setup store with upstream data and get golden reference
        store_, (graph, upstream_features, child_feature_plan), golden_downstream = setup_store_with_data(
            empty_store,
            feature_plan_config,
        )

        # Get the child feature from the graph
        child_key = child_feature_plan.feature.key
        child_version = graph.get_feature_version(child_key)

        with store_.open("w"), graph.use():
            try:
                from datetime import timedelta

                import polars as pl

                from metaxy.models.constants import METAXY_CREATED_AT

                # Add older duplicates to upstream metadata
                for feature_key, upstream_feature in upstream_features.items():
                    # Read existing upstream data
                    existing_df = store_.read(upstream_feature).lazy().collect().to_polars()

                    # Create older duplicates (same IDs, older timestamps)
                    older_df = existing_df.clone()
                    older_df = older_df.with_columns(
                        (pl.col(METAXY_CREATED_AT) - timedelta(hours=2)).alias(METAXY_CREATED_AT)
                    )

                    # Modify a field value to ensure different provenance
                    # This tests that older version is NOT used
                    upstream_id_cols = set(upstream_feature.spec().id_columns)
                    user_fields = [
                        col for col in older_df.columns if not col.startswith("metaxy_") and col not in upstream_id_cols
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
                    store_.write(upstream_feature, older_df)

                    # Now store has 2 versions per sample:
                    # - Original (newer) - should be used
                    # - Duplicate (older, modified) - should be ignored

                # Call resolve_update - should use only latest versions
                increment = store_.resolve_update(
                    child_key,
                    target_version=child_version,
                    project_version=graph.project_version,
                )

            except HashAlgorithmNotSupportedError:
                pytest.skip(f"Hash algorithm {store_.hash_algorithm} not supported by {store_}")

            added_df = increment.new.lazy().collect().to_polars()

            # Exclude metaxy_created_at since it's a timestamp
            common_columns = [
                col for col in added_df.columns if col in golden_downstream.columns and col != METAXY_CREATED_AT
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
        self,
        store: MetadataStore,
        graph: FeatureGraph,
    ) -> None:
        """Test deduplication with all samples having duplicate entries at same timestamp."""
        empty_store = store

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

        child_key = ChildFeature.spec().key
        child_plan = graph.get_feature_plan(child_key)

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
                project_version=graph.project_version,
                hash_algorithm=empty_store.hash_algorithm,
                min_rows=5,
                max_rows=10,
            ).example()

        parent_df = upstream_data["parent"]

        try:
            with empty_store.open("w"):
                from datetime import datetime, timezone

                import polars as pl

                from metaxy.models.constants import METAXY_CREATED_AT

                # Create duplicates with SAME timestamp for ALL samples
                same_timestamp = datetime.now(timezone.utc)

                # Write first version
                version1 = parent_df.with_columns(pl.lit(same_timestamp).alias(METAXY_CREATED_AT))
                empty_store.write(ParentFeature, version1)

                # Create second version with same timestamp but different provenance
                # We modify the provenance columns to simulate different data
                version2 = parent_df.with_columns(
                    pl.lit(same_timestamp).alias(METAXY_CREATED_AT),
                    # Modify provenance to make it different (simulate different underlying data)
                    (pl.col("metaxy_provenance").cast(pl.Utf8) + "_DUP").alias("metaxy_provenance"),
                )

                # Write duplicates with same timestamp
                empty_store.write(ParentFeature, version2)

                # Now every sample has 2 versions with same timestamp
                # Call resolve_update - should pick one deterministically
                increment = empty_store.resolve_update(
                    child_key,
                    target_version=ChildFeature.feature_version(),
                    project_version=graph.project_version,
                )

                # Verify we got results (deterministic even with same timestamps)
                added_df = increment.new.lazy().collect().to_polars()
                assert len(added_df) > 0, "Expected at least some samples after deduplication"

                # With duplicates at same timestamp, we should still get the original count
                # (deduplication picks one version per sample)
                assert len(added_df) == len(parent_df), (
                    f"Expected {len(parent_df)} deduplicated samples, got {len(added_df)}"
                )

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {empty_store.hash_algorithm} not supported by {empty_store}")

    def test_golden_reference_partial_duplicates(
        self,
        store: MetadataStore,
        feature_plan_sequence: FeaturePlanSequence,
    ) -> None:
        """Test golden reference with only some upstream samples having duplicates.

        Uses only the first plan from the sequence (deduplication testing doesn't need
        multi-plan sequences).
        """
        empty_store = store
        feature_plan_config = feature_plan_sequence[0]  # Use first plan only
        # Setup store with upstream data
        store_, (graph, upstream_features, child_feature_plan), golden_downstream = setup_store_with_data(
            empty_store,
            feature_plan_config,
        )

        child_key = child_feature_plan.feature.key
        child_version = graph.get_feature_version(child_key)

        try:
            with store_.open("w"), graph.use():
                from datetime import timedelta

                import polars as pl

                from metaxy.models.constants import METAXY_CREATED_AT

                # Add older duplicates for only HALF of the samples in each upstream
                for feature_key, upstream_feature in upstream_features.items():
                    existing_df = store_.read(upstream_feature).lazy().collect().to_polars()

                    # Get half of samples
                    num_samples = len(existing_df)
                    half_count = num_samples // 2

                    if half_count > 0:
                        # Take first half
                        samples_to_duplicate = existing_df.head(half_count)

                        # Create older version
                        older_df = samples_to_duplicate.with_columns(
                            (pl.col(METAXY_CREATED_AT) - timedelta(hours=1)).alias(METAXY_CREATED_AT)
                        )

                        # Write older duplicates
                        store_.write(upstream_feature, older_df)

                    # Now store has:
                    # - First half of samples: 2 versions each (newer and older)
                    # - Second half of samples: 1 version each (original only)

                # Call resolve_update
                increment = store_.resolve_update(
                    child_key,
                    target_version=child_version,
                    project_version=graph.project_version,
                )

                added_df = increment.new.lazy().collect().to_polars()

                # Exclude timestamp
                common_columns = [
                    col for col in added_df.columns if col in golden_downstream.columns and col != METAXY_CREATED_AT
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
            pytest.skip(f"Hash algorithm {store_.hash_algorithm} not supported by {store_}")

    def test_expansion_changed_rows_not_duplicated(
        self,
        store: MetadataStore,
        graph: FeatureGraph,
    ) -> None:
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
            with store.open("w"):
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
                store.write(Video, video_df)

                # First resolve_update - get parent-level rows
                increment = store.resolve_update(
                    VideoFrames,
                    target_version=VideoFrames.feature_version(),
                    project_version=graph.project_version,
                )
                added_df = increment.new.lazy().collect().to_polars()
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
                store.write(VideoFrames, expanded_df)

                # Verify 6 rows stored (2 videos x 3 frames)
                stored_df = store.read(VideoFrames).lazy().collect().to_polars()
                assert len(stored_df) == 6, f"Expected 6 expanded rows, got {len(stored_df)}"

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
                store.write(Video, updated_video_df)

                # Resolve update after upstream change
                increment_after_change = store.resolve_update(
                    VideoFrames,
                    target_version=VideoFrames.feature_version(),
                    project_version=graph.project_version,
                )

                changed_df = increment_after_change.stale
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
                added_after_change = increment_after_change.new.lazy().collect().to_polars()
                assert len(added_after_change) == 0, f"Expected 0 added rows, got {len(added_after_change)}"

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")

    def test_enable_map_datatype_does_not_affect_versioning(
        self,
        store: MetadataStore,
        feature_plan_sequence: FeaturePlanSequence,
    ) -> None:
        """Toggling enable_map_datatype must not change versioning results.

        Writes upstream data once, then runs resolve_update with the flag off
        and on, asserting both produce identical increments.
        """
        feature_plan_config = feature_plan_sequence[0]
        graph, upstream_features, child_feature_plan = feature_plan_config
        child_key = child_feature_plan.feature.key
        child_version = graph.get_feature_version(child_key)

        upstream_data, _ = generate_plan_data(store, feature_plan_config)

        try:
            with store.open("w"), graph.use():
                write_upstream_to_store(store, feature_plan_config, upstream_data)

                def _resolve_with_map_flag(enabled: bool) -> pl.DataFrame:
                    config = MetaxyConfig.get().model_copy(update={"enable_map_datatype": enabled})
                    with config.use():
                        increment = store.resolve_update(
                            child_key,
                            target_version=child_version,
                            project_version=graph.project_version,
                        )
                        return collect_to_polars(increment.new)

                result_off = _resolve_with_map_flag(False)
                result_on = _resolve_with_map_flag(True)

        except HashAlgorithmNotSupportedError:
            pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")

        assert len(result_off) > 0, "Expected non-empty versioning result"

        from metaxy.models.constants import METAXY_CREATED_AT, METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
        from metaxy.utils._arrow_map import convert_structs_to_maps

        # Normalize result_off's Struct _by_field columns to polars_map.Map so
        # both frames have the same type and values can be compared.
        by_field_columns = [
            c
            for c in (METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD)
            if c in result_off.columns and c in result_on.columns
        ]
        result_off = convert_structs_to_maps(result_off, columns=by_field_columns)

        common_columns = [c for c in result_off.columns if c in result_on.columns and c != METAXY_CREATED_AT]
        pl_testing.assert_frame_equal(
            result_off.select(common_columns).sort(common_columns),
            result_on.select(common_columns).sort(common_columns),
            check_row_order=True,
            check_column_order=False,
        )

    @pytest.mark.parametrize("truncation_length", [16])
    def test_hash_truncation_any_store(
        self, config_with_truncation: MetaxyConfig, store: MetadataStore, graph: FeatureGraph
    ) -> None:
        """Test that hash truncation is applied correctly across store types."""

        class ParentFeature(
            BaseFeature,
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
                fields=["result"],
            ),
        ):
            pass

        child_plan = graph.get_feature_plan(ChildFeature.spec().key)
        feature_versions = {
            "parent": ParentFeature.feature_version(),
            "child": ChildFeature.feature_version(),
        }

        # Config is already set by fixture to truncation_length=16
        truncation_length = config_with_truncation.hash_truncation_length

        # Generate test data
        upstream_data, _ = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            project_version=graph.project_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=5,
            max_rows=10,
        ).example()

        parent_df = upstream_data["parent"]

        with store.open("w"), graph.use():
            # Write parent metadata
            store.write(ParentFeature, parent_df)

            # Compute child metadata
            increment = store.resolve_update(
                ChildFeature,
                target_version=ChildFeature.feature_version(),
                project_version=graph.project_version,
            )

            result = increment.new.lazy().collect().to_polars()

            # Verify all hashes are exactly 16 characters
            hash_col = result["metaxy_provenance"]
            for hash_val in hash_col:
                assert len(hash_val) == truncation_length, (
                    f"Expected hash length {truncation_length}, got {len(hash_val)} in {type(store).__name__}"
                )
