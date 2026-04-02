"""Snapshot tests for provenance computation with deterministic data.

Records the exact provenance values computed for each lineage type,
enabling regression detection when the versioning engine changes.

Uses fixed input data and PolarsVersioningEngine with xxhash64 for determinism.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldSpec,
    LineageRelationship,
)
from metaxy.models.field import SpecialFieldDep
from metaxy.models.types import FieldKey


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

    Uses fixed input data and PolarsVersioningEngine with xxhash64 for determinism.
    """
    import narwhals as nw
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy.ext.polars.versioning import PolarsVersioningEngine
    from metaxy.models.constants import (
        METAXY_CREATED_AT,
        METAXY_DATA_VERSION,
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_FEATURE_VERSION,
        METAXY_PROJECT_VERSION,
        METAXY_PROVENANCE,
        METAXY_PROVENANCE_BY_FIELD,
    )
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
                    METAXY_PROJECT_VERSION: ["snap1", "snap1", "snap1"],
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
                    METAXY_PROJECT_VERSION: ["snap1"] * 5,
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
                    METAXY_PROJECT_VERSION: ["snap1", "snap1"],
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
        upstream_nw = {FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()}

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
        available_cols = [c for c in id_columns + provenance_cols if c in added_df.columns]

        # Sort for deterministic output and convert to dicts
        result_df = added_df.select(available_cols).sort(available_cols)
        # Convert to native Polars for to_dicts()
        result_pl = result_df.to_native()
        snapshot_data = result_pl.to_dicts()

        # Assert against snapshot
        assert snapshot_data == snapshot
