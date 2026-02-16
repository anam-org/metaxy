"""Tests for MetadataStore.calculate_input_progress method.

This module tests the progress percentage calculation for features
with different lineage types (Identity, Aggregation, Expansion).
"""

from __future__ import annotations

import narwhals as nw
import polars as pl
from metaxy_testing import add_metaxy_provenance_column
from metaxy_testing.models import SampleFeatureSpec

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
)
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.models.lineage import LineageRelationship


class TestCalculateInputProgress:
    """Tests for MetadataStore.calculate_input_progress method."""

    def test_returns_none_when_input_is_none(self, graph: FeatureGraph, tmp_path):
        """Returns None when LazyIncrement.input is None (root features)."""

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["root"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Root features require samples argument
            samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "value": ["a", "b", "c"],
                    "metaxy_provenance_by_field": [
                        {"value": "hash1"},
                        {"value": "hash2"},
                        {"value": "hash3"},
                    ],
                }
            )
            lazy_increment = store.resolve_update(RootFeature, samples=nw.from_native(samples), lazy=True)

            # Root features have no input
            assert lazy_increment.input is None
            progress = store.calculate_input_progress(lazy_increment, RootFeature)
            assert progress is None

    def test_returns_100_when_all_input_processed(self, graph: FeatureGraph, tmp_path):
        """Returns 100.0 when all input samples have been processed."""

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["sample_uid"],
                deps=[FeatureDep(feature=Upstream)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write upstream metadata
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "value": ["a", "b", "c"],
                    "metaxy_provenance_by_field": [
                        {"value": "hash1"},
                        {"value": "hash2"},
                        {"value": "hash3"},
                    ],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
            store.write(Upstream, nw.from_native(upstream_data))

            # Write downstream metadata for all samples
            increment = store.resolve_update(Downstream, lazy=False)
            store.write(Downstream, increment.new)

            # Now check progress - should be 100%
            lazy_increment = store.resolve_update(Downstream, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, Downstream)
            assert progress == 100.0

    def test_returns_correct_percentage_when_partially_processed(self, graph: FeatureGraph, tmp_path):
        """Returns correct percentage when some input samples are missing."""

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["sample_uid"],
                deps=[FeatureDep(feature=Upstream)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write upstream metadata for 10 samples
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": list(range(1, 11)),
                    "value": [f"val_{i}" for i in range(1, 11)],
                    "metaxy_provenance_by_field": [{"value": f"hash{i}"} for i in range(1, 11)],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
            store.write(Upstream, nw.from_native(upstream_data))

            # Write downstream metadata for only 3 samples
            increment = store.resolve_update(Downstream, lazy=False)
            partial_data = increment.new.to_polars().head(3)
            store.write(Downstream, partial_data)

            # Check progress - should be 30% (3/10)
            lazy_increment = store.resolve_update(Downstream, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, Downstream)
            assert progress == 30.0

    def test_returns_none_when_no_input(self, graph: FeatureGraph, tmp_path):
        """Returns None when there's no upstream input available."""

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["sample_uid"],
                deps=[FeatureDep(feature=Upstream)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write empty upstream metadata (0 rows) with proper schema
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": pl.Series([], dtype=pl.Int64),
                    "value": pl.Series([], dtype=pl.String),
                    "metaxy_provenance_by_field": pl.Series([], dtype=pl.Struct({"value": pl.String})),
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
            store.write(Upstream, nw.from_native(upstream_data))

            # Resolve update - should have no input to process
            lazy_increment = store.resolve_update(Downstream, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, Downstream)
            assert progress is None  # No input available

    def test_identity_lineage_uses_upstream_id_columns(self, graph: FeatureGraph, tmp_path):
        """Identity (1:1) lineage uses upstream_id_columns for progress."""

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                id_columns=["video_id", "frame_id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["video_id", "frame_id"],
                deps=[FeatureDep(feature=Upstream)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write upstream with composite keys
            upstream_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v1", "v2"],
                    "frame_id": [1, 2, 1],
                    "value": ["a", "b", "c"],
                    "metaxy_provenance_by_field": [
                        {"value": "hash1"},
                        {"value": "hash2"},
                        {"value": "hash3"},
                    ],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
            store.write(Upstream, nw.from_native(upstream_data))

            # Write downstream for 2 out of 3 samples
            increment = store.resolve_update(Downstream, lazy=False)
            partial_data = increment.new.to_polars().head(2)
            store.write(Downstream, partial_data)

            # Check progress - should be 66.67% (2/3)
            lazy_increment = store.resolve_update(Downstream, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, Downstream)
            assert progress is not None
            assert abs(progress - 66.67) < 0.1

    def test_aggregation_lineage_uses_aggregation_columns(self, graph: FeatureGraph, tmp_path):
        """Aggregation (N:1) lineage uses aggregation columns for progress."""

        class SensorReadings(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["sensor_readings"]),
                id_columns=["sensor_id", "timestamp", "reading_id"],
                fields=[FieldSpec(key=FieldKey(["temperature"]), code_version="1")],
            ),
        ):
            pass

        class HourlyStats(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["hourly_stats"]),
                id_columns=["sensor_id", "hour"],
                deps=[
                    FeatureDep(
                        feature=SensorReadings,
                        lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]),
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["avg_temp"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write 6 sensor readings (2 hours × 3 readings each)
            upstream_data = pl.DataFrame(
                {
                    "sensor_id": ["s1"] * 6,
                    "timestamp": [
                        "2024-01-01T10:15:00",
                        "2024-01-01T10:30:00",
                        "2024-01-01T10:45:00",
                        "2024-01-01T11:15:00",
                        "2024-01-01T11:30:00",
                        "2024-01-01T11:45:00",
                    ],
                    "reading_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
                    "hour": [
                        "2024-01-01T10",
                        "2024-01-01T10",
                        "2024-01-01T10",
                        "2024-01-01T11",
                        "2024-01-01T11",
                        "2024-01-01T11",
                    ],
                    "temperature": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                    "metaxy_provenance_by_field": [{"temperature": f"hash{i}"} for i in range(1, 7)],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, SensorReadings)
            store.write(SensorReadings, nw.from_native(upstream_data))

            # Write downstream for only 1 hour (1 out of 2 groups)
            increment = store.resolve_update(HourlyStats, lazy=False)
            partial_data = increment.new.to_polars().head(1)
            store.write(HourlyStats, partial_data)

            # Progress should count by aggregation groups, not individual readings
            # 1 hour processed out of 2 hours = 50%
            lazy_increment = store.resolve_update(HourlyStats, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, HourlyStats)
            assert progress == 50.0

    def test_expansion_lineage_uses_parent_columns(self, graph: FeatureGraph, tmp_path):
        """Expansion (1:N) lineage uses parent columns for progress."""

        class Video(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video"]),
                id_columns=["video_id"],
                fields=[FieldSpec(key=FieldKey(["resolution"]), code_version="1")],
            ),
        ):
            pass

        class VideoFrames(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video_frames"]),
                id_columns=["video_id", "frame_id"],
                deps=[
                    FeatureDep(
                        feature=Video,
                        lineage=LineageRelationship.expansion(on=["video_id"]),
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["embedding"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write 3 videos
            upstream_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v2", "v3"],
                    "resolution": ["1080p", "720p", "4K"],
                    "metaxy_provenance_by_field": [{"resolution": f"hash{i}"} for i in range(1, 4)],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, Video)
            store.write(Video, nw.from_native(upstream_data))

            # Write frames for 2 videos (multiple frames per video)
            # For expansion lineage, user must manually create expanded rows with frame_id
            # Each video has 3 frames
            frames_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v1", "v1", "v2", "v2", "v2"],
                    "frame_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                    "embedding": ["emb1", "emb2", "emb3", "emb4", "emb5", "emb6"],
                }
            )
            # Read upstream from store to get metaxy_data_version_by_field column
            upstream_from_store = store.read(Video).collect().to_polars()
            # Join with upstream to get provenance info
            frames_with_upstream = frames_data.join(
                upstream_from_store.select(
                    "video_id",
                    pl.col("metaxy_data_version_by_field").alias(
                        f"metaxy_data_version_by_field{Video.spec().key.to_column_suffix()}"
                    ),
                ),
                on="video_id",
            )
            # Compute provenance for VideoFrames
            frames_with_prov = store.compute_provenance(VideoFrames, nw.from_native(frames_with_upstream))
            store.write(VideoFrames, frames_with_prov)

            # Progress should count by parent videos, not individual frames
            # 2 videos processed out of 3 = 66.67%
            lazy_increment = store.resolve_update(VideoFrames, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, VideoFrames)
            assert progress is not None
            assert abs(progress - 66.67) < 0.1

    def test_with_renamed_columns(self, graph: FeatureGraph, tmp_path):
        """Progress calculation works with renamed columns."""

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                id_columns=["original_id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["renamed_id"],
                deps=[
                    FeatureDep(
                        feature=Upstream,
                        rename={"original_id": "renamed_id"},
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write upstream
            upstream_data = pl.DataFrame(
                {
                    "original_id": [1, 2, 3, 4],
                    "value": ["a", "b", "c", "d"],
                    "metaxy_provenance_by_field": [{"value": f"hash{i}"} for i in range(1, 5)],
                }
            )
            upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
            store.write(Upstream, nw.from_native(upstream_data))

            # Write downstream for 2 out of 4 samples
            increment = store.resolve_update(Downstream, lazy=False)
            partial_data = increment.new.to_polars().head(2)
            store.write(Downstream, partial_data)

            # Check progress - should be 50% (2/4)
            lazy_increment = store.resolve_update(Downstream, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, Downstream)
            assert progress == 50.0

    def test_multiple_upstreams(self, graph: FeatureGraph, tmp_path):
        """Progress calculation with multiple upstream features."""

        class UpstreamA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_a"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value_a"]), code_version="1")],
            ),
        ):
            pass

        class UpstreamB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_b"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value_b"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["sample_uid"],
                deps=[FeatureDep(feature=UpstreamA), FeatureDep(feature=UpstreamB)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write upstream A
            upstream_a_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "value_a": ["va1", "va2", "va3"],
                    "metaxy_provenance_by_field": [{"value_a": f"hash_a{i}"} for i in range(1, 4)],
                }
            )
            upstream_a_data = add_metaxy_provenance_column(upstream_a_data, UpstreamA)
            store.write(UpstreamA, nw.from_native(upstream_a_data))

            # Write upstream B
            upstream_b_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "value_b": ["vb1", "vb2", "vb3"],
                    "metaxy_provenance_by_field": [{"value_b": f"hash_b{i}"} for i in range(1, 4)],
                }
            )
            upstream_b_data = add_metaxy_provenance_column(upstream_b_data, UpstreamB)
            store.write(UpstreamB, nw.from_native(upstream_b_data))

            # Write downstream for 1 out of 3 samples
            increment = store.resolve_update(Downstream, lazy=False)
            partial_data = increment.new.to_polars().head(1)
            store.write(Downstream, partial_data)

            # Check progress - should be 33.33% (1/3)
            lazy_increment = store.resolve_update(Downstream, lazy=True)
            progress = store.calculate_input_progress(lazy_increment, Downstream)
            assert progress is not None
            assert abs(progress - 33.33) < 0.1
