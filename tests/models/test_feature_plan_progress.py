"""Tests for FeaturePlan progress calculation properties.

This module tests the input_id_columns and upstream_id_columns properties
used for calculating progress percentage in features with different lineage types.
"""

from __future__ import annotations

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
)
from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.lineage import LineageRelationship


class TestUpstreamIdColumns:
    """Tests for FeaturePlan.upstream_id_columns property."""

    def test_no_deps_returns_empty_list(self, graph: FeatureGraph):
        """Root feature with no dependencies returns empty list."""

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["root"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        plan = graph.get_feature_plan(RootFeature.spec().key)
        assert plan.upstream_id_columns == []

    def test_single_upstream_returns_id_columns(self, graph: FeatureGraph):
        """Single upstream feature returns its id_columns."""

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

        plan = graph.get_feature_plan(Downstream.spec().key)
        assert set(plan.upstream_id_columns) == {"video_id", "frame_id"}

    def test_multiple_upstream_union_of_id_columns(self, graph: FeatureGraph):
        """Multiple upstream features return union of their id_columns."""

        class UpstreamA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_a"]),
                id_columns=["video_id", "frame_id"],
                fields=[FieldSpec(key=FieldKey(["value_a"]), code_version="1")],
            ),
        ):
            pass

        class UpstreamB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_b"]),
                id_columns=["video_id", "audio_id"],
                fields=[FieldSpec(key=FieldKey(["value_b"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["video_id", "frame_id", "audio_id"],
                deps=[FeatureDep(feature=UpstreamA), FeatureDep(feature=UpstreamB)],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        plan = graph.get_feature_plan(Downstream.spec().key)
        assert set(plan.upstream_id_columns) == {"video_id", "frame_id", "audio_id"}

    def test_with_rename_applies_column_mapping(self, graph: FeatureGraph):
        """FeatureDep.rename is applied to upstream id_columns."""

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                id_columns=["sample_uid", "original_id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["sample_uid", "renamed_id"],
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

        plan = graph.get_feature_plan(Downstream.spec().key)
        assert set(plan.upstream_id_columns) == {"sample_uid", "renamed_id"}
        assert "original_id" not in plan.upstream_id_columns

    def test_with_partial_rename(self, graph: FeatureGraph):
        """Only renamed columns are affected, others pass through unchanged."""

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                id_columns=["video_id", "frame_id", "extra_id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["video_id", "renamed_frame_id", "extra_id"],
                deps=[
                    FeatureDep(
                        feature=Upstream,
                        rename={"frame_id": "renamed_frame_id"},
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        plan = graph.get_feature_plan(Downstream.spec().key)
        assert set(plan.upstream_id_columns) == {
            "video_id",
            "renamed_frame_id",
            "extra_id",
        }
        assert "frame_id" not in plan.upstream_id_columns


class TestInputIdColumns:
    """Tests for FeaturePlan.input_id_columns property."""

    def test_identity_lineage_returns_upstream_id_columns(self, graph: FeatureGraph):
        """Identity (1:1) lineage returns upstream_id_columns."""

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
                # Default lineage is identity
            ),
        ):
            pass

        plan = graph.get_feature_plan(Downstream.spec().key)
        assert set(plan.input_id_columns) == {"video_id", "frame_id"}
        assert plan.input_id_columns == plan.upstream_id_columns

    def test_aggregation_lineage_returns_aggregation_columns(self, graph: FeatureGraph):
        """Aggregation (N:1) lineage returns aggregation columns."""

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
                        lineage=LineageRelationship.aggregation(
                            on=["sensor_id", "hour"]
                        ),
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["avg_temp"]), code_version="1")],
            ),
        ):
            pass

        plan = graph.get_feature_plan(HourlyStats.spec().key)
        # input_id_columns should be the aggregation columns
        assert set(plan.input_id_columns) == {"sensor_id", "hour"}
        # These should equal the downstream id_columns for aggregation
        assert set(plan.input_id_columns) == set(plan.feature.id_columns)

    def test_expansion_lineage_returns_parent_columns(self, graph: FeatureGraph):
        """Expansion (1:N) lineage returns parent columns from ExpansionRelationship.on."""

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

        plan = graph.get_feature_plan(VideoFrames.spec().key)
        # input_id_columns should be the parent columns (expansion 'on')
        assert plan.input_id_columns == ["video_id"]

    def test_aggregation_with_multiple_upstreams(self, graph: FeatureGraph):
        """Aggregation with multiple upstreams - each dep can have its own lineage.

        With per-dep lineage, input_id_columns is the intersection of input columns
        from all dependencies after their lineage transformations.
        """

        class UpstreamA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_a"]),
                id_columns=["sensor_id", "timestamp", "reading_id"],
                fields=[FieldSpec(key=FieldKey(["temp"]), code_version="1")],
            ),
        ):
            pass

        class UpstreamB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_b"]),
                id_columns=["sensor_id", "hour"],  # UpstreamB also has hour column
                fields=[FieldSpec(key=FieldKey(["humidity"]), code_version="1")],
            ),
        ):
            pass

        class Aggregated(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["aggregated"]),
                id_columns=["sensor_id", "hour"],
                deps=[
                    FeatureDep(
                        feature=UpstreamA,
                        lineage=LineageRelationship.aggregation(
                            on=["sensor_id", "hour"]
                        ),
                    ),
                    # UpstreamB has identity lineage (default) - its input columns are its ID columns
                    FeatureDep(feature=UpstreamB),
                ],
                fields=[FieldSpec(key=FieldKey(["stats"]), code_version="1")],
            ),
        ):
            pass

        plan = graph.get_feature_plan(Aggregated.spec().key)
        # input_id_columns is intersection of:
        # - UpstreamA after aggregation: ["sensor_id", "hour"]
        # - UpstreamB with identity: ["sensor_id", "hour"]
        # Intersection: ["sensor_id", "hour"]
        assert set(plan.input_id_columns) == {"sensor_id", "hour"}

    def test_expansion_with_renamed_upstream(self, graph: FeatureGraph):
        """Expansion lineage with renamed upstream columns."""

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
                id_columns=["vid_id", "frame_id"],
                deps=[
                    FeatureDep(
                        feature=Video,
                        rename={"video_id": "vid_id"},
                        lineage=LineageRelationship.expansion(on=["vid_id"]),
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["embedding"]), code_version="1")],
            ),
        ):
            pass

        plan = graph.get_feature_plan(VideoFrames.spec().key)
        # input_id_columns should use the renamed column
        assert plan.input_id_columns == ["vid_id"]

    def test_identity_with_multiple_upstreams_and_renames(self, graph: FeatureGraph):
        """Identity lineage with multiple upstreams and various renames.

        With per-dep lineage, input_id_columns is the intersection of input columns
        from all dependencies. When deps have different ID columns, the intersection
        is only the common columns.
        """

        class UpstreamA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_a"]),
                id_columns=["sample_uid", "col_a"],
                fields=[FieldSpec(key=FieldKey(["value_a"]), code_version="1")],
            ),
        ):
            pass

        class UpstreamB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_b"]),
                id_columns=["sample_uid", "col_b"],
                fields=[FieldSpec(key=FieldKey(["value_b"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                id_columns=["sample_uid", "renamed_a", "renamed_b"],
                deps=[
                    FeatureDep(feature=UpstreamA, rename={"col_a": "renamed_a"}),
                    FeatureDep(feature=UpstreamB, rename={"col_b": "renamed_b"}),
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        plan = graph.get_feature_plan(Downstream.spec().key)
        # input_id_columns is the intersection of:
        # - UpstreamA (identity): ["sample_uid", "renamed_a"] (after rename)
        # - UpstreamB (identity): ["sample_uid", "renamed_b"] (after rename)
        # Intersection: ["sample_uid"]
        assert set(plan.input_id_columns) == {"sample_uid"}
