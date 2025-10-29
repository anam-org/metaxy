"""Tests for feature graph snapshot push functionality (GitHub issue #86).

This module tests the new metadata-only change detection feature in
record_feature_graph_snapshot() that enables graph snapshot pushes
without triggering migrations.
"""

from __future__ import annotations

import polars as pl
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldKey,
    FieldSpec,
)
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import SnapshotPushResult


def test_snapshot_push_result_model():
    """Test SnapshotPushResult NamedTuple structure and backward compatibility."""
    result = SnapshotPushResult(
        snapshot_version="abc123",
        already_recorded=False,
        metadata_changed=False,
        features_with_spec_changes=[],
    )

    # Test fields
    assert result.snapshot_version == "abc123"
    assert result.already_recorded is False
    assert result.metadata_changed is False
    assert result.features_with_spec_changes == []

    # Test tuple unpacking for backward compatibility
    snapshot_version, already_recorded, metadata_changed, features = result
    assert snapshot_version == "abc123"
    assert already_recorded is False
    assert metadata_changed is False
    assert features == []


def test_record_snapshot_first_time():
    """Test recording a brand new snapshot (computational changes).

    Scenario 1: Empty store, first push.
    Expected: already_recorded=False, metadata_changed=False
    """
    graph = FeatureGraph()

    with graph.use():

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            result = store.record_feature_graph_snapshot()

            # Verify result
            assert isinstance(result, SnapshotPushResult)
            assert result.already_recorded is False
            assert result.metadata_changed is False
            assert result.features_with_spec_changes == []
            assert result.snapshot_version == graph.snapshot_version

            # Verify data was written to feature_versions table
            from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

            versions_lazy = store._read_metadata_native(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            assert versions_df.height == 1
            assert versions_df["feature_key"][0] == "video/files"
            assert versions_df["snapshot_version"][0] == graph.snapshot_version


def test_record_snapshot_metadata_only_changes():
    """Test recording metadata-only changes (rename in FeatureDep).

    Scenario 2: Snapshot exists, but feature_spec_version changed
    (without changing feature_version).

    Example: Adding rename={"old": "new"} to a FeatureDep
    This changes feature_spec_version but NOT feature_version.
    """
    from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

    # Version 1: No rename
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(key=FeatureKey(["upstream"]))],  # No rename yet
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        snapshot_v1 = graph_v1.snapshot_version
        feature_version_v1 = Downstream.feature_version()
        spec_version_v1 = Downstream.feature_spec_version()

        # First push
        with InMemoryMetadataStore() as store:
            result1 = store.record_feature_graph_snapshot()
            assert result1.already_recorded is False
            assert result1.metadata_changed is False

            # Verify initial state
            versions_lazy = store._read_metadata_native(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()
            assert versions_df.height == 2  # upstream + downstream

            # Version 2: Add rename (metadata-only change)
            # This changes spec_version but NOT feature_version or snapshot_version
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class Upstream2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["upstream"]),
                        deps=None,
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class Downstream2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["downstream"]),
                        deps=[
                            FeatureDep(
                                key=FeatureKey(["upstream"]),
                                rename={"value": "renamed_value"},  # Added rename
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                snapshot_v2 = graph_v2.snapshot_version
                feature_version_v2 = Downstream2.feature_version()
                spec_version_v2 = Downstream2.feature_spec_version()

                # Verify: snapshot_version and feature_version are SAME
                assert snapshot_v2 == snapshot_v1, (
                    "Snapshot version should be unchanged (metadata-only change)"
                )
                assert feature_version_v2 == feature_version_v1, (
                    "Feature version should be unchanged (metadata-only change)"
                )

                # But spec_version is DIFFERENT
                assert spec_version_v2 != spec_version_v1, (
                    "Spec version should change (metadata changed)"
                )

                # Second push - should detect metadata-only change
                result2 = store.record_feature_graph_snapshot()

                # Verify result
                assert result2.already_recorded is True
                assert result2.metadata_changed is True
                assert "downstream" in result2.features_with_spec_changes
                assert "upstream" not in result2.features_with_spec_changes
                assert result2.snapshot_version == snapshot_v2

                # Verify new rows were appended
                versions_lazy_after = store._read_metadata_native(FEATURE_VERSIONS_KEY)
                assert versions_lazy_after is not None
                versions_df_after = versions_lazy_after.collect().to_polars()
                # Old rows + new row for downstream (upstream unchanged)
                assert versions_df_after.height == 3

                # Verify only downstream has new row
                downstream_rows = versions_df_after.filter(
                    pl.col("feature_key") == "downstream"
                )
                assert downstream_rows.height == 2  # Two versions of downstream


def test_record_snapshot_no_changes():
    """Test recording when nothing changed.

    Scenario 3: Snapshot exists with identical code.
    Expected: already_recorded=True, metadata_changed=False
    """
    graph = FeatureGraph()

    with graph.use():

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            # First push
            result1 = store.record_feature_graph_snapshot()
            assert result1.already_recorded is False

            # Second push - identical code
            result2 = store.record_feature_graph_snapshot()

            # Verify result
            assert result2.already_recorded is True
            assert result2.metadata_changed is False
            assert result2.features_with_spec_changes == []
            assert result2.snapshot_version == graph.snapshot_version

            # Verify no new rows appended
            from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

            versions_lazy = store._read_metadata_native(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()
            assert versions_df.height == 1  # Still only 1 row


def test_record_snapshot_partial_metadata_changes():
    """Test metadata changes for SOME features (not all).

    Only changed features should appear in features_with_spec_changes.
    """
    from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

    # Version 1: Three features
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class FeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_a"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_b"]),
                deps=[FeatureDep(key=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        class FeatureC(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_c"]),
                deps=[FeatureDep(key=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["output"]), code_version="1")],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            result1 = store.record_feature_graph_snapshot()
            assert result1.already_recorded is False

            # Version 2: Change metadata for FeatureB and FeatureC (not FeatureA)
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class FeatureA2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["feature_a"]),
                        deps=None,
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class FeatureB2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["feature_b"]),
                        deps=[
                            FeatureDep(
                                key=FeatureKey(["feature_a"]),
                                rename={"value": "renamed_b"},  # Changed
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                class FeatureC2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["feature_c"]),
                        deps=[
                            FeatureDep(
                                key=FeatureKey(["feature_a"]),
                                columns=("value",),  # Changed
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["output"]), code_version="1")],
                    ),
                ):
                    pass

                result2 = store.record_feature_graph_snapshot()

                # Verify: only B and C changed, not A
                assert result2.already_recorded is True
                assert result2.metadata_changed is True
                assert "feature_b" in result2.features_with_spec_changes
                assert "feature_c" in result2.features_with_spec_changes
                assert "feature_a" not in result2.features_with_spec_changes
                assert len(result2.features_with_spec_changes) == 2

                # Verify correct rows appended
                versions_lazy = store._read_metadata_native(FEATURE_VERSIONS_KEY)
                assert versions_lazy is not None
                versions_df = versions_lazy.collect().to_polars()
                # Original 3 + 2 new rows (B and C)
                assert versions_df.height == 5


def test_record_snapshot_append_only_behavior():
    """Test append-only behavior: old rows preserved, new rows added with same snapshot_version."""
    from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

    # Start with a proper setup for metadata-only change
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class MyFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["my_feature"]),
                deps=[FeatureDep(key=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        snapshot_v1 = graph_v1.snapshot_version

        with InMemoryMetadataStore() as store:
            # Push v1
            result1 = store.record_feature_graph_snapshot()
            assert result1.snapshot_version == snapshot_v1

            versions_lazy_v1 = store._read_metadata_native(FEATURE_VERSIONS_KEY)
            assert versions_lazy_v1 is not None
            versions_df_v1 = versions_lazy_v1.collect().to_polars()
            assert versions_df_v1.height == 2  # upstream + my_feature

            my_feature_rows_v1 = versions_df_v1.filter(
                pl.col("feature_key") == "my_feature"
            )
            assert my_feature_rows_v1.height == 1
            timestamp_v1 = my_feature_rows_v1["recorded_at"][0]
            my_feature_rows_v1["feature_spec_version"][0]

            # Change metadata (rename in FeatureDep)
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class Upstream2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["upstream"]),
                        deps=None,
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class MyFeature2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["my_feature"]),
                        deps=[
                            FeatureDep(
                                key=FeatureKey(["upstream"]),
                                rename={"value": "new_name"},  # Metadata change
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                snapshot_v2 = graph_v2.snapshot_version
                assert snapshot_v2 == snapshot_v1, (
                    "Metadata-only change should keep same snapshot_version"
                )

                result2 = store.record_feature_graph_snapshot()
                assert result2.metadata_changed is True

                # Verify append-only: old rows still exist
                versions_lazy_v2 = store._read_metadata_native(FEATURE_VERSIONS_KEY)
                assert versions_lazy_v2 is not None
                versions_df_v2 = versions_lazy_v2.collect().to_polars()

                # Total rows: original 2 (upstream + my_feature) + 1 new (my_feature updated)
                assert versions_df_v2.height == 3

                # Find my_feature rows
                my_feature_rows = versions_df_v2.filter(
                    pl.col("feature_key") == "my_feature"
                )

                # Should have 2 rows: old + new
                assert my_feature_rows.height == 2

                # All rows have same snapshot_version
                assert my_feature_rows["snapshot_version"].unique().to_list() == [
                    snapshot_v1
                ]

                # But different spec_versions
                spec_versions = sorted(
                    my_feature_rows["feature_spec_version"].to_list()
                )
                assert len(spec_versions) == 2
                assert spec_versions[0] != spec_versions[1]

                # Timestamps are different (new row has later timestamp)
                timestamps = my_feature_rows.sort("recorded_at")[
                    "recorded_at"
                ].to_list()
                assert timestamps[0] == timestamp_v1  # Old row preserved
                assert timestamps[1] > timestamp_v1  # New row has later timestamp


def test_record_snapshot_computational_change():
    """Test computational change (code_version change) creates NEW snapshot.

    This is NOT a metadata-only change - it should return already_recorded=False.
    """
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class MyFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["my_feature"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        snapshot_v1 = graph_v1.snapshot_version

        with InMemoryMetadataStore() as store:
            result1 = store.record_feature_graph_snapshot()
            assert result1.snapshot_version == snapshot_v1

            # Change code_version (computational change)
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class MyFeature2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["my_feature"]),
                        deps=None,
                        fields=[
                            FieldSpec(key=FieldKey(["value"]), code_version="2")
                        ],  # Changed
                    ),
                ):
                    pass

                snapshot_v2 = graph_v2.snapshot_version

                # Verify snapshot_version changed
                assert snapshot_v2 != snapshot_v1, (
                    "Computational change should create new snapshot_version"
                )

                result2 = store.record_feature_graph_snapshot()

                # Should be treated as NEW snapshot (not metadata-only)
                assert result2.already_recorded is False
                assert result2.metadata_changed is False
                assert result2.snapshot_version == snapshot_v2


def test_snapshot_push_result_snapshot_comparison(snapshot: SnapshotAssertion):
    """Test complete flow with snapshot for regression testing."""
    from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

    # Create features and push multiple times
    results = []

    # Push 1: New snapshot
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(key=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            result1 = store.record_feature_graph_snapshot()
            results.append(
                {
                    "push": 1,
                    "already_recorded": result1.already_recorded,
                    "metadata_changed": result1.metadata_changed,
                    "features_with_spec_changes": result1.features_with_spec_changes,
                }
            )

            # Push 2: No changes
            result2 = store.record_feature_graph_snapshot()
            results.append(
                {
                    "push": 2,
                    "already_recorded": result2.already_recorded,
                    "metadata_changed": result2.metadata_changed,
                    "features_with_spec_changes": result2.features_with_spec_changes,
                }
            )

            # Push 3: Metadata-only change
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class Upstream2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["upstream"]),
                        deps=None,
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class Downstream2(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["downstream"]),
                        deps=[
                            FeatureDep(
                                key=FeatureKey(["upstream"]),
                                rename={"value": "renamed"},
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                result3 = store.record_feature_graph_snapshot()
                results.append(
                    {
                        "push": 3,
                        "already_recorded": result3.already_recorded,
                        "metadata_changed": result3.metadata_changed,
                        "features_with_spec_changes": result3.features_with_spec_changes,
                    }
                )

                # Get final feature_versions table state
                versions_lazy = store._read_metadata_native(FEATURE_VERSIONS_KEY)
                assert versions_lazy is not None
                versions_df = versions_lazy.collect().to_polars()

                # Convert to dict for snapshot (exclude timestamps for stability)
                versions_dict = versions_df.select(
                    [
                        "feature_key",
                        "feature_version",
                        "feature_spec_version",
                        "snapshot_version",
                    ]
                ).to_dicts()

    # Snapshot all results
    assert {
        "results": results,
        "final_versions_count": len(versions_dict),
        "versions": versions_dict,
    } == snapshot
