"""Tests for feature graph snapshot push functionality (GitHub issue #86).

This module tests the new metadata-only change detection feature in
push_graph_snapshot() that enables graph snapshot pushes
without triggering migrations.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from metaxy_testing.models import SampleFeatureSpec
from syrupy.assertion import SnapshotAssertion

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import PushResult


def test_snapshot_push_result_model():
    """Test PushResult NamedTuple structure and backward compatibility."""
    result = PushResult(
        project_version="abc123",
        already_pushed=False,
        updated_features=[],
    )

    # Test fields
    assert result.project_version == "abc123"
    assert result.already_pushed is False
    assert result.updated_features == []

    # Test tuple unpacking
    project_version, already_pushed, features = result
    assert project_version == "abc123"
    assert already_pushed is False
    assert features == []


def test_record_snapshot_first_time(tmp_path: Path):
    """Test recording a brand new snapshot (computational changes).

    Scenario 1: Empty store, first push.
    Expected: already_pushed=False, updated_features=[]
    """
    graph = FeatureGraph()

    with graph.use():

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            result = SystemTableStorage(store).push_graph_snapshot()

            # Verify result
            assert isinstance(result, PushResult)
            assert result.already_pushed is False
            assert result.updated_features == []
            # push_graph_snapshot now returns project-scoped snapshot version
            project = graph.feature_definitions_by_key[FeatureKey(["video", "files"])].project
            assert result.project_version == graph.get_project_version(project)

            # Verify data was written to feature_versions table
            from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

            versions_lazy = store._read_feature(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            assert versions_df.height == 1
            assert versions_df["feature_key"][0] == "video/files"
            assert versions_df["metaxy_project_version"][0] == graph.get_project_version(project)


def test_record_snapshot_metadata_only_changes(tmp_path: Path):
    """Test recording metadata-only changes (rename in FeatureDep).

    Scenario 2: Snapshot exists, but feature_spec_version changed
    (without changing feature_version).

    Example: Adding rename={"old": "new"} to a FeatureDep
    This changes feature_spec_version but NOT feature_version.

    Note: With project-scoped snapshot versions, spec changes (like adding rename)
    now create a NEW snapshot version since we use feature_definition_version
    (which includes the full spec) instead of feature_version (provenance only).
    """
    from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

    # Version 1: No rename
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],  # No rename yet
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        # Get project from the features
        project = graph_v1.feature_definitions_by_key[FeatureKey(["upstream"])].project
        snapshot_v1 = graph_v1.get_project_version(project)
        feature_version_v1 = Downstream.feature_version()
        spec_version_v1 = Downstream.feature_spec_version()

        # First push
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            result1 = SystemTableStorage(store).push_graph_snapshot()
            assert result1.already_pushed is False
            assert result1.project_version == snapshot_v1

            # Verify initial state
            versions_lazy = store._read_feature(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()
            assert versions_df.height == 2  # upstream + downstream

            # Version 2: Add rename (spec change)
            # This changes spec_version and feature_definition_version,
            # which means the project snapshot version also changes
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class Upstream2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["upstream"]),
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class Downstream2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["downstream"]),
                        deps=[
                            FeatureDep(
                                feature=FeatureKey(["upstream"]),
                                rename={"value": "renamed_value"},  # Added rename
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                project2 = graph_v2.feature_definitions_by_key[FeatureKey(["upstream"])].project
                snapshot_v2 = graph_v2.get_project_version(project2)
                feature_version_v2 = Downstream2.feature_version()
                spec_version_v2 = Downstream2.feature_spec_version()

                # feature_version stays the same (rename doesn't affect provenance)
                assert feature_version_v2 == feature_version_v1, (
                    "Feature version should be unchanged (rename doesn't affect provenance)"
                )

                # But spec_version is DIFFERENT (spec changed)
                assert spec_version_v2 != spec_version_v1, "Spec version should change (metadata changed)"

                # And project snapshot version is DIFFERENT (uses definition_version which includes spec)
                assert snapshot_v2 != snapshot_v1, "Project snapshot version should change (definition_version changed)"

                # Second push - this is now a NEW snapshot, not just a metadata update
                result2 = SystemTableStorage(store).push_graph_snapshot()

                # Verify result - this is a new snapshot, not already pushed
                assert result2.already_pushed is False
                assert result2.project_version == snapshot_v2

                # Verify new rows were added (new snapshot)
                versions_lazy_after = store._read_feature(FEATURE_VERSIONS_KEY)
                assert versions_lazy_after is not None
                versions_df_after = versions_lazy_after.collect().to_polars()
                # Old rows (2) + new rows (2) = 4
                assert versions_df_after.height == 4

                # Verify both features have rows for both snapshots
                downstream_rows = versions_df_after.filter(pl.col("feature_key") == "downstream")
                assert downstream_rows.height == 2

                upstream_rows = versions_df_after.filter(pl.col("feature_key") == "upstream")
                assert upstream_rows.height == 2


def test_record_snapshot_no_changes(tmp_path: Path):
    """Test recording when nothing changed.

    Scenario 3: Snapshot exists with identical code.
    Expected: already_pushed=True, updated_features=[]
    """
    graph = FeatureGraph()

    with graph.use():

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # First push
            result1 = SystemTableStorage(store).push_graph_snapshot()
            assert result1.already_pushed is False

            # Second push - identical code
            result2 = SystemTableStorage(store).push_graph_snapshot()

            # Verify result
            project = graph.feature_definitions_by_key[FeatureKey(["video", "files"])].project
            assert result2.already_pushed is True
            assert result2.updated_features == []
            assert result2.project_version == graph.get_project_version(project)

            # Verify no new rows appended
            from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

            versions_lazy = store._read_feature(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()
            assert versions_df.height == 1  # Still only 1 row


def test_record_snapshot_partial_metadata_changes(tmp_path: Path):
    """Test spec changes for SOME features create a new snapshot version.

    With project-scoped snapshot versions, spec changes (like adding rename)
    now create a NEW snapshot version since we use feature_definition_version.
    """
    from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

    # Version 1: Three features
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_a"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_b"]),
                deps=[FeatureDep(feature=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_c"]),
                deps=[FeatureDep(feature=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["output"]), code_version="1")],
            ),
        ):
            pass

        project = graph_v1.feature_definitions_by_key[FeatureKey(["feature_a"])].project
        snapshot_v1 = graph_v1.get_project_version(project)

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            result1 = SystemTableStorage(store).push_graph_snapshot()
            assert result1.already_pushed is False
            assert result1.project_version == snapshot_v1

            # Version 2: Change specs for FeatureB and FeatureC (add rename/columns)
            # This creates a NEW snapshot version since definition_version changes
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class FeatureA2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["feature_a"]),
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class FeatureB2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["feature_b"]),
                        deps=[
                            FeatureDep(
                                feature=FeatureKey(["feature_a"]),
                                rename={"value": "renamed_b"},  # Changed
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                class FeatureC2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["feature_c"]),
                        deps=[
                            FeatureDep(
                                feature=FeatureKey(["feature_a"]),
                                select=("value",),  # Changed
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["output"]), code_version="1")],
                    ),
                ):
                    pass

                project2 = graph_v2.feature_definitions_by_key[FeatureKey(["feature_a"])].project
                snapshot_v2 = graph_v2.get_project_version(project2)

                # Snapshot version should be different (spec changes affect definition_version)
                assert snapshot_v2 != snapshot_v1, "Spec changes should create new project_version"

                result2 = SystemTableStorage(store).push_graph_snapshot()

                # This is a NEW snapshot, not just a metadata update
                assert result2.already_pushed is False
                assert result2.project_version == snapshot_v2

                # Verify correct rows appended (3 original + 3 new = 6)
                versions_lazy = store._read_feature(FEATURE_VERSIONS_KEY)
                assert versions_lazy is not None
                versions_df = versions_lazy.collect().to_polars()
                assert versions_df.height == 6


def test_record_snapshot_append_only_behavior(tmp_path: Path):
    """Test append-only behavior: old rows preserved when new snapshot is pushed.

    With project-scoped snapshot versions, spec changes create new snapshot versions.
    This test verifies that multiple snapshots can coexist in the store.
    """
    from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

    # Start with version 1
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["my_feature"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        project = graph_v1.feature_definitions_by_key[FeatureKey(["upstream"])].project
        snapshot_v1 = graph_v1.get_project_version(project)

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Push v1
            result1 = SystemTableStorage(store).push_graph_snapshot()
            assert result1.project_version == snapshot_v1

            versions_lazy_v1 = store._read_feature(FEATURE_VERSIONS_KEY)
            assert versions_lazy_v1 is not None
            versions_df_v1 = versions_lazy_v1.collect().to_polars()
            assert versions_df_v1.height == 2  # upstream + my_feature

            my_feature_rows_v1 = versions_df_v1.filter(pl.col("feature_key") == "my_feature")
            assert my_feature_rows_v1.height == 1
            timestamp_v1 = my_feature_rows_v1["recorded_at"][0]

            # Change spec (rename in FeatureDep) - creates NEW snapshot version
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class Upstream2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["upstream"]),
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class MyFeature2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["my_feature"]),
                        deps=[
                            FeatureDep(
                                feature=FeatureKey(["upstream"]),
                                rename={"value": "new_name"},  # Spec change
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                project2 = graph_v2.feature_definitions_by_key[FeatureKey(["upstream"])].project
                snapshot_v2 = graph_v2.get_project_version(project2)
                # Spec changes now create new snapshot versions
                assert snapshot_v2 != snapshot_v1, "Spec change should create new project_version"

                result2 = SystemTableStorage(store).push_graph_snapshot()
                assert result2.project_version == snapshot_v2

                # Verify append-only: old rows still exist
                versions_lazy_v2 = store._read_feature(FEATURE_VERSIONS_KEY)
                assert versions_lazy_v2 is not None
                versions_df_v2 = versions_lazy_v2.collect().to_polars()

                # Total rows: original 2 + 2 new = 4
                assert versions_df_v2.height == 4

                # Find my_feature rows
                my_feature_rows = versions_df_v2.filter(pl.col("feature_key") == "my_feature")

                # Should have 2 rows: one for each snapshot
                assert my_feature_rows.height == 2

                # Different project_versions
                project_versions = set(my_feature_rows["metaxy_project_version"].to_list())
                assert project_versions == {snapshot_v1, snapshot_v2}

                # Different definition_versions
                definition_versions = sorted(my_feature_rows["metaxy_definition_version"].to_list())
                assert len(definition_versions) == 2
                assert definition_versions[0] != definition_versions[1]

                # Timestamps are different (new row has later timestamp)
                timestamps = my_feature_rows.sort("recorded_at")["recorded_at"].to_list()
                assert timestamps[0] == timestamp_v1  # Old row preserved
                assert timestamps[1] > timestamp_v1  # New row has later timestamp


def test_record_snapshot_computational_change(tmp_path: Path):
    """Test computational change (code_version change) creates NEW snapshot.

    This is NOT a metadata-only change - it should return already_pushed=False.
    """
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["my_feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        project = graph_v1.feature_definitions_by_key[FeatureKey(["my_feature"])].project
        snapshot_v1 = graph_v1.get_project_version(project)

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            result1 = SystemTableStorage(store).push_graph_snapshot()
            assert result1.project_version == snapshot_v1

            # Change code_version (computational change)
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class MyFeature2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["my_feature"]),
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="2")],  # Changed
                    ),
                ):
                    pass

                project2 = graph_v2.feature_definitions_by_key[FeatureKey(["my_feature"])].project
                snapshot_v2 = graph_v2.get_project_version(project2)

                # Verify project_version changed
                assert snapshot_v2 != snapshot_v1, "Computational change should create new project_version"

                result2 = SystemTableStorage(store).push_graph_snapshot()

                # Should be treated as NEW snapshot (not metadata-only)
                assert result2.already_pushed is False
                assert result2.project_version == snapshot_v2


def test_snapshot_push_result_snapshot_comparison(snapshot: SnapshotAssertion, tmp_path: Path):
    """Test complete flow with snapshot for regression testing."""
    from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

    # Create features and push multiple times
    results = []

    # Push 1: New snapshot
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            result1 = SystemTableStorage(store).push_graph_snapshot()
            results.append(
                {
                    "push": 1,
                    "already_pushed": result1.already_pushed,
                    "updated_features": result1.updated_features,
                }
            )

            # Push 2: No changes
            result2 = SystemTableStorage(store).push_graph_snapshot()
            results.append(
                {
                    "push": 2,
                    "already_pushed": result2.already_pushed,
                    "updated_features": result2.updated_features,
                }
            )

            # Push 3: Metadata-only change
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class Upstream2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["upstream"]),
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    ),
                ):
                    pass

                class Downstream2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["downstream"]),
                        deps=[
                            FeatureDep(
                                feature=FeatureKey(["upstream"]),
                                rename={"value": "renamed"},
                            )
                        ],
                        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                    ),
                ):
                    pass

                result3 = SystemTableStorage(store).push_graph_snapshot()
                results.append(
                    {
                        "push": 3,
                        "already_pushed": result3.already_pushed,
                        "updated_features": result3.updated_features,
                    }
                )

                # Get final feature_versions table state
                versions_lazy = store._read_feature(FEATURE_VERSIONS_KEY)
                assert versions_lazy is not None
                versions_df = versions_lazy.collect().to_polars()

                # Convert to dict for snapshot (exclude timestamps for stability)
                versions_dict = versions_df.select(
                    [
                        "feature_key",
                        "metaxy_feature_version",
                        "metaxy_definition_version",
                        "metaxy_project_version",
                    ]
                ).to_dicts()

                # Snapshot all results
                assert {
                    "results": results,
                    "final_versions_count": len(versions_dict),
                    "versions": versions_dict,
                } == snapshot


def test_feature_info_changes_trigger_repush(tmp_path: Path):
    """Test that changing feature info (metadata in spec) creates a new snapshot.

    With project-scoped snapshot versions using feature_definition_version,
    spec.metadata changes now create NEW snapshot versions since they change
    the spec hash.
    """
    import time

    from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

    # Version 1: Original feature with metadata
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test_feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                metadata={"owner": "team_a", "sla": "24h"},
            ),
        ):
            pass

        project = graph_v1.feature_definitions_by_key[FeatureKey(["test_feature"])].project
        snapshot_v1 = graph_v1.get_project_version(project)

        # First push
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            result1 = SystemTableStorage(store).push_graph_snapshot()
            assert result1.already_pushed is False
            assert result1.project_version == snapshot_v1

            # Get initial recorded_at
            versions_lazy = store._read_feature(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df1 = versions_lazy.collect().to_polars()
            assert versions_df1.height == 1
            recorded_at_1 = versions_df1["recorded_at"][0]

            # Wait a bit to ensure timestamp difference
            time.sleep(0.01)

            # Version 2: Modified feature info (changed metadata)
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class TestFeature2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["test_feature"]),
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                        metadata={
                            "owner": "team_b",
                            "sla": "12h",
                            "pii": True,
                        },  # Changed
                    ),
                ):
                    pass

                project2 = graph_v2.feature_definitions_by_key[FeatureKey(["test_feature"])].project
                snapshot_v2 = graph_v2.get_project_version(project2)

                # Snapshot version should be DIFFERENT (metadata is part of spec which affects definition_version)
                assert snapshot_v1 != snapshot_v2, "Metadata changes should create new project_version"

                # Second push - this is a NEW snapshot
                result2 = SystemTableStorage(store).push_graph_snapshot()

                # Verify result - new snapshot
                assert result2.already_pushed is False
                assert result2.project_version == snapshot_v2

                # Verify new row was appended
                versions_lazy2 = store._read_feature(FEATURE_VERSIONS_KEY)
                assert versions_lazy2 is not None
                versions_df2 = versions_lazy2.collect().to_polars()
                assert versions_df2.height == 2  # Original + new

                # Get both recorded_at timestamps
                feature_rows = versions_df2.filter(pl.col("feature_key") == "test_feature").sort("recorded_at")
                assert feature_rows.height == 2

                recorded_at_old = feature_rows["recorded_at"][0]
                recorded_at_new = feature_rows["recorded_at"][1]

                # Verify recorded_at was updated
                assert recorded_at_new > recorded_at_old
                assert recorded_at_old == recorded_at_1  # Old one unchanged


def test_push_graph_snapshot_requires_project_for_multi_project_graph(tmp_path: Path):
    """Test that push_graph_snapshot raises when project cannot be determined.

    When the graph contains features from multiple projects and no project is
    specified via argument or config, push_graph_snapshot should raise ValueError.
    """
    import pytest

    from metaxy.config import MetaxyConfig
    from metaxy.models.feature_definition import FeatureDefinition
    from metaxy.models.feature_spec import FeatureSpec

    # Create a graph with features from different projects
    graph = FeatureGraph()

    # Manually add features with different projects (simulating multi-project scenario)
    spec_a = FeatureSpec(
        key=FeatureKey(["feature_a"]),
        id_columns=["id"],
        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
    )
    spec_b = FeatureSpec(
        key=FeatureKey(["feature_b"]),
        id_columns=["id"],
        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
    )

    def_a = FeatureDefinition(
        spec=spec_a,
        feature_schema={},
        feature_class_path="test.FeatureA",
        project="project_a",
    )
    def_b = FeatureDefinition(
        spec=spec_b,
        feature_schema={},
        feature_class_path="test.FeatureB",
        project="project_b",
    )

    graph.add_feature_definition(def_a)
    graph.add_feature_definition(def_b)

    # Ensure config has no project set
    MetaxyConfig.reset()

    with graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            with pytest.raises(ValueError, match="Project is required.*multiple projects"):
                SystemTableStorage(store).push_graph_snapshot()


def test_push_ignores_other_project_features_with_unresolved_deps(tmp_path: Path):
    """Test push succeeds even when graph has features from other projects with unresolved deps.

    Regression test for bug where feature discovery picks up features from other projects
    in the same codebase. When those features have dependencies on other features from their
    own project (not in the graph), push should succeed by only computing versions for
    the target project's features.

    The bug occurred because to_snapshot() processed ALL non-external features, including
    features from other projects that were discovered but had unresolved dependencies.
    """
    from metaxy.models.feature_definition import FeatureDefinition
    from metaxy.models.feature_spec import FeatureSpec

    graph = FeatureGraph()

    # Create a feature from "my_project" that we want to push
    my_spec = FeatureSpec(
        key=FeatureKey(["my_feature"]),
        id_columns=["id"],
        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
    )
    my_def = FeatureDefinition(
        spec=my_spec,
        feature_schema={"properties": {"value": {"type": "string"}}},
        feature_class_path="my_project.MyFeature",
        project="my_project",
    )
    graph.add_feature_definition(my_def)

    # Create a feature from "other_project" that depends on a feature NOT in the graph.
    # This simulates the bug scenario: feature discovery picks up a feature from another
    # project that has dependencies on its own project's features (not available here).
    other_spec = FeatureSpec(
        key=FeatureKey(["other_feature"]),
        id_columns=["id"],
        fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        # This depends on a feature that's NOT in the graph
        deps=[FeatureDep(feature=FeatureKey(["missing", "dependency"]))],
    )
    other_def = FeatureDefinition(
        spec=other_spec,
        feature_schema={"properties": {"data": {"type": "string"}}},
        feature_class_path="other_project.OtherFeature",
        project="other_project",
    )
    graph.add_feature_definition(other_def)

    with graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # This should succeed: we're only pushing my_project, so the unresolved
            # dependency in other_project's feature should be ignored
            result = SystemTableStorage(store).push_graph_snapshot(project="my_project")

            assert not result.already_pushed
            # Only my_feature should be in the pushed snapshot
            features_df = SystemTableStorage(store).read_features(current=False, project_version=result.project_version)
            assert features_df.height == 1
            assert features_df["feature_key"][0] == "my_feature"
            assert features_df["project"][0] == "my_project"
