"""Tests for graph CLI commands."""

import re


def test_graph_push_first_time(metaxy_project):
    """Test graph push records snapshot on first run."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli("graph", "push")

        assert result.returncode == 0
        assert "Recorded feature graph" in result.stdout
        assert "Snapshot ID:" in result.stdout


def test_graph_push_already_recorded(metaxy_project):
    """Test graph push shows 'already recorded' on second run."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # First push
        result1 = metaxy_project.run_cli("graph", "push")
        assert "Recorded feature graph" in result1.stdout

        # Second push - should skip
        result2 = metaxy_project.run_cli("graph", "push")
        assert "already recorded" in result2.stdout
        assert "Snapshot ID:" in result2.stdout


def test_graph_history_empty(metaxy_project):
    """Test graph history with no snapshots recorded."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli("graph", "history")

        assert result.returncode == 0
        assert "No graph snapshots recorded yet" in result.stdout


def test_graph_history_with_snapshots(metaxy_project):
    """Test graph history displays recorded snapshots."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push to create snapshot
        metaxy_project.run_cli("graph", "push")

        # Check history
        result = metaxy_project.run_cli("graph", "history")

        assert result.returncode == 0
        assert "Graph Snapshot History" in result.stdout
        assert "Snapshot ID" in result.stdout
        assert "Recorded At" in result.stdout
        assert "Feature Count" in result.stdout
        assert "1" in result.stdout  # 1 feature


def test_graph_history_with_limit(metaxy_project):
    """Test graph history with --limit flag."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push once
        metaxy_project.run_cli("graph", "push")

        # Check history with limit
        result = metaxy_project.run_cli("graph", "history", "--limit", "1")

        assert result.returncode == 0
        assert "Total snapshots: 1" in result.stdout


def test_graph_describe_current(metaxy_project):
    """Test graph describe shows current graph metrics."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli("graph", "describe")

        assert result.returncode == 0
        assert "Describing current graph from code" in result.stdout
        assert "Graph Snapshot:" in result.stdout
        assert "Feature Count" in result.stdout
        assert "Graph Depth" in result.stdout
        assert "Root Features" in result.stdout
        assert "1" in result.stdout  # 1 feature
        assert "video__files" in result.stdout


def test_graph_describe_with_dependencies(metaxy_project):
    """Test graph describe with dependent features shows correct depth."""

    def root_features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def dependent_features():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldDep,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(key=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version=1,
                        deps=[
                            FieldDep(
                                feature_key=FeatureKey(["video", "files"]),
                                fields=[FieldKey(["default"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

    # Load both feature modules
    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            result = metaxy_project.run_cli("graph", "describe")

            assert result.returncode == 0
            assert "Feature Count" in result.stdout
            assert "2" in result.stdout  # 2 features
            assert "Graph Depth" in result.stdout
            # Depth should be 2 (root -> dependent)
            assert "Root Features" in result.stdout
            assert "1" in result.stdout  # 1 root feature
            assert "video__files" in result.stdout


def test_graph_describe_historical_snapshot(metaxy_project):
    """Test graph describe with specific snapshot ID."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push to create snapshot
        push_result = metaxy_project.run_cli("graph", "push")

        # Extract snapshot ID from output
        match = re.search(r"Snapshot ID: ([a-f0-9]+)", push_result.stdout)
        assert match, "Could not find snapshot ID in push output"
        snapshot_id = match.group(1)

        # Describe specific snapshot
        result = metaxy_project.run_cli("graph", "describe", "--snapshot", snapshot_id)

        assert result.returncode == 0
        # Check that output contains "Describing snapshot" and the snapshot_id (may have newlines between them)
        assert "Describing snapshot" in result.stdout
        assert snapshot_id in result.stdout
        assert "Graph Snapshot:" in result.stdout
        assert "Feature Count" in result.stdout


def test_graph_commands_with_store_flag(metaxy_project):
    """Test graph commands work with --store flag."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push with explicit store (uses default "dev" store)
        result = metaxy_project.run_cli("graph", "push", "--store", "dev")

        assert result.returncode == 0
        assert "Snapshot" in result.stdout


def test_graph_workflow_integration(metaxy_project):
    """Test complete workflow: push -> history -> describe."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Step 1: Push
        push_result = metaxy_project.run_cli("graph", "push")
        assert "Recorded feature graph" in push_result.stdout

        # Extract snapshot ID
        match = re.search(r"Snapshot ID: ([a-f0-9]+)", push_result.stdout)
        assert match is not None
        snapshot_id = match.group(1)

        # Step 2: History should show the snapshot
        history_result = metaxy_project.run_cli("graph", "history")
        assert snapshot_id[:13] in history_result.stdout
        assert "2" in history_result.stdout  # 2 features

        # Step 3: Describe should show current graph
        describe_result = metaxy_project.run_cli("graph", "describe")
        assert "Feature Count" in describe_result.stdout
        assert "2" in describe_result.stdout

        # Step 4: Describe historical snapshot
        describe_historical = metaxy_project.run_cli(
            "graph", "describe", "--snapshot", snapshot_id
        )
        # Check that output contains "Describing snapshot" and the snapshot_id (may have newlines between them)
        assert "Describing snapshot" in describe_historical.stdout
        assert snapshot_id in describe_historical.stdout
