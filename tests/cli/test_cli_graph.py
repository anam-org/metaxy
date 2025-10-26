"""Tests for graph CLI commands."""

import re

from metaxy._testing import TempMetaxyProject


def test_graph_push_first_time(metaxy_project: TempMetaxyProject):
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
        assert "Snapshot version:" in result.stdout


def test_graph_push_already_recorded(metaxy_project: TempMetaxyProject):
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
        assert "Snapshot version:" in result2.stdout


def test_graph_history_empty(metaxy_project: TempMetaxyProject):
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


def test_graph_history_with_snapshots(metaxy_project: TempMetaxyProject):
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
        assert "Snapshot version" in result.stdout
        assert "Recorded At" in result.stdout
        assert "Feature Count" in result.stdout
        assert "1" in result.stdout  # 1 feature


def test_graph_history_with_limit(metaxy_project: TempMetaxyProject):
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


def test_graph_describe_current(metaxy_project: TempMetaxyProject):
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
        assert "video/files" in result.stdout


def test_graph_describe_with_dependencies(metaxy_project: TempMetaxyProject):
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
            assert "video/files" in result.stdout


def test_graph_describe_historical_snapshot(metaxy_project: TempMetaxyProject):
    """Test graph describe with specific snapshot version."""

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

        # Extract snapshot version from output
        # The snapshot version might be wrapped to the next line, so use DOTALL flag
        match = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push_result.stdout, re.DOTALL
        )
        assert match, (
            f"Could not find snapshot version in push output. Output: {push_result.stdout}"
        )
        snapshot_version = match.group(1)

        # Describe specific snapshot
        result = metaxy_project.run_cli(
            "graph", "describe", "--snapshot", snapshot_version
        )

        assert result.returncode == 0
        # Check that output contains "Describing snapshot" and the snapshot_version (may have newlines between them)
        assert "Describing snapshot" in result.stdout
        assert snapshot_version in result.stdout
        assert "Graph Snapshot:" in result.stdout
        assert "Feature Count" in result.stdout


def test_graph_commands_with_store_flag(metaxy_project: TempMetaxyProject):
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


def test_graph_workflow_integration(metaxy_project: TempMetaxyProject):
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

        # Extract snapshot version
        # The snapshot version might be wrapped to the next line, so use DOTALL flag
        match = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push_result.stdout, re.DOTALL
        )
        assert match is not None
        snapshot_version = match.group(1)

        # Step 2: History should show the snapshot
        history_result = metaxy_project.run_cli("graph", "history")
        assert snapshot_version[:13] in history_result.stdout
        assert "2" in history_result.stdout  # 2 features

        # Step 3: Describe should show current graph
        describe_result = metaxy_project.run_cli("graph", "describe")
        assert "Feature Count" in describe_result.stdout
        assert "2" in describe_result.stdout

        # Step 4: Describe historical snapshot
        describe_historical = metaxy_project.run_cli(
            "graph", "describe", "--snapshot", snapshot_version
        )
        # Check that output contains "Describing snapshot" and the snapshot_version (may have newlines between them)
        assert "Describing snapshot" in describe_historical.stdout
        assert snapshot_version in describe_historical.stdout


def test_graph_render_terminal_basic(metaxy_project: TempMetaxyProject):
    """Test basic terminal rendering."""

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
        result = metaxy_project.run_cli("graph", "render", "--format", "terminal")

        assert result.returncode == 0
        assert "Graph" in result.stdout
        assert "video/files" in result.stdout
        assert "fields" in result.stdout
        assert "default" in result.stdout


def test_graph_render_cards_format(metaxy_project: TempMetaxyProject):
    """Test cards format rendering."""

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
        result = metaxy_project.run_cli("graph", "render", "--type", "cards")

        assert result.returncode == 0
        assert "Graph" in result.stdout
        assert "video/files" in result.stdout
        assert "Features:" in result.stdout


def test_graph_render_with_dependencies(metaxy_project: TempMetaxyProject):
    """Test rendering graph with dependencies shows edges."""

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

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            # Test terminal format shows dependencies
            result = metaxy_project.run_cli("graph", "render", "--format", "terminal")
            assert result.returncode == 0
            assert "video/files" in result.stdout
            assert "video/processing" in result.stdout
            assert "depends on" in result.stdout

            # Test cards format shows edges
            result_cards = metaxy_project.run_cli("graph", "render", "--type", "cards")
            assert result_cards.returncode == 0
            assert "video/files" in result_cards.stdout
            assert "video/processing" in result_cards.stdout
            assert "→" in result_cards.stdout  # Arrow for dependency


def test_graph_render_mermaid_format(metaxy_project: TempMetaxyProject):
    """Test Mermaid format rendering."""

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
        result = metaxy_project.run_cli("graph", "render", "--format", "mermaid")

        assert result.returncode == 0
        assert "flowchart" in result.stdout
        assert "video/files" in result.stdout
        assert "title:" in result.stdout


def test_graph_render_minimal_preset(metaxy_project: TempMetaxyProject):
    """Test minimal preset hides version information."""

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
        result = metaxy_project.run_cli("graph", "render", "--minimal")

        assert result.returncode == 0
        assert "video/files" in result.stdout
        # Should not show versions in minimal mode
        assert "v:" not in result.stdout


def test_graph_render_verbose_preset(metaxy_project: TempMetaxyProject):
    """Test verbose preset shows all information."""

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
        result = metaxy_project.run_cli("graph", "render", "--verbose")

        assert result.returncode == 0
        assert "video/files" in result.stdout
        # Verbose should show versions
        assert "v:" in result.stdout
        # Verbose should show code versions
        assert "cv:" in result.stdout


def test_graph_render_with_filtering(metaxy_project: TempMetaxyProject):
    """Test graph rendering with focus feature filtering."""

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
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(key=FeatureKey(["video", "files"]))],
                fields=[FieldSpec(key=FieldKey(["frames"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            # Focus on video/processing with upstream dependencies
            result = metaxy_project.run_cli(
                "graph",
                "render",
                "--feature",
                "video/processing",
                "--up",
                "1",
            )

            assert result.returncode == 0
            assert "video/files" in result.stdout
            assert "video/processing" in result.stdout


def test_graph_render_output_to_file(metaxy_project: TempMetaxyProject):
    """Test rendering output to file."""

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
        output_file = metaxy_project.project_dir / "graph.mmd"
        result = metaxy_project.run_cli(
            "graph",
            "render",
            "--format",
            "mermaid",
            "--output",
            str(output_file),
        )

        assert result.returncode == 0
        assert "saved to" in result.stdout
        assert output_file.exists()

        # Check file contents
        content = output_file.read_text()
        assert "flowchart" in content
        assert "video/files" in content


def test_graph_render_field_dependencies(metaxy_project: TempMetaxyProject):
    """Test that field dependencies are shown in rendering."""

    def root_features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["path"]), code_version=1)],
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
                                fields=[FieldKey(["path"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            result = metaxy_project.run_cli("graph", "render", "--format", "terminal")

            assert result.returncode == 0
            # Should show field dependency
            assert "frames" in result.stdout
            assert "video/files.path" in result.stdout or "←" in result.stdout


def test_graph_render_custom_flags(metaxy_project: TempMetaxyProject):
    """Test custom rendering flags."""

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
        # Test with no fields shown
        result = metaxy_project.run_cli(
            "graph", "render", "--no-show-fields", "--no-show-snapshot-version"
        )

        assert result.returncode == 0
        assert "video/files" in result.stdout
        # Should not show "fields" section when disabled
        assert "🔧" not in result.stdout or "fields" not in result.stdout


def test_graph_render_graphviz_format(metaxy_project: TempMetaxyProject, snapshot):
    """Test Graphviz DOT format rendering."""

    def features():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldKey,
            FieldSpec,
        )

        class Parent(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["examples", "parent"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["embeddings"]), code_version=1)],
            ),
        ):
            pass

        class Child(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["examples", "child"]),
                deps=[FeatureDep(key=FeatureKey(["examples", "parent"]))],
                fields=[FieldSpec(key=FieldKey(["predictions"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli("graph", "render", "--format", "graphviz")

        assert result.returncode == 0
        assert result.stdout == snapshot


def test_graph_diff_no_changes(metaxy_project: TempMetaxyProject):
    """Test graph diff shows no changes when snapshots are identical."""

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
        # Push snapshot
        metaxy_project.run_cli("graph", "push")

        # Compare latest with current (should be identical)
        result = metaxy_project.run_cli("graph", "diff", "latest", "current")

        assert result.returncode == 0
        assert "merged view" in result.stdout
        assert "video/files" in result.stdout
        # All features should show as unchanged (no + - ~ symbols for the feature itself)
        assert "+ video/files" not in result.stdout
        assert "- video/files" not in result.stdout
        assert "~ video/files" not in result.stdout


def test_graph_diff_added_feature(metaxy_project: TempMetaxyProject):
    """Test graph diff detects added features."""

    def features_v1():
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

    def features_v2():
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

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        # Extract first snapshot version
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        # Extract second snapshot version
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare the two snapshots
        result = metaxy_project.run_cli("graph", "diff", snapshot1, snapshot2)

        assert result.returncode == 0
        assert (
            "audio/files (added)" in result.stdout or "+ audio/files" in result.stdout
        )


def test_graph_diff_removed_feature(metaxy_project: TempMetaxyProject):
    """Test graph diff detects removed features."""

    def features_v1():
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

    def features_v2():
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

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare the two snapshots
        result = metaxy_project.run_cli("graph", "diff", snapshot1, snapshot2)

        assert result.returncode == 0
        assert (
            "audio/files (removed)" in result.stdout or "- audio/files" in result.stdout
        )


def test_graph_diff_changed_feature(metaxy_project: TempMetaxyProject):
    """Test graph diff detects changed features."""

    def features_v1():
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

    def features_v2():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare the two snapshots
        result = metaxy_project.run_cli("graph", "diff", snapshot1, snapshot2)

        assert result.returncode == 0
        assert (
            "video/files (changed)" in result.stdout or "~ video/files" in result.stdout
        )


def test_graph_diff_with_store_flag(metaxy_project: TempMetaxyProject):
    """Test graph diff works with --store flag."""

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
        # Push with explicit store
        metaxy_project.run_cli("graph", "push", "--store", "dev")

        # Diff with explicit store
        result = metaxy_project.run_cli(
            "graph", "diff", "latest", "current", "--store", "dev"
        )

        assert result.returncode == 0
        # Should show merged view (default) or diff list
        assert (
            "merged view" in result.stdout
            or "Graph Diff:" in result.stdout
            or "No changes" in result.stdout
        )


def test_graph_diff_invalid_snapshot(metaxy_project: TempMetaxyProject):
    """Test graph diff fails gracefully with invalid snapshot."""

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
        result = metaxy_project.run_cli(
            "graph", "diff", "nonexistent_snapshot", "current", check=False
        )

        assert result.returncode == 1
        assert "Error:" in result.stdout


def test_graph_diff_latest_empty_store(metaxy_project: TempMetaxyProject):
    """Test graph diff fails when no snapshots exist in store."""

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
        result = metaxy_project.run_cli(
            "graph", "diff", "latest", "current", check=False
        )

        assert result.returncode == 1
        assert "Error:" in result.stdout
        assert "No snapshots found" in result.stdout


def test_graph_diff_verbose_output(metaxy_project: TempMetaxyProject):
    """Test graph diff verbose mode shows more details."""

    def features_v1():
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

    def features_v2():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare the two snapshots with verbose
        result = metaxy_project.run_cli(
            "graph", "diff", snapshot1, snapshot2, "--verbose"
        )

        assert result.returncode == 0
        assert (
            "video/files (changed)" in result.stdout or "~ video/files" in result.stdout
        )


def test_graph_diff_format_json(metaxy_project: TempMetaxyProject):
    """Test graph diff with JSON format output."""
    import json

    def features_v1():
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

    def features_v2():
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

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with JSON format (default is merged view)
        result = metaxy_project.run_cli(
            "graph", "diff", snapshot1, snapshot2, "--format", "json"
        )

        assert result.returncode == 0

        # Parse JSON output - merged format has "nodes" and "edges"
        data = json.loads(result.stdout)
        assert "nodes" in data
        assert "audio/files" in data["nodes"]
        # Check that audio/files has status "added"
        assert data["nodes"]["audio/files"]["status"] == "added"


def test_graph_diff_format_yaml(metaxy_project: TempMetaxyProject):
    """Test graph diff with YAML format output."""
    import yaml

    def features_v1():
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

    def features_v2():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class AudioFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with YAML format (default is merged view)
        result = metaxy_project.run_cli(
            "graph", "diff", snapshot1, snapshot2, "--format", "yaml"
        )

        assert result.returncode == 0

        # Parse YAML output - merged format has "nodes" and "edges"
        data = yaml.safe_load(result.stdout)
        assert "nodes" in data
        assert "video/files" in data["nodes"]
        assert "audio/files" in data["nodes"]
        assert data["nodes"]["video/files"]["status"] == "removed"
        assert data["nodes"]["audio/files"]["status"] == "added"


def test_graph_diff_format_mermaid(metaxy_project: TempMetaxyProject):
    """Test graph diff with Mermaid format output."""

    def features_v1():
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

    def features_v2():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with Mermaid format
        result = metaxy_project.run_cli(
            "graph", "diff", snapshot1, snapshot2, "--format", "mermaid"
        )

        assert result.returncode == 0
        assert "flowchart TB" in result.stdout
        assert "video/files" in result.stdout


def test_graph_diff_output_to_file(metaxy_project: TempMetaxyProject):
    """Test graph diff outputs to file."""

    def features_v1():
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

    def features_v2():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class AudioFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Record first snapshot (v1)
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot (v2)
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Output to JSON file (merged format)
        output_file = metaxy_project.project_dir / "diff.json"
        result = metaxy_project.run_cli(
            "graph",
            "diff",
            snapshot1,
            snapshot2,
            "--format",
            "json",
            "--output",
            str(output_file),
        )

        assert result.returncode == 0
        assert "saved to" in result.stdout
        assert output_file.exists()

        # Check file contents - merged format has "nodes" and "edges"
        import json

        with open(output_file) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "video/files" in data["nodes"]
        assert "audio/files" in data["nodes"]


def test_graph_diff_invalid_format(metaxy_project: TempMetaxyProject):
    """Test graph diff fails with invalid format."""

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
        # Push snapshot
        metaxy_project.run_cli("graph", "push")

        # Try invalid format
        result = metaxy_project.run_cli(
            "graph", "diff", "latest", "current", "--format", "invalid", check=False
        )

        assert result.returncode == 1
        assert "Error:" in result.stdout
        assert "Invalid format" in result.stdout
