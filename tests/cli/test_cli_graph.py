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
        match = re.search(r"Snapshot version: ([a-f0-9]+)", push_result.stdout)
        assert match, "Could not find snapshot version in push output"
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
        match = re.search(r"Snapshot version: ([a-f0-9]+)", push_result.stdout)
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
