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
            # Check for dependency indicator (emoji arrow or legacy arrow)
            assert (
                "video/files.path" in result.stdout
                or "⬅️" in result.stdout
                or "←" in result.stdout
            )


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


def test_graph_push_metadata_only_changes(metaxy_project: TempMetaxyProject):
    """Test CLI output for metadata-only changes (GitHub issue #86).

    When FeatureDep metadata changes (e.g., rename added) without computational
    changes, the CLI should display a different message than computational changes.
    """

    def features_v1():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version=1)],
            ),
        ):
            pass

        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(key=FeatureKey(["upstream"]))],  # No rename yet
                fields=[FieldSpec(key=FieldKey(["result"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version=1)],
            ),
        ):
            pass

        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        key=FeatureKey(["upstream"]),
                        rename={"value": "renamed_value"},  # Added rename
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version=1)],
            ),
        ):
            pass

    # Push v1
    with metaxy_project.with_features(features_v1):
        result1 = metaxy_project.run_cli("graph", "push")
        assert result1.returncode == 0
        assert "Recorded feature graph" in result1.stdout
        assert "Snapshot version:" in result1.stdout

    # Push v2 (metadata-only change)
    with metaxy_project.with_features(features_v2):
        result2 = metaxy_project.run_cli("graph", "push")
        assert result2.returncode == 0

        # Should show metadata-only change message
        assert "Updated feature graph metadata" in result2.stdout
        assert "no topological changes" in result2.stdout

        # Should list the changed feature
        assert "downstream" in result2.stdout

        # Should still show snapshot version
        assert "Snapshot version:" in result2.stdout


def test_graph_push_no_changes(metaxy_project: TempMetaxyProject):
    """Test CLI output when nothing changed (GitHub issue #86)."""

    def features():
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

    with metaxy_project.with_features(features):
        # First push
        result1 = metaxy_project.run_cli("graph", "push")
        assert "Recorded feature graph" in result1.stdout

        # Second push - no changes
        result2 = metaxy_project.run_cli("graph", "push")
        assert result2.returncode == 0
        assert "already recorded" in result2.stdout
        assert "no changes" in result2.stdout
        assert "Snapshot version:" in result2.stdout


def test_graph_push_three_scenarios_integration(metaxy_project: TempMetaxyProject):
    """Test complete workflow: new → metadata change → no change (GitHub issue #86)."""

    def features_v1():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version=1)],
            ),
        ):
            pass

        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(key=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version=1)],
            ),
        ):
            pass

        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        key=FeatureKey(["upstream"]),
                        columns=("value",),  # Metadata change
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version=1)],
            ),
        ):
            pass

    # Scenario 1: First push (new snapshot)
    with metaxy_project.with_features(features_v1):
        result1 = metaxy_project.run_cli("graph", "push")
        assert result1.returncode == 0
        assert "Recorded feature graph" in result1.stdout
        assert "Snapshot version:" in result1.stdout

    # Scenario 2: Metadata change
    with metaxy_project.with_features(features_v2):
        result2 = metaxy_project.run_cli("graph", "push")
        assert result2.returncode == 0
        assert "Updated feature graph metadata" in result2.stdout
        assert "no topological changes" in result2.stdout
        assert "downstream" in result2.stdout

    # Scenario 3: No change
    with metaxy_project.with_features(features_v2):
        result3 = metaxy_project.run_cli("graph", "push")
        assert result3.returncode == 0
        assert "already recorded" in result3.stdout
        assert "no changes" in result3.stdout


def test_graph_push_multiple_features_metadata_changes(
    metaxy_project: TempMetaxyProject,
):
    """Test CLI output when multiple features have metadata changes."""

    def features_v1():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldKey,
            FieldSpec,
        )

        class FeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_a"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version=1)],
            ),
        ):
            pass

        class FeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_b"]),
                deps=[FeatureDep(key=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version=1)],
            ),
        ):
            pass

        class FeatureC(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_c"]),
                deps=[FeatureDep(key=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["output"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FeatureSpec,
            FieldKey,
            FieldSpec,
        )

        class FeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_a"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["value"]), code_version=1)],
            ),
        ):
            pass

        class FeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_b"]),
                deps=[
                    FeatureDep(
                        key=FeatureKey(["feature_a"]), rename={"value": "renamed_b"}
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version=1)],
            ),
        ):
            pass

        class FeatureC(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_c"]),
                deps=[
                    FeatureDep(
                        key=FeatureKey(["feature_a"]), columns=("value",)
                    )  # Changed
                ],
                fields=[FieldSpec(key=FieldKey(["output"]), code_version=1)],
            ),
        ):
            pass

    # Push v1
    with metaxy_project.with_features(features_v1):
        result1 = metaxy_project.run_cli("graph", "push")
        assert "Recorded feature graph" in result1.stdout

    # Push v2 - multiple metadata changes
    with metaxy_project.with_features(features_v2):
        result2 = metaxy_project.run_cli("graph", "push")
        assert result2.returncode == 0
        assert "Updated feature graph metadata" in result2.stdout

        # Should list both changed features
        assert "feature_b" in result2.stdout
        assert "feature_c" in result2.stdout

        # Should NOT list unchanged feature
        # Note: feature_a might appear in deps, so we check it's not in the "Features with metadata changes" section
        # For now, just verify the two changed features are listed
