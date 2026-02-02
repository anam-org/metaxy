"""Tests for graph CLI commands."""

from metaxy_testing import TempMetaxyProject


def test_graph_push_first_time(metaxy_project: TempMetaxyProject):
    """Test push records snapshot on first run."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["push"])

        assert result.returncode == 0
        assert "Recorded feature graph" in result.stderr
        # Snapshot version now only goes to stdout, not stderr
        assert len(result.stdout.strip()) == 64  # Check hash is in stdout


def test_graph_push_already_recorded(metaxy_project: TempMetaxyProject):
    """Test push shows 'already recorded' on second run."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # First push
        result1 = metaxy_project.run_cli(["push"])
        assert "Recorded feature graph" in result1.stderr

        # Second push - should skip
        result2 = metaxy_project.run_cli(["push"])
        assert "already recorded" in result2.stderr
        # Snapshot version now only goes to stdout, not stderr
        assert len(result2.stdout.strip()) == 64  # Check hash is in stdout


def test_graph_history_empty(metaxy_project: TempMetaxyProject):
    """Test graph history with no snapshots recorded."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["graph", "history"])

        assert result.returncode == 0
        assert "No graph snapshots recorded yet" in result.stderr


def test_graph_history_with_snapshots(metaxy_project: TempMetaxyProject):
    """Test graph history displays recorded snapshots."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push to create snapshot
        metaxy_project.run_cli(["push"])

        # Check history
        result = metaxy_project.run_cli(["graph", "history"])

        assert result.returncode == 0
        assert "Graph Snapshot History" in result.stderr
        assert "Snapshot version" in result.stderr
        assert "Recorded At" in result.stderr
        assert "Feature Count" in result.stderr
        assert "1" in result.stderr  # 1 feature


def test_graph_history_with_limit(metaxy_project: TempMetaxyProject):
    """Test graph history with --limit flag."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push once
        metaxy_project.run_cli(["push"])

        # Check history with limit
        result = metaxy_project.run_cli(["graph", "history", "--limit", "1"])

        assert result.returncode == 0
        assert "Total snapshots: 1" in result.stderr


def test_graph_describe_current(metaxy_project: TempMetaxyProject):
    """Test describe graph shows current graph metrics."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["describe", "graph"])

        assert result.returncode == 0
        assert "Describing current graph from code" in result.stderr
        assert "Graph Snapshot:" in result.stderr
        assert "Total Features" in result.stderr
        assert "Graph Depth" in result.stderr
        assert "Root Features" in result.stderr
        assert "1" in result.stderr  # 1 feature
        assert "video/files" in result.stderr


def test_graph_describe_with_dependencies(metaxy_project: TempMetaxyProject):
    """Test describe graph with dependent features shows correct depth."""

    def root_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldDep,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["video", "files"]),
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
            result = metaxy_project.run_cli(["describe", "graph"])

            assert result.returncode == 0
            assert "Total Features" in result.stderr
            assert "2" in result.stderr  # 2 features
            assert "Graph Depth" in result.stderr
            # Depth should be 2 (root -> dependent)
            assert "Root Features" in result.stderr
            assert "1" in result.stderr  # 1 root feature
            assert "video/files" in result.stderr


def test_graph_describe_historical_snapshot(metaxy_project: TempMetaxyProject):
    """Test describe graph with specific snapshot version."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push to create snapshot
        push_result = metaxy_project.run_cli(["push"])

        # Extract snapshot version from stdout (just the raw hash)
        snapshot_version = push_result.stdout.strip()
        assert snapshot_version, f"Could not find snapshot version in push output. Output: {push_result.stdout}"

        # Describe specific snapshot
        result = metaxy_project.run_cli(["describe", "graph", "--snapshot", snapshot_version])

        assert result.returncode == 0
        # Check that output contains "Describing snapshot" and the snapshot_version (may have newlines between them)
        assert "Describing snapshot" in result.stderr
        assert snapshot_version in result.stderr
        assert "Graph Snapshot:" in result.stderr
        assert "Total Features" in result.stderr


def test_graph_commands_with_store_flag(metaxy_project: TempMetaxyProject):
    """Test graph commands work with --store flag."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push with explicit store (uses default "dev" store)
        result = metaxy_project.run_cli(["push", "--store", "dev"])

        assert result.returncode == 0
        assert result.stdout


def test_graph_workflow_integration(metaxy_project: TempMetaxyProject):
    """Test complete workflow: push -> history -> describe."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class AudioFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Step 1: Push
        push_result = metaxy_project.run_cli(["push"])
        assert "Recorded feature graph" in push_result.stderr

        # Extract snapshot version from stdout (just the raw hash)
        snapshot_version = push_result.stdout.strip()
        assert snapshot_version

        # Step 2: History should show the snapshot (table output goes to stderr)
        history_result = metaxy_project.run_cli(["graph", "history"])
        assert snapshot_version[:13] in history_result.stderr
        assert "2" in history_result.stderr  # 2 features

        # Step 3: Describe should show current graph
        describe_result = metaxy_project.run_cli(["describe", "graph"])
        assert "Total Features" in describe_result.stderr
        assert "2" in describe_result.stderr

        # Step 4: Describe historical snapshot
        describe_historical = metaxy_project.run_cli(["describe", "graph", "--snapshot", snapshot_version])
        # Check that output contains "Describing snapshot" and the snapshot_version (may have newlines between them)
        assert "Describing snapshot" in describe_historical.stderr
        assert snapshot_version in describe_historical.stderr


def test_graph_render_terminal_basic(metaxy_project: TempMetaxyProject):
    """Test basic terminal rendering."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["graph", "render", "--format", "terminal"])

        assert result.returncode == 0
        assert "Graph" in result.stdout
        assert "video/files" in result.stdout
        assert "fields" in result.stdout
        assert "default" in result.stdout


def test_graph_render_cards_format(metaxy_project: TempMetaxyProject):
    """Test cards format rendering."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["graph", "render", "--type", "cards"])

        assert result.returncode == 0
        assert "Graph" in result.stdout
        assert "video/files" in result.stdout
        assert "Features:" in result.stdout


def test_graph_render_with_dependencies(metaxy_project: TempMetaxyProject):
    """Test rendering graph with dependencies shows edges."""

    def root_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldDep,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["video", "files"]),
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
            result = metaxy_project.run_cli(["graph", "render", "--format", "terminal"])
            assert result.returncode == 0
            assert "video/files" in result.stdout
            assert "video/processing" in result.stdout
            assert "depends on" in result.stdout

            # Test cards format shows edges
            result_cards = metaxy_project.run_cli(["graph", "render", "--type", "cards"])
            assert result_cards.returncode == 0
            assert "video/files" in result_cards.stdout
            assert "video/processing" in result_cards.stdout
            assert "→" in result_cards.stdout  # Arrow for dependency


def test_graph_render_mermaid_format(metaxy_project: TempMetaxyProject):
    """Test Mermaid format rendering."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["graph", "render", "--format", "mermaid"])

        assert result.returncode == 0
        assert "flowchart" in result.stdout
        assert "video/files" in result.stdout
        assert "title:" in result.stdout


def test_graph_render_minimal_preset(metaxy_project: TempMetaxyProject):
    """Test minimal preset hides version information."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["graph", "render", "--minimal"])

        assert result.returncode == 0
        assert "video/files" in result.stdout
        # Should not show versions in minimal mode
        assert "v:" not in result.stdout


def test_graph_render_verbose_preset(metaxy_project: TempMetaxyProject):
    """Test verbose preset shows all information."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["graph", "render", "--verbose"])

        assert result.returncode == 0
        assert "video/files" in result.stdout
        # Verbose should show versions
        assert "v:" in result.stdout
        # Verbose should show code versions
        assert "cv:" in result.stdout


def test_graph_render_with_filtering(metaxy_project: TempMetaxyProject):
    """Test graph rendering with focus feature filtering."""

    def root_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[FieldSpec(key=FieldKey(["frames"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            # Focus on video/processing with upstream dependencies
            result = metaxy_project.run_cli(
                [
                    "graph",
                    "render",
                    "--feature",
                    "video/processing",
                    "--up",
                    "1",
                ]
            )

            assert result.returncode == 0
            assert "video/files" in result.stdout
            assert "video/processing" in result.stdout


def test_graph_render_output_to_file(metaxy_project: TempMetaxyProject):
    """Test rendering output to file."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        output_file = metaxy_project.project_dir / "graph.mmd"
        result = metaxy_project.run_cli(
            [
                "graph",
                "render",
                "--format",
                "mermaid",
                "--output",
                str(output_file),
            ]
        )

        assert result.returncode == 0
        assert "saved to" in result.stderr
        assert output_file.exists()

        # Check file contents
        content = output_file.read_text()
        assert "flowchart" in content
        assert "video/files" in content


def test_graph_render_field_dependencies(metaxy_project: TempMetaxyProject):
    """Test that field dependencies are shown in rendering."""

    def root_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldDep,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["video", "files"]),
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
            result = metaxy_project.run_cli(["graph", "render", "--format", "terminal"])

            assert result.returncode == 0
            # Should show field dependency
            assert "frames" in result.stdout
            # Check for dependency indicator (emoji arrow or legacy arrow)
            assert "video/files.path" in result.stdout or "⬅️" in result.stdout or "←" in result.stdout


def test_graph_render_custom_flags(metaxy_project: TempMetaxyProject):
    """Test custom rendering flags."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Test with no fields shown
        result = metaxy_project.run_cli(["graph", "render", "--no-show-fields", "--no-show-snapshot-version"])

        assert result.returncode == 0
        assert "video/files" in result.stdout
        # Should not show "fields" section when disabled
        assert "🔧" not in result.stdout or "fields" not in result.stdout


def test_graph_render_graphviz_format(metaxy_project: TempMetaxyProject, snapshot):
    """Test Graphviz DOT format rendering."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class Parent(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["examples", "parent"]),
                fields=[FieldSpec(key=FieldKey(["embeddings"]), code_version="1")],
            ),
        ):
            pass

        class Child(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["examples", "child"]),
                deps=[FeatureDep(feature=FeatureKey(["examples", "parent"]))],
                fields=[FieldSpec(key=FieldKey(["predictions"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["graph", "render", "--format", "graphviz"])

        assert result.returncode == 0
        assert result.stdout == snapshot


def test_graph_push_metadata_only_changes(metaxy_project: TempMetaxyProject):
    """Test CLI output for metadata-only changes (GitHub issue #86).

    When FeatureDep metadata changes (e.g., rename added) without computational
    changes, the CLI should display a different message than computational changes.
    """

    def features_v1():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],  # No rename yet
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

    def features_v2():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            SampleFeature,
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

    # Push v1
    with metaxy_project.with_features(features_v1):
        result1 = metaxy_project.run_cli(["push"])
        assert result1.returncode == 0
        assert "Recorded feature graph" in result1.stderr
        # Snapshot version now only goes to stdout
        assert len(result1.stdout.strip()) == 64

    with metaxy_project.with_features(features_v2):
        result2 = metaxy_project.run_cli(["push"])
        assert result2.returncode == 0

        # Should show metadata update message (rename doesn't change feature_version)
        assert "Updated feature information" in result2.stderr
        assert "downstream" in result2.stderr

        # Snapshot version should be in stdout
        assert len(result2.stdout.strip()) == 64


def test_graph_push_no_changes(metaxy_project: TempMetaxyProject):
    """Test CLI output when nothing changed (GitHub issue #86)."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # First push
        result1 = metaxy_project.run_cli(["push"])
        assert "Recorded feature graph" in result1.stderr

        # Second push - no changes
        result2 = metaxy_project.run_cli(["push"])
        assert result2.returncode == 0
        assert "already recorded" in result2.stderr
        assert "no changes" in result2.stderr
        # Snapshot version now only goes to stdout
        assert len(result2.stdout.strip()) == 64


def test_graph_push_three_scenarios_integration(metaxy_project: TempMetaxyProject):
    """Test complete workflow: new → metadata change → no change (GitHub issue #86)."""

    def features_v1():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

    def features_v2():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class Upstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["upstream"]),
                        columns=("value",),  # Metadata change
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

    # Scenario 1: First push (new snapshot)
    with metaxy_project.with_features(features_v1):
        result1 = metaxy_project.run_cli(["push"])
        assert result1.returncode == 0
        assert "Recorded feature graph" in result1.stderr
        # Snapshot version now only goes to stdout
        assert len(result1.stdout.strip()) == 64

    # Scenario 2: Metadata change (columns doesn't affect feature_version)
    with metaxy_project.with_features(features_v2):
        result2 = metaxy_project.run_cli(["push"])
        assert result2.returncode == 0
        assert "Updated feature information" in result2.stderr
        assert "downstream" in result2.stderr

    # Scenario 3: No change
    # Note: Re-pushing the same snapshot shows "already recorded"
    with metaxy_project.with_features(features_v2):
        result3 = metaxy_project.run_cli(["push"])
        assert result3.returncode == 0
        assert "already recorded" in result3.stderr


def test_graph_push_logs_store_metadata(metaxy_project: TempMetaxyProject):
    """Test that push logs store metadata (e.g., table name for DuckDB)."""

    def features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["push"])

        assert result.returncode == 0
        # Should log store metadata (DuckDB shows table_name)
        assert "Recorded at:" in result.stderr
        assert "table_name" in result.stderr
        assert "metaxy_system__feature_versions" in result.stderr


def test_graph_push_multiple_features_metadata_changes(
    metaxy_project: TempMetaxyProject,
):
    """Test CLI output when multiple features have metadata changes."""

    def features_v1():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class FeatureA(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_a"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_b"]),
                deps=[FeatureDep(feature=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        class FeatureC(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_c"]),
                deps=[FeatureDep(feature=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["output"]), code_version="1")],
            ),
        ):
            pass

    def features_v2():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import (
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        class FeatureA(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_a"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_b"]),
                deps=[FeatureDep(feature=FeatureKey(["feature_a"]), rename={"value": "renamed_b"})],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        class FeatureC(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_c"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["feature_a"]), columns=("value",))  # Changed
                ],
                fields=[FieldSpec(key=FieldKey(["output"]), code_version="1")],
            ),
        ):
            pass

    # Push v1
    with metaxy_project.with_features(features_v1):
        result1 = metaxy_project.run_cli(["push"])
        assert "Recorded feature graph" in result1.stderr

    # Push v2 - multiple metadata changes (rename and columns don't affect feature_version)
    with metaxy_project.with_features(features_v2):
        result2 = metaxy_project.run_cli(["push"])
        assert result2.returncode == 0
        assert "Updated feature information" in result2.stderr
        # Should list both changed features
        assert "feature_b" in result2.stderr
        assert "feature_c" in result2.stderr


def test_sync_flag_loads_external_features(metaxy_project: TempMetaxyProject):
    """Test that --sync flag loads external feature definitions from the metadata store."""

    def upstream_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["external", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

    def downstream_features():
        """Features that depend on external upstream but don't define it."""
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureDefinition, FeatureDep, FeatureKey, FieldKey, FieldSpec

        # Define external placeholder for upstream
        # Note: project doesn't need to match - sync uses feature keys
        external_upstream = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["external", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={},
            project="any-project",
        )

        from metaxy.models.feature import FeatureGraph

        FeatureGraph.get_active().add_feature_definition(external_upstream)

        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local", "downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["external", "upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

    # First, push the upstream feature to the store
    with metaxy_project.with_features(upstream_features):
        result = metaxy_project.run_cli(["push"])
        assert result.returncode == 0

    # Now run with downstream features that have an external placeholder
    with metaxy_project.with_features(downstream_features):
        # WITHOUT --sync: external feature remains unresolved (shows <external>)
        # Use --all-projects to see the external feature which has a different project
        # Disable auto-sync via METAXY_SYNC=false to test the explicit --sync flag
        result_no_sync = metaxy_project.run_cli(
            ["--all-projects", "list", "features"],
            env={"METAXY_SYNC": "false"},
        )
        assert result_no_sync.returncode == 0
        assert "<external>" in result_no_sync.stdout  # External feature not resolved

        # WITH --sync: external feature should be loaded from store (no <external>)
        result_sync = metaxy_project.run_cli(["--all-projects", "--sync", "list", "features"])
        assert result_sync.returncode == 0
        assert "<external>" not in result_sync.stdout  # External feature resolved
        assert "external/upstream" in result_sync.stdout  # Feature is present
        assert "local/downstream" in result_sync.stdout


def test_sync_flag_warns_on_version_mismatch(metaxy_project: TempMetaxyProject):
    """Test that --sync flag warns when external feature version mismatches."""

    def upstream_features_v1():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["mismatch", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

    def downstream_with_wrong_external():
        """Features with external placeholder that has wrong version."""
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureDefinition, FeatureDep, FeatureKey, FieldKey, FieldSpec

        # Define external placeholder with DIFFERENT code_version than what's stored
        # Note: project doesn't need to match - sync uses feature keys
        external_upstream = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["mismatch", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Wrong!
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="warn",
        )

        from metaxy.models.feature import FeatureGraph

        FeatureGraph.get_active().add_feature_definition(external_upstream)

        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local", "mismatch_downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["mismatch", "upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

    # First, push the upstream feature to the store
    with metaxy_project.with_features(upstream_features_v1):
        result = metaxy_project.run_cli(["push"])
        assert result.returncode == 0

    # Now run with mismatched external placeholder
    with metaxy_project.with_features(downstream_with_wrong_external):
        # With --sync, should warn about version mismatch
        # Disable auto-sync via METAXY_SYNC=false to ensure --sync flag triggers the sync
        result = metaxy_project.run_cli(
            ["--sync", "describe", "graph"],
            env={"METAXY_SYNC": "false"},
        )
        assert result.returncode == 0
        # Should still work but with warning in stderr
        assert "mismatch/upstream" in result.stderr or "Version mismatch" in result.stderr


def test_locked_flag_errors_on_version_mismatch(metaxy_project: TempMetaxyProject):
    """Test that --locked flag raises an error when external feature version mismatches."""

    def upstream_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["assert_mismatch", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

    def downstream_with_wrong_external():
        """Features with external placeholder that has wrong version."""
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureDefinition, FeatureDep, FeatureKey, FieldKey, FieldSpec

        # Define external placeholder with DIFFERENT code_version than what's stored
        # Note: on_version_mismatch="warn" on the feature, but --locked overrides
        # Note: project doesn't need to match - sync uses feature keys
        external_upstream = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["assert_mismatch", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Wrong!
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="warn",  # This gets overridden by --locked
        )

        from metaxy.models.feature import FeatureGraph

        FeatureGraph.get_active().add_feature_definition(external_upstream)

        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local", "assert_downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["assert_mismatch", "upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

    # First, push the upstream feature to the store
    with metaxy_project.with_features(upstream_features):
        result = metaxy_project.run_cli(["push"])
        assert result.returncode == 0

    # Now run with mismatched external placeholder and --locked
    with metaxy_project.with_features(downstream_with_wrong_external):
        # With --sync --locked, should fail due to version mismatch
        # Disable auto-sync via METAXY_SYNC=false to ensure --sync flag triggers the sync
        result = metaxy_project.run_cli(
            ["--sync", "--locked", "describe", "graph"],
            check=False,
            env={"METAXY_SYNC": "false"},
        )
        assert result.returncode != 0
        assert "Version mismatch" in result.stderr


def test_locked_flag_succeeds_when_versions_match(metaxy_project: TempMetaxyProject):
    """Test that --locked flag succeeds when versions match."""

    def upstream_features():
        from metaxy_testing.models import SampleFeature, SampleFeatureSpec

        from metaxy import FeatureKey, FieldKey, FieldSpec

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["assert_match", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

    # First, push the upstream feature to the store
    with metaxy_project.with_features(upstream_features):
        result = metaxy_project.run_cli(["push"])
        assert result.returncode == 0

    # Now run with --sync --locked when there's nothing to mismatch
    # (no external features with wrong versions)
    with metaxy_project.with_features(upstream_features):
        result = metaxy_project.run_cli(["--sync", "--locked", "describe", "graph"])
        assert result.returncode == 0
