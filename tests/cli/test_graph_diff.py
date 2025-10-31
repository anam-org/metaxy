"""Tests for graph-diff CLI commands."""

import json
import re

from metaxy._testing import TempMetaxyProject


def test_graph_diff_render_no_changes(metaxy_project: TempMetaxyProject):
    """Test graph-diff render shows no changes when snapshots are identical."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
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
        result = metaxy_project.run_cli("graph-diff", "render", "latest", "current")

        assert result.returncode == 0
        assert "merged view" in result.stdout
        assert "video/files" in result.stdout
        # All features should show as unchanged (no + - ~ symbols)
        assert "+ video/files" not in result.stdout
        assert "- video/files" not in result.stdout
        assert "~ video/files" not in result.stdout


def test_graph_diff_render_added_feature(metaxy_project: TempMetaxyProject):
    """Test graph-diff render detects added features."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
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

        # Compare the two snapshots
        result = metaxy_project.run_cli("graph-diff", "render", snapshot1, snapshot2)

        assert result.returncode == 0
        # Check for added feature (with Rich markup or plain text)
        assert "audio/files" in result.stdout
        assert "(added)" in result.stdout or "[green]+[/green]" in result.stdout


def test_graph_diff_render_removed_feature(metaxy_project: TempMetaxyProject):
    """Test graph-diff render detects removed features."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
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
        result = metaxy_project.run_cli("graph-diff", "render", snapshot1, snapshot2)

        assert result.returncode == 0
        # Check for removed feature
        assert "audio/files" in result.stdout
        assert "(removed)" in result.stdout or "[red]-[/red]" in result.stdout


def test_graph_diff_render_changed_feature(metaxy_project: TempMetaxyProject):
    """Test graph-diff render detects changed features."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
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
        result = metaxy_project.run_cli("graph-diff", "render", snapshot1, snapshot2)

        assert result.returncode == 0
        # Check for changed feature
        assert "video/files" in result.stdout
        assert "(changed)" in result.stdout or "[yellow]~[/yellow]" in result.stdout
        # Should show version transition
        assert "→" in result.stdout


def test_graph_diff_render_version_transitions(metaxy_project: TempMetaxyProject):
    """Test graph-diff render shows version transitions with proper colors."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["path"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["path"]), code_version=1),
                    FieldSpec(key=FieldKey(["size"]), code_version=1),  # Added field
                ],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare snapshots
        result = metaxy_project.run_cli("graph-diff", "render", snapshot1, snapshot2)

        assert result.returncode == 0
        # Changed feature should show status
        assert "video/files" in result.stdout
        assert "(changed)" in result.stdout or "[yellow]~[/yellow]" in result.stdout
        # Should show version transition
        assert "→" in result.stdout
        # Should show added field
        assert "size" in result.stdout


def test_graph_diff_render_format_json(metaxy_project: TempMetaxyProject):
    """Test graph-diff render with JSON format output."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with JSON format
        result = metaxy_project.run_cli(
            "graph-diff",
            "render",
            snapshot1,
            snapshot2,
            "--format",
            "json",
            check=False,
        )

        # Debug output if failed
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        assert result.returncode == 0
        # Parse JSON output
        data = json.loads(result.stdout)
        assert "nodes" in data
        assert "audio/files" in data["nodes"]
        assert data["nodes"]["audio/files"]["status"] == "added"


def test_graph_diff_render_format_yaml(metaxy_project: TempMetaxyProject):
    """Test graph-diff render with YAML format output."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with YAML format
        result = metaxy_project.run_cli(
            "graph-diff", "render", snapshot1, snapshot2, "--format", "yaml"
        )

        assert result.returncode == 0
        # Just check that output looks like YAML (don't parse it - PyYAML has issues with long strings)
        assert "nodes:" in result.stdout
        assert "video/files" in result.stdout
        assert "audio/files" in result.stdout
        assert "status:" in result.stdout


def test_graph_diff_render_format_mermaid(metaxy_project: TempMetaxyProject):
    """Test graph-diff render with Mermaid format output."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with Mermaid format
        result = metaxy_project.run_cli(
            "graph-diff", "render", snapshot1, snapshot2, "--format", "mermaid"
        )

        assert result.returncode == 0
        assert "flowchart TB" in result.stdout
        assert "video/files" in result.stdout
        # Mermaid should have color styling for changed features
        assert "fill:" in result.stdout or "stroke:" in result.stdout


def test_graph_diff_render_format_cards(metaxy_project: TempMetaxyProject):
    """Test graph-diff render with cards format output."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with cards format
        result = metaxy_project.run_cli(
            "graph-diff", "render", snapshot1, snapshot2, "--format", "cards"
        )

        assert result.returncode == 0
        assert "video/files" in result.stdout
        # Cards format should show feature boxes
        assert "Features:" in result.stdout or "Graph" in result.stdout


def test_graph_diff_render_format_graphviz(metaxy_project: TempMetaxyProject):
    """Test graph-diff render with Graphviz format output."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with Graphviz format
        result = metaxy_project.run_cli(
            "graph-diff", "render", snapshot1, snapshot2, "--format", "graphviz"
        )

        assert result.returncode == 0
        assert "digraph" in result.stdout
        assert "video/files" in result.stdout


def test_graph_diff_render_with_filtering(metaxy_project: TempMetaxyProject):
    """Test graph-diff render with feature filtering."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import (
            Feature,
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
            TestingFeatureSpec,
        )

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class VideoProcessing(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(key=FeatureKey(["video", "files"]))],
                fields=[FieldSpec(key=FieldKey(["frames"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with filtering (focus on video/processing and 1 level up)
        result = metaxy_project.run_cli(
            "graph-diff",
            "render",
            snapshot1,
            snapshot2,
            "--feature",
            "video/processing",
            "--up",
            "1",
        )

        assert result.returncode == 0
        # Should show video/processing and its dependency video/files
        assert "video/processing" in result.stdout
        assert "video/files" in result.stdout
        # Should NOT show audio/files (filtered out)
        assert "audio/files" not in result.stdout


def test_graph_diff_render_output_to_file(metaxy_project: TempMetaxyProject):
    """Test graph-diff render outputs to file."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Output to JSON file
        output_file = metaxy_project.project_dir / "diff.json"
        result = metaxy_project.run_cli(
            "graph-diff",
            "render",
            snapshot1,
            snapshot2,
            "--format",
            "json",
            "--output",
            str(output_file),
        )

        assert result.returncode == 0
        assert "rendered to" in result.stdout or "saved to" in result.stdout
        assert output_file.exists()

        # Check file contents
        with open(output_file) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "video/files" in data["nodes"]
        assert "audio/files" in data["nodes"]


def test_graph_diff_render_with_store_flag(metaxy_project: TempMetaxyProject):
    """Test graph-diff render works with --store flag."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
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
            "graph-diff", "render", "latest", "current", "--store", "dev"
        )

        assert result.returncode == 0
        assert "merged view" in result.stdout or "video/files" in result.stdout


def test_graph_diff_render_invalid_snapshot(metaxy_project: TempMetaxyProject):
    """Test graph-diff render fails gracefully with invalid snapshot."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(
            "graph-diff", "render", "nonexistent_snapshot", "current", check=False
        )

        assert result.returncode == 1
        assert "Error:" in result.stdout


def test_graph_diff_render_latest_empty_store(metaxy_project: TempMetaxyProject):
    """Test graph-diff render fails when no snapshots exist in store."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(
            "graph-diff", "render", "latest", "current", check=False
        )

        assert result.returncode == 1
        assert "Error:" in result.stdout
        assert "No snapshots found" in result.stdout


def test_graph_diff_render_verbose_mode(metaxy_project: TempMetaxyProject):
    """Test graph-diff render verbose mode shows more details."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with verbose mode
        result = metaxy_project.run_cli(
            "graph-diff", "render", snapshot1, snapshot2, "--verbose"
        )

        assert result.returncode == 0
        assert "video/files" in result.stdout


def test_graph_diff_render_minimal_mode(metaxy_project: TempMetaxyProject):
    """Test graph-diff render minimal mode hides version information."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Compare with minimal mode
        result = metaxy_project.run_cli(
            "graph-diff", "render", snapshot1, snapshot2, "--minimal"
        )

        assert result.returncode == 0
        assert "video/files" in result.stdout


def test_graph_diff_render_default_to_current(metaxy_project: TempMetaxyProject):
    """Test graph-diff render defaults to 'current' for to_snapshot."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Push snapshot
        metaxy_project.run_cli("graph", "push")

        # Render diff with only from_snapshot (should default to_snapshot to "current")
        result = metaxy_project.run_cli("graph-diff", "render", "latest")

        assert result.returncode == 0
        assert "video/files" in result.stdout


def test_graph_diff_render_deterministic_ordering(metaxy_project: TempMetaxyProject):
    """Test graph-diff render produces deterministic node ordering."""

    def features_v1():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class Zebra(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["zebra"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class Apple(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["apple"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    def features_v2():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class Zebra(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["zebra"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
        ):
            pass

        class Apple(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["apple"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Record first snapshot
    with metaxy_project.with_features(features_v1):
        push1_result = metaxy_project.run_cli("graph", "push")
        match1 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push1_result.stdout, re.DOTALL
        )
        assert match1
        snapshot1 = match1.group(1)

    # Record second snapshot
    with metaxy_project.with_features(features_v2):
        push2_result = metaxy_project.run_cli("graph", "push")
        match2 = re.search(
            r"Snapshot version:\s+([a-f0-9]+)", push2_result.stdout, re.DOTALL
        )
        assert match2
        snapshot2 = match2.group(1)

        # Render diff multiple times - should get identical output
        result1 = metaxy_project.run_cli("graph-diff", "render", snapshot1, snapshot2)
        result2 = metaxy_project.run_cli("graph-diff", "render", snapshot1, snapshot2)

        assert result1.returncode == 0
        assert result2.returncode == 0
        # Output should be identical (deterministic ordering)
        assert result1.stdout == result2.stdout
        # Apple should appear before Zebra (alphabetical)
        apple_pos = result1.stdout.find("apple")
        zebra_pos = result1.stdout.find("zebra")
        assert apple_pos < zebra_pos, "Features should be in alphabetical order"
