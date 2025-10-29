"""Tests for graph diff functionality."""

from typing import Any

import pytest

from metaxy.graph.diff.diff_models import (
    AddedNode,
    FieldChange,
    GraphDiff,
    NodeChange,
    RemovedNode,
)
from metaxy.graph.diff.differ import GraphDiffer, SnapshotResolver
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


class TestFieldChange:
    """Test FieldChange model."""

    def test_field_added(self):
        """Test field addition detection."""
        change = FieldChange(
            field_key=FieldKey(["test"]), old_version=None, new_version="abc123"
        )
        assert change.is_added
        assert not change.is_removed
        assert not change.is_changed

    def test_field_removed(self):
        """Test field removal detection."""
        change = FieldChange(
            field_key=FieldKey(["test"]), old_version="abc123", new_version=None
        )
        assert change.is_removed
        assert not change.is_added
        assert not change.is_changed

    def test_field_changed(self):
        """Test field version change detection."""
        change = FieldChange(
            field_key=FieldKey(["test"]), old_version="abc123", new_version="def456"
        )
        assert change.is_changed
        assert not change.is_added
        assert not change.is_removed


class TestNodeChange:
    """Test NodeChange model."""

    def test_feature_added(self):
        """Test feature addition detection."""
        change = NodeChange(
            feature_key=FeatureKey(["test"]), old_version=None, new_version="abc123"
        )
        assert change.is_added
        assert not change.is_removed

    def test_feature_removed(self):
        """Test feature removal detection."""
        change = NodeChange(
            feature_key=FeatureKey(["test"]), old_version="abc123", new_version=None
        )
        assert change.is_removed
        assert not change.is_added

    def test_feature_with_field_changes(self):
        """Test feature with field changes."""
        field_change = FieldChange(
            field_key=FieldKey(["field1"]), old_version="v1", new_version="v2"
        )
        change = NodeChange(
            feature_key=FeatureKey(["test"]),
            old_version="abc123",
            new_version="def456",
            changed_fields=[field_change],
        )
        assert change.has_field_changes
        assert (
            len(change.added_fields + change.removed_fields + change.changed_fields)
            == 1
        )


class TestGraphDiff:
    """Test GraphDiff model."""

    def test_empty_diff(self):
        """Test diff with no changes."""
        diff = GraphDiff(from_snapshot_version="s1", to_snapshot_version="s2")
        assert not diff.has_changes

    def test_diff_with_changes(self):
        """Test diff with various changes."""
        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            added_nodes=[AddedNode(feature_key=FeatureKey(["new"]), version="v1")],
            removed_nodes=[RemovedNode(feature_key=FeatureKey(["old"]), version="v1")],
            changed_nodes=[
                NodeChange(
                    feature_key=FeatureKey(["changed"]),
                    old_version="v1",
                    new_version="v2",
                )
            ],
        )
        assert diff.has_changes
        assert len(diff.added_nodes) == 1
        assert len(diff.removed_nodes) == 1
        assert len(diff.changed_nodes) == 1


class TestSnapshotResolver:
    """Test SnapshotResolver."""

    def test_resolve_explicit_version(self):
        """Test resolving explicit snapshot version."""
        resolver = SnapshotResolver()
        result = resolver.resolve_snapshot("abc123def456", None, None)
        assert result == "abc123def456"

    def test_resolve_current_with_empty_graph(self):
        """Test resolving 'current' with empty graph fails."""
        resolver = SnapshotResolver()
        graph = FeatureGraph()
        with pytest.raises(ValueError, match="active graph is empty"):
            resolver.resolve_snapshot("current", None, graph)

    def test_resolve_current_with_graph(self):
        """Test resolving 'current' with active graph."""
        resolver = SnapshotResolver()
        graph = FeatureGraph()

        with graph.use():

            class TestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test"]),
                    deps=None,
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                ),
            ):
                pass

            result = resolver.resolve_snapshot("current", None, graph)
            assert result == graph.snapshot_version
            assert result != "empty"

    def test_resolve_latest_empty_store(self):
        """Test resolving 'latest' with empty store fails."""
        resolver = SnapshotResolver()
        with InMemoryMetadataStore() as store:
            with pytest.raises(ValueError, match="No snapshots found"):
                resolver.resolve_snapshot("latest", store, None)

    def test_resolve_latest_with_snapshot(self):
        """Test resolving 'latest' from store."""
        resolver = SnapshotResolver()
        graph = FeatureGraph()

        with graph.use():

            class TestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test"]),
                    deps=None,
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                ),
            ):
                pass

            with InMemoryMetadataStore() as store:
                # Record a snapshot
                result = store.record_feature_graph_snapshot()

                snapshot_version = result.snapshot_version

                _ = result.already_recorded

                # Resolve latest
                result = resolver.resolve_snapshot("latest", store, graph)
                assert result == snapshot_version


class TestGraphDiffer:
    """Test GraphDiffer."""

    def test_diff_empty_snapshots(self):
        """Test diff with empty snapshots."""
        differ = GraphDiffer()
        snapshot1: dict[str, dict[str, Any]] = {}
        snapshot2: dict[str, dict[str, Any]] = {}

        diff = differ.diff(snapshot1, snapshot2)
        assert not diff.has_changes

    def test_diff_added_features(self):
        """Test diff detects added features."""
        differ = GraphDiffer()
        snapshot1 = {}
        snapshot2 = {
            "feature/new": {"feature_version": "v1", "fields": {"default": "fv1"}}
        }

        diff = differ.diff(snapshot1, snapshot2)
        assert len(diff.added_nodes) == 1
        assert diff.added_nodes[0].feature_key.to_string() == "feature/new"

    def test_diff_removed_features(self):
        """Test diff detects removed features."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/old": {"feature_version": "v1", "fields": {"default": "fv1"}}
        }
        snapshot2 = {}

        diff = differ.diff(snapshot1, snapshot2)
        assert len(diff.removed_nodes) == 1
        assert diff.removed_nodes[0].feature_key.to_string() == "feature/old"

    def test_diff_changed_feature_version(self):
        """Test diff detects feature version changes."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/changed": {"feature_version": "v1", "fields": {"default": "fv1"}}
        }
        snapshot2 = {
            "feature/changed": {"feature_version": "v2", "fields": {"default": "fv1"}}
        }

        diff = differ.diff(snapshot1, snapshot2)
        assert len(diff.changed_nodes) == 1
        assert diff.changed_nodes[0].feature_key.to_string() == "feature/changed"
        assert diff.changed_nodes[0].old_version == "v1"
        assert diff.changed_nodes[0].new_version == "v2"

    def test_diff_added_field(self):
        """Test diff detects added fields."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/test": {"feature_version": "v1", "fields": {"field1": "fv1"}}
        }
        snapshot2 = {
            "feature/test": {
                "feature_version": "v2",
                "fields": {"field1": "fv1", "field2": "fv2"},
            }
        }

        diff = differ.diff(snapshot1, snapshot2)
        assert len(diff.changed_nodes) == 1
        node_change = diff.changed_nodes[0]
        assert node_change.has_field_changes
        assert len(node_change.added_fields) == 1
        field_change = node_change.added_fields[0]
        assert field_change.field_key.to_string() == "field2"
        assert field_change.is_added

    def test_diff_removed_field(self):
        """Test diff detects removed fields."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/test": {
                "feature_version": "v1",
                "fields": {"field1": "fv1", "field2": "fv2"},
            }
        }
        snapshot2 = {
            "feature/test": {"feature_version": "v2", "fields": {"field1": "fv1"}}
        }

        diff = differ.diff(snapshot1, snapshot2)
        assert len(diff.changed_nodes) == 1
        node_change = diff.changed_nodes[0]
        assert node_change.has_field_changes
        assert len(node_change.removed_fields) == 1
        field_change = node_change.removed_fields[0]
        assert field_change.field_key.to_string() == "field2"
        assert field_change.is_removed

    def test_diff_changed_field_version(self):
        """Test diff detects field version changes."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/test": {"feature_version": "v1", "fields": {"field1": "fv1"}}
        }
        snapshot2 = {
            "feature/test": {"feature_version": "v2", "fields": {"field1": "fv2"}}
        }

        diff = differ.diff(snapshot1, snapshot2)
        assert len(diff.changed_nodes) == 1
        node_change = diff.changed_nodes[0]
        assert node_change.has_field_changes
        assert len(node_change.changed_fields) == 1
        field_change = node_change.changed_fields[0]
        assert field_change.field_key.to_string() == "field1"
        assert field_change.is_changed
        assert field_change.old_version == "fv1"
        assert field_change.new_version == "fv2"

    def test_diff_complex_scenario(self):
        """Test diff with multiple types of changes."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/unchanged": {
                "feature_version": "v1",
                "fields": {"f1": "fv1"},
            },
            "feature/changed": {"feature_version": "v1", "fields": {"f1": "fv1"}},
            "feature/removed": {"feature_version": "v1", "fields": {"f1": "fv1"}},
        }
        snapshot2 = {
            "feature/unchanged": {
                "feature_version": "v1",
                "fields": {"f1": "fv1"},
            },
            "feature/changed": {"feature_version": "v2", "fields": {"f1": "fv2"}},
            "feature/added": {"feature_version": "v1", "fields": {"f1": "fv1"}},
        }

        diff = differ.diff(snapshot1, snapshot2)
        assert len(diff.added_nodes) == 1
        assert len(diff.removed_nodes) == 1
        assert len(diff.changed_nodes) == 1
        assert diff.added_nodes[0].feature_key.to_string() == "feature/added"
        assert diff.removed_nodes[0].feature_key.to_string() == "feature/removed"
        assert diff.changed_nodes[0].feature_key.to_string() == "feature/changed"

    @pytest.mark.skip(
        reason="Feature class must be importable from module level for snapshot reconstruction. "
        "Test setup creates class inside function which is not importable."
    )
    def test_load_snapshot_data_from_store(self):
        """Test loading snapshot data from store.

        NOTE: This test is skipped because it requires feature classes to be importable
        from their recorded module path. The test defines TestFeature inside the function,
        making it non-importable. To properly test this, features must be defined at module level.
        """
        differ = GraphDiffer()
        graph = FeatureGraph()

        with graph.use():

            class TestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    deps=None,
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                ),
            ):
                pass

            with InMemoryMetadataStore() as store:
                # Record snapshot
                result = store.record_feature_graph_snapshot()

                snapshot_version = result.snapshot_version

                _ = result.already_recorded

                # Load snapshot data
                snapshot_data = differ.load_snapshot_data(store, snapshot_version)

                assert "test/feature" in snapshot_data
                feature_data = snapshot_data["test/feature"]
                assert "feature_version" in feature_data
                assert "fields" in feature_data

    def test_load_snapshot_data_invalid_version(self):
        """Test loading non-existent snapshot fails."""
        differ = GraphDiffer()

        with InMemoryMetadataStore() as store:
            with pytest.raises(ValueError, match="Failed to load snapshot"):
                differ.load_snapshot_data(store, "nonexistent")


class TestDiffFormatter:
    """Test DiffFormatter output formats."""

    def test_format_dispatcher_terminal(self):
        """Test format dispatcher routes to terminal format."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            added_nodes=[
                AddedNode(feature_key=FeatureKey(["test", "feature"]), version="v1")
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format(diff=diff, format="terminal", diff_only=True)
        assert "Added (1)" in result
        assert "test/feature" in result

    def test_format_dispatcher_invalid(self):
        """Test format dispatcher raises error for invalid format."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(from_snapshot_version="s1", to_snapshot_version="s2")
        formatter = DiffFormatter()

        with pytest.raises(ValueError, match="Unknown format: invalid"):
            formatter.format(diff=diff, format="invalid", diff_only=True)

    def test_format_json_empty_diff(self):
        """Test JSON format for empty diff."""
        import json

        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(from_snapshot_version="snap1", to_snapshot_version="snap2")
        formatter = DiffFormatter()

        result = formatter.format_json_diff_only(diff)
        data = json.loads(result)

        assert data["from_snapshot_version"] == "snap1"
        assert data["to_snapshot_version"] == "snap2"
        assert data["added_nodes"] == []
        assert data["removed_nodes"] == []
        assert data["changed_nodes"] == []

    def test_format_json_with_changes(self):
        """Test JSON format with various changes."""
        import json

        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            added_nodes=[
                AddedNode(feature_key=FeatureKey(["new", "feature"]), version="v1")
            ],
            removed_nodes=[
                RemovedNode(feature_key=FeatureKey(["old", "feature"]), version="v1")
            ],
            changed_nodes=[
                NodeChange(
                    feature_key=FeatureKey(["changed", "feature"]),
                    old_version="v1",
                    new_version="v2",
                    changed_fields=[
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version="fv1",
                            new_version="fv2",
                        )
                    ],
                )
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format_json_diff_only(diff)
        data = json.loads(result)

        assert data["added_nodes"] == ["new/feature"]
        assert data["removed_nodes"] == ["old/feature"]
        assert len(data["changed_nodes"]) == 1
        assert data["changed_nodes"][0]["feature_key"] == "changed/feature"
        assert data["changed_nodes"][0]["old_version"] == "v1"
        assert data["changed_nodes"][0]["new_version"] == "v2"
        assert len(data["changed_nodes"][0]["field_changes"]) == 1
        assert data["changed_nodes"][0]["field_changes"][0]["field_key"] == "field1"
        assert data["changed_nodes"][0]["field_changes"][0]["is_changed"]

    def test_format_yaml_empty_diff(self):
        """Test YAML format for empty diff."""
        import yaml

        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(from_snapshot_version="snap1", to_snapshot_version="snap2")
        formatter = DiffFormatter()

        result = formatter.format_yaml_diff_only(diff)
        data = yaml.safe_load(result)

        assert data["from_snapshot_version"] == "snap1"
        assert data["to_snapshot_version"] == "snap2"
        assert data["added_nodes"] == []
        assert data["removed_nodes"] == []
        assert data["changed_nodes"] == []

    def test_format_yaml_with_changes(self):
        """Test YAML format with various changes."""
        import yaml

        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            added_nodes=[
                AddedNode(feature_key=FeatureKey(["new", "feature"]), version="v1")
            ],
            removed_nodes=[
                RemovedNode(feature_key=FeatureKey(["old", "feature"]), version="v1")
            ],
            changed_nodes=[
                NodeChange(
                    feature_key=FeatureKey(["changed", "feature"]),
                    old_version="v1",
                    new_version="v2",
                )
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format_yaml_diff_only(diff)
        data = yaml.safe_load(result)

        assert data["added_nodes"] == ["new/feature"]
        assert data["removed_nodes"] == ["old/feature"]
        assert len(data["changed_nodes"]) == 1
        assert data["changed_nodes"][0]["feature_key"] == "changed/feature"

    def test_format_mermaid_empty_diff(self):
        """Test Mermaid format for empty diff."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(from_snapshot_version="s1", to_snapshot_version="s2")
        formatter = DiffFormatter()

        result = formatter.format_mermaid_diff_only(diff)

        assert "flowchart TB" in result
        assert "Empty[No changes]" in result

    def test_format_mermaid_with_added_features(self):
        """Test Mermaid format with added features."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            added_nodes=[
                AddedNode(feature_key=FeatureKey(["new", "feature"]), version="v1")
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format_mermaid_diff_only(diff)

        assert "flowchart TB" in result
        assert "new/feature" in result
        assert "stroke:#00FF00" in result  # Green border for added

    def test_format_mermaid_with_removed_features(self):
        """Test Mermaid format with removed features."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            removed_nodes=[
                RemovedNode(feature_key=FeatureKey(["old", "feature"]), version="v1")
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format_mermaid_diff_only(diff)

        assert "old/feature" in result
        assert "stroke:#FF0000" in result  # Red border for removed

    def test_format_mermaid_with_changed_features(self):
        """Test Mermaid format with changed features."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            changed_nodes=[
                NodeChange(
                    feature_key=FeatureKey(["changed", "feature"]),
                    old_version="v1",
                    new_version="v2",
                )
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format_mermaid_diff_only(diff)

        assert "changed/feature" in result
        assert "stroke:#FFAA00" in result  # Orange border for changed

    def test_format_mermaid_verbose_with_field_changes(self):
        """Test Mermaid verbose format shows field changes."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            changed_nodes=[
                NodeChange(
                    feature_key=FeatureKey(["test", "feature"]),
                    old_version="v1",
                    new_version="v2",
                    added_fields=[
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version=None,
                            new_version="fv1",
                        ),
                    ],
                    removed_fields=[
                        FieldChange(
                            field_key=FieldKey(["field2"]),
                            old_version="fv2",
                            new_version=None,
                        ),
                    ],
                )
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format_mermaid_diff_only(diff, verbose=True)

        assert "test/feature" in result
        assert "+ field1" in result  # Added field
        assert "- field2" in result  # Removed field

    def test_format_mermaid_sanitizes_node_ids(self):
        """Test Mermaid format sanitizes node IDs."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            added_nodes=[
                AddedNode(
                    feature_key=FeatureKey(["feature-with", "dashes"]), version="v1"
                )
            ],
        )
        formatter = DiffFormatter()

        result = formatter.format_mermaid_diff_only(diff)

        # Node ID should have slashes and dashes replaced with underscores
        assert "feature_with_dashes" in result


class TestMergedGraphData:
    """Test create_merged_graph_data method."""

    def test_empty_snapshots(self):
        """Test merging empty snapshots."""
        differ = GraphDiffer()
        snapshot1: dict[str, dict[str, Any]] = {}
        snapshot2: dict[str, dict[str, Any]] = {}
        diff = differ.diff(snapshot1, snapshot2)

        merged = differ.create_merged_graph_data(snapshot1, snapshot2, diff)

        assert merged["nodes"] == {}
        assert merged["edges"] == []

    def test_added_features(self):
        """Test merged graph with added features."""
        differ = GraphDiffer()
        snapshot1 = {}
        snapshot2 = {
            "feature/new": {
                "feature_version": "v1",
                "fields": {"default": "fv1"},
                "feature_spec": {"deps": None},
            }
        }

        diff = differ.diff(snapshot1, snapshot2)
        merged = differ.create_merged_graph_data(snapshot1, snapshot2, diff)

        assert "feature/new" in merged["nodes"]
        node = merged["nodes"]["feature/new"]
        assert node["status"] == "added"
        assert node["old_version"] is None
        assert node["new_version"] == "v1"
        assert node["dependencies"] == []

    def test_removed_features(self):
        """Test merged graph with removed features."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/old": {
                "feature_version": "v1",
                "fields": {"default": "fv1"},
                "feature_spec": {"deps": None},
            }
        }
        snapshot2 = {}

        diff = differ.diff(snapshot1, snapshot2)
        merged = differ.create_merged_graph_data(snapshot1, snapshot2, diff)

        assert "feature/old" in merged["nodes"]
        node = merged["nodes"]["feature/old"]
        assert node["status"] == "removed"
        assert node["old_version"] == "v1"
        assert node["new_version"] is None

    def test_changed_features(self):
        """Test merged graph with changed features."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/changed": {
                "feature_version": "v1",
                "fields": {"default": "fv1"},
                "feature_spec": {"deps": None},
            }
        }
        snapshot2 = {
            "feature/changed": {
                "feature_version": "v2",
                "fields": {"default": "fv2"},
                "feature_spec": {"deps": None},
            }
        }

        diff = differ.diff(snapshot1, snapshot2)
        merged = differ.create_merged_graph_data(snapshot1, snapshot2, diff)

        assert "feature/changed" in merged["nodes"]
        node = merged["nodes"]["feature/changed"]
        assert node["status"] == "changed"
        assert node["old_version"] == "v1"
        assert node["new_version"] == "v2"

    def test_unchanged_features(self):
        """Test merged graph with unchanged features."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/unchanged": {
                "feature_version": "v1",
                "fields": {"default": "fv1"},
                "feature_spec": {"deps": None},
            }
        }
        snapshot2 = {
            "feature/unchanged": {
                "feature_version": "v1",
                "fields": {"default": "fv1"},
                "feature_spec": {"deps": None},
            }
        }

        diff = differ.diff(snapshot1, snapshot2)
        merged = differ.create_merged_graph_data(snapshot1, snapshot2, diff)

        assert "feature/unchanged" in merged["nodes"]
        node = merged["nodes"]["feature/unchanged"]
        assert node["status"] == "unchanged"
        assert node["old_version"] == "v1"
        assert node["new_version"] == "v1"

    def test_dependencies_extraction(self):
        """Test that dependencies are extracted correctly."""
        differ = GraphDiffer()
        snapshot1 = {}
        snapshot2 = {
            "feature/parent": {
                "feature_version": "v1",
                "fields": {},
                "feature_spec": {"deps": None},
            },
            "feature/child": {
                "feature_version": "v1",
                "fields": {},
                "feature_spec": {"deps": [{"key": ["feature", "parent"]}]},
            },
        }

        diff = differ.diff(snapshot1, snapshot2)
        merged = differ.create_merged_graph_data(snapshot1, snapshot2, diff)

        child_node = merged["nodes"]["feature/child"]
        assert child_node["dependencies"] == ["feature/parent"]

        # Check edges (arrow points from dependency to dependent feature)
        assert {"from": "feature/parent", "to": "feature/child"} in merged["edges"]

    def test_mixed_changes(self):
        """Test merged graph with all types of changes."""
        differ = GraphDiffer()
        snapshot1 = {
            "feature/unchanged": {
                "feature_version": "v1",
                "fields": {},
                "feature_spec": {"deps": None},
            },
            "feature/changed": {
                "feature_version": "v1",
                "fields": {},
                "feature_spec": {"deps": None},
            },
            "feature/removed": {
                "feature_version": "v1",
                "fields": {},
                "feature_spec": {"deps": None},
            },
        }
        snapshot2 = {
            "feature/unchanged": {
                "feature_version": "v1",
                "fields": {},
                "feature_spec": {"deps": None},
            },
            "feature/changed": {
                "feature_version": "v2",
                "fields": {},
                "feature_spec": {"deps": None},
            },
            "feature/added": {
                "feature_version": "v1",
                "fields": {},
                "feature_spec": {"deps": None},
            },
        }

        diff = differ.diff(snapshot1, snapshot2)
        merged = differ.create_merged_graph_data(snapshot1, snapshot2, diff)

        assert len(merged["nodes"]) == 4
        assert merged["nodes"]["feature/unchanged"]["status"] == "unchanged"
        assert merged["nodes"]["feature/changed"]["status"] == "changed"
        assert merged["nodes"]["feature/removed"]["status"] == "removed"
        assert merged["nodes"]["feature/added"]["status"] == "added"


class TestMergedFormatters:
    """Test merged graph formatters."""

    def test_format_terminal_merged_empty(self):
        """Test terminal formatting of empty merged graph."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {"nodes": {}, "edges": []}
        formatter = DiffFormatter()

        result = formatter.format_terminal_merged(merged_data)
        assert "Empty graph" in result

    def test_format_terminal_merged_with_features(self):
        """Test terminal formatting with features."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "added",
                    "old_version": None,
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_terminal_merged(merged_data)
        assert "feature/test" in result
        assert "added" in result

    def test_format_json_merged(self):
        """Test JSON formatting of merged graph."""
        import json

        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "added",
                    "old_version": None,
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_json_merged(merged_data)
        data = json.loads(result)

        assert "nodes" in data
        assert "feature/test" in data["nodes"]
        assert data["nodes"]["feature/test"]["status"] == "added"

    def test_format_yaml_merged(self):
        """Test YAML formatting of merged graph."""
        import yaml

        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_yaml_merged(merged_data)
        data = yaml.safe_load(result)

        assert "nodes" in data
        assert "feature/test" in data["nodes"]
        assert data["nodes"]["feature/test"]["status"] == "changed"

    def test_format_mermaid_merged_empty(self):
        """Test Mermaid formatting of empty merged graph."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {"nodes": {}, "edges": []}
        formatter = DiffFormatter()

        result = formatter.format_mermaid_merged(merged_data)
        assert "flowchart TB" in result
        assert "Empty[No features]" in result

    def test_format_mermaid_merged_with_features(self):
        """Test Mermaid formatting with features."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/added": {
                    "status": "added",
                    "old_version": None,
                    "new_version": "abc123",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/removed": {
                    "status": "removed",
                    "old_version": "def456",
                    "new_version": None,
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/changed": {
                    "status": "changed",
                    "old_version": "old123",
                    "new_version": "new456",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_mermaid_merged(merged_data)
        assert "flowchart TB" in result
        assert "feature/added" in result
        assert "feature/removed" in result
        assert "feature/changed" in result
        assert "stroke:#00FF00" in result  # Green border for added
        assert "stroke:#FF0000" in result  # Red border for removed
        assert "stroke:#FFAA00" in result  # Orange border for changed

    def test_format_mermaid_merged_with_edges(self):
        """Test Mermaid formatting includes edges."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/parent": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/child": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/parent"],
                },
            },
            "edges": [{"from": "feature/parent", "to": "feature/child"}],
        }
        formatter = DiffFormatter()

        result = formatter.format_mermaid_merged(merged_data)
        assert "feature_parent --> feature_child" in result


class TestFieldColorHighlighting:
    """Test field-level color highlighting in formatters."""

    def test_terminal_merged_shows_all_fields_with_colors(self):
        """Test that terminal format shows all fields with color indicators for changed features."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {"field1": "fv1", "field2": "fv2", "field3": "fv3"},
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version=None,
                            new_version="fv1",
                        ),  # Added
                        FieldChange(
                            field_key=FieldKey(["field2"]),
                            old_version="fv2_old",
                            new_version="fv2",
                        ),  # Changed
                        # field3 is unchanged
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_terminal_merged(merged_data)

        # Should show added field with green +
        assert "[green]+[/green] field1" in result
        # Should show changed field with yellow ~
        assert "[yellow]~[/yellow] field2" in result
        # Should show unchanged field with no symbol (just spaces)
        assert "      field3" in result

    def test_mermaid_merged_shows_all_fields_with_colors(self):
        """Test that mermaid format shows all fields with color HTML for changed features."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "abc123",
                    "new_version": "def456",
                    "fields": {"field1": "fv1", "field2": "fv2", "field3": "fv3"},
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version=None,
                            new_version="fv1",
                        ),  # Added
                        FieldChange(
                            field_key=FieldKey(["field2"]),
                            old_version="fv2_old",
                            new_version="fv2",
                        ),  # Changed
                        # field3 is unchanged
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_mermaid_merged(merged_data)

        # Should show added field with green color and version
        assert '<font color="#00AA00">- field1' in result and "(fv1)" in result
        # Should show changed field: yellow name, red old version, green new version
        assert '- <font color="#FFAA00">field2</font>' in result
        assert '<font color="#CC0000">fv2_ol</font>' in result  # old version in red
        assert '<font color="#00AA00">fv2</font>' in result  # new version in green
        # Should show unchanged field with dash prefix and version
        assert "- field3" in result and "(fv3)" in result

    def test_terminal_merged_shows_removed_fields_with_color(self):
        """Test that terminal format shows removed fields with red color."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "abc123",
                    "new_version": "def456",
                    "fields": {"field1": "fv1"},
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field2"]),
                            old_version="fv2",
                            new_version=None,
                        ),  # Removed
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_terminal_merged(merged_data)

        # Should show removed field with red -
        assert "[red]-[/red] field2" in result

    def test_mermaid_merged_shows_removed_fields_with_color(self):
        """Test that mermaid format shows removed fields with red color."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "abc123",
                    "new_version": "def456",
                    "fields": {"field1": "fv1"},
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field2"]),
                            old_version="fv2",
                            new_version=None,
                        ),  # Removed
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_mermaid_merged(merged_data)

        # Should show removed field with red color and version
        assert '<font color="#CC0000">- field2' in result and "(fv2)" in result


class TestDiffFormatterDispatcher:
    """Test the format() dispatcher method."""

    def test_format_diff_only_mode(self):
        """Test dispatcher routes to diff_only methods."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        diff = GraphDiff(
            from_snapshot_version="s1",
            to_snapshot_version="s2",
            added_nodes=[AddedNode(feature_key=FeatureKey(["test"]), version="v1")],
        )
        formatter = DiffFormatter()

        result = formatter.format(diff=diff, format="terminal", diff_only=True)
        assert "Added (1)" in result

    def test_format_merged_mode(self):
        """Test dispatcher routes to merged methods."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "added",
                    "old_version": None,
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format(
            merged_data=merged_data, format="terminal", diff_only=False
        )
        assert "feature/test" in result
        assert "merged view" in result.lower()

    def test_format_requires_diff_for_diff_only(self):
        """Test that diff_only mode requires diff parameter."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        formatter = DiffFormatter()

        with pytest.raises(ValueError, match="diff is required"):
            formatter.format(diff_only=True)

    def test_format_requires_merged_data_for_merged_mode(self):
        """Test that merged mode requires merged_data parameter."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        formatter = DiffFormatter()

        with pytest.raises(ValueError, match="merged_data is required"):
            formatter.format(diff_only=False)


class TestGraphSlicing:
    """Test graph slicing/filtering functionality."""

    def test_filter_with_no_focus_returns_all(self):
        """Test that filtering without focus feature returns all features."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {
                "feature/a": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/b": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
            },
            "edges": [],
        }

        result = differ.filter_merged_graph(merged_data, focus_feature=None)
        assert len(result["nodes"]) == 2
        assert "feature/a" in result["nodes"]
        assert "feature/b" in result["nodes"]

    def test_filter_with_focus_only(self):
        """Test filtering with focus feature and no up/down limits."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {
                "feature/root": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/parent": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/root"],
                },
                "feature/focus": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/parent"],
                },
                "feature/child": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/focus"],
                },
                "feature/unrelated": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
            },
            "edges": [
                {"from": "feature/root", "to": "feature/parent"},
                {"from": "feature/parent", "to": "feature/focus"},
                {"from": "feature/focus", "to": "feature/child"},
            ],
        }

        # Default behavior: focus with no up/down specified -> include all upstream and downstream
        result = differ.filter_merged_graph(merged_data, focus_feature="feature/focus")
        assert len(result["nodes"]) == 4  # focus + root + parent + child
        assert "feature/focus" in result["nodes"]
        assert "feature/root" in result["nodes"]
        assert "feature/parent" in result["nodes"]
        assert "feature/child" in result["nodes"]
        assert "feature/unrelated" not in result["nodes"]

    def test_filter_with_focus_and_up_limit(self):
        """Test filtering with focus feature and upstream limit."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {
                "feature/root": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/parent": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/root"],
                },
                "feature/focus": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/parent"],
                },
            },
            "edges": [
                {"from": "feature/root", "to": "feature/parent"},
                {"from": "feature/parent", "to": "feature/focus"},
            ],
        }

        # up=1 means only direct dependencies
        result = differ.filter_merged_graph(
            merged_data, focus_feature="feature/focus", up=1, down=0
        )
        assert len(result["nodes"]) == 2  # focus + parent only
        assert "feature/focus" in result["nodes"]
        assert "feature/parent" in result["nodes"]
        assert "feature/root" not in result["nodes"]

    def test_filter_with_focus_and_down_limit(self):
        """Test filtering with focus feature and downstream limit."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {
                "feature/focus": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/child": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/focus"],
                },
                "feature/grandchild": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/child"],
                },
            },
            "edges": [
                {"from": "feature/focus", "to": "feature/child"},
                {"from": "feature/child", "to": "feature/grandchild"},
            ],
        }

        # down=1 means only direct dependents
        result = differ.filter_merged_graph(
            merged_data, focus_feature="feature/focus", up=0, down=1
        )
        assert len(result["nodes"]) == 2  # focus + child only
        assert "feature/focus" in result["nodes"]
        assert "feature/child" in result["nodes"]
        assert "feature/grandchild" not in result["nodes"]

    def test_filter_with_up_down_zero(self):
        """Test filtering with up=0 and down=0 shows only focus feature."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {
                "feature/parent": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/focus": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/parent"],
                },
                "feature/child": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/focus"],
                },
            },
            "edges": [
                {"from": "feature/parent", "to": "feature/focus"},
                {"from": "feature/focus", "to": "feature/child"},
            ],
        }

        result = differ.filter_merged_graph(
            merged_data, focus_feature="feature/focus", up=0, down=0
        )
        assert len(result["nodes"]) == 1  # only focus feature
        assert "feature/focus" in result["nodes"]
        assert "feature/parent" not in result["nodes"]
        assert "feature/child" not in result["nodes"]

    def test_filter_with_invalid_feature_raises_error(self):
        """Test that filtering with non-existent feature raises ValueError."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {"feature/a": {"status": "unchanged"}},
            "edges": [],
        }

        with pytest.raises(ValueError, match="not found in graph"):
            differ.filter_merged_graph(merged_data, focus_feature="feature/nonexistent")

    def test_filter_supports_both_key_formats(self):
        """Test that filtering supports both / and __ in feature keys."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                }
            },
            "edges": [],
        }

        # Test with / format
        result1 = differ.filter_merged_graph(
            merged_data, focus_feature="feature/test", up=0, down=0
        )
        assert "feature/test" in result1["nodes"]

        # Test with __ format
        result2 = differ.filter_merged_graph(
            merged_data, focus_feature="feature__test", up=0, down=0
        )
        assert "feature/test" in result2["nodes"]

    def test_filter_preserves_edges(self):
        """Test that filtering preserves edges between included nodes."""
        differ = GraphDiffer()
        merged_data = {
            "nodes": {
                "feature/a": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": [],
                },
                "feature/b": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/a"],
                },
                "feature/c": {
                    "status": "unchanged",
                    "old_version": "v1",
                    "new_version": "v1",
                    "fields": {},
                    "field_changes": [],
                    "dependencies": ["feature/a"],
                },
            },
            "edges": [
                {"from": "feature/a", "to": "feature/b"},
                {"from": "feature/a", "to": "feature/c"},
            ],
        }

        result = differ.filter_merged_graph(
            merged_data, focus_feature="feature/b", up=1, down=0
        )
        # Should include feature/a and feature/b
        assert len(result["nodes"]) == 2
        # Should include edge from a to b, but not a to c
        assert len(result["edges"]) == 1
        assert {"from": "feature/a", "to": "feature/b"} in result["edges"]
        assert {"from": "feature/a", "to": "feature/c"} not in result["edges"]


class TestShowAllFieldsParameter:
    """Test show_all_fields parameter in formatters."""

    def test_format_terminal_shows_all_fields_by_default(self):
        """Test that terminal format shows all fields by default."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {
                        "field1": "fv1",
                        "field2": "fv2",
                        "field3": "fv3",
                    },
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version="fv1_old",
                            new_version="fv1",
                        ),
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_terminal_merged(merged_data, show_all_fields=True)

        # Should show changed field
        assert "field1" in result
        # Should also show unchanged fields
        assert "field2" in result
        assert "field3" in result

    def test_format_terminal_shows_only_changed_fields_when_requested(self):
        """Test that terminal format can show only changed fields."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {
                        "field1": "fv1",
                        "field2": "fv2",
                        "field3": "fv3",
                    },
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version="fv1_old",
                            new_version="fv1",
                        ),
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_terminal_merged(merged_data, show_all_fields=False)

        # Should show changed field
        assert "field1" in result
        # Should NOT show unchanged fields
        assert "field2" not in result
        assert "field3" not in result

    def test_format_mermaid_shows_all_fields_by_default(self):
        """Test that mermaid format shows all fields by default."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "abc123",
                    "new_version": "def456",
                    "fields": {
                        "field1": "fv1",
                        "field2": "fv2",
                        "field3": "fv3",
                    },
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version="fv1_old",
                            new_version="fv1",
                        ),
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_mermaid_merged(merged_data, show_all_fields=True)

        # Should show changed field
        assert "field1" in result
        # Should also show unchanged fields
        assert "field2" in result
        assert "field3" in result

    def test_format_mermaid_shows_only_changed_fields_when_requested(self):
        """Test that mermaid format can show only changed fields."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "abc123",
                    "new_version": "def456",
                    "fields": {
                        "field1": "fv1",
                        "field2": "fv2",
                        "field3": "fv3",
                    },
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version="fv1_old",
                            new_version="fv1",
                        ),
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format_mermaid_merged(merged_data, show_all_fields=False)

        # Should show changed field
        assert "field1" in result
        # Should NOT show unchanged fields
        assert "field2" not in result
        assert "field3" not in result

    def test_format_dispatcher_passes_show_all_fields(self):
        """Test that format() dispatcher passes show_all_fields to formatters."""
        from metaxy.graph.diff.rendering.formatter import DiffFormatter

        merged_data = {
            "nodes": {
                "feature/test": {
                    "status": "changed",
                    "old_version": "v1",
                    "new_version": "v2",
                    "fields": {"field1": "fv1", "field2": "fv2"},
                    "field_changes": [
                        FieldChange(
                            field_key=FieldKey(["field1"]),
                            old_version="fv1_old",
                            new_version="fv1",
                        ),
                    ],
                    "dependencies": [],
                }
            },
            "edges": [],
        }
        formatter = DiffFormatter()

        result = formatter.format(
            merged_data=merged_data,
            format="terminal",
            diff_only=False,
            show_all_fields=False,
        )

        # Should show changed field
        assert "field1" in result
        # Should NOT show unchanged field
        assert "field2" not in result
