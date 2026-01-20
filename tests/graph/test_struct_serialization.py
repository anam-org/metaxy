"""Tests for GraphData and GraphDiff struct serialization."""

import pytest

from metaxy.graph.diff.diff_models import (
    AddedNode,
    FieldChange,
    GraphDiff,
    NodeChange,
    RemovedNode,
)
from metaxy.graph.diff.models import EdgeData, FieldNode, GraphData, GraphNode
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.utils.exceptions import MetaxyEmptyCodeVersionError


def test_graphdata_to_struct():
    """Test GraphData serialization to struct."""
    graph_data = GraphData(
        nodes={
            "feature/a": GraphNode(
                key=FeatureKey(["feature", "a"]),
                version="v1",
                code_version="1",
                fields=[FieldNode(key=FieldKey(["field1"]), version="f1", code_version="1")],
                dependencies=[FeatureKey(["feature", "b"])],
            )
        },
        edges=[
            EdgeData(
                from_key=FeatureKey(["feature", "b"]),
                to_key=FeatureKey(["feature", "a"]),
            )
        ],
    )

    struct = graph_data.to_struct()

    assert "nodes" in struct
    assert "edges" in struct
    assert len(struct["nodes"]) == 1
    assert len(struct["edges"]) == 1

    # Check node
    node = struct["nodes"][0]
    assert node["key"] == "feature/a"
    assert node["version"] == "v1"
    assert node["code_version"] == "1"
    assert len(node["fields"]) == 1
    assert node["fields"][0]["key"] == "field1"
    assert node["fields"][0]["version"] == "f1"
    assert node["dependencies"] == ["feature/b"]

    # Check edge
    edge = struct["edges"][0]
    assert edge["from_key"] == "feature/b"
    assert edge["to_key"] == "feature/a"


def test_graphdata_from_struct():
    """Test GraphData deserialization from struct."""
    struct = {
        "nodes": [
            {
                "key": "feature/a",
                "version": "v1",
                "code_version": "1",
                "fields": [{"key": "field1", "version": "f1", "code_version": "1"}],
                "dependencies": ["feature/b"],
            }
        ],
        "edges": [{"from_key": "feature/b", "to_key": "feature/a"}],
    }

    graph_data = GraphData.from_struct(struct)

    assert len(graph_data.nodes) == 1
    assert len(graph_data.edges) == 1

    node = graph_data.nodes["feature/a"]
    assert node.key == FeatureKey(["feature", "a"])
    assert node.version == "v1"
    assert node.code_version == "1"
    assert len(node.fields) == 1
    assert node.fields[0].key == FieldKey(["field1"])
    assert node.fields[0].version == "f1"
    assert node.dependencies == [FeatureKey(["feature", "b"])]

    edge = graph_data.edges[0]
    assert edge.from_key == FeatureKey(["feature", "b"])
    assert edge.to_key == FeatureKey(["feature", "a"])


def test_graphdata_roundtrip():
    """Test GraphData round-trip serialization."""
    original = GraphData(
        nodes={
            "feature/a": GraphNode(
                key=FeatureKey(["feature", "a"]),
                version="v1",
                code_version="1",
                fields=[
                    FieldNode(key=FieldKey(["field1"]), version="f1", code_version="1"),
                    FieldNode(key=FieldKey(["field2"]), version="f2", code_version="2"),
                ],
                dependencies=[FeatureKey(["feature", "b"])],
            ),
            "feature/b": GraphNode(
                key=FeatureKey(["feature", "b"]),
                version="v2",
                code_version="2",
                fields=[],
                dependencies=[],
            ),
        },
        edges=[
            EdgeData(
                from_key=FeatureKey(["feature", "b"]),
                to_key=FeatureKey(["feature", "a"]),
            )
        ],
    )

    struct = original.to_struct()
    restored = GraphData.from_struct(struct)

    assert len(restored.nodes) == len(original.nodes)
    assert len(restored.edges) == len(original.edges)

    # Verify all nodes restored correctly
    for key, orig_node in original.nodes.items():
        rest_node = restored.nodes[key]
        assert rest_node.key == orig_node.key
        assert rest_node.version == orig_node.version
        assert rest_node.code_version == orig_node.code_version
        assert len(rest_node.fields) == len(orig_node.fields)
        assert rest_node.dependencies == orig_node.dependencies


def test_graphdiff_to_struct():
    """Test GraphDiff serialization to struct."""
    graph_diff = GraphDiff(
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        added_nodes=[
            AddedNode(
                feature_key=FeatureKey(["feature", "a"]),
                version="v1",
                code_version="1",
                fields=[{"key": "field1", "version": "f1", "code_version": "1"}],
                dependencies=[],
            )
        ],
        removed_nodes=[
            RemovedNode(
                feature_key=FeatureKey(["feature", "b"]),
                version="v2",
                code_version="2",
                fields=[],
                dependencies=[],
            )
        ],
        changed_nodes=[
            NodeChange(
                feature_key=FeatureKey(["feature", "c"]),
                old_version="v1",
                new_version="v2",
                old_code_version="1",
                new_code_version="2",
                added_fields=[
                    FieldChange(
                        field_key=FieldKey(["field1"]),
                        new_version="f1",
                        new_code_version="1",
                    )
                ],
                removed_fields=[],
                changed_fields=[],
            )
        ],
    )

    struct = graph_diff.to_struct()

    assert "added_nodes" in struct
    assert "removed_nodes" in struct
    assert "changed_nodes" in struct

    # Check added node
    assert len(struct["added_nodes"]) == 1
    added = struct["added_nodes"][0]
    assert added["key"] == "feature/a"
    assert added["version"] == "v1"

    # Check removed node
    assert len(struct["removed_nodes"]) == 1
    removed = struct["removed_nodes"][0]
    assert removed["key"] == "feature/b"

    # Check changed node
    assert len(struct["changed_nodes"]) == 1
    changed = struct["changed_nodes"][0]
    assert changed["key"] == "feature/c"
    assert changed["old_version"] == "v1"
    assert changed["new_version"] == "v2"
    assert len(changed["added_fields"]) == 1


def test_graphdiff_from_struct():
    """Test GraphDiff deserialization from struct."""
    struct = {
        "added_nodes": [
            {
                "key": "feature/a",
                "version": "v1",
                "code_version": "1",
                "fields": [{"key": "field1", "version": "f1", "code_version": "1"}],
                "dependencies": [],
            }
        ],
        "removed_nodes": [
            {
                "key": "feature/b",
                "version": "v2",
                "code_version": "2",
                "fields": [],
                "dependencies": [],
            }
        ],
        "changed_nodes": [
            {
                "key": "feature/c",
                "old_version": "v1",
                "new_version": "v2",
                "old_code_version": "1",
                "new_code_version": "2",
                "added_fields": [
                    {
                        "key": "field1",
                        "version": "f1",
                        "code_version": "1",
                    }
                ],
                "removed_fields": [],
                "changed_fields": [],
            }
        ],
    }

    graph_diff = GraphDiff.from_struct(struct, "snap1", "snap2")

    assert graph_diff.from_snapshot_version == "snap1"
    assert graph_diff.to_snapshot_version == "snap2"
    assert len(graph_diff.added_nodes) == 1
    assert len(graph_diff.removed_nodes) == 1
    assert len(graph_diff.changed_nodes) == 1

    # Check added node
    added = graph_diff.added_nodes[0]
    assert added.feature_key == FeatureKey(["feature", "a"])
    assert added.version == "v1"

    # Check removed node
    removed = graph_diff.removed_nodes[0]
    assert removed.feature_key == FeatureKey(["feature", "b"])

    # Check changed node
    changed = graph_diff.changed_nodes[0]
    assert changed.feature_key == FeatureKey(["feature", "c"])
    assert changed.old_version == "v1"
    assert changed.new_version == "v2"
    assert len(changed.added_fields) == 1


def test_graphdiff_roundtrip():
    """Test GraphDiff round-trip serialization."""
    original = GraphDiff(
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        added_nodes=[
            AddedNode(
                feature_key=FeatureKey(["feature", "a"]),
                version="v1",
                code_version="1",
                fields=[
                    {"key": "field1", "version": "f1", "code_version": "1"},
                    {"key": "field2", "version": "f2", "code_version": "2"},
                ],
                dependencies=[FeatureKey(["feature", "x"])],
            )
        ],
        removed_nodes=[
            RemovedNode(
                feature_key=FeatureKey(["feature", "b"]),
                version="v2",
                code_version="2",
                fields=[],
                dependencies=[],
            )
        ],
        changed_nodes=[
            NodeChange(
                feature_key=FeatureKey(["feature", "c"]),
                old_version="v1",
                new_version="v2",
                old_code_version="1",
                new_code_version="2",
                added_fields=[
                    FieldChange(
                        field_key=FieldKey(["field1"]),
                        new_version="f1",
                        new_code_version="1",
                    )
                ],
                removed_fields=[
                    FieldChange(
                        field_key=FieldKey(["field2"]),
                        old_version="f2",
                        old_code_version="2",
                    )
                ],
                changed_fields=[
                    FieldChange(
                        field_key=FieldKey(["field3"]),
                        old_version="f3a",
                        new_version="f3b",
                        old_code_version="3",
                        new_code_version="4",
                    )
                ],
            )
        ],
    )

    struct = original.to_struct()
    restored = GraphDiff.from_struct(struct, "snap1", "snap2")

    assert restored.from_snapshot_version == original.from_snapshot_version
    assert restored.to_snapshot_version == original.to_snapshot_version
    assert len(restored.added_nodes) == len(original.added_nodes)
    assert len(restored.removed_nodes) == len(original.removed_nodes)
    assert len(restored.changed_nodes) == len(original.changed_nodes)

    # Verify changed node details
    orig_changed = original.changed_nodes[0]
    rest_changed = restored.changed_nodes[0]
    assert rest_changed.feature_key == orig_changed.feature_key
    assert rest_changed.old_version == orig_changed.old_version
    assert rest_changed.new_version == orig_changed.new_version
    assert len(rest_changed.added_fields) == len(orig_changed.added_fields)
    assert len(rest_changed.removed_fields) == len(orig_changed.removed_fields)
    assert len(rest_changed.changed_fields) == len(orig_changed.changed_fields)


def test_graphdata_handles_none_values():
    """Test GraphData handles None values correctly."""
    graph_data = GraphData(
        nodes={
            "feature/a": GraphNode(
                key=FeatureKey(["feature", "a"]),
                version=None,  # None version
                code_version=None,  # None code_version
                fields=[
                    FieldNode(
                        key=FieldKey(["field1"]),
                        version=None,
                        code_version=None,
                    )
                ],
                dependencies=[],
            )
        },
        edges=[],
    )

    with pytest.raises(
        MetaxyEmptyCodeVersionError,
        match="Field field1 in feature feature/a has empty code_version",
    ):
        graph_data.to_struct()
