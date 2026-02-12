"""Tests for field-level dependency Mermaid rendering."""

from metaxy.graph.diff.rendering.field_deps import (
    render_field_deps_mermaid,
    resolve_field_dependencies_from_snapshot,
)


def _make_snapshot() -> dict:
    """Build a graph snapshot with three features and field-level lineage.

    video/raw  (root, fields: audio, frames)
      ↓ (expansion 1:N on video_id, default field mapping)
    video/chunk (fields: audio, frames)
      ↓ (identity 1:1, specific mapping: faces→[frames])
    video/faces (fields: faces)
    """
    return {
        "video/raw": {
            "metaxy_feature_version": "aaaa1111",
            "fields": {"audio": "f1a1", "frames": "f1a2"},
            "feature_spec": {
                "key": "video/raw",
                "id_columns": ["video_id"],
                "deps": [],
                "fields": [
                    {"key": "audio", "code_version": "1", "deps": []},
                    {"key": "frames", "code_version": "1", "deps": []},
                ],
                "metadata": {},
                "description": None,
            },
        },
        "video/chunk": {
            "metaxy_feature_version": "bbbb2222",
            "fields": {"audio": "f2a1", "frames": "f2a2"},
            "feature_spec": {
                "key": "video/chunk",
                "id_columns": ["chunk_id"],
                "deps": [
                    {
                        "feature": "video/raw",
                        "select": None,
                        "rename": None,
                        "fields_mapping": {
                            "mapping": {
                                "type": "default",
                                "match_suffix": False,
                                "exclude_fields": [],
                            }
                        },
                        "sql_filters": None,
                        "lineage": {"relationship": {"type": "1:N", "on": ["video_id"], "id_generation_pattern": None}},
                        "optional": False,
                    }
                ],
                "fields": [
                    {"key": "audio", "code_version": "1", "deps": []},
                    {"key": "frames", "code_version": "1", "deps": []},
                ],
                "metadata": {},
                "description": None,
            },
        },
        "video/faces": {
            "metaxy_feature_version": "cccc3333",
            "fields": {"faces": "f3a1"},
            "feature_spec": {
                "key": "video/faces",
                "id_columns": ["chunk_id"],
                "deps": [
                    {
                        "feature": "video/chunk",
                        "select": None,
                        "rename": None,
                        "fields_mapping": {
                            "mapping": {
                                "type": "specific",
                                "mapping": {"faces": [["frames"]]},
                            }
                        },
                        "sql_filters": None,
                        "lineage": {"relationship": {"type": "1:1"}},
                        "optional": False,
                    }
                ],
                "fields": [
                    {"key": "faces", "code_version": "1", "deps": []},
                ],
                "metadata": {},
                "description": None,
            },
        },
    }


class TestResolveFieldDependencies:
    def test_root_feature_has_no_deps(self):
        snapshot = _make_snapshot()
        result = resolve_field_dependencies_from_snapshot(snapshot)
        assert result["video/raw"] == {}

    def test_default_mapping_resolves_by_name(self):
        snapshot = _make_snapshot()
        result = resolve_field_dependencies_from_snapshot(snapshot)

        chunk_deps = result["video/chunk"]
        assert "audio" in chunk_deps
        assert chunk_deps["audio"] == {"video/raw": ["audio"]}
        assert "frames" in chunk_deps
        assert chunk_deps["frames"] == {"video/raw": ["frames"]}

    def test_specific_mapping_resolves_correctly(self):
        snapshot = _make_snapshot()
        result = resolve_field_dependencies_from_snapshot(snapshot)

        faces_deps = result["video/faces"]
        assert "faces" in faces_deps
        assert faces_deps["faces"] == {"video/chunk": ["frames"]}


class TestRenderFieldDepsMermaid:
    def test_renders_all_features(self):
        snapshot = _make_snapshot()
        mermaid = render_field_deps_mermaid(snapshot, direction="LR")

        assert mermaid.startswith("flowchart LR")

        # Subgraphs present
        assert 'subgraph video_raw["video/raw"]' in mermaid
        assert 'subgraph video_chunk["video/chunk"]' in mermaid
        assert 'subgraph video_faces["video/faces"]' in mermaid

        # Field nodes inside subgraphs
        assert 'video_raw_audio["audio"]' in mermaid
        assert 'video_raw_frames["frames"]' in mermaid
        assert 'video_chunk_audio["audio"]' in mermaid
        assert 'video_chunk_frames["frames"]' in mermaid
        assert 'video_faces_faces["faces"]' in mermaid

    def test_field_to_field_edges(self):
        snapshot = _make_snapshot()
        mermaid = render_field_deps_mermaid(snapshot)

        # Default mapping: audio→audio, frames→frames
        assert "video_raw_audio --> video_chunk_audio" in mermaid
        assert "video_raw_frames --> video_chunk_frames" in mermaid

        # Specific mapping: chunk/frames → faces/faces
        assert "video_chunk_frames --> video_faces_faces" in mermaid

    def test_root_has_no_incoming_edges(self):
        snapshot = _make_snapshot()
        mermaid = render_field_deps_mermaid(snapshot)

        # No edges should target video_raw fields
        for line in mermaid.split("\n"):
            if "-->" in line:
                target = line.split("-->")[1].strip()
                assert not target.startswith("video_raw_"), f"Root feature has incoming edge: {line}"

    def test_style_lines_present(self):
        snapshot = _make_snapshot()
        mermaid = render_field_deps_mermaid(snapshot)

        assert "style video_raw stroke:" in mermaid
        assert "style video_chunk stroke:" in mermaid
        assert "style video_faces stroke:" in mermaid
        assert "stroke-width:2px" in mermaid

    def test_features_filter(self):
        snapshot = _make_snapshot()
        mermaid = render_field_deps_mermaid(snapshot, features=["video/raw", "video/chunk"])

        # Only filtered features should appear as subgraphs
        assert 'subgraph video_raw["video/raw"]' in mermaid
        assert 'subgraph video_chunk["video/chunk"]' in mermaid
        assert "video/faces" not in mermaid

        # Edges between raw→chunk should still exist
        assert "video_raw_audio --> video_chunk_audio" in mermaid

    def test_default_direction_is_tb(self):
        snapshot = _make_snapshot()
        mermaid = render_field_deps_mermaid(snapshot)
        assert mermaid.startswith("flowchart TB")

    def test_all_subgraphs_same_color(self):
        snapshot = _make_snapshot()
        mermaid = render_field_deps_mermaid(snapshot)
        style_lines = [line for line in mermaid.split("\n") if line.strip().startswith("style ")]
        assert len(style_lines) == 3
        colors = [line.split("stroke:")[1].split(",")[0] for line in style_lines]
        assert len(set(colors)) == 1
