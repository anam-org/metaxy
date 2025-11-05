"""LanceDB-specific tests."""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("lancedb")

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.lancedb import LanceDBMetadataStore


def test_lancedb_write_and_read(tmp_path, test_graph, test_features) -> None:
    """Write metadata and read it back."""
    database_path = tmp_path / "lancedb"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(database_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            }
        )

        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_lancedb_persistence_across_instances(
    tmp_path, test_graph, test_features
) -> None:
    """Ensure data persists across store instances."""
    database_path = tmp_path / "lancedb"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(database_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

    with LanceDBMetadataStore(database_path) as store:
        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 2


def test_lancedb_drop_feature(tmp_path, test_graph, test_features) -> None:
    """Dropping a feature removes the Lance table and allows re-creation."""
    database_path = tmp_path / "lancedb"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(database_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        feature_key = feature_cls.spec().key  # type: ignore[attr-defined]
        assert feature_key in store.list_features()

        store.drop_feature_metadata(feature_cls)
        assert feature_key not in store.list_features()

        fresh = pl.DataFrame(
            {
                "sample_uid": [2],
                "provenance_by_field": [
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, fresh)

        result = collect_to_polars(store.read_metadata(feature_cls))
        assert result["sample_uid"].to_list() == [2]


def test_lancedb_display(tmp_path) -> None:
    """Display includes path and feature count."""
    database_path = tmp_path / "lancedb"
    store = LanceDBMetadataStore(database_path)

    closed_display = store.display()
    assert "LanceDBMetadataStore" in closed_display
    assert str(database_path) in closed_display

    with store:
        open_display = store.display()
        assert "features=0" in open_display
