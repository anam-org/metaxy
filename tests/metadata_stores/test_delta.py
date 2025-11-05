"""Delta Lake-specific tests that don't apply to other stores."""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

from deltalake import DeltaTable

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.delta import DeltaMetadataStore


def _delta_log_versions(path) -> set[str]:
    log_dir = path / "_delta_log"
    if not log_dir.exists():
        return set()
    return {file.name for file in log_dir.glob("*.json") if file.is_file()}


def test_delta_write_and_read(tmp_path, test_graph, test_features) -> None:
    """Write metadata and read it back from Delta store."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path) as store:
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

        # Delta log should contain version 0 after initial write
        feature_path = store._feature_path(feature_key)
        assert "00000000000000000000.json" in _delta_log_versions(feature_path)

        delta_table = DeltaTable(str(feature_path))
        assert delta_table.version() == 0
        assert delta_table.to_pyarrow_table().num_rows == 3


def test_delta_persistence_across_instances(
    tmp_path, test_graph, test_features
) -> None:
    """Data written in one instance is visible in another."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path) as store:
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

    with DeltaMetadataStore(store_path) as store:
        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 2


def test_delta_drop_feature(tmp_path, test_graph, test_features) -> None:
    """Dropping metadata removes the underlying Delta table and allows fresh writes."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path) as store:
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

        feature_path = store._feature_path(feature_key)
        assert (feature_path / "_delta_log").exists()

        store.drop_feature_metadata(feature_cls)
        assert not (feature_path / "_delta_log").exists()

    # Fresh instance should see no data
    with DeltaMetadataStore(store_path) as store:
        assert store.list_features() == []

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

        # Delta log recreated with new version 0
        feature_path = store._feature_path(feature_key)
        assert "00000000000000000000.json" in _delta_log_versions(feature_path)


def test_delta_lists_features(tmp_path, test_graph, test_features) -> None:
    """Verify feature discovery in Delta store."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path) as store:
        assert store.list_features() == []

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

        assert store.list_features() == [feature_key]


def test_delta_display(tmp_path) -> None:
    """Display output includes path and feature count when open."""
    store_path = tmp_path / "delta"
    store = DeltaMetadataStore(store_path, storage_options={"AWS_ACCESS_KEY_ID": "x"})

    closed_display = store.display()
    assert "DeltaMetadataStore" in closed_display
    assert str(store_path) in closed_display
    assert "storage_options=***" in closed_display

    with store:
        open_display = store.display()
        assert "features=0" in open_display
