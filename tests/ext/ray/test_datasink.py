"""Tests for MetaxyDatasink Ray Data integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

import metaxy as mx
from metaxy.metadata_store.delta import DeltaMetadataStore

if TYPE_CHECKING:
    import ray
else:
    ray = pytest.importorskip("ray")


from metaxy_testing import RAY_FEATURES_MODULE

from .conftest import FEATURE_KEY, make_test_data


def test_datasink_writes_metadata(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that MetaxyDatasink writes metadata to the store."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    ds.write_datasink(datasink)

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()

        assert len(df) == 5
        assert set(df["sample_uid"].to_list()) == {"a", "b", "c", "d", "e"}
        assert set(df["value"].to_list()) == {1, 2, 3, 4, 5}


def test_datasink_with_multiple_blocks(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
):
    """Test that MetaxyDatasink handles multiple data blocks."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    # Create multiple small datasets and concatenate to get multiple blocks
    data1 = make_test_data(sample_uids=["a", "b"], values=[1, 2])
    data2 = make_test_data(sample_uids=["c", "d"], values=[3, 4])
    data3 = make_test_data(sample_uids=["e", "f"], values=[5, 6])

    ds = ray.data.from_arrow_refs(
        [
            ray.put(data1.to_arrow()),
            ray.put(data2.to_arrow()),
            ray.put(data3.to_arrow()),
        ]
    )

    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    ds.write_datasink(datasink)

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()

        assert len(df) == 6
        assert set(df["sample_uid"].to_list()) == {"a", "b", "c", "d", "e", "f"}
        assert set(df["value"].to_list()) == {1, 2, 3, 4, 5, 6}


@pytest.mark.parametrize(
    "feature_param",
    [
        pytest.param(FEATURE_KEY, id="feature_key_list"),
        pytest.param("test/ray_feature", id="feature_key_string"),
        pytest.param(mx.FeatureKey(["test", "ray_feature"]), id="feature_key_object"),
    ],
)
def test_datasink_feature_key_formats(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
    feature_param,
):
    """Test that MetaxyDatasink accepts different feature key formats."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    datasink = MetaxyDatasink(
        feature=feature_param,
        store=delta_store,
        config=ray_config,
    )

    ds.write_datasink(datasink)

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()
        assert len(df) == 5


def test_datasink_with_config_auto_discovery(
    ray_context,
    test_data: pl.DataFrame,
    tmp_path: Path,
):
    """Test that MetaxyDatasink auto-discovers config when not provided."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    # Create a fresh delta store for this test
    delta_root = tmp_path / "delta_auto"
    delta_store = DeltaMetadataStore(root_path=delta_root)

    # Write config to a TOML file
    config_path = tmp_path / "metaxy.toml"
    config_content = f'''project = "test"
store = "dev"
auto_create_tables = true
entrypoints = ["{RAY_FEATURES_MODULE}"]

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{delta_root}"
'''
    config_path.write_text(config_content)

    # Load config manually for the test process
    config = mx.MetaxyConfig.load(config_path)
    mx.init_metaxy(config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    # Create datasink with explicit config (simulating what would happen in a Ray worker)
    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=config,
    )

    ds.write_datasink(datasink)

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()
        assert len(df) == 5


def test_datasink_single_row(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
):
    """Test that MetaxyDatasink handles single row datasets correctly."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    # Create a single-row dataset
    single_row_data = make_test_data(sample_uids=["single"], values=[42])
    ds = ray.data.from_arrow(single_row_data.to_arrow())

    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    ds.write_datasink(datasink)

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()

        assert len(df) == 1
        assert df["sample_uid"].to_list() == ["single"]
        assert df["value"].to_list() == [42]


def test_datasink_feature_key_coercion(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
):
    """Test that the datasink correctly coerces feature keys during initialization."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    # Test with string path
    datasink_str = MetaxyDatasink(
        feature="test/ray_feature",
        store=delta_store,
        config=ray_config,
    )
    assert datasink_str._feature_key == mx.FeatureKey(["test", "ray_feature"])

    # Test with list
    datasink_list = MetaxyDatasink(
        feature=["test", "ray_feature"],
        store=delta_store,
        config=ray_config,
    )
    assert datasink_list._feature_key == mx.FeatureKey(["test", "ray_feature"])

    # Test with FeatureKey object
    datasink_key = MetaxyDatasink(
        feature=mx.FeatureKey(["test", "ray_feature"]),
        store=delta_store,
        config=ray_config,
    )
    assert datasink_key._feature_key == mx.FeatureKey(["test", "ray_feature"])


def test_datasink_stores_config(
    ray_config: mx.MetaxyConfig,
    delta_store: DeltaMetadataStore,
):
    """Test that datasink stores the provided configuration and store."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    assert datasink.store is delta_store
    assert datasink.config is ray_config


def test_datasink_result_tracks_written_rows(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that datasink.result tracks the number of written rows."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    ds.write_datasink(datasink)

    assert datasink.result.rows_written == 5
    assert datasink.result.rows_failed == 0


def test_datasink_result_aggregates_across_multiple_write_tasks(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
):
    """Test that datasink.result correctly aggregates stats from multiple parallel write tasks."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    # Create multiple blocks - each will be processed by a separate write task
    data1 = make_test_data(sample_uids=["a", "b"], values=[1, 2])
    data2 = make_test_data(sample_uids=["c", "d", "e"], values=[3, 4, 5])
    data3 = make_test_data(sample_uids=["f", "g", "h", "i"], values=[6, 7, 8, 9])

    ds = ray.data.from_arrow_refs(
        [
            ray.put(data1.to_arrow()),
            ray.put(data2.to_arrow()),
            ray.put(data3.to_arrow()),
        ]
    )

    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    # Allow parallel writes - Delta Lake handles concurrent writers
    ds.write_datasink(datasink, concurrency=3)

    # Total: 2 + 3 + 4 = 9 rows across 3 write tasks
    assert datasink.result.rows_written == 9
    assert datasink.result.rows_failed == 0

    # Verify the data was actually written
    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()
        assert len(df) == 9


def test_datasink_result_not_available_before_write(
    ray_config: mx.MetaxyConfig,
    delta_store: DeltaMetadataStore,
):
    """Test that accessing result before write completes raises an error."""
    from metaxy.ext.ray.datasink import MetaxyDatasink

    mx.init_metaxy(ray_config)

    datasink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    with pytest.raises(RuntimeError, match="Write operation has not completed yet"):
        _ = datasink.result
