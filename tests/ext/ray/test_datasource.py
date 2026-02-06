"""Tests for MetaxyDatasource Ray Data integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import narwhals as nw
import polars as pl
import pytest

import metaxy as mx
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

if TYPE_CHECKING:
    import ray
else:
    ray = pytest.importorskip("ray")


from metaxy_testing import RAY_FEATURES_MODULE

from .conftest import DERIVED_FEATURE_KEY, FEATURE_KEY, make_test_data


def test_datasource_reads_metadata(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that MetaxyDatasource reads metadata from the store."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # First, write some data to the store
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Read using datasource
    datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    assert len(result) == 5
    assert set(result["sample_uid"].to_list()) == {"a", "b", "c", "d", "e"}
    assert set(result["value"].to_list()) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize(
    "feature_param",
    [
        pytest.param(FEATURE_KEY, id="feature_key_list"),
        pytest.param("test/ray_feature", id="feature_key_string"),
        pytest.param(mx.FeatureKey(["test", "ray_feature"]), id="feature_key_object"),
    ],
)
def test_datasource_feature_key_formats(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
    feature_param,
):
    """Test that MetaxyDatasource accepts different feature key formats."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Write data to the store
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Read using different feature key format
    datasource = MetaxyDatasource(
        feature=feature_param,
        store=delta_store,
        config=ray_config,
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    assert len(result) == 5


def test_datasource_single_row(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
):
    """Test that MetaxyDatasource handles single row datasets correctly."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    single_row_data = make_test_data(sample_uids=["single"], values=[42])

    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, single_row_data.to_arrow())

    datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    assert len(result) == 1
    assert result["sample_uid"].to_list() == ["single"]
    assert result["value"].to_list() == [42]


def test_datasource_feature_key_coercion(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
):
    """Test that the datasource correctly coerces feature keys during initialization."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Test with string path
    datasource_str = MetaxyDatasource(
        feature="test/ray_feature",
        store=delta_store,
        config=ray_config,
    )
    assert datasource_str._feature_key == mx.FeatureKey(["test", "ray_feature"])

    # Test with list
    datasource_list = MetaxyDatasource(
        feature=["test", "ray_feature"],
        store=delta_store,
        config=ray_config,
    )
    assert datasource_list._feature_key == mx.FeatureKey(["test", "ray_feature"])

    # Test with FeatureKey object
    datasource_key = MetaxyDatasource(
        feature=mx.FeatureKey(["test", "ray_feature"]),
        store=delta_store,
        config=ray_config,
    )
    assert datasource_key._feature_key == mx.FeatureKey(["test", "ray_feature"])


def test_datasource_stores_config(
    ray_config: mx.MetaxyConfig,
    delta_store: DeltaMetadataStore,
):
    """Test that datasource stores the provided configuration and store."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
    )

    assert datasource.store is delta_store
    assert datasource.config is ray_config


def test_datasource_with_filters(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that MetaxyDatasource applies filters correctly."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Write data to the store
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Read with filter: value > 2
    datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
        filters=[nw.col("value") > 2],
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    # Should only have rows where value > 2 (i.e., values 3, 4, 5)
    assert len(result) == 3
    assert set(result["value"].to_list()) == {3, 4, 5}
    assert set(result["sample_uid"].to_list()) == {"c", "d", "e"}


def test_datasource_with_columns(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that MetaxyDatasource selects columns correctly."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Write data to the store
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Read with column selection
    datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
        columns=["sample_uid", "value"],
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    assert len(result) == 5
    # Should have only the requested columns
    assert "sample_uid" in result.columns
    assert "value" in result.columns
    assert set(result.columns) == {"sample_uid", "value"}


def test_datasource_with_filters_and_columns(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that MetaxyDatasource applies both filters and column selection."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Write data to the store
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Read with filter and column selection
    datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=delta_store,
        config=ray_config,
        filters=[nw.col("value") >= 3],
        columns=["sample_uid", "value"],
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    # Should only have rows where value >= 3 (i.e., values 3, 4, 5)
    assert len(result) == 3
    assert set(result["value"].to_list()) == {3, 4, 5}
    assert "sample_uid" in result.columns
    assert "value" in result.columns


def test_datasource_with_config_auto_discovery(
    ray_context,
    test_data: pl.DataFrame,
    tmp_path: Path,
):
    """Test that MetaxyDatasource auto-discovers config when not provided."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

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
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{delta_root}"
'''
    config_path.write_text(config_content)

    # Load config manually for the test process
    config = mx.MetaxyConfig.load(config_path)
    mx.init(config)

    # Write data first
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Create datasource with explicit config
    datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=delta_store,
        config=config,
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    assert len(result) == 5


def test_datasource_and_datasink_end_to_end(
    ray_config: mx.MetaxyConfig,
    ray_context,
    tmp_path: Path,
):
    """End-to-end test: write with datasink, read with datasource, transform, write back."""
    from metaxy.ext.ray.datasink import MetaxyDatasink
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Create two separate stores for source and destination
    source_store = DeltaMetadataStore(root_path=tmp_path / "source")
    dest_store = DeltaMetadataStore(root_path=tmp_path / "dest")

    # Create initial test data
    initial_data = make_test_data(
        sample_uids=["a", "b", "c"],
        values=[10, 20, 30],
    )

    # Write initial data to source store using datasink
    initial_ds = ray.data.from_arrow(initial_data.to_arrow())

    source_sink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=source_store,
        config=ray_config,
    )
    initial_ds.write_datasink(source_sink)

    # Read from source using datasource
    source_datasource = MetaxyDatasource(
        feature=FEATURE_KEY,
        store=source_store,
        config=ray_config,
    )

    read_ds = ray.data.read_datasource(source_datasource)

    # Transform: double the values
    def transform_row(row):
        row["value"] = row["value"] * 2
        # Update provenance to reflect the transformation
        row["metaxy_provenance"] = f"transformed_{row['metaxy_provenance']}"
        row["metaxy_provenance_by_field"] = {"value": f"transformed_{row['metaxy_provenance_by_field']['value']}"}
        return row

    transformed_ds = read_ds.map(transform_row)

    # Write transformed data to destination store using datasink
    dest_sink = MetaxyDatasink(
        feature=FEATURE_KEY,
        store=dest_store,
        config=ray_config,
    )
    transformed_ds.write_datasink(dest_sink)

    # Verify the result by reading from destination
    with dest_store:
        result = dest_store.read(FEATURE_KEY)
        df = result.collect()

        assert len(df) == 3
        assert set(df["sample_uid"].to_list()) == {"a", "b", "c"}
        # Values should be doubled
        assert set(df["value"].to_list()) == {20, 40, 60}


def test_datasource_incremental_all_new(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test incremental mode returns all samples as 'new' when derived feature has no data."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Write upstream data (root feature)
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Read incrementally from derived feature - all samples should be "new"
    datasource = MetaxyDatasource(
        feature=DERIVED_FEATURE_KEY,
        store=delta_store,
        config=ray_config,
        incremental=True,
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    assert len(result) == 5
    assert set(result["sample_uid"].to_list()) == {"a", "b", "c", "d", "e"}
    assert set(result["metaxy_status"].to_list()) == {"new"}


def test_datasource_incremental_up_to_date(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test incremental mode returns empty when derived data is up-to-date."""
    from metaxy.ext.ray.datasource import MetaxyDatasource

    mx.init(ray_config)

    # Write upstream data (root feature)
    with delta_store.open("w"):
        delta_store.write(FEATURE_KEY, test_data.to_arrow())

    # Compute and write derived data with correct provenance
    with delta_store.open("w"):
        changes = delta_store.resolve_update(DERIVED_FEATURE_KEY)
        derived_data = changes.new.with_columns(nw.lit(100).alias("derived_value"))
        delta_store.write(DERIVED_FEATURE_KEY, derived_data.to_arrow())

    # Read incrementally - should return empty since data is up-to-date
    datasource = MetaxyDatasource(
        feature=DERIVED_FEATURE_KEY,
        store=delta_store,
        config=ray_config,
        incremental=True,
    )

    ds = ray.data.read_datasource(datasource)
    result = pl.from_pandas(ds.to_pandas())

    assert len(result) == 0
