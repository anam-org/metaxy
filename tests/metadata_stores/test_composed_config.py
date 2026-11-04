"""Tests for configuring a composed `MetadataStore` from Metaxy config."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from metaxy.config import InvalidConfigError, MetaxyConfig, StoreConfig
from metaxy.ext.duckdb.engine import DuckDBEngine
from metaxy.ext.ibis.engine import IbisStorageConfig
from metaxy.ext.polars.engine import PolarsComputeEngine
from metaxy.ext.polars.handlers.delta import DeltaStorageConfig
from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.utils import collect_to_polars

DELTA_STORAGE = "metaxy.ext.polars.handlers.delta.DeltaStorageConfig"
POLARS_ENGINE = "metaxy.ext.polars.engine.PolarsComputeEngine"


@pytest.fixture
def make_config() -> Callable[..., MetaxyConfig]:
    def make(**stores: dict[str, Any]) -> MetaxyConfig:
        return MetaxyConfig(
            stores={name: StoreConfig(type="metaxy.MetadataStore", config=config) for name, config in stores.items()}
        )

    return make


@pytest.fixture
def upstream(test_features: dict[str, Any]) -> Any:
    return test_features["UpstreamFeatureA"]


@pytest.mark.delta
@pytest.mark.polars
@pytest.mark.usefixtures("test_graph")
def test_store_is_built_from_engine_and_storages(
    make_config: Callable[..., MetaxyConfig], tmp_path: Path, upstream: Any
) -> None:
    prod, dev = str(tmp_path / "prod"), str(tmp_path / "dev")
    config = make_config(
        prod={"engine": {"type": POLARS_ENGINE}, "storage": {"type": DELTA_STORAGE, "location": prod}},
        dev={
            "engine": {"type": POLARS_ENGINE},
            "storage": {"type": DELTA_STORAGE, "location": dev, "layout": "flat"},
            "fallback_storage": [{"type": DELTA_STORAGE, "location": prod}],
        },
    )

    with config.get_store("prod").open("w") as prod_store:
        prod_store.write(
            upstream, pl.DataFrame({"sample_uid": [1], METAXY_PROVENANCE_BY_FIELD: [{"frames": "h", "audio": "h"}]})
        )

    dev_store = config.get_store("dev")
    assert isinstance(dev_store._engine, PolarsComputeEngine)
    assert dev_store._storage == DeltaStorageConfig(location=dev, layout="flat")
    assert dev_store._fallback_storage == (DeltaStorageConfig(location=prod),)
    with dev_store:
        assert collect_to_polars(dev_store.read(upstream))["sample_uid"].to_list() == [1]


@pytest.mark.duckdb
def test_storage_defaults_to_engine_native_storage(make_config: Callable[..., MetaxyConfig], tmp_path: Path) -> None:
    database = str(tmp_path / "metadata.duckdb")
    config = make_config(dev={"engine": {"type": "metaxy.ext.duckdb.engine.DuckDBEngine", "database": database}})

    store = config.get_store("dev")

    assert isinstance(store, MetadataStore)
    assert isinstance(store._engine, DuckDBEngine)
    assert store._storage == IbisStorageConfig(location=database)


@pytest.mark.parametrize(
    ("store_config", "message"),
    [
        pytest.param({"engine": {"type": DELTA_STORAGE, "location": "x"}}, "must build a ComputeEngine", id="engine"),
        pytest.param(
            {"engine": {"type": POLARS_ENGINE}, "storage": {"type": POLARS_ENGINE}},
            "must build a StorageConfig",
            id="storage",
        ),
    ],
)
def test_components_of_the_wrong_kind_are_rejected(
    make_config: Callable[..., MetaxyConfig], store_config: dict[str, Any], message: str
) -> None:
    with pytest.raises(InvalidConfigError, match=message):
        make_config(dev=store_config).get_store("dev")
