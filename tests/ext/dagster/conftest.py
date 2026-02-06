"""Common fixtures for Dagster integration tests."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dagster as dg
import pytest
from pytest_cases import fixture, parametrize_with_cases

import metaxy as mx
import metaxy.ext.dagster as mxd


class DagsterStoreConfigCases:
    """Store configuration cases for Dagster tests."""

    @pytest.mark.delta
    def case_delta(self, tmp_path: Path) -> mx.StoreConfig:
        """DeltaMetadataStore configuration."""
        return mx.StoreConfig(
            type="metaxy.ext.metadata_stores.delta.DeltaMetadataStore",
            config={"root_path": tmp_path / "delta_store"},
        )

    @pytest.mark.clickhouse
    def case_clickhouse(self, request) -> mx.StoreConfig:
        """ClickHouseMetadataStore configuration."""
        return mx.StoreConfig(
            type="metaxy.ext.metadata_stores.clickhouse.ClickHouseMetadataStore",
            config={"connection_string": request.getfixturevalue("clickhouse_db")},
        )


@fixture
@parametrize_with_cases("store_config", cases=DagsterStoreConfigCases)
def metaxy_config(store_config: mx.StoreConfig) -> Iterator[mx.MetaxyConfig]:
    """Parametrized metaxy config fixture.

    This fixture is parametrized to run tests with both:
    - DeltaMetadataStore (file-based)
    - ClickHouseMetadataStore (SQL-based, requires clickhouse binary)
    """
    with mx.MetaxyConfig(project="test", stores={"dev": store_config}).use() as config:
        yield config


@pytest.fixture
def resources(metaxy_config: mx.MetaxyConfig) -> dict[str, Any]:
    """Dagster resources using the parametrized store."""
    store = mxd.MetaxyStoreFromConfigResource(name="dev")
    return {
        "store": store,
        "metaxy_io_manager": mxd.MetaxyIOManager(store=store),
    }


@pytest.fixture
def instance() -> Iterator[dg.DagsterInstance]:
    """Ephemeral Dagster instance for testing."""
    with dg.DagsterInstance.ephemeral() as instance:
        yield instance
