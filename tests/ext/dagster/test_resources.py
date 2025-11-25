from collections.abc import Iterator
from typing import Any

import dagster as dg
import pytest

import metaxy as mx
import metaxy.ext.dagster as mxd


@pytest.fixture(autouse=True)
def metaxy_config(tmp_path):
    with mx.MetaxyConfig(
        stores={
            "dev": mx.StoreConfig(
                type="metaxy.metadata_store.delta.DeltaMetadataStore",
                config={"root_path": tmp_path},
            )
        }
    ).use() as config:
        yield config


@pytest.fixture
def resources(metaxy_config: mx.MetaxyConfig):
    return {"store_from_config": mxd.MetaxyStoreFromConfigResource(name="dev")}


@pytest.fixture
def instance() -> Iterator[dg.DagsterInstance]:
    with dg.DagsterInstance.ephemeral() as instance:
        yield instance


def test_metaxy_store_from_config(
    metaxy_config: mx.MetaxyConfig,
    instance: dg.DagsterInstance,
    resources: dict[str, Any],
):
    @dg.asset
    def my_asset(
        context: dg.AssetExecutionContext,
        store_from_config: dg.ResourceParam[mx.MetadataStore],
    ):
        assert store_from_config._materialization_id == context.run_id

    dg.materialize_to_memory([my_asset], resources=resources)
