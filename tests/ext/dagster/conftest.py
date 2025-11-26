from collections.abc import Iterator

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
def resources():
    store = mxd.MetaxyStoreFromConfigResource(name="dev")
    return {
        "store": store,
        "metaxy_io_manager": mxd.MetaxyIOManager(store=store),
    }


@pytest.fixture
def instance() -> Iterator[dg.DagsterInstance]:
    with dg.DagsterInstance.ephemeral() as instance:
        yield instance
