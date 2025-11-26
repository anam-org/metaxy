from typing import Any

import dagster as dg

import metaxy as mx


def test_metaxy_store_from_config(
    metaxy_config: mx.MetaxyConfig,
    instance: dg.DagsterInstance,
    resources: dict[str, Any],
):
    @dg.asset
    def my_asset(
        context: dg.AssetExecutionContext,
        store: dg.ResourceParam[mx.MetadataStore],
    ):
        assert store._materialization_id == context.run_id

    dg.materialize_to_memory([my_asset], resources=resources)
