"""
Ported from metaxy/ext/dagster/resources.py (MetaxyStoreFromConfigResource).

CONFIRMED against python\\rivers\\resource.py:
    class Resource(BaseSettings):
        def setup(self) -> None: ...
        def teardown(self) -> None: ...
    Resources are injected into asset functions via type-hinted parameters
    (see _worker_call_with_resources: resources are deserialized via
    model_validate_json, setup() is called, then injected as an arg).

KNOWN LIMITATION vs. the Dagster version (confirmed, not a guess):
    Dagster's create_resource(context) receives an InitResourceContext and
    passes `materialization_id=context.run.run_id` so every write from a
    run is tagged with that run's id. Rivers' `setup(self) -> None` takes
    NO context/run-id argument at all -- there is currently no way to get
    a run id at resource-setup time. This means per-run materialization
    tracking (the `metaxy/materialized_in_run` metadata Dagster's
    MetaxyIOManager reports) cannot be reproduced the same way here.
    Left out rather than faked with a wrong value.
"""

from __future__ import annotations

import metaxy as mx
from rivers import Resource


class MetaxyResource(Resource):
    """Resource for asset functions that want direct `MetadataStore` access
    (e.g. to call `store.resolve_update(...)`), mirroring
    `metaxy.ext.dagster.MetaxyStoreFromConfigResource`.

    Usage (resources are injected via type-hinted parameters in Rivers):

        @Asset(...)
        def my_asset(context, metaxy: MetaxyResource):
            with metaxy.store:
                increment = metaxy.store.resolve_update("my/feature")
            ...

    If `name` is not provided, the default store will be used (set via
    `store = "my_name"` in `metaxy.toml` or the `$METAXY_STORE` env var).
    """

    name: str | None = None

    _store: mx.MetadataStore | None = None

    def setup(self) -> None:
        self._store = mx.MetaxyConfig.get().get_store(self.name)

    def teardown(self) -> None:
        self._store = None

    @property
    def store(self) -> mx.MetadataStore:
        if self._store is None:
            raise RuntimeError(
                "MetaxyResource.store accessed before setup() was called "
                "by the Rivers runtime."
            )
        return self._store
