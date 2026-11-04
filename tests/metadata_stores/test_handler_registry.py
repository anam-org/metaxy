"""Tests for the multi-handler registry on ComputeEngine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from metaxy.ext.duckdb.engine import DuckDBEngine
from metaxy.ext.duckdb.handlers.lance import DuckDBLanceHandler
from metaxy.ext.ibis.engine import IbisSQLHandler, IbisStorageConfig
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.exceptions import NoHandlerError
from metaxy.metadata_store.storage_config import LanceStorageConfig, StorageConfig
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.utils import collect_to_polars


@pytest.mark.ibis
@pytest.mark.duckdb
class TestHandlerRegistry:
    """Test the flat handler registry on ComputeEngine."""

    def test_default_handlers_loaded_lazily(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        assert not engine._defaults_loaded
        sc = IbisStorageConfig(format="duckdb", location=":memory:")
        handler = engine.get_handler(sc)
        assert engine._defaults_loaded
        assert isinstance(handler, IbisSQLHandler)

    def test_get_handler_returns_capable_handler(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        sc = IbisStorageConfig(format="duckdb", location=":memory:")
        handler = engine.get_handler(sc)
        assert isinstance(handler, IbisSQLHandler)

    def test_get_handler_raises_no_handler_error(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        sc = StorageConfig(format="unknown_format", location="/tmp/nope")
        with pytest.raises(NoHandlerError, match="No handler registered for format 'unknown_format'"):
            engine.get_handler(sc)

    def test_user_override_takes_priority(self) -> None:
        custom_handler = IbisSQLHandler(auto_create_tables=True)
        engine = DuckDBEngine(
            database=":memory:",
            handlers=[custom_handler],
        )
        sc = IbisStorageConfig(format="duckdb", location=":memory:")
        # User-provided handler comes first
        assert engine.get_handler(sc) is custom_handler

    def test_lance_default_handler(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        sc = LanceStorageConfig(format="lance", location="/tmp/lance")
        handler = engine.get_handler(sc)
        assert isinstance(handler, DuckDBLanceHandler)

    def test_lance_user_override(self) -> None:
        custom_lance = DuckDBLanceHandler()
        engine = DuckDBEngine(
            database=":memory:",
            handlers=[custom_lance],
        )
        sc = LanceStorageConfig(format="lance", location="/tmp/lance")
        assert engine.get_handler(sc) is custom_lance

    def test_defaults_loaded_only_once(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        engine._ensure_defaults()
        count_after_first = len(engine._handlers)
        engine._ensure_defaults()
        assert len(engine._handlers) == count_after_first

    def test_can_handle(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        assert engine.can_handle(IbisStorageConfig(format="duckdb", location=":memory:"))
        assert engine.can_handle(LanceStorageConfig(format="lance", location="/tmp/lance"))


@pytest.mark.ibis
@pytest.mark.duckdb
class TestMultiStorageHandlerResolution:
    """Test that a DuckDB engine can work with both SQL and Lance storage configs."""

    @pytest.fixture
    def multi_store(self, tmp_path: Path) -> MetadataStore:
        lance_handler = DuckDBLanceHandler()
        engine = DuckDBEngine(
            database=":memory:",
            extensions=[*lance_handler.required_extensions()],
        )
        return MetadataStore(
            engine=engine,
            storage=IbisStorageConfig(format="duckdb", location=":memory:"),
            fallback_storage=[LanceStorageConfig(format="lance", location=str(tmp_path / "lance_data"))],
        )

    def test_handler_resolution_for_sql_storage(self, multi_store: MetadataStore) -> None:
        assert multi_store._storage is not None
        handler = multi_store._engine.get_handler(multi_store._storage)
        assert isinstance(handler, IbisSQLHandler)

    def test_handler_resolution_for_lance_storage(self, multi_store: MetadataStore) -> None:
        handler = multi_store._engine.get_handler(multi_store._fallback_storage[0])
        assert isinstance(handler, DuckDBLanceHandler)


def _metadata(sample_uids: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_uid": sample_uids,
            METAXY_PROVENANCE_BY_FIELD: [{"frames": f"h{uid}", "audio": f"h{uid}"} for uid in sample_uids],
        }
    )


@pytest.mark.ibis
@pytest.mark.duckdb
@pytest.mark.usefixtures("test_graph")
class TestFallbackStorage:
    """A store reads from its primary storage first, then from its fallback storages, and writes only to the primary."""

    @pytest.fixture
    def database(self, tmp_path: Path) -> Path:
        return tmp_path / "metadata.duckdb"

    @pytest.fixture
    def prod_storage(self, database: Path) -> IbisStorageConfig:
        return IbisStorageConfig(format="duckdb", location=str(database), table_prefix="prod_")

    @pytest.fixture
    def dev_storage(self, database: Path) -> IbisStorageConfig:
        return IbisStorageConfig(format="duckdb", location=str(database), table_prefix="dev_")

    @pytest.fixture
    def make_store(self, database: Path) -> Callable[..., MetadataStore]:
        def make(storage: StorageConfig | None = None, fallback_storage: Sequence[StorageConfig] = ()) -> MetadataStore:
            return MetadataStore(
                engine=DuckDBEngine(database=database, auto_create_tables=True),
                storage=storage,
                fallback_storage=fallback_storage,
                auto_create_tables=True,
            )

        return make

    @pytest.fixture
    def upstream(self, test_features: dict[str, Any]) -> Any:
        return test_features["UpstreamFeatureA"]

    @pytest.fixture
    def prod_with_upstream(
        self, make_store: Callable[..., MetadataStore], prod_storage: IbisStorageConfig, upstream: Any
    ) -> None:
        with make_store(prod_storage).open("w") as prod:
            prod.write(upstream, _metadata([1, 2]))

    @pytest.fixture
    def dev_store(
        self,
        make_store: Callable[..., MetadataStore],
        dev_storage: IbisStorageConfig,
        prod_storage: IbisStorageConfig,
    ) -> MetadataStore:
        return make_store(dev_storage, [prod_storage])

    @pytest.mark.usefixtures("prod_with_upstream")
    def test_reads_from_fallback_storage(self, dev_store: MetadataStore, upstream: Any) -> None:
        with dev_store:
            assert dev_store.has_feature(upstream)
            assert sorted(collect_to_polars(dev_store.read(upstream))["sample_uid"].to_list()) == [1, 2]

    @pytest.mark.usefixtures("prod_with_upstream")
    def test_primary_storage_takes_precedence(self, dev_store: MetadataStore, upstream: Any) -> None:
        with dev_store.open("w"):
            dev_store.write(upstream, _metadata([3]))
            assert collect_to_polars(dev_store.read(upstream))["sample_uid"].to_list() == [3]

    def test_writes_go_to_primary_storage_only(
        self,
        dev_store: MetadataStore,
        make_store: Callable[..., MetadataStore],
        prod_storage: IbisStorageConfig,
        upstream: Any,
    ) -> None:
        with dev_store.open("w"):
            dev_store.write(upstream, _metadata([1]))

        with make_store(prod_storage) as prod:
            assert not prod.has_feature(upstream)

    def test_has_feature_is_false_when_no_storage_has_it(self, dev_store: MetadataStore, upstream: Any) -> None:
        with dev_store:
            assert not dev_store.has_feature(upstream)

    def test_storage_defaults_to_engine_native_storage(
        self, make_store: Callable[..., MetadataStore], database: Path, upstream: Any
    ) -> None:
        store = make_store()
        assert store._storage == IbisStorageConfig(format="duckdb", location=str(database))

        with store.open("w"):
            store.write(upstream, _metadata([1]))
            assert collect_to_polars(store.read(upstream))["sample_uid"].to_list() == [1]

    def test_unhandled_fallback_storage_fails_at_construction(
        self, make_store: Callable[..., MetadataStore], dev_storage: IbisStorageConfig
    ) -> None:
        with pytest.raises(NoHandlerError, match="unknown_format"):
            make_store(dev_storage, [StorageConfig(format="unknown_format", location="/tmp/nope")])
