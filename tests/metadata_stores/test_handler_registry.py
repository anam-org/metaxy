"""Tests for the multi-handler registry on ComputeEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from metaxy.ext.duckdb.engine import DuckDBEngine
from metaxy.ext.duckdb.handlers.lance import DuckDBLanceHandler
from metaxy.ext.ibis.engine import IbisSQLHandler, IbisStorageConfig
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.exceptions import NoHandlerError
from metaxy.metadata_store.storage_config import LanceStorageConfig, StorageConfig


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
        lance_path = tmp_path / "lance_data"
        lance_handler = DuckDBLanceHandler()
        engine = DuckDBEngine(
            database=":memory:",
            extensions=[*lance_handler.required_extensions()],
        )
        return MetadataStore(
            engine=engine,
            storage=[
                IbisStorageConfig(format="duckdb", location=":memory:"),
                LanceStorageConfig(format="lance", location=str(lance_path)),
            ],
        )

    def test_handler_resolution_for_sql_storage(self, multi_store: MetadataStore) -> None:
        sc = multi_store._storage[0]
        handler = multi_store._engine.get_handler(sc)
        assert isinstance(handler, IbisSQLHandler)

    def test_handler_resolution_for_lance_storage(self, multi_store: MetadataStore) -> None:
        sc = multi_store._storage[1]
        handler = multi_store._engine.get_handler(sc)
        assert isinstance(handler, DuckDBLanceHandler)
