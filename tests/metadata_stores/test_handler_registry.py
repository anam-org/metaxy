"""Tests for the multi-handler registry on ComputeEngine."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

import narwhals as nw
import polars as pl
import pytest
from metaxy.ext.duckdb.engine import DuckDBEngine
from metaxy.ext.duckdb.handlers.lance import DuckDBLanceHandler
from metaxy.ext.ibis.engine import IbisSQLHandler, IbisStorageConfig
from metaxy.ext.polars.engine import PolarsComputeEngine
from metaxy.ext.polars.handlers.delta import DeltaHandler, DeltaStorageConfig
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.exceptions import NoHandlerError
from metaxy.metadata_store.storage_config import LanceStorageConfig, StorageConfig
from metaxy.metadata_store.table_ref import DeltaTableIdentifier
from metaxy.metadata_store.types import AccessMode
from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
from metaxy.utils import collect_to_polars
from narwhals.typing import Frame


@pytest.mark.ibis
@pytest.mark.duckdb
class TestHandlerRegistry:
    """Test the flat handler registry on ComputeEngine."""

    def test_default_handlers_loaded_lazily(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        assert not engine._defaults_loaded
        sc = IbisStorageConfig(location=":memory:")
        handler = engine.get_handler(sc)
        assert engine._defaults_loaded
        assert isinstance(handler, IbisSQLHandler)

    def test_get_handler_returns_capable_handler(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        sc = IbisStorageConfig(location=":memory:")
        handler = engine.get_handler(sc)
        assert isinstance(handler, IbisSQLHandler)

    def test_get_handler_raises_no_handler_error(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        sc = StorageConfig(location="/tmp/nope")
        with pytest.raises(NoHandlerError, match="No handler registered for StorageConfig"):
            engine.get_handler(sc)

    def test_user_override_takes_priority(self) -> None:
        custom_handler = IbisSQLHandler(auto_create_tables=True)
        engine = DuckDBEngine(
            database=":memory:",
            handlers=[custom_handler],
        )
        sc = IbisStorageConfig(location=":memory:")
        # User-provided handler comes first
        assert engine.get_handler(sc) is custom_handler

    def test_lance_default_handler(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        sc = LanceStorageConfig(location="/tmp/lance")
        handler = engine.get_handler(sc)
        assert isinstance(handler, DuckDBLanceHandler)

    def test_lance_user_override(self) -> None:
        custom_lance = DuckDBLanceHandler()
        engine = DuckDBEngine(
            database=":memory:",
            handlers=[custom_lance],
        )
        sc = LanceStorageConfig(location="/tmp/lance")
        assert engine.get_handler(sc) is custom_lance

    def test_defaults_loaded_only_once(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        engine._ensure_defaults()
        count_after_first = len(engine._handlers)
        engine._ensure_defaults()
        assert len(engine._handlers) == count_after_first

    def test_can_handle(self) -> None:
        engine = DuckDBEngine(database=":memory:")
        assert engine.can_handle(IbisStorageConfig(location=":memory:"))
        assert engine.can_handle(LanceStorageConfig(location="/tmp/lance"))


@pytest.mark.ibis
@pytest.mark.duckdb
class TestMultiStorageHandlerResolution:
    """Test that a DuckDB engine can work with both SQL and Lance storage configs."""

    @pytest.fixture
    def multi_store(self, tmp_path: Path) -> MetadataStore:
        engine = DuckDBEngine(database=":memory:")
        return MetadataStore(
            engine=engine,
            storage=IbisStorageConfig(location=":memory:"),
            fallback_storage=[LanceStorageConfig(location=str(tmp_path / "lance_data"))],
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
        return IbisStorageConfig(location=str(database), table_prefix="prod_")

    @pytest.fixture
    def dev_storage(self, database: Path) -> IbisStorageConfig:
        return IbisStorageConfig(location=str(database), table_prefix="dev_")

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
        assert store._storage == IbisStorageConfig(location=str(database))

        with store.open("w"):
            store.write(upstream, _metadata([1]))
            assert collect_to_polars(store.read(upstream))["sample_uid"].to_list() == [1]

    def test_table_name_comes_from_primary_storage(self, dev_store: MetadataStore, upstream: Any) -> None:
        assert dev_store.get_table_name(upstream.spec.key) == f"dev_{upstream.spec.key.table_name}"

    def test_table_name_requires_sql_storage(self, tmp_path: Path, upstream: Any) -> None:
        store = MetadataStore(
            engine=DuckDBEngine(database=":memory:"), storage=LanceStorageConfig(location=str(tmp_path))
        )
        with pytest.raises(TypeError, match="LanceStorageConfig does not store features in SQL tables"):
            store.get_table_name(upstream.spec.key)

    def test_unhandled_fallback_storage_fails_at_construction(
        self, make_store: Callable[..., MetadataStore], dev_storage: IbisStorageConfig
    ) -> None:
        with pytest.raises(NoHandlerError, match="StorageConfig"):
            make_store(dev_storage, [StorageConfig(location="/tmp/nope")])


class UnusedStorageConfig(IbisStorageConfig):
    """Storage that no store in these tests uses."""


class RecordingSQLHandler(IbisSQLHandler):
    """SQL handler that records its lifecycle events."""

    def __init__(self, events: list[str], *, storage_type: type[StorageConfig] = IbisStorageConfig) -> None:
        super().__init__(auto_create_tables=True)
        self.events = events
        self.storage_type = storage_type

    def can_handle(self, storage_config: StorageConfig) -> bool:
        return type(storage_config) is self.storage_type

    def on_open(self, conn: Any, mode: AccessMode) -> None:  # noqa: ARG002
        self.events.append(f"on_open:{mode}")

    def on_close(self, conn: Any) -> None:  # noqa: ARG002
        self.events.append("on_close")

    @contextmanager
    def write_scope(self, conn: Any) -> Generator[Self]:  # noqa: ARG002
        self.events.append("write_scope:enter")
        yield self
        self.events.append("write_scope:exit")


@pytest.mark.ibis
@pytest.mark.duckdb
@pytest.mark.usefixtures("test_graph")
class TestHandlerLifecycle:
    """The engine opens the handlers that serve the store's storages once per connection and scopes each write."""

    @pytest.fixture
    def events(self) -> list[str]:
        return []

    @pytest.fixture
    def unused_events(self) -> list[str]:
        return []

    @pytest.fixture
    def engine(self, tmp_path: Path, events: list[str], unused_events: list[str]) -> DuckDBEngine:
        return DuckDBEngine(
            database=tmp_path / "metadata.duckdb",
            handlers=[
                RecordingSQLHandler(unused_events, storage_type=UnusedStorageConfig),
                RecordingSQLHandler(events),
            ],
        )

    @pytest.fixture
    def store(self, engine: DuckDBEngine) -> MetadataStore:
        return MetadataStore(engine=engine, auto_create_tables=True)

    def test_hooks_wrap_the_connection_and_each_write(
        self, store: MetadataStore, events: list[str], test_features: dict[str, Any]
    ) -> None:
        upstream = test_features["UpstreamFeatureA"]
        with store.open("w"):
            store.write(upstream, _metadata([1]))
            store.write(upstream, _metadata([2]))
            assert store.has_feature(upstream)
            collect_to_polars(store.read(upstream))

        assert events == [
            "on_open:w",
            *["write_scope:enter", "write_scope:exit"] * 2,
            "on_close",
        ]

    def test_handlers_without_a_storage_are_not_opened(self, store: MetadataStore, unused_events: list[str]) -> None:
        with store:
            pass

        assert unused_events == []

    def test_handler_resolved_while_open_is_opened(self, engine: DuckDBEngine, unused_events: list[str]) -> None:
        with engine:
            engine.get_handler(UnusedStorageConfig(location=""))
            assert unused_events == ["on_open:r"]

        assert unused_events == ["on_open:r", "on_close"]


class CapturingDeltaHandler(DeltaHandler):
    """Delta handler that records the schema of every frame it writes."""

    def __init__(self) -> None:
        super().__init__()
        self.schemas: list[nw.Schema] = []

    def write(self, conn: None, table_id: DeltaTableIdentifier, df: Frame, **kwargs: Any) -> None:
        self.schemas.append(nw.from_native(df).collect_schema())
        super().write(conn, table_id, df, **kwargs)


@pytest.mark.delta
@pytest.mark.polars
@pytest.mark.usefixtures("test_graph")
class TestWriteFrameContract:
    """Handlers receive `*_by_field` columns as `Map`, whatever form the caller passes."""

    @pytest.fixture
    def handler(self) -> CapturingDeltaHandler:
        return CapturingDeltaHandler()

    @pytest.fixture
    def store(self, tmp_path: Path, handler: CapturingDeltaHandler) -> MetadataStore:
        return MetadataStore(
            engine=PolarsComputeEngine(handlers=[handler]), storage=DeltaStorageConfig(location=str(tmp_path))
        )

    @pytest.mark.parametrize(
        "to_native",
        [
            pytest.param(lambda df: df, id="polars"),
            pytest.param(pl.DataFrame.lazy, id="polars-lazy"),
            pytest.param(pl.DataFrame.to_arrow, id="pyarrow"),
        ],
    )
    def test_struct_columns_arrive_as_maps(
        self,
        store: MetadataStore,
        handler: CapturingDeltaHandler,
        test_features: dict[str, Any],
        to_native: Callable[[pl.DataFrame], Any],
    ) -> None:
        upstream = test_features["UpstreamFeatureA"]
        with store.open("w"):
            store.write(upstream, to_native(_metadata([1])))
            read_schema = store.read(upstream).collect_schema()

        (written_schema,) = handler.schemas
        for schema in (written_schema, read_schema):
            assert isinstance(schema[METAXY_PROVENANCE_BY_FIELD], nw.Map)
            assert isinstance(schema[METAXY_DATA_VERSION_BY_FIELD], nw.Map)
