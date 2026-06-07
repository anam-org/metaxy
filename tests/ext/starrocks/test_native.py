"""StarRocks metadata store tests."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import ibis.expr.datatypes as idt
import narwhals as nw
import polars as pl
import pytest
from metaxy import BaseFeature, FeatureDefinition, FeatureSpec, HashAlgorithm
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.ext.polars.versioning import PolarsVersioningEngine
from metaxy.ext.starrocks import (
    StarRocksMetadataStore,
    StarRocksMetadataStoreConfig,
    StarRocksMySQLMetadataStore,
    StarRocksVersioningEngine,
)
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_UPDATED_AT,
)
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey
from metaxy.utils import collect_to_polars
from tests.metadata_stores.shared import (
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.fixture
def dummy_starrocks_store() -> StarRocksMetadataStore:
    return StarRocksMetadataStore(connection_string="mysql://root@127.0.0.1:9030/metaxy")


def test_starrocks_config_defaults() -> None:
    config = StarRocksMetadataStoreConfig(connection_string="mysql://root@127.0.0.1:9030/metaxy")

    assert config.replication_num == 1
    assert config.buckets is None
    assert config.group_concat_max_len == 16_777_216
    assert config.enable_latest_aggregate_table is True


def test_starrocks_requires_connection_info() -> None:
    with pytest.raises(ValueError, match="Must provide either connection_string or connection_params"):
        StarRocksMetadataStore()


def test_starrocks_connection_params_default_mysql_port() -> None:
    store = StarRocksMetadataStore(connection_params={"host": "127.0.0.1", "database": "metaxy"})

    assert store.connection_params["port"] == 9030


def test_starrocks_config_instantiation() -> None:
    config = MetaxyConfig(
        stores={
            "starrocks": StoreConfig(
                type="metaxy.ext.starrocks.StarRocksMetadataStore",
                config={
                    "connection_string": "mysql://root@127.0.0.1:9030/metaxy",
                    "replication_num": 1,
                    "buckets": 8,
                    "group_concat_max_len": 2048,
                    "enable_latest_aggregate_table": True,
                },
            ),
            "starrocks_mysql": StoreConfig(
                type="metaxy.ext.starrocks.StarRocksMySQLMetadataStore",
                config={
                    "connection_string": "mysql://root@127.0.0.1:9030/metaxy",
                    "enable_latest_aggregate_table": True,
                },
            ),
        }
    )

    optimized = config.get_store("starrocks")
    baseline = config.get_store("starrocks_mysql")

    assert isinstance(optimized, StarRocksMetadataStore)
    assert optimized.buckets == 8
    assert isinstance(baseline, StarRocksMySQLMetadataStore)
    assert baseline.enable_latest_aggregate_table is False


def test_starrocks_store_modes_are_distinct() -> None:
    baseline = StarRocksMySQLMetadataStore(connection_string="mysql://root@127.0.0.1:9030/metaxy")
    optimized = StarRocksMetadataStore(connection_string="mysql://root@127.0.0.1:9030/metaxy")

    assert baseline.native_implementation() == nw.Implementation.POLARS
    assert baseline.versioning_engine_cls is PolarsVersioningEngine
    assert baseline.hash_algorithm == HashAlgorithm.XXH3_64
    assert baseline.enable_latest_aggregate_table is False

    assert optimized.native_implementation() == nw.Implementation.IBIS
    assert optimized.versioning_engine_cls is StarRocksVersioningEngine
    assert optimized.hash_algorithm == HashAlgorithm.XXH3_64
    assert optimized.enable_latest_aggregate_table is True


def test_starrocks_duplicate_table_ddl(dummy_starrocks_store: StarRocksMetadataStore) -> None:
    schema = nw.Schema(
        {
            "sample_uid": nw.Int64(),
            "label": nw.String(),
            METAXY_PROVENANCE_BY_FIELD: nw.Struct({"frames": nw.String(), "audio": nw.String()}),
            METAXY_FEATURE_VERSION: nw.String(),
            METAXY_UPDATED_AT: nw.Datetime(time_zone="UTC"),
            METAXY_DELETED_AT: nw.Datetime(time_zone="UTC"),
        }
    )

    sql = dummy_starrocks_store._build_create_duplicate_table_sql(
        "test_stores__upstream_a",
        schema,
        ["sample_uid"],
    )

    assert sql == (
        "CREATE TABLE IF NOT EXISTS `test_stores__upstream_a` (\n"
        "  `sample_uid` BIGINT,\n"
        "  `label` STRING,\n"
        "  `metaxy_provenance_by_field` STRING,\n"
        "  `metaxy_feature_version` STRING,\n"
        "  `metaxy_updated_at` DATETIME,\n"
        "  `metaxy_deleted_at` DATETIME\n"
        ")\n"
        "DUPLICATE KEY(`sample_uid`)\n"
        "DISTRIBUTED BY HASH(`sample_uid`)\n"
        'PROPERTIES ("replication_num" = "1")'
    )


def test_starrocks_latest_table_ddl(dummy_starrocks_store: StarRocksMetadataStore) -> None:
    schema = nw.Schema({"sample_uid": nw.Int64(), METAXY_FEATURE_VERSION: nw.String()})

    sql = dummy_starrocks_store._build_create_latest_table_sql(
        "test_stores__upstream_a",
        schema,
        ["sample_uid"],
    )

    assert sql == (
        "CREATE TABLE IF NOT EXISTS `test_stores__upstream_a__metaxy_latest` (\n"
        "  `sample_uid` BIGINT,\n"
        "  `metaxy_feature_version` VARCHAR(255),\n"
        "  `metaxy_latest_lifecycle_at` DATETIME MAX\n"
        ")\n"
        "AGGREGATE KEY(`sample_uid`, `metaxy_feature_version`)\n"
        "DISTRIBUTED BY HASH(`sample_uid`, `metaxy_feature_version`)\n"
        'PROPERTIES ("replication_num" = "1")'
    )


def test_starrocks_json_encode_decode_roundtrip(
    dummy_starrocks_store: StarRocksMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    feature = test_features["UpstreamFeatureA"]
    df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"frames": "f1", "audio": "a1"}],
                METAXY_DATA_VERSION_BY_FIELD: [{"frames": "df1", "audio": "da1"}],
            }
        )
    )

    encoded = dummy_starrocks_store._encode_json_columns(df)
    encoded_pl = collect_to_polars(encoded)

    assert encoded_pl.schema[METAXY_PROVENANCE_BY_FIELD] == pl.String
    assert encoded_pl[METAXY_PROVENANCE_BY_FIELD].to_list() == ['{"frames":"f1","audio":"a1"}']

    decoded = dummy_starrocks_store._decode_json_frame(encoded.lazy(), feature.key)
    decoded_pl = collect_to_polars(decoded)

    assert decoded_pl.schema[METAXY_PROVENANCE_BY_FIELD] == pl.Struct({"frames": pl.String, "audio": pl.String})
    assert decoded_pl[METAXY_PROVENANCE_BY_FIELD].to_list() == [{"frames": "f1", "audio": "a1"}]


def test_starrocks_json_decode_empty_result_preserves_metaxy_struct_schema(
    dummy_starrocks_store: StarRocksMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    feature = test_features["UpstreamFeatureA"]
    empty = nw.from_native(
        pl.DataFrame(
            schema={
                "sample_uid": pl.Int64,
                METAXY_PROVENANCE_BY_FIELD: pl.String,
                METAXY_DATA_VERSION_BY_FIELD: pl.String,
            }
        ).lazy()
    )

    decoded = dummy_starrocks_store._decode_json_frame(empty, feature.key)
    decoded_schema = collect_to_polars(decoded).schema

    expected = pl.Struct({"frames": pl.String, "audio": pl.String})
    assert decoded_schema[METAXY_PROVENANCE_BY_FIELD] == expected
    assert decoded_schema[METAXY_DATA_VERSION_BY_FIELD] == expected


def _starrocks_engine(
    test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> StarRocksVersioningEngine:
    plan = test_graph.get_feature_plan(test_features["UpstreamFeatureA"].key)
    return StarRocksVersioningEngine(
        plan, StarRocksMetadataStore("mysql://root@127.0.0.1:9030/metaxy")._create_hash_functions()
    )


def _compile_mysql(frame: nw.DataFrame[Any] | nw.LazyFrame[Any]) -> str:
    import ibis

    return ibis.to_sql(frame.to_native(), dialect="mysql")


class _FakeLatestTableConnection:
    def __init__(self, latest_table: Any, latest_table_name: str) -> None:
        self.latest_table = latest_table
        self.latest_table_name = latest_table_name
        self.table_calls: list[str] = []

    def list_tables(self) -> list[str]:
        return [self.latest_table_name]

    def table(self, table_name: str) -> Any:
        self.table_calls.append(table_name)
        return self.latest_table


class _FakeStarRocksConnection:
    def __init__(self, tables: list[str] | None = None, table_objects: dict[str, Any] | None = None) -> None:
        self.tables = set(tables or [])
        self.table_objects = table_objects or {}
        self.raw_sql_calls: list[str] = []
        self.inserts: list[tuple[str, Any]] = []
        self.drops: list[str] = []

    def list_tables(self) -> list[str]:
        return list(self.tables)

    def raw_sql(self, sql: str) -> None:
        self.raw_sql_calls.append(sql)

    def insert(self, table_name: str, obj: Any) -> None:
        self.inserts.append((table_name, obj))
        self.tables.add(table_name)

    def drop_table(self, table_name: str) -> None:
        self.drops.append(table_name)
        self.tables.discard(table_name)

    def table(self, table_name: str) -> Any:
        return self.table_objects[table_name]


def test_starrocks_type_mapping_and_distribution_options() -> None:
    store = StarRocksMetadataStore(
        connection_string="mysql://root@127.0.0.1:9030/metaxy",
        replication_num=None,
        buckets=8,
    )
    schema = nw.Schema(
        {
            "id": nw.String(),
            "flag": nw.Boolean(),
            "count": nw.Int32(),
            "ratio32": nw.Float32(),
            "ratio64": nw.Float64(),
            "day": nw.Date(),
            "seen_at": nw.Datetime(),
            "items": nw.List(nw.String()),
            METAXY_DATA_VERSION_BY_FIELD: nw.Struct({"default": nw.String()}),
        }
    )

    sql = store._build_create_duplicate_table_sql("typed_table", schema, ["id"])

    assert "`id` VARCHAR(255)" in sql
    assert "`flag` BOOLEAN" in sql
    assert "`count` INT" in sql
    assert "`ratio32` FLOAT" in sql
    assert "`ratio64` DOUBLE" in sql
    assert "`day` DATE" in sql
    assert "`seen_at` DATETIME" in sql
    assert "`items` STRING" in sql
    assert "`metaxy_data_version_by_field` STRING" in sql
    assert "DISTRIBUTED BY HASH(`id`) BUCKETS 8" in sql
    assert "PROPERTIES" not in sql
    assert store._column_sql_type("fallback", object(), is_key=False) == "STRING"


def test_starrocks_resolve_key_columns_falls_back_for_system_tables(
    dummy_starrocks_store: StarRocksMetadataStore,
) -> None:
    schema = nw.Schema({"first_col": nw.Int64(), "second_col": nw.String()})

    assert dummy_starrocks_store._resolve_table_key_columns(FeatureKey(["metaxy-system", "events"]), schema) == [
        "first_col"
    ]

    with pytest.raises(ValueError, match="Cannot create StarRocks table"):
        dummy_starrocks_store._resolve_table_key_columns(FeatureKey(["metaxy-system", "empty"]), nw.Schema({}))


def test_starrocks_create_table_executes_canonical_and_latest_ddl(
    dummy_starrocks_store: StarRocksMetadataStore,
    test_features: dict[str, FeatureDefinition],
) -> None:
    feature = test_features["UpstreamFeatureA"]
    fake_conn = _FakeStarRocksConnection()
    dummy_starrocks_store._conn = fake_conn  # ty: ignore[invalid-assignment]
    df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "value": [10],
                METAXY_FEATURE_VERSION: ["v1"],
                METAXY_UPDATED_AT: [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                METAXY_DELETED_AT: [None],
            }
        )
    )

    with pytest.warns(UserWarning, match="AUTO_CREATE_TABLES"):
        statements = dummy_starrocks_store._create_starrocks_table(feature.key, "feature_table", df)

    assert len(statements) == 2
    assert len(fake_conn.raw_sql_calls) == 2
    assert "DUPLICATE KEY(`sample_uid`)" in fake_conn.raw_sql_calls[0]
    assert "AGGREGATE KEY(`sample_uid`, `metaxy_feature_version`)" in fake_conn.raw_sql_calls[1]


def test_starrocks_write_creates_and_populates_latest_projection(
    dummy_starrocks_store: StarRocksMetadataStore,
    test_features: dict[str, FeatureDefinition],
) -> None:
    feature = test_features["UpstreamFeatureA"]
    fake_conn = _FakeStarRocksConnection()
    dummy_starrocks_store._conn = fake_conn  # ty: ignore[invalid-assignment]
    dummy_starrocks_store.auto_create_tables = True
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    deleted_at = datetime(2024, 1, 3, tzinfo=timezone.utc)
    df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "value": [10, 20],
                METAXY_PROVENANCE_BY_FIELD: [{"default": "h1"}, {"default": "h2"}],
                METAXY_FEATURE_VERSION: ["v1", "v1"],
                METAXY_UPDATED_AT: [updated_at, updated_at],
                METAXY_DELETED_AT: [None, deleted_at],
            }
        )
    )

    with pytest.warns(UserWarning, match="AUTO_CREATE_TABLES"):
        dummy_starrocks_store._write_feature(feature.key, df)

    canonical_insert = fake_conn.inserts[0]
    latest_insert = fake_conn.inserts[1]
    latest_df = latest_insert[1]

    assert canonical_insert[0] == dummy_starrocks_store.get_table_name(feature.key)
    assert latest_insert[0].endswith("__metaxy_latest")
    assert isinstance(latest_df, pl.DataFrame)
    assert latest_df["metaxy_latest_lifecycle_at"].to_list() == [updated_at, deleted_at]


def test_starrocks_write_requires_existing_table_when_auto_create_disabled(
    dummy_starrocks_store: StarRocksMetadataStore,
    test_features: dict[str, FeatureDefinition],
) -> None:
    fake_conn = _FakeStarRocksConnection()
    dummy_starrocks_store._conn = fake_conn  # ty: ignore[invalid-assignment]
    dummy_starrocks_store.auto_create_tables = False

    with pytest.raises(Exception, match="does not exist"):
        dummy_starrocks_store._write_feature(
            test_features["UpstreamFeatureA"].key,
            nw.from_native(pl.DataFrame({"sample_uid": [1]})),
        )


def test_starrocks_drop_feature_drops_latest_projection_first(dummy_starrocks_store: StarRocksMetadataStore) -> None:
    feature_key = FeatureKey(["test", "stores", "upstream_a"])
    table_name = dummy_starrocks_store.get_table_name(feature_key)
    latest_table_name = dummy_starrocks_store.get_latest_table_name(table_name)
    fake_conn = _FakeStarRocksConnection([table_name, latest_table_name])
    dummy_starrocks_store._conn = fake_conn  # ty: ignore[invalid-assignment]

    dummy_starrocks_store._drop_feature(feature_key)

    assert fake_conn.drops == [latest_table_name, table_name]


def test_starrocks_refresh_latest_table_sql(
    dummy_starrocks_store: StarRocksMetadataStore,
    test_features: dict[str, FeatureDefinition],
) -> None:
    import ibis

    feature_key = test_features["UpstreamFeatureA"].key
    table_name = dummy_starrocks_store.get_table_name(feature_key)
    latest_table_name = dummy_starrocks_store.get_latest_table_name(table_name)
    table = pl.DataFrame(
        {
            "sample_uid": [1],
            METAXY_FEATURE_VERSION: ["v1"],
            METAXY_UPDATED_AT: [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            METAXY_DELETED_AT: [None],
        }
    )
    fake_conn = _FakeStarRocksConnection(
        [table_name, latest_table_name],
        table_objects={table_name: ibis.memtable(table)},
    )
    dummy_starrocks_store._conn = fake_conn  # ty: ignore[invalid-assignment]

    dummy_starrocks_store._refresh_latest_table(feature_key)

    assert fake_conn.raw_sql_calls[0] == f"TRUNCATE TABLE `{latest_table_name}`"
    assert "INSERT INTO" in fake_conn.raw_sql_calls[1]
    assert "MAX(COALESCE(`metaxy_deleted_at`, `metaxy_updated_at`))" in fake_conn.raw_sql_calls[1]


def test_starrocks_decode_json_skips_invalid_and_system_frames(dummy_starrocks_store: StarRocksMetadataStore) -> None:
    invalid_json = nw.from_native(pl.DataFrame({METAXY_PROVENANCE_BY_FIELD: ["not-json"]}).lazy())
    system_frame = nw.from_native(pl.DataFrame({METAXY_PROVENANCE_BY_FIELD: ['{"a":"b"}']}).lazy())

    decoded = dummy_starrocks_store._decode_json_frame(invalid_json, FeatureKey(["test", "feature"]))
    system_decoded = dummy_starrocks_store._decode_json_frame(system_frame, FeatureKey(["metaxy-system", "events"]))

    assert collect_to_polars(decoded).schema[METAXY_PROVENANCE_BY_FIELD] == pl.String
    assert system_decoded is system_frame


def test_starrocks_decode_json_noop_when_no_json_columns(dummy_starrocks_store: StarRocksMetadataStore) -> None:
    frame = nw.from_native(pl.DataFrame({"sample_uid": [1]}).lazy())

    assert dummy_starrocks_store._decode_json_frame(frame, FeatureKey(["test", "feature"])) is frame


def test_starrocks_read_decodes_own_results_and_leaves_fallback_results(
    dummy_starrocks_store: StarRocksMetadataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_key = FeatureKey(["test", "feature"])
    raw = nw.from_native(pl.DataFrame({METAXY_PROVENANCE_BY_FIELD: ['{"default":"h"}']}).lazy())
    fallback_store = object()

    monkeypatch.setattr(
        "metaxy.ext.ibis.metadata_store.IbisMetadataStore.read",
        lambda *args, **kwargs: (raw, fallback_store) if kwargs["with_store_info"] else raw,
    )
    monkeypatch.setattr(dummy_starrocks_store, "_resolve_feature_key", lambda feature: feature_key)

    decoded = dummy_starrocks_store.read(feature_key)
    fallback_result = dummy_starrocks_store.read(feature_key, with_store_info=True)

    assert collect_to_polars(decoded).schema[METAXY_PROVENANCE_BY_FIELD] == pl.Struct({"default": pl.String})
    assert fallback_result == (raw, fallback_store)


def test_starrocks_raw_json_reads_restores_flag(dummy_starrocks_store: StarRocksMetadataStore) -> None:
    assert dummy_starrocks_store._decode_json_reads is True

    with dummy_starrocks_store._raw_json_reads():
        assert dummy_starrocks_store._decode_json_reads is False

    assert dummy_starrocks_store._decode_json_reads is True


def test_starrocks_ibis_timestamp_type_maps_to_polars_datetime(dummy_starrocks_store: StarRocksMetadataStore) -> None:
    assert dummy_starrocks_store.ibis_type_to_polars(idt.Timestamp(timezone="UTC")) == pl.Datetime("us")


def test_starrocks_baseline_hash_validation_rejects_unsupported_algorithm() -> None:
    store = StarRocksMySQLMetadataStore(
        connection_string="mysql://root@127.0.0.1:9030/metaxy",
        hash_algorithm=HashAlgorithm.FARMHASH,
    )

    with pytest.raises(Exception, match="not supported"):
        store._validate_hash_algorithm_support()


def test_starrocks_native_hash_validation_rejects_unsupported_algorithm() -> None:
    store = StarRocksMetadataStore(
        connection_string="mysql://root@127.0.0.1:9030/metaxy",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with pytest.raises(Exception, match="not supported"):
        store._validate_hash_algorithm_support()


def test_starrocks_keep_latest_falls_back_without_projection(
    dummy_starrocks_store: StarRocksMetadataStore,
) -> None:
    import ibis

    feature_key = FeatureKey(["test", "stores", "upstream_a"])
    canonical = ibis.memtable(
        {
            "sample_uid": [1, 1],
            METAXY_FEATURE_VERSION: ["v1", "v1"],
            METAXY_UPDATED_AT: [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
        }
    )
    dummy_starrocks_store._conn = _FakeStarRocksConnection()  # ty: ignore[invalid-assignment]

    result = dummy_starrocks_store._keep_latest_by_group(
        df=nw.from_native(canonical, eager_only=False),
        feature_key=feature_key,
        group_columns=["sample_uid"],
        timestamp_columns=[METAXY_UPDATED_AT],
        feature_version="v1",
    )

    assert "ROW_NUMBER" in _compile_mysql(result).upper()


def test_starrocks_versioning_hash_sql(
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    import ibis

    engine = _starrocks_engine(test_graph, test_features)
    df = nw.from_native(ibis.memtable({"value": ["abc"]}), eager_only=False)

    xxh3_result = engine.hash_string_column(df, "value", "hashed", HashAlgorithm.XXH3_64)
    md5_result = engine.hash_string_column(df, "value", "hashed", HashAlgorithm.MD5)
    sha_result = engine.hash_string_column(df, "value", "hashed", HashAlgorithm.SHA256)

    assert "XX_HASH3_64" in _compile_mysql(xxh3_result).upper()
    assert "MD5" in _compile_mysql(md5_result).upper()
    sha_sql = _compile_mysql(sha_result).upper()
    assert "SHA2" in sha_sql
    assert "256" in sha_sql


def test_starrocks_versioning_json_build_and_extract_sql(
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    import ibis

    engine = _starrocks_engine(test_graph, test_features)
    df = nw.from_native(ibis.memtable({"frames": ["f1"], "audio": ["a1"]}), eager_only=False)

    built = engine.build_struct_column(df, "versions", {"frames": "frames", "audio": "audio"})
    build_sql = _compile_mysql(built).upper()

    assert "JSON_OBJECT" in build_sql
    assert "JSON_STRING" in build_sql

    extracted = engine._extract_metadata_fields(built, "versions", {"frames": "frames_out"})
    extract_sql = _compile_mysql(extracted).upper()

    assert "GET_JSON_STRING" in extract_sql
    assert '$."FRAMES"' in extract_sql


def test_starrocks_versioning_group_concat_sql(
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    import ibis

    engine = _starrocks_engine(test_graph, test_features)
    df = nw.from_native(
        ibis.memtable({"sample_uid": [1, 1], "value": ["b", "a"], "sort_key": [2, 1]}),
        eager_only=False,
    )

    result = engine.concat_strings_over_groups(
        df,
        source_column="value",
        target_column="joined",
        group_by_columns=["sample_uid"],
        order_by_columns=["sort_key"],
        separator="|",
    )
    sql = _compile_mysql(result).upper()

    assert "GROUP_CONCAT" in sql
    assert "SEPARATOR '|'" in sql
    assert "ORDER BY" in sql


def test_starrocks_versioning_row_number_latest_sql(
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    import ibis

    engine = _starrocks_engine(test_graph, test_features)
    df = nw.from_native(
        ibis.memtable(
            {
                "sample_uid": [1, 1],
                "value": ["old", "new"],
                "updated_at": [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                ],
            }
        ),
        eager_only=False,
    )

    result = engine.keep_latest_by_group(df, ["sample_uid"], ["updated_at"])
    sql = _compile_mysql(result).upper()

    assert "ROW_NUMBER" in sql
    assert "ORDER BY" in sql


def test_starrocks_keep_latest_uses_aggregate_latest_table(
    dummy_starrocks_store: StarRocksMetadataStore,
) -> None:
    import ibis

    feature_key = FeatureKey(["test", "stores", "upstream_a"])
    latest_table_name = dummy_starrocks_store.get_latest_table_name(dummy_starrocks_store.get_table_name(feature_key))
    canonical = ibis.memtable(
        {
            "sample_uid": [1, 1],
            "value": ["old", "new"],
            METAXY_FEATURE_VERSION: ["v1", "v1"],
            METAXY_UPDATED_AT: [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            METAXY_DELETED_AT: [None, None],
        }
    )
    latest = ibis.memtable(
        {
            "sample_uid": [1],
            METAXY_FEATURE_VERSION: ["v1"],
            "metaxy_latest_lifecycle_at": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
        }
    )
    fake_conn = _FakeLatestTableConnection(latest, latest_table_name)
    dummy_starrocks_store._conn = fake_conn  # ty: ignore[invalid-assignment]

    result = dummy_starrocks_store._keep_latest_by_group(
        df=nw.from_native(canonical, eager_only=False),
        feature_key=feature_key,
        group_columns=["sample_uid"],
        timestamp_columns=[METAXY_DELETED_AT, METAXY_UPDATED_AT],
        feature_version="v1",
    )

    sql = _compile_mysql(result).upper()
    assert fake_conn.table_calls == [latest_table_name]
    assert "INNER JOIN" in sql
    assert "METAXY_LATEST_LIFECYCLE_AT" in sql
    assert "ROW_NUMBER" in sql


@pytest.mark.ibis
@pytest.mark.polars
@pytest.mark.starrocks
class TestStarRocksBaselineMySQL(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, starrocks_db: str, starrocks_table_prefix: str):
        return StarRocksMySQLMetadataStore(
            connection_string=starrocks_db,
            table_prefix=starrocks_table_prefix,
            hash_algorithm=HashAlgorithm.XXH3_64,
        )

    @pytest.fixture
    def named_store(self, starrocks_db: str, starrocks_table_prefix: str):
        return StarRocksMySQLMetadataStore(
            connection_string=starrocks_db,
            table_prefix=starrocks_table_prefix,
            hash_algorithm=HashAlgorithm.XXH3_64,
            name="starrocks-mysql-test",
        )


@pytest.mark.ibis
@pytest.mark.native
@pytest.mark.starrocks
class TestStarRocksNativeOptimized(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, starrocks_db: str, starrocks_table_prefix: str):
        return StarRocksMetadataStore(
            connection_string=starrocks_db,
            table_prefix=starrocks_table_prefix,
            hash_algorithm=HashAlgorithm.XXH3_64,
        )

    @pytest.fixture
    def named_store(self, starrocks_db: str, starrocks_table_prefix: str):
        return StarRocksMetadataStore(
            connection_string=starrocks_db,
            table_prefix=starrocks_table_prefix,
            hash_algorithm=HashAlgorithm.XXH3_64,
            name="starrocks-test",
        )

    def test_latest_aggregate_table_is_populated_and_used(self, store: StarRocksMetadataStore) -> None:
        key = FeatureKey(["starrocks_latest_projection"])

        class LatestProjectionFeature(
            BaseFeature,
            spec=FeatureSpec(key=key, id_columns=["id"]),
        ):
            id: str
            value: int

        with store.open("w"):
            store.write(
                LatestProjectionFeature,
                pl.DataFrame(
                    {
                        "id": ["a", "b"],
                        "value": [1, 10],
                        METAXY_PROVENANCE_BY_FIELD: [{"default": "a1"}, {"default": "b1"}],
                    }
                ),
            )
            time.sleep(0.01)
            store.write(
                LatestProjectionFeature,
                pl.DataFrame(
                    {
                        "id": ["a"],
                        "value": [2],
                        METAXY_PROVENANCE_BY_FIELD: [{"default": "a2"}],
                    }
                ),
            )

            table_name = store.get_table_name(key)
            latest_table_name = store.get_latest_table_name(table_name)
            assert latest_table_name in store.conn.list_tables()

            latest_projection = collect_to_polars(nw.from_native(store.conn.table(latest_table_name), eager_only=False))
            assert latest_projection.height == 2
            assert set(latest_projection["id"].to_list()) == {"a", "b"}

            with store._raw_json_reads():
                read_plan = store.read(LatestProjectionFeature)
            sql = _compile_mysql(read_plan).upper()
            assert latest_table_name.upper() in sql
            assert "INNER JOIN" in sql
            assert "METAXY_LATEST_LIFECYCLE_AT" in sql
            assert "ROW_NUMBER" in sql
