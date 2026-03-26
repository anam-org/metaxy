"""Tests for shared metadata store utilities."""

from pathlib import Path

import ibis.expr.datatypes as dt
import polars as pl
import pytest
from ibis.common.collections import FrozenOrderedDict
from ibis.common.temporal import IntervalUnit

from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.metadata_store.utils import is_local_path


@pytest.fixture
def store(tmp_path: Path) -> DuckDBMetadataStore:
    """Minimal DuckDBMetadataStore for testing ibis_type_to_polars."""
    return DuckDBMetadataStore(database=tmp_path / "test.db")


def test_is_local_path() -> None:
    """is_local_path should correctly detect local vs remote URIs."""
    assert is_local_path("./local/path")
    assert is_local_path("/absolute/path")
    assert is_local_path("relative/path")
    assert is_local_path("C:\\Windows\\Path")
    assert is_local_path("file:///absolute/path")
    assert is_local_path("local://path")

    assert not is_local_path("s3://bucket/path")
    assert not is_local_path("db://database-name")
    assert not is_local_path("http://remote-server/db")
    assert not is_local_path("https://remote-server/db")
    assert not is_local_path("gs://bucket/path")
    assert not is_local_path("az://container/path")


# -- ibis_type_to_polars: simple types delegated to Ibis's to_polars() --


@pytest.mark.parametrize(
    ("ibis_type", "expected_polars_type"),
    [
        (dt.String(), pl.String),
        (dt.Int8(), pl.Int8),
        (dt.Int16(), pl.Int16),
        (dt.Int32(), pl.Int32),
        (dt.Int64(), pl.Int64),
        (dt.UInt8(), pl.UInt8),
        (dt.UInt16(), pl.UInt16),
        (dt.UInt32(), pl.UInt32),
        (dt.UInt64(), pl.UInt64),
        (dt.Float32(), pl.Float32),
        (dt.Float64(), pl.Float64),
        (dt.Boolean(), pl.Boolean),
        (dt.Date(), pl.Date),
        (dt.Time(), pl.Time),
        (dt.Null(), pl.Null),
        (dt.Binary(), pl.Binary),
    ],
    ids=lambda v: str(v),
)
def test_ibis_type_to_polars_simple(
    store: IbisMetadataStore, ibis_type: dt.DataType, expected_polars_type: pl.DataType
) -> None:
    assert store.ibis_type_to_polars(ibis_type) == expected_polars_type


# -- ibis_type_to_polars: custom-handled fallback types --


@pytest.mark.parametrize(
    ("ibis_type", "expected_polars_type"),
    [
        (dt.Float16(), pl.Float32),
        (dt.UUID(), pl.String),
        (dt.JSON(), pl.String),
        (dt.MACADDR(), pl.String),
        (dt.INET(), pl.String),
        (dt.GeoSpatial(), pl.Binary),
        (dt.Point(), pl.Binary),
    ],
    ids=lambda v: str(v),
)
def test_ibis_type_to_polars_fallback(
    store: IbisMetadataStore, ibis_type: dt.DataType, expected_polars_type: pl.DataType
) -> None:
    assert store.ibis_type_to_polars(ibis_type) == expected_polars_type


# -- ibis_type_to_polars: parametric types --


def test_ibis_type_to_polars_timestamp_no_tz(store: IbisMetadataStore) -> None:
    result = store.ibis_type_to_polars(dt.Timestamp())
    assert result == pl.Datetime("ns")


def test_ibis_type_to_polars_timestamp_with_tz(store: IbisMetadataStore) -> None:
    result = store.ibis_type_to_polars(dt.Timestamp(timezone="UTC"))
    assert result == pl.Datetime("ns", "UTC")


def test_ibis_type_to_polars_decimal(store: IbisMetadataStore) -> None:
    result = store.ibis_type_to_polars(dt.Decimal(precision=10, scale=2))
    assert result == pl.Decimal(precision=10, scale=2)


def test_ibis_type_to_polars_interval(store: IbisMetadataStore) -> None:
    result = store.ibis_type_to_polars(dt.Interval(unit=IntervalUnit.MICROSECOND))
    assert result == pl.Duration("us")


# -- ibis_type_to_polars: recursive / compound types --


def test_ibis_type_to_polars_array(store: IbisMetadataStore) -> None:
    result = store.ibis_type_to_polars(dt.Array(value_type=dt.Int64()))
    assert result == pl.List(pl.Int64)


def test_ibis_type_to_polars_nested_array(store: IbisMetadataStore) -> None:
    result = store.ibis_type_to_polars(dt.Array(value_type=dt.Array(value_type=dt.String())))
    assert result == pl.List(pl.List(pl.String))


def test_ibis_type_to_polars_struct(store: IbisMetadataStore) -> None:
    ibis_struct = dt.Struct(fields=FrozenOrderedDict({"name": dt.String(), "age": dt.Int32()}))
    result = store.ibis_type_to_polars(ibis_struct)
    assert result == pl.Struct({"name": pl.String, "age": pl.Int32})


# -- ibis_type_to_polars: Map raises at base level --


def test_ibis_type_to_polars_map_raises(store: IbisMetadataStore) -> None:
    with pytest.raises(NotImplementedError):
        store.ibis_type_to_polars(dt.Map(key_type=dt.String(), value_type=dt.Int64()))


def test_ibis_type_to_polars_unknown_raises(store: IbisMetadataStore) -> None:
    with pytest.raises(NotImplementedError):
        store.ibis_type_to_polars(dt.Unknown())
