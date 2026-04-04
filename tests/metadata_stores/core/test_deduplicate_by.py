"""Tests for Deduplication runtime deduplication in the read path."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing.models import SampleFeature, SampleFeatureSpec

from metaxy import Deduplication, FeatureGraph, FieldKey, FieldSpec
from metaxy.ext.polars.handlers.delta import DeltaMetadataStore
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.constants import ALL_SYSTEM_COLUMNS
from metaxy.utils import collect_to_polars


@contextmanager
def fixed_transaction_time(store: MetadataStore, timestamp: datetime) -> Iterator[None]:
    """Set a deterministic transaction timestamp for writes within this context."""
    store._transaction_timestamp = timestamp  # noqa: SLF001
    try:
        yield
    finally:
        store._transaction_timestamp = None  # noqa: SLF001


# ========== Fixtures ==========


@pytest.fixture
def feature_factory(graph: FeatureGraph) -> Callable[..., type[SampleFeature]]:
    _ = graph

    def build_feature(
        *,
        key: str,
        deduplication: Deduplication | None = None,
    ) -> type[SampleFeature]:
        class TestFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=key,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deduplication=deduplication,
            ),
        ):
            content_hash: str | None = None

        return TestFeature

    return build_feature


@pytest.fixture
def dedup_feature(feature_factory: Callable[..., type[SampleFeature]]) -> type[SampleFeature]:
    return feature_factory(key="test/deduplicated", deduplication=Deduplication(by=("content_hash",)))


@pytest.fixture
def no_dedup_feature(feature_factory: Callable[..., type[SampleFeature]]) -> type[SampleFeature]:
    return feature_factory(key="test/no_dedup")


@pytest.fixture
def delta_store(tmp_path: Path) -> DeltaMetadataStore:
    return DeltaMetadataStore(root_path=tmp_path / "delta")


def _make_data(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        [{**row, "metaxy_provenance_by_field": {"default": f"prov_{row['sample_uid']}"}} for row in rows]
    )


@pytest.fixture
def data_factory() -> Callable[[list[dict[str, Any]]], pl.DataFrame]:
    return _make_data


@pytest.fixture
def three_rows_data(data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame]) -> pl.DataFrame:
    return data_factory(
        [
            {"sample_uid": "s1", "content_hash": "aaa"},
            {"sample_uid": "s2", "content_hash": "aaa"},
            {"sample_uid": "s3", "content_hash": "bbb"},
        ]
    )


@pytest.fixture
def two_duplicate_rows_data(data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame]) -> pl.DataFrame:
    return data_factory(
        [
            {"sample_uid": "s1", "content_hash": "aaa"},
            {"sample_uid": "s2", "content_hash": "aaa"},
        ]
    )


def _write_and_read(
    store: DeltaMetadataStore,
    feature: type[SampleFeature],
    data: pl.DataFrame,
    /,
    **read_kwargs: Any,
) -> pl.DataFrame:
    with store.open("w"):
        store.write(feature, data)
        return collect_to_polars(store.read(feature, **read_kwargs))


# ========== _internal_read_columns unit tests ==========


@pytest.mark.parametrize(
    ("columns", "additional_columns", "target_column"),
    [
        pytest.param(["metaxy_feature_version"], None, "metaxy_feature_version", id="user-column-overlaps-system"),
        pytest.param(["content_hash"], ["content_hash"], "content_hash", id="user-column-overlaps-dedup"),
        pytest.param(["col1"], ["metaxy_feature_version"], "metaxy_feature_version", id="dedup-overlaps-system"),
    ],
)
def test_internal_read_columns_avoids_duplicates(
    columns: list[str],
    additional_columns: list[str] | None,
    target_column: str,
) -> None:
    result = MetadataStore._internal_read_columns(columns, additional_columns=additional_columns)
    assert result.count(target_column) == 1


class TestInternalReadColumns:
    """Unit tests for MetadataStore._internal_read_columns."""

    def test_adds_missing_system_columns(self) -> None:
        result = MetadataStore._internal_read_columns(["col1"])
        for sys_col in ALL_SYSTEM_COLUMNS:
            assert sys_col in result

    def test_preserves_user_column_order(self) -> None:
        result = MetadataStore._internal_read_columns(["b", "a", "c"])
        assert result[:3] == ["b", "a", "c"]

    def test_adds_additional_columns(self) -> None:
        result = MetadataStore._internal_read_columns(["col1"], additional_columns=["dedup_col"])
        assert "dedup_col" in result

    def test_none_additional_columns_equivalent_to_omitted(self) -> None:
        assert MetadataStore._internal_read_columns(
            ["col1"], additional_columns=None
        ) == MetadataStore._internal_read_columns(["col1"])


# ========== Integration tests ==========


def test_deduplicate_by_removes_content_duplicates(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, dedup_feature, three_rows_data)

    assert len(result) == 2
    assert set(result["content_hash"].to_list()) == {"aaa", "bbb"}


def test_deduplicate_by_none_keeps_all_rows(
    no_dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, no_dedup_feature, three_rows_data)

    assert len(result) == 3


@pytest.mark.parametrize(
    ("with_sample_history", "include_soft_deleted", "expect_soft_delete"),
    [
        pytest.param(True, False, False, id="with-sample-history"),
        pytest.param(False, True, True, id="include-soft-deleted"),
    ],
)
def test_deduplicate_by_can_be_disabled_by_read_mode(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
    with_sample_history: bool,
    include_soft_deleted: bool,
    expect_soft_delete: bool,
) -> None:
    with delta_store.open("w"):
        delta_store.write(dedup_feature, two_duplicate_rows_data)
        if expect_soft_delete:
            delta_store.delete(dedup_feature, filters=nw.col("sample_uid") == "s1", soft=True)
        result = collect_to_polars(
            delta_store.read(
                dedup_feature,
                with_sample_history=with_sample_history,
                include_soft_deleted=include_soft_deleted,
            )
        )

    assert len(result) == 2


def test_deduplicate_by_stable_after_soft_delete(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
) -> None:
    with delta_store.open("w"):
        delta_store.write(dedup_feature, two_duplicate_rows_data)
        delta_store.delete(dedup_feature, filters=nw.col("sample_uid") == "s1", soft=True)
        result = collect_to_polars(delta_store.read(dedup_feature))

    assert len(result) == 1
    assert result["sample_uid"].to_list() == ["s2"]


def test_deduplicate_by_works_with_column_projection(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, dedup_feature, three_rows_data, columns=["sample_uid"])

    assert len(result) == 2
    assert "content_hash" not in result.columns


def test_deduplicate_by_works_with_column_projection_excluding_id_columns(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, dedup_feature, three_rows_data, columns=["content_hash"])

    assert len(result) == 2
    assert result.columns == ["content_hash"]
    assert set(result["content_hash"].to_list()) == {"aaa", "bbb"}


def test_deduplicate_by_keep_latest(
    feature_factory: Callable[..., type[SampleFeature]],
    delta_store: DeltaMetadataStore,
    data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame],
) -> None:
    latest_feature = feature_factory(
        key="test/dedup_latest",
        deduplication=Deduplication(by=("content_hash",), keep="latest"),
    )

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    with delta_store.open("w"), fixed_transaction_time(delta_store, t0):
        delta_store.write(latest_feature, data_factory([{"sample_uid": "s1", "content_hash": "aaa"}]))

    with delta_store.open("w"), fixed_transaction_time(delta_store, t0 + timedelta(seconds=1)):
        delta_store.write(latest_feature, data_factory([{"sample_uid": "s2", "content_hash": "aaa"}]))
        result = collect_to_polars(delta_store.read(latest_feature))

    assert len(result) == 1
    assert result["sample_uid"].to_list() == ["s2"]
