"""Tests for FeatureSpec.unique runtime behavior."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import narwhals as nw
import polars as pl
import pytest
from metaxy import FeatureGraph, FieldKey, FieldSpec, Unique
from metaxy.ext.polars.handlers.delta import DeltaMetadataStore
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.constants import (
    ALL_SYSTEM_COLUMNS,
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROJECT_VERSION,
    METAXY_PROVENANCE,
    METAXY_UPDATED_AT,
)
from metaxy.utils import collect_to_polars
from metaxy.utils.constants import TEMP_TABLE_NAME
from metaxy_testing.models import SampleFeature, SampleFeatureSpec

# ========== Fixtures ==========


@pytest.fixture
def feature_factory(graph: FeatureGraph) -> Callable[..., type[SampleFeature]]:
    _ = graph

    def build_feature(
        *,
        key: str,
        unique: Unique | None = None,
    ) -> type[SampleFeature]:
        class TestFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=key,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                unique=unique,
            ),
        ):
            content_hash: str | None = None
            ordinal: int | None = None
            status: str | None = None
            value: str | None = None

        return TestFeature

    return build_feature


@pytest.fixture
def dedup_feature(feature_factory: Callable[..., type[SampleFeature]]) -> type[SampleFeature]:
    return feature_factory(key="test/deduplicated", unique=Unique(subset=("content_hash",)))


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
        pytest.param(["content_hash"], ["content_hash"], "content_hash", id="user-column-overlaps-unique"),
        pytest.param(["col1"], ["metaxy_feature_version"], "metaxy_feature_version", id="unique-overlaps-system"),
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

    def test_deduplicates_additional_columns_while_preserving_order(self) -> None:
        result = MetadataStore._internal_read_columns(
            ["col1"],
            additional_columns=["sample_uid", "sample_uid", "dedup_col"],
        )
        assert result.count("sample_uid") == 1
        assert result[-2:] == ["sample_uid", "dedup_col"]


def test_apply_unique_preserves_subset_order_for_any(
    graph: FeatureGraph,
    delta_store: DeltaMetadataStore,
) -> None:
    class OrderedUniqueFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="test/ordered_unique_any",
            unique=Unique(subset=("secondary_key", "primary_key")),
        ),
    ):
        primary_key: str | None = None
        secondary_key: str | None = None

    lazy_frame = nw.from_native(
        pl.DataFrame(
            [
                {
                    "sample_uid": "s1",
                    "primary_key": "a",
                    "secondary_key": "b",
                    METAXY_DELETED_AT: None,
                    METAXY_UPDATED_AT: datetime(2025, 1, 1, tzinfo=timezone.utc),
                }
            ]
        ).lazy()
    )
    plan = graph.get_feature_plan(OrderedUniqueFeature.spec().key)
    captured: dict[str, Any] = {}

    def fake_unique(
        self: nw.LazyFrame[Any],
        subset: str | list[str] | None = None,
        *,
        keep: str = "any",
        order_by: str | list[str] | None = None,
    ) -> nw.LazyFrame[Any]:
        captured["subset"] = subset
        captured["keep"] = keep
        captured["order_by"] = order_by
        return self

    with patch.object(type(lazy_frame), "unique", autospec=True, side_effect=fake_unique):
        result = delta_store._apply_unique(lazy_frame, plan=plan)

    assert TEMP_TABLE_NAME not in result.collect_schema().names()
    assert captured == {
        "subset": ["secondary_key", "primary_key"],
        "keep": "last",
        "order_by": [
            "sample_uid",
            TEMP_TABLE_NAME,
            METAXY_DELETED_AT,
            METAXY_UPDATED_AT,
            METAXY_DATA_VERSION,
            METAXY_PROVENANCE,
            METAXY_FEATURE_VERSION,
            METAXY_PROJECT_VERSION,
            METAXY_CREATED_AT,
            METAXY_MATERIALIZATION_ID,
        ],
    }


def test_apply_unique_preserves_subset_order_for_latest(
    graph: FeatureGraph,
    delta_store: DeltaMetadataStore,
) -> None:
    class OrderedLatestFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="test/ordered_unique_latest",
            unique=Unique(
                subset=("secondary_key", "primary_key"),
                keep="latest",
                order_by=("priority",),
            ),
        ),
    ):
        primary_key: str | None = None
        secondary_key: str | None = None
        priority: int | None = None

    lazy_frame = nw.from_native(
        pl.DataFrame(
            [
                {
                    "sample_uid": "s1",
                    "primary_key": "a",
                    "secondary_key": "b",
                    "priority": 1,
                    METAXY_DELETED_AT: None,
                    METAXY_UPDATED_AT: datetime(2025, 1, 1, tzinfo=timezone.utc),
                }
            ]
        ).lazy()
    )
    plan = graph.get_feature_plan(OrderedLatestFeature.spec().key)
    captured: dict[str, Any] = {}

    def fake_unique(
        self: nw.LazyFrame[Any],
        subset: str | list[str] | None = None,
        *,
        keep: str = "any",
        order_by: str | list[str] | None = None,
    ) -> nw.LazyFrame[Any]:
        captured["subset"] = subset
        captured["keep"] = keep
        captured["order_by"] = order_by
        return self

    with patch.object(type(lazy_frame), "unique", autospec=True, side_effect=fake_unique):
        result = delta_store._apply_unique(lazy_frame, plan=plan)

    assert TEMP_TABLE_NAME not in result.collect_schema().names()
    assert captured == {
        "subset": ["secondary_key", "primary_key"],
        "keep": "last",
        "order_by": [
            "priority",
            "sample_uid",
            TEMP_TABLE_NAME,
            METAXY_DELETED_AT,
            METAXY_UPDATED_AT,
            METAXY_DATA_VERSION,
            METAXY_PROVENANCE,
            METAXY_FEATURE_VERSION,
            METAXY_PROJECT_VERSION,
            METAXY_CREATED_AT,
            METAXY_MATERIALIZATION_ID,
        ],
    }


def test_apply_unique_rejects_unsupported_keep(delta_store: DeltaMetadataStore) -> None:
    lazy_frame = nw.from_native(pl.DataFrame([{"sample_uid": "s1", "content_hash": "aaa"}]).lazy())
    invalid_plan = SimpleNamespace(
        feature=SimpleNamespace(
            id_columns=("sample_uid",),
            unique=Unique.model_construct(subset=("content_hash",), keep="unsupported"),
        )
    )

    with pytest.raises(ValueError, match="Unsupported unique.keep value"):
        delta_store._apply_unique(lazy_frame, plan=invalid_plan)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


# ========== Integration tests ==========


def test_unique_removes_content_duplicates(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, dedup_feature, three_rows_data)

    assert len(result) == 2
    assert set(result["content_hash"].to_list()) == {"aaa", "bbb"}


def test_unique_none_keeps_all_rows(
    no_dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, no_dedup_feature, three_rows_data)

    assert len(result) == 3


def test_unique_applied_with_sample_history(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
) -> None:
    with delta_store.open("w"):
        delta_store.write(dedup_feature, two_duplicate_rows_data)
        result = collect_to_polars(
            delta_store.read(
                dedup_feature,
                with_feature_history=True,
                with_sample_history=True,
            )
        )

    assert len(result) == 1


def test_unique_can_be_disabled_explicitly(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
) -> None:
    with delta_store.open("w"):
        delta_store.write(dedup_feature, two_duplicate_rows_data)
        result = collect_to_polars(delta_store.read(dedup_feature, apply_unique=False))

    assert len(result) == 2


def test_unique_applied_with_include_soft_deleted(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
) -> None:
    """Unique is a feature property and is applied even when include_soft_deleted=True."""
    result = _write_and_read(delta_store, dedup_feature, two_duplicate_rows_data, include_soft_deleted=True)

    assert len(result) == 1


def test_live_duplicate_survives_soft_delete(
    feature_factory: Callable[..., type[SampleFeature]],
    delta_store: DeltaMetadataStore,
    data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame],
) -> None:
    feature = feature_factory(key="test/dedup_soft", unique=Unique(subset=("content_hash",)))

    with delta_store.open("w"):
        delta_store.write(
            feature,
            data_factory(
                [
                    {"sample_uid": "s1", "content_hash": "shared"},
                    {"sample_uid": "s2", "content_hash": "shared"},
                ]
            ),
        )
        delta_store.delete(feature, filters=nw.col("sample_uid") == "s2", soft=True)
        result = collect_to_polars(delta_store.read(feature))
        historical = collect_to_polars(delta_store.read(feature, with_sample_history=True))
        including_deleted = collect_to_polars(delta_store.read(feature, include_soft_deleted=True))

    assert len(result) == 1
    assert result["sample_uid"].to_list() == ["s1"]
    assert result["content_hash"].to_list() == ["shared"]
    assert len(historical) == 1
    assert historical[METAXY_DELETED_AT].is_null().all()
    assert including_deleted["sample_uid"].to_list() == ["s2"]
    assert including_deleted[METAXY_DELETED_AT].is_not_null().all()


@pytest.mark.parametrize(
    "unique",
    [
        pytest.param(Unique(subset=("content_hash",)), id="any"),
        pytest.param(
            Unique(subset=("content_hash",), keep="latest", order_by=("ordinal",)),
            id="latest",
        ),
    ],
)
def test_stale_materialization_cannot_hide_current_duplicate(
    feature_factory: Callable[..., type[SampleFeature]],
    delta_store: DeltaMetadataStore,
    data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame],
    unique: Unique,
) -> None:
    feature = feature_factory(key=f"test/dedup_current_{unique.keep}", unique=unique)

    with delta_store.open("w"):
        delta_store.write(
            feature,
            data_factory([{"sample_uid": "s2", "content_hash": "aaa", "ordinal": 1}]),
            materialization_id="a-first",
        )
        delta_store.write(
            feature,
            data_factory([{"sample_uid": "s1", "content_hash": "aaa", "ordinal": 2}]),
            materialization_id="b-second",
        )
        delta_store.write(
            feature,
            data_factory([{"sample_uid": "s1", "content_hash": "ccc", "ordinal": 3}]),
            materialization_id="c-third",
        )
        result = collect_to_polars(delta_store.read(feature))

    assert dict(zip(result["sample_uid"], result["content_hash"], strict=True)) == {
        "s1": "ccc",
        "s2": "aaa",
    }


def test_unique_works_with_column_projection(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, dedup_feature, three_rows_data, columns=["sample_uid"])

    assert len(result) == 2
    assert "content_hash" not in result.columns


def test_unique_works_with_column_projection_excluding_id_columns(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    three_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(delta_store, dedup_feature, three_rows_data, columns=["content_hash"])

    assert len(result) == 2
    assert result.columns == ["content_hash"]
    assert set(result["content_hash"].to_list()) == {"aaa", "bbb"}


def test_unique_works_with_column_projection_and_sample_history(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(
        delta_store,
        dedup_feature,
        two_duplicate_rows_data,
        with_sample_history=True,
        columns=["sample_uid"],
    )

    assert len(result) == 1
    assert result.columns == ["sample_uid"]


def test_unique_works_with_column_projection_and_sample_history_without_id_columns(
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
) -> None:
    result = _write_and_read(
        delta_store,
        dedup_feature,
        two_duplicate_rows_data,
        with_sample_history=True,
        columns=["content_hash"],
    )

    assert len(result) == 1
    assert result.columns == ["content_hash"]


def test_unique_latest_loads_order_by_for_exact_projection(
    feature_factory: Callable[..., type[SampleFeature]],
    delta_store: DeltaMetadataStore,
    data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame],
) -> None:
    feature = feature_factory(
        key="test/dedup_latest_projection",
        unique=Unique(subset=("content_hash",), keep="latest", order_by=("ordinal",)),
    )
    result = _write_and_read(
        delta_store,
        feature,
        data_factory(
            [
                {"sample_uid": "s1", "content_hash": "aaa", "ordinal": 2, "value": "winner"},
                {"sample_uid": "s2", "content_hash": "aaa", "ordinal": 1, "value": "older"},
            ]
        ),
        columns=["value"],
    )

    assert result.columns == ["value"]
    assert result["value"].to_list() == ["winner"]


def test_unique_latest_applies_filters_after_selection(
    feature_factory: Callable[..., type[SampleFeature]],
    delta_store: DeltaMetadataStore,
    data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame],
) -> None:
    feature = feature_factory(
        key="test/dedup_latest_filter",
        unique=Unique(subset=("content_hash",), keep="latest", order_by=("ordinal",)),
    )
    result = _write_and_read(
        delta_store,
        feature,
        data_factory(
            [
                {"sample_uid": "s1", "content_hash": "aaa", "ordinal": 1, "status": "active"},
                {"sample_uid": "s2", "content_hash": "aaa", "ordinal": 2, "status": "deleted"},
            ]
        ),
        filters=[nw.col("status") == "active"],
    )

    assert result.is_empty()


def test_unique_latest_uses_user_order_before_write_time(
    feature_factory: Callable[..., type[SampleFeature]],
    delta_store: DeltaMetadataStore,
    data_factory: Callable[[list[dict[str, Any]]], pl.DataFrame],
) -> None:
    latest_feature = feature_factory(
        key="test/dedup_latest",
        unique=Unique(subset=("content_hash",), keep="latest", order_by=("ordinal",)),
    )

    with delta_store.open("w"):
        delta_store.write(
            latest_feature,
            data_factory([{"sample_uid": "s1", "content_hash": "aaa", "ordinal": 2}]),
            materialization_id="a-first",
        )

    with delta_store.open("w"):
        delta_store.write(
            latest_feature,
            data_factory([{"sample_uid": "s2", "content_hash": "aaa", "ordinal": 1}]),
            materialization_id="z-later",
        )
        result = collect_to_polars(delta_store.read(latest_feature))

    assert len(result) == 1
    assert result["sample_uid"].to_list() == ["s1"]


def test_unique_history_without_registered_feature_plan_requires_explicit_raw_read(
    graph: FeatureGraph,
    dedup_feature: type[SampleFeature],
    delta_store: DeltaMetadataStore,
    two_duplicate_rows_data: pl.DataFrame,
) -> None:
    with delta_store.open("w"):
        delta_store.write(dedup_feature, two_duplicate_rows_data)
        graph.remove_feature(dedup_feature.spec().key)

        with pytest.raises(
            RuntimeError,
            match=r"Cannot apply FeatureSpec\.unique.*set apply_unique=False",
        ):
            delta_store.read(
                dedup_feature.spec().key,
                with_feature_history=True,
                with_sample_history=True,
            )

        raw = collect_to_polars(
            delta_store.read(
                dedup_feature.spec().key,
                with_feature_history=True,
                with_sample_history=True,
                apply_unique=False,
            )
        )

    assert len(raw) == 2
