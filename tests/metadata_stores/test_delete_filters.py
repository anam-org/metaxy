"""Tests for delete filter expressions."""

from __future__ import annotations

from datetime import datetime, timezone

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from metaxy_testing.predicate_cases import predicate_cases

from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD

PREDICATE_CASES = predicate_cases()


@pytest.fixture
def delete_filter_feature():
    class DeleteFilterFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="delete_filter_feature",
            fields=["value", "price", "status", "ts_tz", "ts_naive"],
        ),
    ):
        value: int | None = None
        price: float | None = None
        status: str | None = None
        ts_tz: datetime | None = None
        ts_naive: datetime | None = None

    return DeleteFilterFeature


@pytest.fixture
def base_delete_filter_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_uid": [str(i) for i in range(10)],
            "value": list(range(10)),
            "price": [1.0, 10.0, 3.0, 7.0, 2.5, 9.5, 4.0, 8.0, 0.5, 6.0],
            "status": [
                "active",
                "inactive",
                "pending",
                "active",
                "inactive",
                "pending",
                "active",
                "inactive",
                "pending",
                "active",
            ],
            "ts_tz": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc),
                datetime(2024, 1, 4, tzinfo=timezone.utc),
                datetime(2024, 1, 5, tzinfo=timezone.utc),
                datetime(2024, 1, 6, tzinfo=timezone.utc),
                datetime(2024, 1, 7, tzinfo=timezone.utc),
                datetime(2024, 1, 8, tzinfo=timezone.utc),
                datetime(2024, 1, 9, tzinfo=timezone.utc),
                datetime(2024, 1, 10, tzinfo=timezone.utc),
            ],
            "ts_naive": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
                datetime(2024, 1, 7),
                datetime(2024, 1, 8),
                datetime(2024, 1, 9),
                datetime(2024, 1, 10),
            ],
            METAXY_PROVENANCE_BY_FIELD: [
                {
                    "value": "p",
                    "price": "p",
                    "status": "p",
                    "ts_tz": "p",
                    "ts_naive": "p",
                }
            ]
            * 10,
        }
    )


@pytest.mark.parametrize(
    "case",
    PREDICATE_CASES,
    ids=[case.name for case in PREDICATE_CASES],
)
@pytest.mark.parametrize("soft", [True, False], ids=["soft", "hard"])
def test_delete_metadata_accepts_predicate_cases(
    any_store: MetadataStore,
    case,
    delete_filter_feature,
    base_delete_filter_df: pl.DataFrame,
    soft: bool,
) -> None:
    with any_store.open("write"):
        any_store.write_metadata(delete_filter_feature, base_delete_filter_df)
        any_store.delete_metadata(
            delete_filter_feature,
            filters=case.exprs,
            soft=soft,
            current_only=True,
        )


def test_delete_metadata_datetime_filter_hard_delete(
    any_store: MetadataStore,
    delete_filter_feature,
    base_delete_filter_df: pl.DataFrame,
) -> None:
    cutoff = datetime(2024, 1, 5, tzinfo=timezone.utc)
    with any_store.open("write"):
        any_store.write_metadata(delete_filter_feature, base_delete_filter_df)
        any_store.delete_metadata(
            delete_filter_feature,
            filters=nw.col("ts_tz") > cutoff,
            soft=False,
            current_only=True,
        )

    with any_store:
        remaining = any_store.read_metadata(delete_filter_feature).collect().to_polars()
        assert remaining.height == 5


@pytest.mark.parametrize("soft", [True, False], ids=["soft", "hard"])
def test_delete_metadata_with_none_filters(
    any_store: MetadataStore,
    delete_filter_feature,
    base_delete_filter_df: pl.DataFrame,
    soft: bool,
) -> None:
    """Test that delete_metadata accepts None as filters to delete all records."""
    with any_store.open("write"):
        any_store.write_metadata(delete_filter_feature, base_delete_filter_df)
        any_store.delete_metadata(
            delete_filter_feature,
            filters=None,
            soft=soft,
            current_only=True,
        )

    with any_store:
        remaining = any_store.read_metadata(delete_filter_feature).collect().to_polars()
        if soft:
            # Soft delete: records still exist when include_soft_deleted=True
            all_data = any_store.read_metadata(delete_filter_feature, include_soft_deleted=True).collect().to_polars()
            assert all_data.height == 10
            # But not returned by default
            assert remaining.height == 0
        else:
            # Hard delete: records are physically removed
            assert remaining.height == 0
