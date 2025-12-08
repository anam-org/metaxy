"""Tests for delete filter expressions."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.predicate_cases import predicate_cases
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


def _base_delete_filter_df() -> pl.DataFrame:
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
def test_delete_metadata_accepts_predicate_cases(
    any_store: MetadataStore,
    case,
    delete_filter_feature,
) -> None:
    df = _base_delete_filter_df()
    with any_store.open("write"):
        any_store.write_metadata(delete_filter_feature, df)
        any_store.delete_metadata(
            delete_filter_feature,
            filters=case.exprs,
            soft=True,
            current_only=True,
        )
