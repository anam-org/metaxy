"""Shared FeatureSpec.unique tests."""

import narwhals as nw
import polars as pl
import pytest
from metaxy import BaseFeature, FeatureSpec, Unique
from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import (
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROJECT_VERSION,
)
from metaxy.utils import collect_to_polars


def _metadata(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                **row,
                "metaxy_provenance_by_field": {"default": f"prov_{row['id']}"},
            }
            for row in rows
        ]
    )


class UniqueTests:
    """Backend contract for read-time uniqueness."""

    def test_unique_any_uses_stable_feature_id_order(self, store: MetadataStore) -> None:
        class StableAnyFeature(
            BaseFeature,
            spec=FeatureSpec(
                key="test/shared_unique_any",
                id_columns=("id",),
                unique=Unique(subset=("group",)),
            ),
        ):
            id: str
            group: str
            value: str

        with store.open("w"):
            store.write(
                StableAnyFeature,
                _metadata(
                    [
                        {"id": "z", "group": "g", "value": "stable-winner"},
                        {"id": "a", "group": "g", "value": "physical-last"},
                    ]
                ),
            )
            result = collect_to_polars(
                store.read(
                    StableAnyFeature,
                    columns=["id", "value"],
                )
            )

        assert result.to_dicts() == [{"id": "z", "value": "stable-winner"}]

    def test_unique_latest_uses_explicit_order(self, store: MetadataStore) -> None:
        class OrderedFeature(
            BaseFeature,
            spec=FeatureSpec(
                key="test/shared_unique_latest",
                id_columns=("group", "ordinal"),
                unique=Unique(
                    subset=("group",),
                    keep="latest",
                    order_by=("ordinal",),
                ),
            ),
        ):
            id: str
            group: str
            ordinal: int
            value: str

        with store.open("w"):
            store.write(
                OrderedFeature,
                _metadata(
                    [
                        {"id": "winner", "group": "g", "ordinal": 2, "value": "new"},
                        {"id": "older", "group": "g", "ordinal": 1, "value": "old"},
                    ]
                ),
            )
            result = collect_to_polars(
                store.read(
                    OrderedFeature,
                    columns=["id", "ordinal", "value"],
                )
            )
            retained = collect_to_polars(
                store.read(
                    OrderedFeature,
                    columns=["id", "ordinal", "value"],
                    apply_unique=False,
                )
            )

        assert result.to_dicts() == [{"id": "winner", "ordinal": 2, "value": "new"}]
        assert sorted(retained["ordinal"].to_list()) == [1, 2]

    def test_unique_latest_treats_null_order_as_older(self, store: MetadataStore) -> None:
        class NullableOrderFeature(
            BaseFeature,
            spec=FeatureSpec(
                key="test/shared_unique_latest_null",
                id_columns=("id",),
                unique=Unique(
                    subset=("group",),
                    keep="latest",
                    order_by=("ordinal",),
                ),
            ),
        ):
            id: str
            group: str
            ordinal: int | None

        with store.open("w"):
            store.write(
                NullableOrderFeature,
                _metadata(
                    [
                        {"id": "ordered", "group": "g", "ordinal": 1},
                        {"id": "missing", "group": "g", "ordinal": None},
                    ]
                ),
            )
            result = collect_to_polars(
                store.read(
                    NullableOrderFeature,
                    columns=["id", "ordinal"],
                )
            )

        assert result.to_dicts() == [{"id": "ordered", "ordinal": 1}]

    def test_unique_latest_tie_keeps_whole_row(self, store: MetadataStore) -> None:
        class TiedFeature(
            BaseFeature,
            spec=FeatureSpec(
                key="test/shared_unique_latest_tie",
                id_columns=("id",),
                unique=Unique(
                    subset=("group",),
                    keep="latest",
                    order_by=("ordinal",),
                ),
            ),
        ):
            id: str
            group: str
            ordinal: int
            value: str
            paired_value: str

        with store.open("w"):
            store.write(
                TiedFeature,
                _metadata(
                    [
                        {
                            "id": "z",
                            "group": "g",
                            "ordinal": 1,
                            "value": "z-value",
                            "paired_value": "z-pair",
                        },
                        {
                            "id": "a",
                            "group": "g",
                            "ordinal": 1,
                            "value": "a-value",
                            "paired_value": "a-pair",
                        },
                    ]
                ),
            )
            result = collect_to_polars(
                store.read(
                    TiedFeature,
                    columns=["id", "value", "paired_value"],
                )
            )

        assert result.to_dicts() == [{"id": "z", "value": "z-value", "paired_value": "z-pair"}]

    def test_current_view_tie_keeps_whole_explicitly_ordered_row(self, store: MetadataStore) -> None:
        class CurrentTieFeature(
            BaseFeature,
            spec=FeatureSpec(
                key="test/shared_unique_current_tie",
                id_columns=("id",),
                unique=Unique(
                    subset=("group",),
                    keep="latest",
                    order_by=("ordinal",),
                ),
            ),
        ):
            id: str
            group: str
            ordinal: int
            value: str
            paired_value: str

        with store.open("w"):
            store.write(
                CurrentTieFeature,
                _metadata(
                    [
                        {
                            "id": "same-id",
                            "group": "g",
                            "ordinal": 2,
                            "value": "winner",
                            "paired_value": "winner-pair",
                        },
                        {
                            "id": "same-id",
                            "group": "g",
                            "ordinal": 1,
                            "value": "physical-last",
                            "paired_value": "physical-last-pair",
                        },
                    ]
                ),
            )
            result = collect_to_polars(
                store.read(
                    CurrentTieFeature,
                    columns=["ordinal", "value", "paired_value"],
                )
            )
            retained = collect_to_polars(
                store.read(
                    CurrentTieFeature,
                    columns=["value", "paired_value"],
                    apply_unique=False,
                )
            )

        assert result.to_dicts() == [
            {
                "ordinal": 2,
                "value": "winner",
                "paired_value": "winner-pair",
            }
        ]
        assert retained.to_dicts() == [
            {
                "value": "winner",
                "paired_value": "winner-pair",
            }
        ]

    def test_unique_history_reappend_uses_recency_after_logical_key(self, store: MetadataStore) -> None:
        class ReappendFeature(
            BaseFeature,
            spec=FeatureSpec(
                key="test/shared_unique_reappend",
                id_columns=("id",),
                unique=Unique(subset=("group",)),
            ),
        ):
            id: str
            group: str
            value: str

        with store.open("w"):
            data = _metadata([{"id": "same-id", "group": "g", "value": "same-value"}])
            store.write(ReappendFeature, data, materialization_id="a-older")
            store.write(ReappendFeature, data, materialization_id="z-latest")
            result = collect_to_polars(
                store.read(
                    ReappendFeature,
                    with_sample_history=True,
                    columns=[METAXY_MATERIALIZATION_ID],
                )
            )

        assert result[METAXY_MATERIALIZATION_ID].to_list() == ["z-latest"]

    def test_unique_history_uses_versions_after_identical_logical_key(self, store: MetadataStore) -> None:
        class VersionedHistoryFeature(
            BaseFeature,
            spec=FeatureSpec(
                key="test/shared_unique_versioned_history",
                id_columns=("id",),
                unique=Unique(subset=("group",)),
            ),
        ):
            id: str
            group: str
            value: str

        with store.open("w"):
            store.write(
                VersionedHistoryFeature,
                _metadata(
                    [
                        {
                            "id": "same-id",
                            "group": "g",
                            "value": "same-value",
                            METAXY_FEATURE_VERSION: "version-z",
                            METAXY_PROJECT_VERSION: "project-z",
                        },
                        {
                            "id": "same-id",
                            "group": "g",
                            "value": "same-value",
                            METAXY_FEATURE_VERSION: "version-a",
                            METAXY_PROJECT_VERSION: "project-a",
                        },
                    ]
                ),
                preserve_feature_version=True,
            )
            result = collect_to_polars(
                store.read(
                    VersionedHistoryFeature,
                    with_feature_history=True,
                    with_sample_history=True,
                    columns=[METAXY_FEATURE_VERSION, METAXY_PROJECT_VERSION],
                )
            )
            retained = collect_to_polars(
                store.read(
                    VersionedHistoryFeature,
                    with_feature_history=True,
                    apply_unique=False,
                    columns=[METAXY_FEATURE_VERSION, METAXY_PROJECT_VERSION],
                )
            )

        assert result.select(METAXY_FEATURE_VERSION, METAXY_PROJECT_VERSION).to_dicts() == [
            {
                METAXY_FEATURE_VERSION: "version-z",
                METAXY_PROJECT_VERSION: "project-z",
            }
        ]
        assert sorted(retained[METAXY_FEATURE_VERSION].to_list()) == [
            "version-a",
            "version-z",
        ]

    @pytest.mark.parametrize(
        "unique",
        [
            pytest.param(Unique(subset=("group",)), id="any"),
            pytest.param(
                Unique(
                    subset=("group",),
                    keep="latest",
                    order_by=("ordinal",),
                ),
                id="latest",
            ),
        ],
    )
    def test_unique_soft_delete_contract(self, store: MetadataStore, unique: Unique) -> None:
        class SoftDeleteFeature(
            BaseFeature,
            spec=FeatureSpec(
                key=f"test/shared_unique_soft_delete_{unique.keep}",
                id_columns=("id",),
                unique=unique,
            ),
        ):
            id: str
            group: str
            ordinal: int

        with store.open("w"):
            store.write(
                SoftDeleteFeature,
                _metadata(
                    [
                        {"id": "a-live", "group": "g", "ordinal": 1},
                        {"id": "z-deleted", "group": "g", "ordinal": 2},
                    ]
                ),
            )
            store.delete(
                SoftDeleteFeature,
                filters=nw.col("id") == "z-deleted",
                soft=True,
            )
            result = collect_to_polars(store.read(SoftDeleteFeature, columns=["id"]))
            including_deleted = collect_to_polars(
                store.read(
                    SoftDeleteFeature,
                    columns=["id", METAXY_DELETED_AT],
                    include_soft_deleted=True,
                )
            )

        assert result["id"].to_list() == ["a-live"]
        assert including_deleted["id"].to_list() == ["z-deleted"]
        assert including_deleted[METAXY_DELETED_AT].is_not_null().all()
