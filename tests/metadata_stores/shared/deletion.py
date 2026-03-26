"""Deletion test pack for metadata stores."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from metaxy_testing.predicate_cases import PredicateCase, predicate_cases

from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DELETED_AT,
    METAXY_PROVENANCE_BY_FIELD,
)

PREDICATE_CASES = predicate_cases()


class DeletionTests:
    """Tests for soft/hard delete, restore, timestamps, and delete filter expressions."""

    # ---- fixtures from test_deletion.py ----

    @pytest.fixture
    def base_time(self) -> datetime:
        return datetime(2024, 1, 1, tzinfo=timezone.utc)

    @pytest.fixture
    def make_value_df(self) -> Callable[..., pl.DataFrame]:
        def _make(
            *,
            sample_uids: list[str],
            values: list[int],
            provenance: list[str],
            created_at: list[datetime],
            deleted_at: list[datetime] | None = None,
        ) -> pl.DataFrame:
            data = {
                "sample_uid": sample_uids,
                "value": values,
                METAXY_PROVENANCE_BY_FIELD: [{"value": prov} for prov in provenance],
                METAXY_CREATED_AT: created_at,
            }
            if deleted_at is not None:
                data[METAXY_DELETED_AT] = deleted_at
            # Note: metaxy_updated_at is set by write, not manually
            return pl.DataFrame(data)

        return _make

    @pytest.fixture
    def user_profile_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "sample_uid": ["u1", "u2", "u3"],
                "email": ["a@example.com", "b@example.com", "c@example.com"],
                "status": ["active", "active", "inactive"],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"email": "p1", "status": "p1"},
                    {"email": "p2", "status": "p2"},
                    {"email": "p3", "status": "p3"},
                ],
            }
        )

    # ---- fixtures from test_delete_filters.py ----

    @pytest.fixture
    def delete_filter_feature(self) -> type[SampleFeature]:
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
    def base_delete_filter_df(self) -> pl.DataFrame:
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

    # ---- tests from test_deletion.py ----

    def test_soft_deleted_rows_filtered_by_default(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Soft-deleted metadata rows should be hidden by default and opt-in with include_deleted."""

        class SoftDeleteFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="soft_delete",
                fields=["value"],
            ),
        ):
            value: int | None = None

        initial_df = make_value_df(
            sample_uids=["a", "b"],
            values=[1, 2],
            provenance=["p1", "p2"],
            created_at=[base_time, base_time],
        )

        with store.open("w"):
            store.write(SoftDeleteFeature, initial_df)
            # Use delete to soft-delete row "a" instead of manually writing a deleted row
            store.delete(SoftDeleteFeature, filters=nw.col("sample_uid") == "a", soft=True)

            active = store.read(SoftDeleteFeature).collect().to_polars()
            assert active.filter(pl.col("sample_uid") == "a").is_empty()
            assert active[METAXY_DELETED_AT].is_null().all()

            with_deleted = store.read(SoftDeleteFeature, include_soft_deleted=True).collect().to_polars()
            assert set(with_deleted["sample_uid"]) == {"a", "b"}
            deleted_row = with_deleted.filter(pl.col("sample_uid") == "a")
            assert deleted_row[METAXY_DELETED_AT].is_null().any() is False
            active_row = with_deleted.filter(pl.col("sample_uid") == "b")
            assert active_row[METAXY_DELETED_AT].is_null().all()

    def test_write_delete_write_sequence(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Test write->delete->write sequence: latest write is visible, soft deletes is hidden by default."""

        class WriteDeleteWriteFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="write_delete_write",
                fields=["value"],
            ),
        ):
            value: int | None = None

        write1_df = make_value_df(
            sample_uids=["x"],
            values=[100],
            provenance=["p_write1"],
            created_at=[base_time],
        )

        delete_df = make_value_df(
            sample_uids=["x"],
            values=[100],
            provenance=["p_delete"],
            created_at=[base_time + timedelta(seconds=1)],
            deleted_at=[base_time + timedelta(seconds=1)],
        )

        write2_df = make_value_df(
            sample_uids=["x"],
            values=[200],
            provenance=["p_write2"],
            created_at=[base_time + timedelta(seconds=2)],
        )

        with store.open("w"):
            store.write(WriteDeleteWriteFeature, write1_df)
            store.write(WriteDeleteWriteFeature, delete_df)
            store.write(WriteDeleteWriteFeature, write2_df)

            active = store.read(WriteDeleteWriteFeature).collect().to_polars()
            assert len(active) == 1
            assert active["sample_uid"][0] == "x"
            assert active["value"][0] == 200
            assert active[METAXY_DELETED_AT].is_null().all()

            with_deleted = (
                store.read(WriteDeleteWriteFeature, include_soft_deleted=True)
                .collect()
                .to_polars()
                .sort(METAXY_CREATED_AT)
            )
            assert len(with_deleted) == 1
            assert with_deleted["sample_uid"][0] == "x"
            assert with_deleted["value"][0] == 200
            assert with_deleted[METAXY_DELETED_AT].is_null().all()

    def test_write_delete_write_delete_sequence(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Test write->delete->write->delete: no active rows by default, latest write visible with filter."""

        class WriteDeleteWriteDeleteFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="write_delete_write_delete",
                fields=["value"],
            ),
        ):
            value: int | None = None

        write1_df = make_value_df(
            sample_uids=["y"],
            values=[100],
            provenance=["p_write1"],
            created_at=[base_time],
        )

        write2_df = make_value_df(
            sample_uids=["y"],
            values=[200],
            provenance=["p_write2"],
            created_at=[base_time + timedelta(seconds=2)],
        )

        with store.open("w"):
            # Write first value
            store.write(WriteDeleteWriteDeleteFeature, write1_df)
            # Soft delete it
            store.delete(WriteDeleteWriteDeleteFeature, filters=nw.col("sample_uid") == "y", soft=True)
            # Write second value
            store.write(WriteDeleteWriteDeleteFeature, write2_df)
            # Soft delete it again
            store.delete(WriteDeleteWriteDeleteFeature, filters=nw.col("sample_uid") == "y", soft=True)

            active = store.read(WriteDeleteWriteDeleteFeature).collect().to_polars()
            assert active.is_empty()

            with_deleted = store.read(WriteDeleteWriteDeleteFeature, include_soft_deleted=True).collect().to_polars()
            assert len(with_deleted) == 1
            assert with_deleted["sample_uid"][0] == "y"
            assert with_deleted["value"][0] == 200
            assert with_deleted[METAXY_DELETED_AT].is_not_null().all()

    def test_soft_delete_historical_version_preserves_latest(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Soft-deleting old versions should not hide the current version."""

        class VersionedFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="soft_delete_versions",
                fields=["value"],
            ),
        ):
            value: int | None = None

        newer_time = base_time + timedelta(minutes=5)

        first_version = make_value_df(
            sample_uids=["a"],
            values=[1],
            provenance=["p1"],
            created_at=[base_time],
        )
        latest_version = make_value_df(
            sample_uids=["a"],
            values=[2],
            provenance=["p2"],
            created_at=[newer_time],
        )

        with store.open("w"):
            store.write(VersionedFeature, first_version)
            store.write(VersionedFeature, latest_version)

            store.delete(
                VersionedFeature,
                filters=nw.col(METAXY_CREATED_AT) == base_time,
                with_feature_history=True,
                with_sample_history=True,
            )

            active = store.read(VersionedFeature).collect().to_polars()
            assert active.is_empty()

            with_deleted = (
                store.read(VersionedFeature, include_soft_deleted=True, with_sample_history=True).collect().to_polars()
            )
            non_deleted_values = with_deleted.filter(pl.col(METAXY_DELETED_AT).is_null())["value"].to_list()
            assert 2 in non_deleted_values
            assert with_deleted.filter(pl.col(METAXY_DELETED_AT).is_not_null()).filter(pl.col("value") == 1).height >= 1

    def test_soft_delete_then_overwrite_restores_row(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Writing new metadata after a soft delete should make the row active again."""

        class RestoreFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="soft_delete_restore",
                fields=["value"],
            ),
        ):
            value: int | None = None

        initial = make_value_df(
            sample_uids=["a"],
            values=[1],
            provenance=["p1"],
            created_at=[base_time],
        )

        with store.open("w"):
            store.write(RestoreFeature, initial)
            store.delete(RestoreFeature, filters=nw.col("sample_uid") == "a")

            soft_deleted = store.read(RestoreFeature).collect().to_polars()
            assert soft_deleted.is_empty()

            replacement_time = datetime.now(timezone.utc) + timedelta(seconds=1)
            replacement = make_value_df(
                sample_uids=["a"],
                values=[2],
                provenance=["p2"],
                created_at=[replacement_time],
            )
            store.write(RestoreFeature, replacement)

            active = store.read(RestoreFeature).collect().to_polars()
            assert active.height == 1
            assert active["value"].to_list() == [2]
            assert active[METAXY_DELETED_AT].is_null().all()

            with_deleted = (
                store.read(RestoreFeature, include_soft_deleted=True, with_sample_history=True).collect().to_polars()
            )
            assert 2 in set(with_deleted["value"])
            assert with_deleted.filter(pl.col("value") == 1).filter(pl.col(METAXY_DELETED_AT).is_not_null()).height >= 1
            assert with_deleted.filter(pl.col("value") == 2)[METAXY_DELETED_AT].is_null().all()

    def test_hard_delete_memory_store_only(
        self,
        store: MetadataStore,
        user_profile_df: pl.DataFrame,
    ) -> None:
        """Hard delete currently implemented for in-memory and delta stores."""

        class UserProfile(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="hard_delete_profile",
                fields=["email", "status"],
            ),
        ):
            email: str | None = None
            status: str | None = None

        with store.open("w"):
            store.write(UserProfile, user_profile_df)
            store.delete(
                UserProfile,
                filters=nw.col("status") == "inactive",
                with_feature_history=True,
            )

        with store:
            remaining = store.read(UserProfile).collect().to_polars()
            assert set(remaining["sample_uid"]) == {"u1", "u2"}

            with_deleted = store.read(UserProfile, include_soft_deleted=True).collect().to_polars()
            assert set(with_deleted["sample_uid"]) == {"u1", "u2", "u3"}

    def test_hard_delete(
        self,
        store: MetadataStore,
        user_profile_df: pl.DataFrame,
    ) -> None:
        """Hard delete removes rows from storage."""

        class UserProfile(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="hard_delete_profile",
                fields=["email", "status"],
            ),
        ):
            email: str | None = None
            status: str | None = None

        with store.open("w"):
            store.write(UserProfile, user_profile_df)
            store.delete(
                UserProfile,
                filters=nw.col("status") == "inactive",
                soft=False,
                with_feature_history=True,
            )

        with store:
            remaining = store.read(UserProfile).collect().to_polars()
            assert remaining.height == 2
            assert set(remaining["sample_uid"]) == {"u1", "u2"}

    def test_hard_delete_historical_version_preserves_latest(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Hard-deleting old versions should keep the latest version intact."""

        class VersionedHardDeleteFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="hard_delete_versions",
                fields=["value"],
            ),
        ):
            value: int | None = None

        newer_time = base_time + timedelta(minutes=5)

        first_version = make_value_df(
            sample_uids=["a"],
            values=[1],
            provenance=["p1"],
            created_at=[base_time],
        )
        latest_version = make_value_df(
            sample_uids=["a"],
            values=[2],
            provenance=["p2"],
            created_at=[newer_time],
        )

        with store.open("w"):
            store.write(VersionedHardDeleteFeature, first_version)
            store.write(VersionedHardDeleteFeature, latest_version)

            store.delete(
                VersionedHardDeleteFeature,
                filters=nw.col(METAXY_CREATED_AT) == base_time,
                soft=False,
                with_feature_history=True,
                with_sample_history=True,
            )

            remaining = store.read(VersionedHardDeleteFeature).collect().to_polars()
            assert remaining.height == 1
            assert remaining["value"].to_list() == [2]
            assert METAXY_DELETED_AT not in remaining.columns or remaining[METAXY_DELETED_AT].is_null().all()

    def test_hard_delete_then_overwrite_restores_row(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Writing new metadata after a hard delete should reintroduce the row."""

        class RestoreHardDeleteFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="hard_delete_restore",
                fields=["value"],
            ),
        ):
            value: int | None = None

        initial = make_value_df(
            sample_uids=["a"],
            values=[1],
            provenance=["p1"],
            created_at=[base_time],
        )

        with store.open("w"):
            store.write(RestoreHardDeleteFeature, initial)
            store.delete(
                RestoreHardDeleteFeature,
                filters=nw.col("sample_uid") == "a",
                soft=False,
                with_feature_history=True,
            )

            after_delete = store.read(RestoreHardDeleteFeature).collect().to_polars()
            assert after_delete.is_empty()

            replacement = make_value_df(
                sample_uids=["a"],
                values=[2],
                provenance=["p2"],
                created_at=[base_time + timedelta(minutes=1)],
            )
            store.write(RestoreHardDeleteFeature, replacement)

            active = store.read(RestoreHardDeleteFeature).collect().to_polars()
            assert active.height == 1
            assert active["value"].to_list() == [2]
            assert METAXY_DELETED_AT not in active.columns or active[METAXY_DELETED_AT].is_null().all()

    def test_soft_delete_preserves_original_created_at(
        self,
        store: MetadataStore,
        base_time: datetime,
        make_value_df: Callable[..., pl.DataFrame],
    ) -> None:
        """Soft delete should preserve the original created_at and set deleted_at.

        When delete with soft=True is called, it:
        1. Reads the existing row (with its original created_at)
        2. Sets metaxy_deleted_at on it
        3. Writes it back (preserving the original created_at)
        """

        class SoftDeleteTimestampFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="soft_delete_timestamp",
                fields=["value"],
            ),
        ):
            value: int | None = None

        initial_df = make_value_df(
            sample_uids=["a"],
            values=[1],
            provenance=["p1"],
            created_at=[base_time],
        )

        with store.open("w"):
            store.write(SoftDeleteTimestampFeature, initial_df)

            # Perform soft delete
            store.delete(
                SoftDeleteTimestampFeature,
                filters=nw.col("sample_uid") == "a",
                soft=True,
            )

            # Read all rows including soft deleted to see the deletion marker
            all_rows = (
                store.read(SoftDeleteTimestampFeature, include_soft_deleted=True, with_sample_history=True)
                .collect()
                .to_polars()
                .sort(METAXY_CREATED_AT)
            )

            # Should have 2 rows: original and soft delete marker
            assert all_rows.height == 2

            # Get the soft delete marker row (the one with deleted_at set)
            soft_delete_row = all_rows.filter(pl.col(METAXY_DELETED_AT).is_not_null())
            assert soft_delete_row.height == 1

            # The created_at should be preserved from the original row
            created_at = soft_delete_row[METAXY_CREATED_AT][0]
            deleted_at = soft_delete_row[METAXY_DELETED_AT][0]
            assert created_at == base_time, f"created_at ({created_at}) should be preserved as {base_time}"
            assert deleted_at > base_time, f"deleted_at ({deleted_at}) should be after base_time ({base_time})"

    # ---- tests from test_delete_filters.py ----

    @pytest.mark.parametrize(
        "case",
        PREDICATE_CASES,
        ids=[case.name for case in PREDICATE_CASES],
    )
    @pytest.mark.parametrize("soft", [True, False], ids=["soft", "hard"])
    def test_delete_accepts_predicate_cases(
        self,
        store: MetadataStore,
        case: PredicateCase,
        delete_filter_feature: type[SampleFeature],
        base_delete_filter_df: pl.DataFrame,
        soft: bool,
    ) -> None:
        with store.open("w"):
            store.write(delete_filter_feature, base_delete_filter_df)
            store.delete(
                delete_filter_feature,
                filters=case.exprs,
                soft=soft,
                with_feature_history=False,
            )

    def test_delete_datetime_filter_hard_delete(
        self,
        store: MetadataStore,
        delete_filter_feature: type[SampleFeature],
        base_delete_filter_df: pl.DataFrame,
    ) -> None:
        cutoff = datetime(2024, 1, 5, tzinfo=timezone.utc)
        with store.open("w"):
            store.write(delete_filter_feature, base_delete_filter_df)
            store.delete(
                delete_filter_feature,
                filters=nw.col("ts_tz") > cutoff,
                soft=False,
                with_feature_history=False,
            )

        with store:
            remaining = store.read(delete_filter_feature).collect().to_polars()
            assert remaining.height == 5

    @pytest.mark.parametrize("soft", [True, False], ids=["soft", "hard"])
    def test_delete_with_none_filters(
        self,
        store: MetadataStore,
        delete_filter_feature: type[SampleFeature],
        base_delete_filter_df: pl.DataFrame,
        soft: bool,
    ) -> None:
        """Test that delete accepts None as filters to delete all records."""
        with store.open("w"):
            store.write(delete_filter_feature, base_delete_filter_df)
            store.delete(
                delete_filter_feature,
                filters=None,
                soft=soft,
                with_feature_history=False,
            )

        with store:
            remaining = store.read(delete_filter_feature).collect().to_polars()
            if soft:
                # Soft delete: records still exist when include_soft_deleted=True
                all_data = store.read(delete_filter_feature, include_soft_deleted=True).collect().to_polars()
                assert all_data.height == 10
                # But not returned by default
                assert remaining.height == 0
            else:
                # Hard delete: records are physically removed
                assert remaining.height == 0
