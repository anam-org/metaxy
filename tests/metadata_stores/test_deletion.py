"""Tests for metadata deletion behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.pytest_helpers import skip_exception
from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DELETED_AT,
    METAXY_PROVENANCE_BY_FIELD,
)


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def make_value_df():
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
        return pl.DataFrame(data)

    return _make


@pytest.fixture
def user_profile_df() -> pl.DataFrame:
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


def test_soft_deleted_rows_filtered_by_default(
    any_store: MetadataStore,
    base_time: datetime,
    make_value_df,
):
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
    soft_deletes_df = make_value_df(
        sample_uids=["a"],
        values=[1],
        provenance=["p_del"],
        created_at=[base_time + timedelta(seconds=1)],
        deleted_at=[base_time + timedelta(seconds=1)],
    )

    with any_store:
        any_store.write_metadata(SoftDeleteFeature, initial_df)
        any_store.write_metadata(SoftDeleteFeature, soft_deletes_df)

        active = any_store.read_metadata(SoftDeleteFeature).collect().to_polars()
        assert active.filter(pl.col("sample_uid") == "a").is_empty()
        assert active[METAXY_DELETED_AT].is_null().all()

        with_deleted = (
            any_store.read_metadata(SoftDeleteFeature, include_soft_deleted=True)
            .collect()
            .to_polars()
        )
        assert set(with_deleted["sample_uid"]) == {"a", "b"}
        deleted_row = with_deleted.filter(pl.col("sample_uid") == "a")
        assert deleted_row[METAXY_DELETED_AT].is_null().any() is False
        active_row = with_deleted.filter(pl.col("sample_uid") == "b")
        assert active_row[METAXY_DELETED_AT].is_null().all()


def test_write_delete_write_sequence(
    any_store: MetadataStore,
    base_time: datetime,
    make_value_df,
):
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

    with any_store:
        any_store.write_metadata(WriteDeleteWriteFeature, write1_df)
        any_store.write_metadata(WriteDeleteWriteFeature, delete_df)
        any_store.write_metadata(WriteDeleteWriteFeature, write2_df)

        active = any_store.read_metadata(WriteDeleteWriteFeature).collect().to_polars()
        assert len(active) == 1
        assert active["sample_uid"][0] == "x"
        assert active["value"][0] == 200
        assert active[METAXY_DELETED_AT].is_null().all()

        with_deleted = (
            any_store.read_metadata(WriteDeleteWriteFeature, include_soft_deleted=True)
            .collect()
            .to_polars()
            .sort(METAXY_CREATED_AT)
        )
        assert len(with_deleted) == 1
        assert with_deleted["sample_uid"][0] == "x"
        assert with_deleted["value"][0] == 200
        assert with_deleted[METAXY_DELETED_AT].is_null().all()


def test_write_delete_write_delete_sequence(
    any_store: MetadataStore,
    base_time: datetime,
    make_value_df,
):
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

    delete1_df = make_value_df(
        sample_uids=["y"],
        values=[100],
        provenance=["p_delete1"],
        created_at=[base_time + timedelta(seconds=1)],
        deleted_at=[base_time + timedelta(seconds=1)],
    )

    write2_df = make_value_df(
        sample_uids=["y"],
        values=[200],
        provenance=["p_write2"],
        created_at=[base_time + timedelta(seconds=2)],
    )

    delete2_df = make_value_df(
        sample_uids=["y"],
        values=[200],
        provenance=["p_delete2"],
        created_at=[base_time + timedelta(seconds=3)],
        deleted_at=[base_time + timedelta(seconds=3)],
    )

    with any_store:
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, write1_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, delete1_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, write2_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, delete2_df)

        active = (
            any_store.read_metadata(WriteDeleteWriteDeleteFeature).collect().to_polars()
        )
        assert active.is_empty()

        with_deleted = (
            any_store.read_metadata(
                WriteDeleteWriteDeleteFeature, include_soft_deleted=True
            )
            .collect()
            .to_polars()
        )
        assert len(with_deleted) == 1
        assert with_deleted["sample_uid"][0] == "y"
        assert with_deleted["value"][0] == 200
        assert with_deleted[METAXY_DELETED_AT].is_not_null().all()


@skip_exception(NotImplementedError, "unsupported delete store")
def test_soft_delete_historical_version_preserves_latest(
    any_store: MetadataStore,
    base_time: datetime,
    make_value_df,
):
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

    with any_store:
        any_store.write_metadata(VersionedFeature, first_version)
        any_store.write_metadata(VersionedFeature, latest_version)

        any_store.delete_metadata(
            VersionedFeature,
            filters=nw.col(METAXY_CREATED_AT) == base_time,
            current_only=False,
            latest_only=False,
        )

        active = any_store.read_metadata(VersionedFeature).collect().to_polars()
        assert active.is_empty()

        with_deleted = (
            any_store.read_metadata(
                VersionedFeature, include_soft_deleted=True, latest_only=False
            )
            .collect()
            .to_polars()
        )
        non_deleted_values = with_deleted.filter(pl.col(METAXY_DELETED_AT).is_null())[
            "value"
        ].to_list()
        assert 2 in non_deleted_values
        assert (
            with_deleted.filter(pl.col(METAXY_DELETED_AT).is_not_null())
            .filter(pl.col("value") == 1)
            .height
            >= 1
        )


@skip_exception(NotImplementedError, "unsupported delete store")
def test_soft_delete_then_overwrite_restores_row(
    any_store: MetadataStore,
    base_time: datetime,
    make_value_df,
):
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

    with any_store:
        any_store.write_metadata(RestoreFeature, initial)
        any_store.delete_metadata(RestoreFeature, filters=nw.col("sample_uid") == "a")

        soft_deleted = any_store.read_metadata(RestoreFeature).collect().to_polars()
        assert soft_deleted.is_empty()

        replacement_time = datetime.now(timezone.utc) + timedelta(seconds=1)
        replacement = make_value_df(
            sample_uids=["a"],
            values=[2],
            provenance=["p2"],
            created_at=[replacement_time],
        )
        any_store.write_metadata(RestoreFeature, replacement)

        active = any_store.read_metadata(RestoreFeature).collect().to_polars()
        assert active.height == 1
        assert active["value"].to_list() == [2]
        assert active[METAXY_DELETED_AT].is_null().all()

        with_deleted = (
            any_store.read_metadata(
                RestoreFeature, include_soft_deleted=True, latest_only=False
            )
            .collect()
            .to_polars()
        )
        assert 2 in set(with_deleted["value"])
        assert (
            with_deleted.filter(pl.col("value") == 1)
            .filter(pl.col(METAXY_DELETED_AT).is_not_null())
            .height
            >= 1
        )
        assert (
            with_deleted.filter(pl.col("value") == 2)[METAXY_DELETED_AT].is_null().all()
        )


@skip_exception(NotImplementedError, "unsupported delete store")
def test_hard_delete_memory_store_only(
    any_store: MetadataStore,
    user_profile_df: pl.DataFrame,
):
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

    with any_store.open("write"):
        any_store.write_metadata(UserProfile, user_profile_df)
        any_store.delete_metadata(
            UserProfile,
            filters=nw.col("status") == "inactive",
            current_only=False,
        )

    with any_store:
        remaining = any_store.read_metadata(UserProfile).collect().to_polars()
        assert set(remaining["sample_uid"]) == {"u1", "u2"}

        with_deleted = (
            any_store.read_metadata(UserProfile, include_soft_deleted=True)
            .collect()
            .to_polars()
        )
        assert set(with_deleted["sample_uid"]) == {"u1", "u2", "u3"}


@skip_exception(NotImplementedError, "unsupported delete store")
def test_hard_delete(
    any_store: MetadataStore,
    user_profile_df: pl.DataFrame,
):
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

    with any_store.open("write"):
        any_store.write_metadata(UserProfile, user_profile_df)
        any_store.delete_metadata(
            UserProfile,
            filters=nw.col("status") == "inactive",
            soft=False,
            current_only=False,
        )

    with any_store:
        remaining = any_store.read_metadata(UserProfile).collect().to_polars()
        assert remaining.height == 2
        assert set(remaining["sample_uid"]) == {"u1", "u2"}


@skip_exception(NotImplementedError, "unsupported delete store")
def test_hard_delete_historical_version_preserves_latest(
    any_store: MetadataStore,
    base_time: datetime,
    make_value_df,
):
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

    with any_store:
        any_store.write_metadata(VersionedHardDeleteFeature, first_version)
        any_store.write_metadata(VersionedHardDeleteFeature, latest_version)

        any_store.delete_metadata(
            VersionedHardDeleteFeature,
            filters=nw.col(METAXY_CREATED_AT) == base_time,
            soft=False,
            current_only=False,
            latest_only=False,
        )

        remaining = (
            any_store.read_metadata(VersionedHardDeleteFeature).collect().to_polars()
        )
        assert remaining.height == 1
        assert remaining["value"].to_list() == [2]
        assert (
            METAXY_DELETED_AT not in remaining.columns
            or remaining[METAXY_DELETED_AT].is_null().all()
        )


@skip_exception(NotImplementedError, "unsupported delete store")
def test_hard_delete_then_overwrite_restores_row(
    any_store: MetadataStore,
    base_time: datetime,
    make_value_df,
):
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

    with any_store:
        any_store.write_metadata(RestoreHardDeleteFeature, initial)
        any_store.delete_metadata(
            RestoreHardDeleteFeature,
            filters=nw.col("sample_uid") == "a",
            soft=False,
            current_only=False,
        )

        after_delete = (
            any_store.read_metadata(RestoreHardDeleteFeature).collect().to_polars()
        )
        assert after_delete.is_empty()

        replacement = make_value_df(
            sample_uids=["a"],
            values=[2],
            provenance=["p2"],
            created_at=[base_time + timedelta(minutes=1)],
        )
        any_store.write_metadata(RestoreHardDeleteFeature, replacement)

        active = any_store.read_metadata(RestoreHardDeleteFeature).collect().to_polars()
        assert active.height == 1
        assert active["value"].to_list() == [2]
        assert (
            METAXY_DELETED_AT not in active.columns
            or active[METAXY_DELETED_AT].is_null().all()
        )
