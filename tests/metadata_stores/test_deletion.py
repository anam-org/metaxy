"""Tests for metadata deletion behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import narwhals as nw
import polars as pl

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.pytest_helpers import skip_exception
from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DELETED_AT,
    METAXY_PROVENANCE_BY_FIELD,
)


def _require_supported_delete_store(
    any_store: MetadataStore, supported: set[str], reason: str
) -> None:
    if any_store.__class__.__name__ not in supported:
        raise NotImplementedError(reason)


def test_soft_deleted_rows_filtered_by_default(any_store: MetadataStore):
    """Soft-deleted metadata rows should be hidden by default and opt-in with include_deleted."""

    class SoftDeleteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="soft_delete",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    initial_df = pl.DataFrame(
        {
            "sample_uid": ["a", "b"],
            "value": [1, 2],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p1"}, {"value": "p2"}],
            METAXY_CREATED_AT: [base_time, base_time],
        }
    )

    soft_deletes_df = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [1],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_del"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=1)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=1)],
        }
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


def test_write_delete_write_sequence(any_store: MetadataStore):
    """Test write->delete->write sequence: latest write is visible, soft deletes is hidden by default."""

    class WriteDeleteWriteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="write_delete_write",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # First write
    write1_df = pl.DataFrame(
        {
            "sample_uid": ["x"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )

    # Delete (soft delete)
    delete_df = pl.DataFrame(
        {
            "sample_uid": ["x"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_delete"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=1)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=1)],
        }
    )

    # Second write (resurrection)
    write2_df = pl.DataFrame(
        {
            "sample_uid": ["x"],
            "value": [200],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write2"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=2)],
        }
    )

    with any_store:
        any_store.write_metadata(WriteDeleteWriteFeature, write1_df)
        any_store.write_metadata(WriteDeleteWriteFeature, delete_df)
        any_store.write_metadata(WriteDeleteWriteFeature, write2_df)

        # Default: should see only the latest write (value=200)
        active = any_store.read_metadata(WriteDeleteWriteFeature).collect().to_polars()
        assert len(active) == 1
        assert active["sample_uid"][0] == "x"
        assert active["value"][0] == 200
        assert active[METAXY_DELETED_AT].is_null().all()

        # With deleted: should see latest (write2) because deduplication picks it over soft deletes
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


def test_write_delete_write_delete_sequence(any_store: MetadataStore):
    """Test write->delete->write->delete: no active rows by default, latest write visible with filter."""

    class WriteDeleteWriteDeleteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="write_delete_write_delete",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # First write
    write1_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )

    # First delete
    delete1_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [100],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_delete1"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=1)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=1)],
        }
    )

    # Second write
    write2_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [200],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_write2"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=2)],
        }
    )

    # Second delete
    delete2_df = pl.DataFrame(
        {
            "sample_uid": ["y"],
            "value": [200],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p_delete2"}],
            METAXY_CREATED_AT: [base_time + timedelta(seconds=3)],
            METAXY_DELETED_AT: [base_time + timedelta(seconds=3)],
        }
    )

    with any_store:
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, write1_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, delete1_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, write2_df)
        any_store.write_metadata(WriteDeleteWriteDeleteFeature, delete2_df)

        # Default: no active rows (latest is a soft delete)
        active = (
            any_store.read_metadata(WriteDeleteWriteDeleteFeature).collect().to_polars()
        )
        assert active.is_empty()

        # With deleted: should see the latest soft delete (value=200 deleted)
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
def test_soft_delete_historical_version_preserves_latest(any_store: MetadataStore):
    """Soft-deleting old versions should not hide the current version."""

    _require_supported_delete_store(
        any_store,
        {"ClickHouseMetadataStore", "DuckDBMetadataStore", "InMemoryMetadataStore"},
        "Soft delete restoration pending hard delete support for non-memory backends",
    )

    class VersionedFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="soft_delete_versions",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    newer_time = base_time + timedelta(minutes=5)

    first_version = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [1],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )
    latest_version = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [2],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p2"}],
            METAXY_CREATED_AT: [newer_time],
        }
    )

    with any_store:
        any_store.write_metadata(VersionedFeature, first_version)
        any_store.write_metadata(VersionedFeature, latest_version)

        any_store.delete_metadata(
            VersionedFeature,
            filters=nw.col(METAXY_CREATED_AT) == base_time,
            current_only=False,
        )

        active = any_store.read_metadata(VersionedFeature).collect().to_polars()
        assert active.height == 1
        assert active["value"].to_list() == [2]
        assert active[METAXY_DELETED_AT].is_null().all()

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
def test_soft_delete_then_overwrite_restores_row(any_store: MetadataStore):
    """Writing new metadata after a soft delete should make the row active again."""

    _require_supported_delete_store(
        any_store,
        {"ClickHouseMetadataStore", "DuckDBMetadataStore", "InMemoryMetadataStore"},
        "Soft delete restoration pending hard delete support for non-memory backends",
    )

    class RestoreFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="soft_delete_restore",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    restored_time = base_time + timedelta(minutes=1)

    initial = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [1],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )
    replacement = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [2],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p2"}],
            METAXY_CREATED_AT: [restored_time],
        }
    )

    with any_store:
        any_store.write_metadata(RestoreFeature, initial)
        any_store.delete_metadata(RestoreFeature, filters=nw.col("sample_uid") == "a")

        soft_deleted = any_store.read_metadata(RestoreFeature).collect().to_polars()
        assert soft_deleted.is_empty()

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
def test_hard_delete_memory_store_only(any_store: MetadataStore):
    """Hard delete currently implemented for in-memory and delta stores."""

    _require_supported_delete_store(
        any_store,
        {"ClickHouseMetadataStore", "DuckDBMetadataStore", "InMemoryMetadataStore"},
        "Hard delete pending for non-memory backends",
    )

    class UserProfile(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="hard_delete_profile",
            fields=["email", "status"],
        ),
    ):
        email: str | None = None
        status: str | None = None

    df = pl.DataFrame(
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

    with any_store.open("write"):
        any_store.write_metadata(UserProfile, df)
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
def test_hard_delete(any_store: MetadataStore):
    """Hard delete removes rows from storage."""

    _require_supported_delete_store(
        any_store,
        {"ClickHouseMetadataStore", "DuckDBMetadataStore", "InMemoryMetadataStore"},
        "Hard delete pending for non-memory backends",
    )

    class UserProfile(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="hard_delete_profile",
            fields=["email", "status"],
        ),
    ):
        email: str | None = None
        status: str | None = None

    df = pl.DataFrame(
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

    with any_store.open("write"):
        any_store.write_metadata(UserProfile, df)
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
def test_hard_delete_historical_version_preserves_latest(any_store: MetadataStore):
    """Hard-deleting old versions should keep the latest version intact."""

    _require_supported_delete_store(
        any_store,
        {"ClickHouseMetadataStore", "DuckDBMetadataStore", "InMemoryMetadataStore"},
        "Hard delete pending for non-memory backends",
    )

    class VersionedHardDeleteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="hard_delete_versions",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    newer_time = base_time + timedelta(minutes=5)

    first_version = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [1],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )
    latest_version = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [2],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p2"}],
            METAXY_CREATED_AT: [newer_time],
        }
    )

    with any_store:
        any_store.write_metadata(VersionedHardDeleteFeature, first_version)
        any_store.write_metadata(VersionedHardDeleteFeature, latest_version)

        any_store.delete_metadata(
            VersionedHardDeleteFeature,
            filters=nw.col(METAXY_CREATED_AT) == base_time,
            soft=False,
            current_only=False,
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
def test_hard_delete_then_overwrite_restores_row(any_store: MetadataStore):
    """Writing new metadata after a hard delete should reintroduce the row."""

    _require_supported_delete_store(
        any_store,
        {"ClickHouseMetadataStore", "DuckDBMetadataStore", "InMemoryMetadataStore"},
        "Hard delete pending for non-memory backends",
    )

    class RestoreHardDeleteFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="hard_delete_restore",
            fields=["value"],
        ),
    ):
        value: int | None = None

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    restored_time = base_time + timedelta(minutes=1)

    initial = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [1],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p1"}],
            METAXY_CREATED_AT: [base_time],
        }
    )
    replacement = pl.DataFrame(
        {
            "sample_uid": ["a"],
            "value": [2],
            METAXY_PROVENANCE_BY_FIELD: [{"value": "p2"}],
            METAXY_CREATED_AT: [restored_time],
        }
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

        any_store.write_metadata(RestoreHardDeleteFeature, replacement)

        active = any_store.read_metadata(RestoreHardDeleteFeature).collect().to_polars()
        assert active.height == 1
        assert active["value"].to_list() == [2]
        assert (
            METAXY_DELETED_AT not in active.columns
            or active[METAXY_DELETED_AT].is_null().all()
        )
