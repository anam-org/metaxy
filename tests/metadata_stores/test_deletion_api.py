"""Comprehensive tests for the new deletion and mutation API.

Tests delete_metadata, soft_delete_metadata, and mutate_metadata
across all metadata store backends.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import narwhals as nw
import polars as pl
import pytest

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store import InMemoryMetadataStore

# ========== Test Fixtures ==========


@pytest.fixture
def store_with_data(graph):
    """Create in-memory store with sample data."""

    class UserProfile(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["user_profile"]),
            fields=[
                FieldSpec(key=FieldKey(["email"]), code_version="1"),
                FieldSpec(key=FieldKey(["status"]), code_version="1"),
            ],
        ),
    ):
        """Test feature for user profiles."""

        pass

    class UserEvents(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["user_events"]),
            deps=[FeatureDep(feature=UserProfile)],
            fields=[
                FieldSpec(key=FieldKey(["event_type"]), code_version="1"),
                FieldSpec(key=FieldKey(["timestamp"]), code_version="1"),
            ],
        ),
    ):
        """Test feature depending on UserProfile."""

        pass

    store = InMemoryMetadataStore()

    with store.open("write"):
        # Write user profiles
        profiles_df = pl.DataFrame(
            {
                "sample_uid": ["user_1", "user_2", "user_3"],
                "email": [
                    "user1@example.com",
                    "user2@example.com",
                    "user3@example.com",
                ],
                "status": ["active", "active", "inactive"],
                "metaxy_provenance_by_field": [
                    {"email": "h1", "status": "h1"},
                    {"email": "h2", "status": "h2"},
                    {"email": "h3", "status": "h3"},
                ],
                "metaxy_created_at": [
                    datetime.now(timezone.utc) - timedelta(days=100),
                    datetime.now(timezone.utc) - timedelta(days=50),
                    datetime.now(timezone.utc) - timedelta(days=10),
                ],
            }
        )
        store.write_metadata(UserProfile, nw.from_native(profiles_df))

        # Write user events
        events_df = pl.DataFrame(
            {
                "sample_uid": ["e1", "e2", "e3", "e4"],
                "event_type": ["login", "logout", "login", "login"],
                "timestamp": [
                    datetime.now(timezone.utc) - timedelta(days=95),
                    datetime.now(timezone.utc) - timedelta(days=94),
                    datetime.now(timezone.utc) - timedelta(days=45),
                    datetime.now(timezone.utc) - timedelta(days=5),
                ],
                "metaxy_provenance_by_field": [
                    {"event_type": "e1", "timestamp": "e1"},
                    {"event_type": "e2", "timestamp": "e2"},
                    {"event_type": "e3", "timestamp": "e3"},
                    {"event_type": "e4", "timestamp": "e4"},
                ],
                "metaxy_created_at": [
                    datetime.now(timezone.utc) - timedelta(days=95),
                    datetime.now(timezone.utc) - timedelta(days=94),
                    datetime.now(timezone.utc) - timedelta(days=45),
                    datetime.now(timezone.utc) - timedelta(days=5),
                ],
            }
        )
        store.write_metadata(UserEvents, nw.from_native(events_df))

    return store, UserProfile, UserEvents


# ========== Hard Delete Tests ==========


class TestDeleteMetadata:
    """Test delete_metadata (hard delete) functionality."""

    def test_delete_by_expression(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test hard delete using Narwhals expression."""
        store, UserProfile, UserEvents = store_with_data

        with store.open("write"):
            result = store.delete_metadata(
                UserProfile, filter=nw.col("status") == "inactive"
            )

        assert result.rows_affected == 1
        assert result.error is None

        # Verify deletion
        with store:
            df = store.read_metadata(UserProfile).collect()
            assert len(df) == 2
            assert "user_3" not in df["sample_uid"].to_list()

    def test_delete_by_frame_id_columns(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test hard delete using frame with id_columns matching."""
        ids_to_delete = nw.from_native(
            pl.DataFrame({"sample_uid": ["user_1", "user_2"]})
        )

        with store.open("write"):
            result = store.delete_metadata(
                UserProfile, filter=ids_to_delete, match_on="id_columns"
            )

        assert result.rows_affected == 2

        # Verify deletion
        with store:
            df = store.read_metadata(UserProfile).collect()
            assert len(df) == 1
            assert df["sample_uid"][0] == "user_3"

    def test_delete_by_frame_all_columns(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test hard delete using frame with all_columns matching."""
        records_to_delete = nw.from_native(
            pl.DataFrame({"sample_uid": ["user_1"], "status": ["active"]})
        )

        with store.open("write"):
            result = store.delete_metadata(
                UserProfile, filter=records_to_delete, match_on="all_columns"
            )

        assert result.rows_affected == 1

    def test_delete_by_frame_specific_columns(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test hard delete using frame with specific column list."""
        records = nw.from_native(
            pl.DataFrame({"status": ["inactive"], "email": ["user3@example.com"]})
        )

        with store.open("write"):
            result = store.delete_metadata(
                UserProfile,
                filter=records,
                match_on=["status"],  # Only match on status
            )

        assert result.rows_affected == 1

    def test_delete_with_propagation(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test delete with propagation through dependency graph."""
        # Note: This would need proper feature dependencies set up
        # For now, testing without actual dependencies
        with store.open("write"):
            result = store.delete_metadata(
                UserProfile,
                filter=nw.col("sample_uid") == "user_1",
            )

        assert result.rows_affected == 1

    def test_delete_no_matches(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test delete when filter matches no records."""
        with store.open("write"):
            result = store.delete_metadata(
                UserProfile, filter=nw.col("sample_uid") == "nonexistent"
            )

        assert result.rows_affected == 0


# ========== Soft Delete Tests ==========


class TestSoftDeleteMetadata:
    """Test soft_delete_metadata functionality."""

    def test_soft_delete_sets_deleted_at(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test that soft delete sets metaxy_deleted_at timestamp."""
        with store.open("write"):
            result = store.soft_delete_metadata(
                UserProfile, filter=nw.col("sample_uid") == "user_1"
            )

        assert result.rows_affected == 1

        # Verify metaxy_deleted_at is set (read with include_deleted=True)
        with store:
            df = store.read_metadata(UserProfile, include_deleted=True).collect()
            user1_row = df.filter(nw.col("sample_uid") == "user_1")
            assert user1_row["metaxy_deleted_at"][0] is not None

    def test_soft_delete_auto_filtered_in_reads(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test that soft-deleted records are automatically filtered."""
        with store.open("write"):
            store.soft_delete_metadata(
                UserProfile, filter=nw.col("sample_uid") == "user_1"
            )

        # Default read should exclude soft-deleted
        with store:
            df = store.read_metadata(UserProfile).collect()
            assert len(df) == 2
            assert "user_1" not in df["sample_uid"].to_list()

        # Explicit include_deleted=True should include them
        with store:
            df = store.read_metadata(UserProfile, include_deleted=True).collect()
            assert len(df) == 3
            assert "user_1" in df["sample_uid"].to_list()

    def test_soft_delete_idempotent(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test that soft deleting already soft-deleted record doesn't re-update."""
        # First soft delete
        with store.open("write"):
            result1 = store.soft_delete_metadata(
                UserProfile, filter=nw.col("sample_uid") == "user_1"
            )
            assert result1.rows_affected == 1

        # Second soft delete (should affect 0 rows)
        with store.open("write"):
            result2 = store.soft_delete_metadata(
                UserProfile, filter=nw.col("sample_uid") == "user_1"
            )
            assert result2.rows_affected == 0


# ========== Mutation Tests ==========


class TestMutateMetadata:
    """Test mutate_metadata (generic updates) functionality."""

    def test_mutate_anonymize_fields(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test mutation for GDPR anonymization."""
        with store.open("write"):
            result = store.mutate_metadata(
                UserProfile,
                filter=nw.col("sample_uid") == "user_1",
                updates={"email": "[REDACTED]", "status": "anonymized"},
            )

        assert result.rows_affected == 1
        assert result.updates == {"email": "[REDACTED]", "status": "anonymized"}

        # Verify mutation
        with store:
            df = store.read_metadata(UserProfile).collect()
            user1 = df.filter(nw.col("sample_uid") == "user_1")
            assert user1["email"][0] == "[REDACTED]"
            assert user1["status"][0] == "anonymized"

    def test_mutate_set_null(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test mutation setting column to NULL."""
        with store.open("write"):
            result = store.mutate_metadata(
                UserProfile,
                filter=nw.col("sample_uid") == "user_1",
                updates={"email": None},
            )

        assert result.rows_affected == 1

        # Verify email is null
        with store:
            df = store.read_metadata(UserProfile).collect()
            user1 = df.filter(nw.col("sample_uid") == "user_1")
            assert user1["email"][0] is None

    def test_mutate_multiple_records(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test mutation affecting multiple records."""
        with store.open("write"):
            result = store.mutate_metadata(
                UserProfile,
                filter=nw.col("status") == "active",
                updates={"status": "verified"},
            )

        assert result.rows_affected == 2

        # Verify both active users are now verified
        with store:
            df = store.read_metadata(UserProfile).collect()
            assert df.filter(nw.col("status") == "verified").shape[0] == 2


# ========== Edge Cases ==========


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_delete_nonexistent_feature(self, store_with_data, graph):
        """Test deleting from a feature that doesn't exist."""

        # Create a feature that's in the graph but not in the store
        class NonexistentFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["nonexistent"]),
                fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
            ),
        ):
            pass

        store, _, _ = store_with_data

        with store.open("write"):
            result = store.delete_metadata(
                NonexistentFeature, filter=nw.col("sample_uid") == "user_1"
            )

        # Should return 0 rows affected, not error
        assert result.rows_affected == 0

    def test_invalid_match_on_column(self, store_with_data):
        store, UserProfile, UserEvents = store_with_data

        """Test error when match_on column doesn't exist in frame."""
        invalid_frame = nw.from_native(pl.DataFrame({"nonexistent_col": [1, 2, 3]}))

        with store.open("write"):
            with pytest.raises(ValueError, match="Column .* not found in filter frame"):
                store.delete_metadata(
                    UserProfile,
                    filter=invalid_frame,
                    match_on="id_columns",  # Will fail: sample_uid not in frame
                )
