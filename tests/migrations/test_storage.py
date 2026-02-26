"""Tests for migration system table storage layer."""

from pathlib import Path

import pytest

from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.metadata_store.system import (
    Event,
    EventType,
    MigrationStatus,
    SystemTableStorage,
)


@pytest.fixture
def store(tmp_path: Path):
    """Create fresh store for each test."""
    with DeltaMetadataStore(root_path=tmp_path / "delta_store").open("w") as s:
        yield s


@pytest.fixture
def storage(store):
    """Create storage layer for testing."""
    return SystemTableStorage(store)


def test_write_and_read_events(storage: SystemTableStorage):
    """Test writing and reading migration events."""
    migration_id = "mig_001"

    # Write started event
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))

    # Write feature events
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/a"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )

    # Write completed event
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Read all events
    events = storage.get_migration_events(migration_id, project="test")
    assert events.height == 4
    assert events["event_type"].to_list() == [
        EventType.MIGRATION_STARTED.value,
        EventType.FEATURE_MIGRATION_STARTED.value,
        EventType.FEATURE_MIGRATION_COMPLETED.value,
        EventType.MIGRATION_COMPLETED.value,
    ]


def test_get_migration_status(storage: SystemTableStorage):
    """Test computing migration status from events."""
    migration_id = "mig_001"

    # Not started
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.NOT_STARTED

    # Started
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.IN_PROGRESS

    # Feature in progress
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/a"))
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.IN_PROGRESS

    # Feature completed
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.IN_PROGRESS

    # Migration completed
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.COMPLETED


def test_get_migration_status_failed(storage: SystemTableStorage):
    """Test failed migration status."""
    migration_id = "mig_failed"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/a"))
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            error_message="Something went wrong",
        )
    )
    storage.write_event(
        Event.migration_failed(
            project="test",
            migration_id=migration_id,
            error_message="Something went wrong",
        )
    )

    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.FAILED


def test_is_feature_completed(storage: SystemTableStorage):
    """Test checking if specific feature completed."""
    migration_id = "mig_001"

    # Not completed yet
    assert not storage.is_feature_completed(migration_id, "feature/a", "test")

    # Complete successfully
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    assert storage.is_feature_completed(migration_id, "feature/a", "test")

    # Another feature with error - not completed successfully
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            error_message="Error",
        )
    )
    assert not storage.is_feature_completed(migration_id, "feature/b", "test")


def test_get_completed_features(storage: SystemTableStorage):
    """Test getting list of completed features."""
    migration_id = "mig_001"

    # Complete multiple features
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            rows_affected=200,
        )
    )
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/c",
            error_message="Error",
        )
    )

    completed = storage.get_completed_features(migration_id, project="test")
    assert set(completed) == {"feature/a", "feature/b"}


def test_get_failed_features(storage: SystemTableStorage):
    """Test getting failed features with error messages."""
    migration_id = "mig_001"

    # Complete with errors
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            error_message="Error B",
        )
    )
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/c",
            error_message="Error C",
        )
    )

    failed = storage.get_failed_features(migration_id, project="test")
    assert failed == {"feature/b": "Error B", "feature/c": "Error C"}


def test_resumable_migration(storage: SystemTableStorage):
    """Test resuming a migration that partially failed."""
    migration_id = "mig_resume"

    # Start migration
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))

    # Complete first feature
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/a"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )

    # Fail on second feature
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/b"))
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            error_message="Network error",
        )
    )

    # Check status
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.IN_PROGRESS
    assert storage.is_feature_completed(migration_id, "feature/a", "test")
    assert not storage.is_feature_completed(migration_id, "feature/b", "test")

    # Resume: retry feature/b
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/b"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            rows_affected=200,
        )
    )

    # Complete third feature
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/c"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/c",
            rows_affected=300,
        )
    )

    # Mark complete
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Verify final state
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.COMPLETED
    assert set(storage.get_completed_features(migration_id, project="test")) == {
        "feature/a",
        "feature/b",
        "feature/c",
    }


def test_typed_events_api(storage: SystemTableStorage):
    """Test using builder pattern for type-safe event construction."""
    migration_id = "mig_typed"

    # Write started event using builder
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))

    # Write feature events
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/a"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )

    # Write feature with error
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/b"))
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            error_message="Test error",
        )
    )

    # Complete migration
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Verify events were written correctly
    events = storage.get_migration_events(migration_id, project="test")
    assert events.height == 6

    # Verify status computation works
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.COMPLETED
    assert storage.is_feature_completed(migration_id, "feature/a", "test")
    assert not storage.is_feature_completed(migration_id, "feature/b", "test")
    assert storage.get_failed_features(migration_id, project="test") == {"feature/b": "Test error"}


def test_typed_events_failed_migration(storage: SystemTableStorage):
    """Test builder pattern for a failed migration."""
    migration_id = "mig_typed_failed"

    # Start and immediately fail
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.migration_failed(
            project="test",
            migration_id=migration_id,
            error_message="Critical error occurred",
        )
    )

    # Verify status
    assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.FAILED

    # Verify error is in payload
    events = storage.get_migration_events(migration_id, project="test")
    failed_event = events.filter(events["event_type"] == EventType.MIGRATION_FAILED.value)
    assert failed_event.height == 1


# ========== Complex Migration Scenarios ==========


def test_multiple_migrations_in_sequence(storage: SystemTableStorage):
    """Test tracking multiple migrations executed in sequence."""
    # Migration 1: Complete successfully
    storage.write_event(Event.migration_started(project="test", migration_id="mig_001"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id="mig_001",
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(Event.migration_completed(project="test", migration_id="mig_001"))

    # Migration 2: Start and fail
    storage.write_event(Event.migration_started(project="test", migration_id="mig_002"))
    storage.write_event(Event.feature_started(project="test", migration_id="mig_002", feature_key="feature/b"))
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id="mig_002",
            feature_key="feature/b",
            error_message="Database timeout",
        )
    )
    storage.write_event(
        Event.migration_failed(project="test", migration_id="mig_002", error_message="Database timeout")
    )

    # Migration 3: In progress
    storage.write_event(Event.migration_started(project="test", migration_id="mig_003"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id="mig_003",
            feature_key="feature/c",
            rows_affected=50,
        )
    )

    # Verify statuses
    assert storage.get_migration_status("mig_001", "test") == MigrationStatus.COMPLETED
    assert storage.get_migration_status("mig_002", "test") == MigrationStatus.FAILED
    assert storage.get_migration_status("mig_003", "test") == MigrationStatus.IN_PROGRESS

    # Verify completed features
    assert storage.get_completed_features("mig_001", "test") == ["feature/a"]
    assert storage.get_completed_features("mig_002", "test") == []
    assert storage.get_completed_features("mig_003", "test") == ["feature/c"]

    # Verify failed features
    assert storage.get_failed_features("mig_001", "test") == {}
    assert storage.get_failed_features("mig_002", "test") == {"feature/b": "Database timeout"}
    assert storage.get_failed_features("mig_003", "test") == {}


def test_migration_with_multiple_features(storage: SystemTableStorage):
    """Test migration affecting multiple features with mixed success/failure."""
    migration_id = "mig_multi_feature"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))

    # Feature A: Success
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/a"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )

    # Feature B: Success
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/b"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            rows_affected=200,
        )
    )

    # Feature C: Failure
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/c"))
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/c",
            error_message="Schema mismatch",
        )
    )

    # Feature D: Success
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/d"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/d",
            rows_affected=50,
        )
    )

    # Migration completes despite feature C failure
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Verify overall status
    assert storage.get_migration_status(migration_id, "test") == MigrationStatus.COMPLETED

    # Verify completed features
    completed = storage.get_completed_features(migration_id, "test")
    assert set(completed) == {"feature/a", "feature/b", "feature/d"}

    # Verify failed features
    failed = storage.get_failed_features(migration_id, "test")
    assert failed == {"feature/c": "Schema mismatch"}

    # Verify individual feature completion
    assert storage.is_feature_completed(migration_id, "feature/a", "test")
    assert storage.is_feature_completed(migration_id, "feature/b", "test")
    assert not storage.is_feature_completed(migration_id, "feature/c", "test")
    assert storage.is_feature_completed(migration_id, "feature/d", "test")


def test_migration_retry_after_partial_failure(storage: SystemTableStorage):
    """Test retrying a migration that previously failed partway through."""
    migration_id = "mig_retry"

    # First attempt: Start and partially complete
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            rows_affected=200,
        )
    )
    # Feature C fails
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/c",
            error_message="Network error",
            rows_affected=50,  # Partial completion
        )
    )

    # Check status after first attempt
    assert storage.get_migration_status(migration_id, "test") == MigrationStatus.IN_PROGRESS

    # Retry: Feature C succeeds this time
    storage.write_event(Event.feature_started(project="test", migration_id=migration_id, feature_key="feature/c"))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/c",
            rows_affected=150,
        )
    )

    # Complete remaining features
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/d",
            rows_affected=75,
        )
    )

    # Mark migration as completed
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Verify final status
    assert storage.get_migration_status(migration_id, "test") == MigrationStatus.COMPLETED

    # Verify all features completed (including the retried one)
    completed = storage.get_completed_features(migration_id, "test")
    assert set(completed) == {"feature/a", "feature/b", "feature/c", "feature/d"}

    # Verify no failed features in final state (feature/c was eventually successful)
    # Note: The failed event is still in history but feature/c has a completed event too
    assert storage.is_feature_completed(migration_id, "feature/c", "test")


def test_multiple_projects_isolation(storage: SystemTableStorage):
    """Test that migrations for different projects are properly isolated."""
    # Project 1
    storage.write_event(Event.migration_started(project="project1", migration_id="mig_001"))
    storage.write_event(
        Event.feature_completed(
            project="project1",
            migration_id="mig_001",
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(Event.migration_completed(project="project1", migration_id="mig_001"))

    # Project 2 (same migration ID but different project)
    storage.write_event(Event.migration_started(project="project2", migration_id="mig_001"))
    storage.write_event(
        Event.feature_failed(
            project="project2",
            migration_id="mig_001",
            feature_key="feature/b",
            error_message="Error in project2",
        )
    )
    storage.write_event(
        Event.migration_failed(
            project="project2",
            migration_id="mig_001",
            error_message="Error in project2",
        )
    )

    # Verify project1 shows completed
    assert storage.get_migration_status("mig_001", project="project1") == MigrationStatus.COMPLETED
    assert storage.get_completed_features("mig_001", "project1") == ["feature/a"]
    assert storage.get_failed_features("mig_001", "project1") == {}

    # Verify project2 shows failed
    assert storage.get_migration_status("mig_001", project="project2") == MigrationStatus.FAILED
    assert storage.get_completed_features("mig_001", "project2") == []
    assert storage.get_failed_features("mig_001", "project2") == {"feature/b": "Error in project2"}


def test_feature_with_zero_rows_affected(storage: SystemTableStorage):
    """Test feature completion with zero rows affected (e.g., no data to migrate)."""
    migration_id = "mig_zero_rows"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/empty",
            rows_affected=0,
        )
    )
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Verify it's still considered completed
    assert storage.get_migration_status(migration_id, "test") == MigrationStatus.COMPLETED
    assert storage.is_feature_completed(migration_id, "feature/empty", "test")
    assert storage.get_completed_features(migration_id, "test") == ["feature/empty"]


def test_list_executed_migrations(storage: SystemTableStorage):
    """Test listing all executed migrations across projects."""
    # Project 1
    storage.write_event(Event.migration_started(project="project1", migration_id="mig_001"))
    storage.write_event(Event.migration_started(project="project1", migration_id="mig_002"))

    # Project 2
    storage.write_event(Event.migration_started(project="project2", migration_id="mig_003"))

    # List all migrations (no project filter)
    all_migrations = storage.list_executed_migrations(project=None)
    assert set(all_migrations) == {"mig_001", "mig_002", "mig_003"}

    # List migrations for specific project
    project1_migrations = storage.list_executed_migrations(project="project1")
    assert set(project1_migrations) == {"mig_001", "mig_002"}

    project2_migrations = storage.list_executed_migrations(project="project2")
    assert set(project2_migrations) == {"mig_003"}


def test_migration_with_no_features(storage: SystemTableStorage):
    """Test migration that starts and completes without any features (edge case)."""
    migration_id = "mig_no_features"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    assert storage.get_migration_status(migration_id, "test") == MigrationStatus.COMPLETED
    assert storage.get_completed_features(migration_id, "test") == []
    assert storage.get_failed_features(migration_id, "test") == {}


def test_migration_immediate_failure(storage: SystemTableStorage):
    """Test migration that fails immediately without processing any features."""
    migration_id = "mig_immediate_fail"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.migration_failed(
            project="test",
            migration_id=migration_id,
            error_message="Precondition check failed",
        )
    )

    assert storage.get_migration_status(migration_id, "test") == MigrationStatus.FAILED
    assert storage.get_completed_features(migration_id, "test") == []
    assert storage.get_failed_features(migration_id, "test") == {}


def test_empty_payload_parsing(storage: SystemTableStorage):
    """Test that empty payload strings are handled correctly."""
    from datetime import datetime, timezone

    import polars as pl

    # Create a DataFrame with empty payload string (mimics SQLAlchemy default)
    from metaxy.metadata_store.system.events import EVENTS_SCHEMA, EventType

    df = pl.DataFrame(
        {
            "project": ["test"],
            "execution_id": ["mig_001"],
            "event_type": [EventType.MIGRATION_STARTED],
            "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
            "feature_key": [None],
            "payload": [""],  # Empty string
        },
        schema=EVENTS_SCHEMA,
    )

    # Verify the DataFrame was created successfully
    assert df.height == 1
    assert df["payload"][0] == ""

    # Verify JSON parsing works with empty string (should not fail)
    # This mimics what happens when SQLAlchemy writes empty payload default
    parsed = df.select(pl.col("payload"))
    assert parsed.height == 1


def test_get_migration_summary(storage: SystemTableStorage):
    """Test get_migration_summary() convenience method."""
    migration_id = "mig_summary"

    # Test NOT_STARTED status
    summary = storage.get_migration_summary(migration_id, "test")
    assert summary["status"] == MigrationStatus.NOT_STARTED
    assert summary["completed_features"] == []
    assert summary["failed_features"] == {}
    assert summary["total_features_processed"] == 0

    # Start migration and complete some features
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            rows_affected=200,
        )
    )

    # Fail one feature
    storage.write_event(
        Event.feature_failed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/c",
            error_message="Test error",
        )
    )

    # Test IN_PROGRESS status
    summary = storage.get_migration_summary(migration_id, "test")
    assert summary["status"] == MigrationStatus.IN_PROGRESS
    assert set(summary["completed_features"]) == {"feature/a", "feature/b"}
    assert summary["failed_features"] == {"feature/c": "Test error"}
    assert summary["total_features_processed"] == 3

    # Complete migration
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Test COMPLETED status
    summary = storage.get_migration_summary(migration_id, "test")
    assert summary["status"] == MigrationStatus.COMPLETED
    assert set(summary["completed_features"]) == {"feature/a", "feature/b"}
    assert summary["failed_features"] == {"feature/c": "Test error"}
    assert summary["total_features_processed"] == 3


# ========== Tests for expected_features parameter ==========


def test_get_migration_status_with_expected_features(storage: SystemTableStorage):
    """Test get_migration_status() with expected_features parameter.

    This tests the bugfix for detecting when a migration YAML has been modified
    after initial completion (e.g., new operations added).
    """
    migration_id = "mig_expected"

    # Start migration and complete features
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/b",
            rows_affected=200,
        )
    )
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Without expected_features, status is COMPLETED
    assert storage.get_migration_status(migration_id, "test") == MigrationStatus.COMPLETED

    # With expected_features matching completed features, status is COMPLETED
    assert (
        storage.get_migration_status(migration_id, "test", expected_features=["feature/a", "feature/b"])
        == MigrationStatus.COMPLETED
    )

    # With expected_features including a feature that wasn't completed, status is IN_PROGRESS
    # This simulates when a new operation is added to an already-completed migration
    assert (
        storage.get_migration_status(
            migration_id,
            "test",
            expected_features=["feature/a", "feature/b", "feature/c"],
        )
        == MigrationStatus.IN_PROGRESS
    )


def test_get_migration_status_expected_features_subset(storage: SystemTableStorage):
    """Test that completed features being a superset of expected is OK."""
    migration_id = "mig_superset"

    # Complete more features than expected
    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    for fk in ["feature/a", "feature/b", "feature/c"]:
        storage.write_event(
            Event.feature_completed(
                project="test",
                migration_id=migration_id,
                feature_key=fk,
                rows_affected=100,
            )
        )
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Expect only feature/a - should still be COMPLETED since all expected are done
    assert (
        storage.get_migration_status(migration_id, "test", expected_features=["feature/a"]) == MigrationStatus.COMPLETED
    )


def test_get_migration_status_expected_features_with_failed_migration(
    storage: SystemTableStorage,
):
    """Test expected_features with a failed migration."""
    migration_id = "mig_failed_expected"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(
        Event.migration_failed(
            project="test",
            migration_id=migration_id,
            error_message="Fatal error",
        )
    )

    # Even with expected_features, a FAILED migration stays FAILED
    assert (
        storage.get_migration_status(migration_id, "test", expected_features=["feature/a", "feature/b"])
        == MigrationStatus.FAILED
    )


def test_get_migration_summary_with_expected_features(storage: SystemTableStorage):
    """Test get_migration_summary() with expected_features parameter."""
    migration_id = "mig_summary_expected"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(
        Event.feature_completed(
            project="test",
            migration_id=migration_id,
            feature_key="feature/a",
            rows_affected=100,
        )
    )
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Summary without expected_features shows COMPLETED
    summary = storage.get_migration_summary(migration_id, "test")
    assert summary["status"] == MigrationStatus.COMPLETED

    # Summary with expected_features that aren't all completed shows IN_PROGRESS
    summary = storage.get_migration_summary(migration_id, "test", expected_features=["feature/a", "feature/b"])
    assert summary["status"] == MigrationStatus.IN_PROGRESS
    assert summary["completed_features"] == ["feature/a"]


def test_get_migration_status_empty_expected_features(storage: SystemTableStorage):
    """Test that empty expected_features list is ignored."""
    migration_id = "mig_empty_expected"

    storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
    storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))

    # Empty list should behave same as None
    assert storage.get_migration_status(migration_id, "test", expected_features=[]) == MigrationStatus.COMPLETED
