"""Tests for migration system table storage layer."""

import pytest

from metaxy import InMemoryMetadataStore
from metaxy.metadata_store.system_tables import SystemTableStorage


@pytest.fixture
def store():
    """Create fresh store for each test."""
    return InMemoryMetadataStore()


@pytest.fixture
def storage(store):
    """Create storage layer for testing."""
    with store:
        yield SystemTableStorage(store)


def test_write_and_read_events(storage):
    """Test writing and reading migration events."""
    migration_id = "mig_001"

    # Write started event
    storage.write_event(migration_id, "started", project="test")

    # Write feature events
    storage.write_event(
        migration_id, "feature_started", project="test", feature_key="feature/a"
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/a",
        rows_affected=100,
    )

    # Write completed event
    storage.write_event(migration_id, "completed", project="test")

    # Read all events
    events = (
        storage.get_migration_events(migration_id, project="test").collect().to_polars()
    )
    assert events.height == 4
    assert events["event_type"].to_list() == [
        "started",
        "feature_started",
        "feature_completed",
        "completed",
    ]


def test_get_migration_status(storage):
    """Test computing migration status from events."""
    migration_id = "mig_001"

    # Not started
    assert storage.get_migration_status(migration_id, project="test") == "not_started"

    # Started
    storage.write_event(migration_id, "started", project="test")
    assert storage.get_migration_status(migration_id, project="test") == "in_progress"

    # Feature in progress
    storage.write_event(
        migration_id, "feature_started", project="test", feature_key="feature/a"
    )
    assert storage.get_migration_status(migration_id, project="test") == "in_progress"

    # Feature completed
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/a",
        rows_affected=100,
    )
    assert storage.get_migration_status(migration_id, project="test") == "in_progress"

    # Migration completed
    storage.write_event(migration_id, "completed", project="test")
    assert storage.get_migration_status(migration_id, project="test") == "completed"


def test_get_migration_status_failed(storage):
    """Test failed migration status."""
    migration_id = "mig_failed"

    storage.write_event(migration_id, "started", project="test")
    storage.write_event(
        migration_id, "feature_started", project="test", feature_key="feature/a"
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/a",
        error_message="Something went wrong",
    )
    storage.write_event(migration_id, "failed", project="test")

    assert storage.get_migration_status(migration_id, project="test") == "failed"


def test_is_feature_completed(storage):
    """Test checking if specific feature completed."""
    migration_id = "mig_001"

    # Not completed yet
    assert not storage.is_feature_completed(migration_id, "feature/a", "test")

    # Complete successfully
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/a",
        rows_affected=100,
    )
    assert storage.is_feature_completed(migration_id, "feature/a", "test")

    # Another feature with error - not completed successfully
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/b",
        error_message="Error",
    )
    assert not storage.is_feature_completed(migration_id, "feature/b", "test")


def test_get_completed_features(storage):
    """Test getting list of completed features."""
    migration_id = "mig_001"

    # Complete multiple features
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/a",
        rows_affected=100,
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/b",
        rows_affected=200,
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/c",
        error_message="Error",
    )

    completed = storage.get_completed_features(migration_id, project="test")
    assert set(completed) == {"feature/a", "feature/b"}


def test_get_failed_features(storage):
    """Test getting failed features with error messages."""
    migration_id = "mig_001"

    # Complete with errors
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/a",
        rows_affected=100,
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/b",
        error_message="Error B",
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/c",
        error_message="Error C",
    )

    failed = storage.get_failed_features(migration_id, project="test")
    assert failed == {"feature/b": "Error B", "feature/c": "Error C"}


def test_resumable_migration(storage):
    """Test resuming a migration that partially failed."""
    migration_id = "mig_resume"

    # Start migration
    storage.write_event(migration_id, "started", project="test")

    # Complete first feature
    storage.write_event(
        migration_id, "feature_started", project="test", feature_key="feature/a"
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/a",
        rows_affected=100,
    )

    # Fail on second feature
    storage.write_event(
        migration_id, "feature_started", project="test", feature_key="feature/b"
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/b",
        error_message="Network error",
    )

    # Check status
    assert storage.get_migration_status(migration_id, project="test") == "in_progress"
    assert storage.is_feature_completed(migration_id, "feature/a", "test")
    assert not storage.is_feature_completed(migration_id, "feature/b", "test")

    # Resume: retry feature/b
    storage.write_event(
        migration_id, "feature_started", project="test", feature_key="feature/b"
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/b",
        rows_affected=200,
    )

    # Complete third feature
    storage.write_event(
        migration_id, "feature_started", project="test", feature_key="feature/c"
    )
    storage.write_event(
        migration_id,
        "feature_completed",
        project="test",
        feature_key="feature/c",
        rows_affected=300,
    )

    # Mark complete
    storage.write_event(migration_id, "completed", project="test")

    # Verify final state
    assert storage.get_migration_status(migration_id, project="test") == "completed"
    assert set(storage.get_completed_features(migration_id, project="test")) == {
        "feature/a",
        "feature/b",
        "feature/c",
    }
