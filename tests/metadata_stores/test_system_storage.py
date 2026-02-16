"""Tests for system table storage layer across all store backends."""

from pytest_cases import parametrize_with_cases

from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.system import (
    Event,
    EventType,
    MigrationStatus,
    SystemTableStorage,
)

from .conftest import AllStoresCases


@parametrize_with_cases("store", cases=AllStoresCases)
def test_write_and_read_system_events(store: MetadataStore):
    """Test writing and reading migration events across all store types."""
    with store:
        storage = SystemTableStorage(store)
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


@parametrize_with_cases("store", cases=AllStoresCases)
def test_get_migration_status_across_stores(store: MetadataStore):
    """Test computing migration status from events across all store types."""
    with store:
        storage = SystemTableStorage(store)
        migration_id = "mig_002"

        # Not started
        assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.NOT_STARTED

        # Started
        storage.write_event(Event.migration_started(project="test", migration_id=migration_id))
        assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.IN_PROGRESS

        # Feature completed
        storage.write_event(
            Event.feature_completed(
                project="test",
                migration_id=migration_id,
                feature_key="feature/a",
                rows_affected=50,
            )
        )
        assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.IN_PROGRESS

        # Migration completed
        storage.write_event(Event.migration_completed(project="test", migration_id=migration_id))
        assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.COMPLETED


@parametrize_with_cases("store", cases=AllStoresCases)
def test_migration_failed_status(store: MetadataStore):
    """Test migration failed status across all store types."""
    with store:
        storage = SystemTableStorage(store)
        migration_id = "mig_003"

        # Start migration
        storage.write_event(Event.migration_started(project="test", migration_id=migration_id))

        # Fail migration
        storage.write_event(
            Event.migration_failed(
                project="test",
                migration_id=migration_id,
                error_message="Something went wrong",
            )
        )

        assert storage.get_migration_status(migration_id, project="test") == MigrationStatus.FAILED


@parametrize_with_cases("store", cases=AllStoresCases)
def test_multiple_migrations_sequence(store: MetadataStore):
    """Test multiple migrations in sequence across all store types."""
    with store:
        storage = SystemTableStorage(store)

        # First migration
        storage.write_event(Event.migration_started(project="test", migration_id="m1"))
        storage.write_event(
            Event.feature_completed(
                project="test",
                migration_id="m1",
                feature_key="feature/a",
                rows_affected=10,
            )
        )
        storage.write_event(Event.migration_completed(project="test", migration_id="m1"))

        # Second migration
        storage.write_event(Event.migration_started(project="test", migration_id="m2"))
        storage.write_event(
            Event.feature_completed(
                project="test",
                migration_id="m2",
                feature_key="feature/b",
                rows_affected=20,
            )
        )
        storage.write_event(Event.migration_completed(project="test", migration_id="m2"))

        # Both should be completed
        assert storage.get_migration_status("m1", project="test") == MigrationStatus.COMPLETED
        assert storage.get_migration_status("m2", project="test") == MigrationStatus.COMPLETED

        # List all executed migrations
        executed = storage.list_executed_migrations(project="test")
        assert set(executed) == {"m1", "m2"}
