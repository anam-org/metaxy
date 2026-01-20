from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    """Metaxy event types."""

    MIGRATION_STARTED = "migration_started"
    MIGRATION_COMPLETED = "migration_completed"
    MIGRATION_FAILED = "migration_failed"
    FEATURE_MIGRATION_STARTED = "feature_migration_started"
    FEATURE_MIGRATION_COMPLETED = "feature_migration_completed"
    FEATURE_MIGRATION_FAILED = "feature_migration_failed"


class PayloadType(str, Enum):
    """Payload types for event payloads."""

    EMPTY = "empty"
    ERROR = "error"
    ROWS_AFFECTED = "rows_affected"


class MigrationStatus(str, Enum):
    """Migration execution status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# Column name constants (to avoid drift between Event model and storage)
COL_PROJECT = "project"
COL_EXECUTION_ID = "execution_id"
COL_EVENT_TYPE = "event_type"
COL_TIMESTAMP = "timestamp"
COL_FEATURE_KEY = "feature_key"
COL_PAYLOAD = "payload"

# Events schema (for Polars storage)
EVENTS_SCHEMA = {
    COL_PROJECT: pl.String,
    COL_EXECUTION_ID: pl.String,
    COL_EVENT_TYPE: pl.Enum(EventType),
    COL_TIMESTAMP: pl.Datetime("us"),
    COL_FEATURE_KEY: pl.String,
    COL_PAYLOAD: pl.String,  # JSON string with arbitrary event data
}


class EmptyPayload(BaseModel):
    """Empty payload for events with no additional data."""

    model_config = ConfigDict(frozen=True)
    type: Literal[PayloadType.EMPTY] = PayloadType.EMPTY


class ErrorPayload(BaseModel):
    """Payload for events with error information."""

    model_config = ConfigDict(frozen=True)
    type: Literal[PayloadType.ERROR] = PayloadType.ERROR
    error_message: str
    rows_affected: int | None = None  # Optional: rows processed before failure


class RowsAffectedPayload(BaseModel):
    """Payload for events tracking rows affected."""

    model_config = ConfigDict(frozen=True)
    type: Literal[PayloadType.ROWS_AFFECTED] = PayloadType.ROWS_AFFECTED
    rows_affected: int


# Discriminated union for payloads
Payload = EmptyPayload | ErrorPayload | RowsAffectedPayload


class Event(BaseModel):
    """Migration event with typed payload.

    All event types use this single class and are distinguished by event_type and payload.type fields.
    """

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    project: str
    execution_id: str  # Generic ID for the execution (migration, job, etc.)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_key: str | None = None  # Feature key for feature-level events, empty for execution-level events
    payload: Annotated[Payload, Field(default_factory=EmptyPayload, discriminator="type")]

    def to_polars(self) -> pl.DataFrame:
        """Convert this model instance to a single-row Polars DataFrame.

        Returns:
            Polars DataFrame with one row matching EVENTS_SCHEMA
        """
        data = {
            COL_PROJECT: self.project,
            COL_EXECUTION_ID: self.execution_id,
            COL_EVENT_TYPE: self.event_type,
            COL_TIMESTAMP: self.timestamp,
            COL_FEATURE_KEY: self.feature_key,
            COL_PAYLOAD: self.payload.model_dump_json(),
        }
        return pl.DataFrame([data], schema=EVENTS_SCHEMA)

    @classmethod
    def migration_started(cls, project: str, migration_id: str) -> Event:
        """Create a migration started event.

        Args:
            project: Project name
            migration_id: Migration ID (maps to execution_id internally)

        Returns:
            Event with started type and empty payload
        """
        return cls(
            project=project,
            execution_id=migration_id,
            event_type=EventType.MIGRATION_STARTED,
            payload=EmptyPayload(),
        )

    @classmethod
    def migration_completed(cls, project: str, migration_id: str) -> Event:
        """Create a migration completed event.

        Args:
            project: Project name
            migration_id: Migration ID (maps to execution_id internally)

        Returns:
            Event with migration_completed type and empty payload
        """
        return cls(
            project=project,
            execution_id=migration_id,
            event_type=EventType.MIGRATION_COMPLETED,
            payload=EmptyPayload(),
        )

    @classmethod
    def migration_failed(
        cls,
        project: str,
        migration_id: str,
        error_message: str,
        rows_affected: int | None = None,
    ) -> Event:
        """Create a migration failed event.

        Args:
            project: Project name
            migration_id: Migration ID (maps to execution_id internally)
            error_message: Error message describing the failure
            rows_affected: Optional number of rows processed before failure

        Returns:
            Event with migration_failed type and error payload
        """
        return cls(
            project=project,
            execution_id=migration_id,
            event_type=EventType.MIGRATION_FAILED,
            payload=ErrorPayload(error_message=error_message, rows_affected=rows_affected),
        )

    @classmethod
    def feature_migration_started(cls, project: str, migration_id: str, feature_key: str) -> Event:
        """Create a feature started event.

        Args:
            project: Project name
            migration_id: Migration ID (maps to execution_id internally)
            feature_key: Feature key being processed

        Returns:
            Event with feature_started type and empty payload
        """
        return cls(
            project=project,
            execution_id=migration_id,
            event_type=EventType.FEATURE_MIGRATION_STARTED,
            payload=EmptyPayload(),
            feature_key=feature_key,
        )

    @classmethod
    def feature_migration_completed(
        cls, project: str, migration_id: str, feature_key: str, rows_affected: int
    ) -> Event:
        """Create a feature completed event (successful).

        Args:
            project: Project name
            migration_id: Migration ID (maps to execution_id internally)
            feature_key: Feature key that was processed
            rows_affected: Number of rows affected

        Returns:
            Event with feature_completed type and rows_affected payload
        """
        return cls(
            project=project,
            execution_id=migration_id,
            event_type=EventType.FEATURE_MIGRATION_COMPLETED,
            feature_key=feature_key,
            payload=RowsAffectedPayload(rows_affected=rows_affected),
        )

    @classmethod
    def feature_migration_failed(
        cls,
        project: str,
        migration_id: str,
        feature_key: str,
        error_message: str,
        rows_affected: int | None = None,
    ) -> Event:
        """Create a feature failed event.

        Args:
            project: Project name
            migration_id: Migration ID (maps to execution_id internally)
            feature_key: Feature key that failed
            error_message: Error message describing the failure
            rows_affected: Optional number of rows processed before failure

        Returns:
            Event with feature_failed type and error payload
        """
        return cls(
            project=project,
            execution_id=migration_id,
            event_type=EventType.FEATURE_MIGRATION_FAILED,
            feature_key=feature_key,
            payload=ErrorPayload(error_message=error_message, rows_affected=rows_affected),
        )

    # Shorter aliases for convenience
    feature_started = feature_migration_started
    feature_completed = feature_migration_completed
    feature_failed = feature_migration_failed
