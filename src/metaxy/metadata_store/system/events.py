from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    """Metaxy event types."""


class PayloadType(str, Enum):
    """Payload types for event payloads."""

    EMPTY = "empty"
    ERROR = "error"
    ROWS_AFFECTED = "rows_affected"


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
    """Event with typed payload.

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
