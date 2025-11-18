"""System table components for metadata store.

This package provides system table functionality for Metaxy:
- events: Migration event types with builder pattern
- keys: System table keys and constants
- models: Pydantic models and schemas for system tables
- storage: Storage layer for system tables
"""

from metaxy.metadata_store.system.events import (
    COL_EVENT_TYPE,
    COL_EXECUTION_ID,
    COL_FEATURE_KEY,
    COL_PAYLOAD,
    COL_PROJECT,
    COL_TIMESTAMP,
    Event,
    EventType,
    MigrationStatus,
    PayloadType,
)
from metaxy.metadata_store.system.keys import (
    EVENTS_KEY,
    FEATURE_VERSIONS_KEY,
    METAXY_SYSTEM_KEY_PREFIX,
)
from metaxy.metadata_store.system.models import (
    FEATURE_VERSIONS_SCHEMA,
    FeatureVersionsModel,
)
from metaxy.metadata_store.system.storage import (
    SystemTableStorage,
)

__all__ = [
    # Events
    "Event",
    "EventType",
    "MigrationStatus",
    "PayloadType",
    # Column names
    "COL_PROJECT",
    "COL_EXECUTION_ID",
    "COL_EVENT_TYPE",
    "COL_TIMESTAMP",
    "COL_FEATURE_KEY",
    "COL_PAYLOAD",
    # Keys
    "METAXY_SYSTEM_KEY_PREFIX",
    "FEATURE_VERSIONS_KEY",
    "EVENTS_KEY",
    # Models
    "FEATURE_VERSIONS_SCHEMA",
    "FeatureVersionsModel",
    # Storage
    "SystemTableStorage",
]
