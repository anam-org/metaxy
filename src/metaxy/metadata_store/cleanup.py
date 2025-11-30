"""Data deletion and mutation models and types for metadata stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from metaxy.models.types import FeatureKey

DeletionMode = Literal["hard", "soft"]
"""Deletion mode: 'hard' for physical removal, 'soft' for logical deletion."""


@dataclass
class DeletionResult:
    """Result of a delete operation (hard or soft).

    Attributes:
        feature_key: Feature that was updated.
        rows_affected: Total rows affected.
        timestamp: When the deletion operation was executed.
        error: Error message if deletion failed, None if successful.

    Example:
        ```python
        result = store.delete_metadata(
            UserEvents,
            filter=nw.col("timestamp") < cutoff
        )
        print(f"Deleted {result.rows_affected} rows")
        ```
    """

    feature_key: FeatureKey
    rows_affected: int
    timestamp: datetime
    error: str | None = None


@dataclass
class MutationResult:
    """Result of a mutate operation.

    Attributes:
        feature_key: Feature that was updated.
        rows_affected: Total rows affected.
        updates: Column -> value mapping that was applied.
        timestamp: When the updated rows were appended.
        timestamp: When the mutation operation was executed.
        error: Error message if mutation failed, None if successful.

    Example:
        ```python
        result = store.mutate_metadata(
            UserProfile,
            filter=nw.col("user_id") == "user_123",
            updates={"email": "[REDACTED]"}
        )
        print(f"Anonymized {result.rows_affected} rows")
        ```
    """

    feature_key: FeatureKey
    rows_affected: int
    updates: dict[str, Any]
    timestamp: datetime
    error: str | None = None
