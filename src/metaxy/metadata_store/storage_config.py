"""Passive descriptor for a storage backend's location and format.

`StorageConfig` is a frozen Pydantic model that answers "where is the data?"
without any behaviour.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StorageConfig(BaseModel):
    """Base storage configuration.

    Subclass per backend to add typed fields instead of relying on
    an untyped options dict.

    Attributes:
        format: Short identifier for the storage format.
        location: Connection URI, directory path, or object store URI.
        schema: Optional database schema or dataset qualifier.
    """

    model_config = ConfigDict(frozen=True)

    format: str
    location: str
    schema_: str | None = Field(default=None, alias="schema")
