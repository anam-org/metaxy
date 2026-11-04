"""Passive descriptor for a storage backend's location and format.

`StorageConfig` is a frozen Pydantic model that answers "where is the data?"
without any behaviour. Each subclass implements ``resolve()`` to translate a
``FeatureKey`` into a backend-specific ``TableRef``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from metaxy.metadata_store.table_ref import TableRef
    from metaxy.models.types import FeatureKey


class StorageConfig(BaseModel):
    """Base storage configuration.

    Subclass per backend to add typed fields instead of relying on
    an untyped options dict.

    Attributes:
        location: Connection URI, directory path, or object store URI.
        schema: Optional database schema or dataset qualifier.
    """

    model_config = ConfigDict(frozen=True)

    location: str
    schema_: str | None = Field(default=None, alias="schema")

    def resolve(self, key: FeatureKey) -> TableRef:
        """Translate a feature key into a backend-specific table identifier."""
        raise NotImplementedError(f"{type(self).__name__} must implement resolve()")


class LanceStorageConfig(StorageConfig):
    """Storage configuration for Lance tables.

    Attributes:
        connect_kwargs: Extra keyword arguments passed to `lancedb.connect`.
    """

    connect_kwargs: dict[str, object] = Field(default_factory=dict)

    def resolve(self, key: FeatureKey) -> TableRef:
        from metaxy.metadata_store.table_ref import LanceTableIdentifier

        return LanceTableIdentifier(
            uri=f"{self.location.rstrip('/')}/{key.table_name}.lance",
            table_name=key.table_name,
            database=self.location,
            connect_kwargs=dict(self.connect_kwargs),
        )
