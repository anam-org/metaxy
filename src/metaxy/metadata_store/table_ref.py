"""Table identifier types for metadata store backends.

Each storage backend has its own way of identifying tables. These frozen
dataclasses provide a type-safe, backend-specific table reference that
TableAdapter methods operate on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class TableRef(ABC):
    """Base class for table identifiers."""

    @abstractmethod
    def table_parts(self) -> tuple[str, ...]:
        """Return a tuple of table parts."""


@dataclass(frozen=True, slots=True)
class SQLTableIdentifier(TableRef):
    """Identifies a table in a SQL database."""

    table_name: str

    def table_parts(self) -> tuple[str, ...]:
        return (self.table_name,)


@dataclass(frozen=True, slots=True)
class DeltaTableIdentifier(TableRef):
    """Identifies a Delta Lake table by its URI."""

    uri: str | Path

    def table_parts(self) -> tuple[str, ...]:
        return (str(self.uri),)


@dataclass(frozen=True, slots=True)
class LanceTableIdentifier(TableRef):
    """Identifies a Lance table by its URI."""

    uri: str
    table_name: str

    def table_parts(self) -> tuple[str, ...]:
        return (self.uri,)
