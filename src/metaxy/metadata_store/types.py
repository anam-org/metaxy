"""Types for metadata stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from metaxy.metadata_store.system.keys import METAXY_SYSTEM_KEY_PREFIX

# Access mode for metadata store connections.
#
# Controls whether the store is opened in read-only or read-write mode.
# This is particularly important for stores like DuckDB that lock the database in write mode by default.
# Specific store implementations should handle this parameter accordingly.
AccessMode = Literal["r", "w"]


@dataclass(frozen=True, slots=True)
class TableIdentifier:
    """Storage-layer table identifier, decoupled from feature-domain concepts.

    Backends use this to locate and name tables without needing to know
    about features, feature keys, or the feature graph.
    """

    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalise parts to an immutable tuple."""
        object.__setattr__(self, "parts", tuple(self.parts))

    @property
    def table_name(self) -> str:
        """SQL-compatible table name derived from parts."""
        return "__".join(part.replace("-", "_") for part in self.parts)

    @property
    def is_system_table(self) -> bool:
        """Whether this identifier refers to a Metaxy system table."""
        return len(self.parts) >= 1 and self.parts[0] == METAXY_SYSTEM_KEY_PREFIX

    def to_string(self) -> str:
        """Slash-separated string representation (for display/error messages)."""
        return "/".join(self.parts)
