"""Types for metadata stores."""

from typing import Literal

# Access mode for metadata store connections.
#
# Controls whether the store is opened in read-only or read-write mode.
# This is particularly important for stores like DuckDB that lock the database in write mode by default.
# Specific store implementations should handle this parameter accordingly.
AccessMode = Literal["read", "write"]
