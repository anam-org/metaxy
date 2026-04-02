"""Abstract base class for storage handlers.

A `IOHandler` knows how to perform I/O for a specific combination of
compute engine and storage format. The compute engine's connection is passed
to each method call.

Examples of concrete handlers:

- `IbisSQLHandler` — Ibis engine reading/writing SQL tables
- `DuckDBDuckLakeHandler` — DuckDB engine reading DuckLake tables
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

import narwhals as nw
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.table_ref import TableRef
from metaxy.metadata_store.types import AccessMode

ConnectionT = TypeVar("ConnectionT")
TableRefT = TypeVar("TableRefT", bound=TableRef)


class IOHandler(ABC, Generic[ConnectionT, TableRefT]):
    """Handles I/O for a specific storage format via a specific compute engine.

    Type parameters:

    - `ConnectionT`: the compute engine's connection type (e.g. `SQLBackend`)

    - `TableRefT`: the table reference type (e.g. `SQLTableIdentifier`)
    """

    @abstractmethod
    def can_handle(self, storage_config: StorageConfig) -> bool:
        """Whether this handler can handle I/O for `storage_config`."""
        ...

    @abstractmethod
    def read(
        self,
        conn: ConnectionT,
        table_id: TableRefT,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read from `table_id` using `conn`.

        Returns `None` when the table does not exist.
        """
        ...

    @abstractmethod
    def write(
        self,
        conn: ConnectionT,
        table_id: TableRefT,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write `df` to `table_id` using `conn`."""
        ...

    @abstractmethod
    def has_feature(
        self,
        conn: ConnectionT,
        table_id: TableRefT,
    ) -> bool:
        """Whether `table_id` exists."""
        ...

    @abstractmethod
    def drop(
        self,
        conn: ConnectionT,
        table_id: TableRefT,
    ) -> None:
        """Drop the table identified by `table_id`."""
        ...

    @abstractmethod
    def delete(
        self,
        conn: ConnectionT,
        table_id: TableRefT,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        """Hard-delete rows matching `filters` from `table_id`."""
        ...

    def get_store_metadata(
        self,
        table_id: TableRefT,
    ) -> dict[str, Any]:
        """Return backend-specific metadata for `table_id`.

        Default returns an empty dict.
        """
        return {}

    @contextmanager
    def open(self, conn: ConnectionT, mode: AccessMode) -> Iterator[Self]:  # noqa: ARG002
        """Per-operation lifecycle hook around I/O calls.

        Override to acquire/release resources (e.g. catalogs, transactions)
        scoped to a single CRUD operation.
        """
        yield self
