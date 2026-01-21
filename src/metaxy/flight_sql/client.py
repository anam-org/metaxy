"""Flight SQL client for remote metadata store access.

Provides a MetadataStore implementation that connects to remote
metaxy Flight SQL servers, enabling distributed metadata access.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pyarrow.flight as flight

from metaxy.metadata_store.base import CoercibleToFeatureKey, MetadataStore, MetadataStoreConfig
from metaxy.models.feature import FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from collections.abc import Sequence

    import narwhals as nw

    from metaxy.models.plan import FeaturePlan
    from metaxy.versioning import VersioningEngine


class FlightSQLClientConfig(MetadataStoreConfig):
    """Configuration for Flight SQL client connections.

    Args:
        url: Flight SQL server URL (e.g., "grpc://localhost:8815")
        **kwargs: Additional MetadataStore configuration
    """

    url: str


class FlightSQLMetadataStore(MetadataStore):
    """Read-only metadata store client for remote Flight SQL servers.

    Connects to a remote metaxy Flight SQL server and provides
    read access to metadata via SQL queries. This enables federation
    across multiple metaxy instances.

    Args:
        url: Flight SQL server URL (e.g., "grpc://localhost:8815")
        **kwargs: Additional MetadataStore configuration

    Example:
        ```python
        from metaxy.flight_sql import FlightSQLMetadataStore

        # Connect to remote Flight SQL server
        remote_store = FlightSQLMetadataStore(
            url="grpc://prod-metadata.example.com:8815"
        )

        with remote_store:
            # Query remote metadata
            df = remote_store.read_metadata_sql(
                "SELECT * FROM my_feature__key WHERE sample_uid < 1000"
            )
        ```

    Note:
        This is a read-only store. Write operations are not supported
        in this version.
    """

    _client: flight.FlightClient | None = None

    def __init__(self, url: str, **kwargs: Any) -> None:
        """Initialize Flight SQL client.

        Args:
            url: Flight SQL server URL
            **kwargs: Additional MetadataStore configuration
        """
        super().__init__(**kwargs)
        self.url = url

    @contextmanager
    def open(self, mode: str = "read") -> Iterator[FlightSQLMetadataStore]:
        """Open connection to Flight SQL server.

        Args:
            mode: Access mode (only "read" is supported)

        Yields:
            Self
        """
        if self._is_open:
            yield self
            return

        # Create Flight client
        location = flight.Location(self.url)
        self._client = flight.FlightClient(location)
        self._is_open = True

        try:
            yield self
        finally:
            self._client = None
            self._is_open = False

    def read_metadata_sql(self, query: str) -> nw.DataFrame[Any]:
        """Execute SQL query on remote server and return results.

        Args:
            query: SQL query to execute

        Returns:
            Narwhals DataFrame with query results

        Raises:
            RuntimeError: If store is not open
            flight.FlightError: If query execution fails

        Example:
            ```python
            with store:
                df = store.read_metadata_sql(
                    "SELECT * FROM my_feature__key LIMIT 100"
                )
            ```
        """
        import narwhals as nw

        if not self._is_open or self._client is None:
            raise RuntimeError("Store is not open")

        # Create command for query
        command = json.dumps({"query": query}).encode("utf-8")
        descriptor = flight.FlightDescriptor.for_command(command)

        # Get flight info
        info = self._client.get_flight_info(descriptor)

        # Get data from first endpoint
        reader = self._client.do_get(info.endpoints[0].ticket)

        # Read all data as Arrow table
        table = reader.read_all()

        # Convert to Narwhals DataFrame
        return nw.from_native(table, eager_only=True)

    @classmethod
    def config_model(cls) -> type[FlightSQLClientConfig]:
        """Return configuration model for this store."""
        return FlightSQLClientConfig

    # Abstract method implementations

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm (not used for remote client)."""
        return HashAlgorithm.XXHASH64

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[VersioningEngine]:
        """Create versioning engine (not used for remote client)."""
        yield PolarsVersioningEngine(plan=plan)

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Check if feature exists (not supported for remote client)."""
        raise NotImplementedError(
            "Flight SQL client does not support _has_feature_impl. Use read_metadata_sql() for custom queries."
        )

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from store (not supported - use read_metadata_sql)."""
        raise NotImplementedError(
            "Flight SQL client does not support read_metadata_in_store. Use read_metadata_sql() instead."
        )

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Any,
        **kwargs: Any,
    ) -> None:
        """Write metadata to store (not supported - read-only client)."""
        raise NotImplementedError("Flight SQL client is read-only. Write operations are not supported.")

    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        current_only: bool,
    ) -> None:
        """Delete metadata (not supported - read-only client)."""
        raise NotImplementedError("Flight SQL client is read-only. Delete operations are not supported.")

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop feature metadata (not supported - read-only client)."""
        raise NotImplementedError("Flight SQL client is read-only. Drop operations are not supported.")

    def display(self) -> str:
        """Return display string for store."""
        return f"FlightSQLMetadataStore(url={self.url!r})"
