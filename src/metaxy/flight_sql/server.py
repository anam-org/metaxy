"""Arrow Flight SQL server implementation for metaxy.

Provides a Flight server that exposes metaxy metadata stores via the
Arrow Flight protocol, enabling remote access and integration with
external tools.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow.flight as flight

if TYPE_CHECKING:
    from collections.abc import Iterator

    from metaxy.metadata_store import MetadataStore


class MetaxyFlightSQLServer(flight.FlightServerBase):
    """Arrow Flight server for metaxy metadata stores.

    Wraps a metaxy MetadataStore and exposes it via the Arrow Flight protocol,
    allowing remote clients to query metadata using SQL.

    This is a read-only server - writes are not supported in this version.
    Use the native MetadataStore API for write operations.

    Args:
        location: Flight server location (e.g., "grpc://0.0.0.0:8815")
        store: Metadata store to expose
        **kwargs: Additional arguments passed to FlightServerBase

    Example:
        ```python
        from metaxy.flight_sql import MetaxyFlightSQLServer
        from metaxy.metadata_store import DuckDBMetadataStore

        store = DuckDBMetadataStore("metadata.duckdb")

        server = MetaxyFlightSQLServer(
            location="grpc://0.0.0.0:8815",
            store=store,
        )

        # Start server (blocks)
        server.serve()
        ```

    Note:
        The server requires the metadata store to be opened before serving.
        Use the store as a context manager or call open() explicitly.
    """

    def __init__(
        self,
        location: str,
        store: MetadataStore,
        **kwargs: Any,
    ) -> None:
        """Initialize the Flight SQL server.

        Args:
            location: Flight server location
            store: Metadata store to expose
            **kwargs: Additional arguments for FlightServerBase
        """
        # Parse location into FlightServerBase format
        self.location = flight.Location(location)
        super().__init__(self.location, **kwargs)

        self._store = store
        self._store_opened_by_server = False

    def __enter__(self) -> MetaxyFlightSQLServer:
        """Enter context manager - opens store if needed."""
        if not self._store._is_open:
            self._store.__enter__()
            self._store_opened_by_server = True
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager - closes store if opened by server."""
        if self._store_opened_by_server:
            self._store.__exit__(*args)
            self._store_opened_by_server = False

    def get_flight_info(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
    ) -> flight.FlightInfo:
        """Get information about a query flight.

        Args:
            context: Server call context
            descriptor: Flight descriptor containing the SQL query

        Returns:
            FlightInfo with query metadata
        """
        # Parse command from descriptor
        command = self._parse_command(descriptor)
        sql_query = command.get("query")

        if not sql_query:
            raise flight.FlightServerError("No SQL query provided")

        # Execute query to get schema (without fetching data)
        # We'll use a LIMIT 0 query to get schema efficiently
        schema_query = f"SELECT * FROM ({sql_query}) LIMIT 0"

        try:
            result_df = self._store.read_metadata_sql(schema_query)
            # Convert to Arrow schema
            arrow_table = result_df.to_arrow()
            schema = arrow_table.schema
        except Exception as e:
            raise flight.FlightInternalError(f"Failed to get query schema: {e}") from e

        # Create ticket for do_get
        ticket = flight.Ticket(descriptor.command)

        # Create endpoint
        endpoint = flight.FlightEndpoint(ticket, [self.location])

        return flight.FlightInfo(
            schema=schema,
            descriptor=descriptor,
            endpoints=[endpoint],
            total_records=-1,  # Unknown until execution
            total_bytes=-1,
        )

    def do_get(
        self,
        context: flight.ServerCallContext,
        ticket: flight.Ticket,
    ) -> flight.RecordBatchStream:
        """Execute a query and stream results.

        Args:
            context: Server call context
            ticket: Ticket containing the SQL query

        Returns:
            RecordBatchReader streaming query results
        """
        # Parse command from ticket
        command = json.loads(ticket.ticket.decode("utf-8"))
        sql_query = command.get("query")

        if not sql_query:
            raise flight.FlightServerError("No SQL query provided")

        try:
            # Execute query
            result_df = self._store.read_metadata_sql(sql_query)

            # Convert to Arrow table
            arrow_table = result_df.to_arrow()

            # Return as RecordBatchReader
            return flight.RecordBatchStream(arrow_table)

        except Exception as e:
            raise flight.FlightInternalError(f"Query execution failed: {e}") from e

    def list_flights(
        self,
        context: flight.ServerCallContext,
        criteria: bytes,
    ) -> Iterator[flight.FlightInfo]:
        """List available flights (not implemented for metadata stores).

        Args:
            context: Server call context
            criteria: Listing criteria

        Yields:
            FlightInfo objects (empty for metadata stores)
        """
        # Metadata stores don't have a fixed set of "flights"
        # Clients should use get_flight_info with specific queries
        return iter([])

    def _parse_command(self, descriptor: flight.FlightDescriptor) -> dict[str, Any]:
        """Parse command from FlightDescriptor.

        Args:
            descriptor: Flight descriptor

        Returns:
            Parsed command dictionary
        """
        if descriptor.descriptor_type == flight.DescriptorType.CMD:
            try:
                return json.loads(descriptor.command.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise flight.FlightServerError(f"Invalid command format: {e}") from e
        else:
            raise flight.FlightServerError("Only CMD descriptors are supported for queries")

    def serve(self) -> None:
        """Start the Flight server and block until shutdown.

        This method opens the metadata store if needed and serves
        Flight requests until interrupted.

        Example:
            ```python
            server = MetaxyFlightSQLServer("grpc://0.0.0.0:8815", store)
            server.serve()  # Blocks until Ctrl+C
            ```
        """
        with self:
            super().serve()
