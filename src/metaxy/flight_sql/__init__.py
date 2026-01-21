"""Arrow Flight SQL server and client for metaxy metadata stores.

Experimental support for exposing metaxy metadata stores via the Arrow Flight SQL
protocol, enabling access from external tools and remote metaxy instances.

Server Example:
    ```python
    from metaxy.flight_sql import MetaxyFlightSQLServer
    from metaxy.metadata_store import DuckDBMetadataStore

    # Create backend store
    store = DuckDBMetadataStore("metadata.duckdb")

    # Start Flight SQL server
    server = MetaxyFlightSQLServer(
        location="grpc://0.0.0.0:8815",
        store=store,
    )

    server.serve()  # Blocks until shutdown
    ```

Client Example:
    ```python
    from metaxy.flight_sql import FlightSQLMetadataStore

    # Connect to remote server
    remote_store = FlightSQLMetadataStore(
        url="grpc://localhost:8815"
    )

    with remote_store:
        df = remote_store.read_metadata_sql("SELECT * FROM my_feature__key")
    ```
"""

from __future__ import annotations

from metaxy.flight_sql.client import FlightSQLMetadataStore
from metaxy.flight_sql.server import MetaxyFlightSQLServer

__all__ = ["FlightSQLMetadataStore", "MetaxyFlightSQLServer"]
