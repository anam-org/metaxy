"""Arrow Flight SQL server for metaxy metadata stores.

Experimental support for exposing metaxy metadata stores via the Arrow Flight SQL
protocol, enabling access from external tools and JDBC/ADBC clients.

Example:
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
"""

from __future__ import annotations

from metaxy.flight_sql.server import MetaxyFlightSQLServer

__all__ = ["MetaxyFlightSQLServer"]
