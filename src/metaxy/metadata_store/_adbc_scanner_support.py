"""ADBC Scanner support for DuckDB federation.

Experimental support for querying remote ADBC data sources from DuckDB using
the community `adbc_scanner` extension.

Warning:
    This feature requires the DuckDB `adbc_scanner` community extension:
    ```sql
    INSTALL adbc_scanner FROM community;
    LOAD adbc_scanner;
    ```

Example:
    ```python
    from metaxy.metadata_store import DuckDBMetadataStore

    # DuckDB can query remote PostgreSQL via ADBC
    store = DuckDBMetadataStore("local.db")

    with store:
        # Install and load extension
        store.install_adbc_scanner()

        # Connect to remote PostgreSQL
        conn_handle = store.adbc_connect_postgres(
            host="prod-db.example.com",
            database="features",
            user="readonly",
            password="secret",
        )

        # Query remote data
        remote_df = store.adbc_scan(
            conn_handle,
            "SELECT * FROM my_feature__key WHERE sample_uid < 1000"
        )

        # Disconnect
        store.adbc_disconnect(conn_handle)
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import narwhals as nw


class ADBCScannerNotAvailableError(Exception):
    """Raised when adbc_scanner extension is not available."""

    pass


def install_adbc_scanner(conn: Any) -> None:
    """Install and load the adbc_scanner extension.

    Args:
        conn: DuckDB connection

    Raises:
        ADBCScannerNotAvailableError: If extension cannot be installed
    """
    try:
        conn.execute("INSTALL adbc_scanner FROM community")
        conn.execute("LOAD adbc_scanner")
    except Exception as e:
        raise ADBCScannerNotAvailableError(
            "Failed to install adbc_scanner extension. "
            "This is a DuckDB community extension that may not be available in all environments. "
            f"Error: {e}"
        ) from e


def adbc_connect(conn: Any, options: dict[str, Any]) -> int:
    """Connect to an ADBC data source.

    Args:
        conn: DuckDB connection
        options: ADBC connection options (driver-specific)

    Returns:
        Connection handle (integer)

    Example:
        ```python
        handle = adbc_connect(conn, {
            'driver': 'postgresql',
            'uri': 'postgresql://host:5432/db',
        })
        ```
    """
    result = conn.execute("SELECT adbc_connect(?)", [options]).fetchone()
    if result is None:
        raise RuntimeError("adbc_connect returned no result")
    return result[0]


def adbc_disconnect(conn: Any, handle: int) -> None:
    """Disconnect from an ADBC data source.

    Args:
        conn: DuckDB connection
        handle: Connection handle from adbc_connect()
    """
    conn.execute("SELECT adbc_disconnect(?)", [handle])


def adbc_scan(conn: Any, handle: int, query: str) -> nw.DataFrame[Any]:
    """Execute a query on an ADBC connection and return results.

    Args:
        conn: DuckDB connection
        handle: Connection handle
        query: SQL query to execute on the remote database

    Returns:
        Narwhals DataFrame with query results
    """
    import narwhals as nw

    result = conn.execute("SELECT * FROM adbc_scan(?, ?)", [handle, query]).fetchdf()
    return nw.from_native(result)


def adbc_scan_table(conn: Any, handle: int, table_name: str) -> nw.DataFrame[Any]:
    """Scan an entire table from an ADBC connection.

    Args:
        conn: DuckDB connection
        handle: Connection handle
        table_name: Name of table to scan

    Returns:
        Narwhals DataFrame with table data
    """
    import narwhals as nw

    result = conn.execute("SELECT * FROM adbc_scan_table(?, ?)", [handle, table_name]).fetchdf()
    return nw.from_native(result)
