"""Arrow Flight SQL server CLI commands."""

from typing import Annotated

import cyclopts

from metaxy.cli.console import console

app = cyclopts.App(
    name="flight-sql",
    help="Arrow Flight SQL server commands for remote metadata access.",
    console=console,
)


@app.default
def serve(
    location: Annotated[
        str,
        cyclopts.Parameter(
            help="Server location (e.g., grpc://0.0.0.0:8815)",
        ),
    ] = "grpc://0.0.0.0:8815",
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Metadata store name from config. If not provided, uses the default project store.",
        ),
    ] = None,
):
    """Start Arrow Flight SQL server.

    Exposes the metadata store via Arrow Flight protocol, enabling:
    - Remote SQL queries from external tools (DBeaver, etc.)
    - Cross-instance metadata federation
    - JDBC/ADBC client access

    The server requires the metadata store to support SQL queries
    (DuckDB, PostgreSQL, ClickHouse, etc.).

    Example:
        ```bash
        # Start server on default port with project's default store
        metaxy flight-sql serve

        # Start on custom port
        metaxy flight-sql serve grpc://0.0.0.0:9000

        # Use specific store from config
        metaxy flight-sql serve --store production
        ```

    Note:
        Press Ctrl+C to stop the server.
    """
    from metaxy.cli.context import AppContext
    from metaxy.flight_sql import MetaxyFlightSQLServer

    # Get context
    ctx = AppContext.get()

    # Get the metadata store using context helper
    try:
        metadata_store = ctx.get_store(store)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from e

    # Create Flight SQL server
    console.print(f"[cyan]Starting Flight SQL server at {location}[/cyan]")
    console.print(f"[dim]Store: {metadata_store.display()}[/dim]")
    console.print()

    server = MetaxyFlightSQLServer(
        location=location,
        store=metadata_store,
    )

    try:
        with metadata_store:
            console.print("[green]✓[/green] Server started. Press Ctrl+C to stop.")
            console.print()
            console.print("[dim]Connect using:[/dim]")
            console.print(f"[dim]  URL: {location}[/dim]")
            console.print()

            server.serve()
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Server stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from e
