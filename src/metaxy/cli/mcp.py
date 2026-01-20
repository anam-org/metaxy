"""MCP server CLI commands."""

import cyclopts

from metaxy.cli.console import console

app = cyclopts.App(
    name="mcp",
    help="MCP (Model Context Protocol) server commands.",
    console=console,
)


@app.default
def serve():
    """Start the MCP server.

    Exposes Metaxy's feature graph and metadata store operations to AI assistants.
    Requires the `mcp` extra: `pip install metaxy[mcp]`
    """
    try:
        from metaxy.ext.mcp import run_server
    except ImportError as e:
        console.print(
            "[red]Error:[/red] MCP dependencies not installed. Install with: [cyan]pip install metaxy[mcp][/cyan]"
        )
        raise SystemExit(1) from e

    run_server()
