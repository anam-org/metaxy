"""History command for Metaxy CLI."""

from typing import Annotated

import cyclopts
from rich.table import Table

from metaxy.cli.console import console, error_console

# History subcommand app
app = cyclopts.App(
    name="history",
    help="Show history of recorded graph snapshots from the metadata store.",
    console=console,
    error_console=error_console,
)


@app.default
def history(
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (defaults to configured default store)",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        cyclopts.Parameter(
            name=["--limit"],
            help="Limit number of snapshots to show (defaults to all)",
        ),
    ] = None,
):
    """Show history of recorded graph snapshots.

    Displays all recorded graph snapshots from the metadata store,
    showing project versions, when they were recorded, and feature counts.

    Examples:
        ```console
        $ metaxy history
        ```

        ```console
        $ metaxy history --store prod
        ```

        ```console
        $ metaxy history --limit 5
        ```
    """
    from metaxy.cli.context import AppContext

    context = AppContext.get()
    metadata_store = context.get_store(store)

    from metaxy.metadata_store.system.storage import SystemTableStorage

    with metadata_store:
        storage = SystemTableStorage(metadata_store)
        snapshots_df = storage.read_graph_snapshots(project=context.project)

        if snapshots_df.height == 0:
            console.print("[yellow]No graph snapshots recorded yet[/yellow]")
            return

        if limit is not None:
            snapshots_df = snapshots_df.head(limit)

        table = Table(title="Graph Snapshot History")
        table.add_column("Project version", style="cyan", no_wrap=False, overflow="fold")
        table.add_column("Recorded At", style="green", no_wrap=False)
        table.add_column("Feature Count", style="yellow", justify="right", no_wrap=False)

        for row in snapshots_df.iter_rows(named=True):
            table.add_row(
                row["metaxy_project_version"],
                row["recorded_at"].strftime("%Y-%m-%d %H:%M:%S"),
                str(row["feature_count"]),
            )

        console.print(table)
        console.print(f"\nTotal snapshots: {snapshots_df.height}")
