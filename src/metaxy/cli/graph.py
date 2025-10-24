"""Graph management commands for Metaxy CLI."""

from __future__ import annotations

from typing import Annotated

import cyclopts
from rich.console import Console
from rich.table import Table

# Rich console for formatted output
console = Console()

# Graph subcommand app
app = cyclopts.App(
    name="graph",  # pyrefly: ignore[unexpected-keyword]
    help="Manage feature graphs",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
)


@app.command()
def push(
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (defaults to configured default store)",
        ),
    ] = None,
):
    """Record all feature versions (push graph snapshot).

    Records all features in the active graph to the metadata store
    with a deterministic snapshot ID. This should be run after deploying
    new feature definitions.

    Example:
        $ metaxy graph push

        ✓ Recorded feature graph
          Snapshot ID: abc123def456...

        # Or if already recorded:
        ℹ Snapshot already recorded (skipped)
          Snapshot ID: abc123def456...
    """
    from metaxy.cli.context import get_store
    from metaxy.entrypoints import load_features

    # Load features from entrypoints
    load_features()

    metadata_store = get_store(store)

    with metadata_store:
        snapshot_id, was_already_recorded = (
            metadata_store.record_feature_graph_snapshot()
        )

        if was_already_recorded:
            console.print("[blue]ℹ[/blue] Snapshot already recorded (skipped)")
            console.print(f"  Snapshot ID: {snapshot_id}")
        else:
            console.print("[green]✓[/green] Recorded feature graph")
            console.print(f"  Snapshot ID: {snapshot_id}")


@app.command()
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
    showing snapshot IDs, when they were recorded, and feature counts.

    Example:
        $ metaxy graph history

        Graph Snapshot History
        ┌──────────────┬─────────────────────┬───────────────┐
        │ Snapshot ID  │ Recorded At         │ Feature Count │
        ├──────────────┼─────────────────────┼───────────────┤
        │ abc123...    │ 2025-01-15 10:30:00 │ 42            │
        │ def456...    │ 2025-01-14 09:15:00 │ 40            │
        └──────────────┴─────────────────────┴───────────────┘
    """
    from metaxy.cli.context import get_store
    from metaxy.entrypoints import load_features

    # Load features from entrypoints
    load_features()

    metadata_store = get_store(store)

    with metadata_store:
        # Read snapshot history
        snapshots_df = metadata_store.read_graph_snapshots()

        if snapshots_df.height == 0:
            console.print("[yellow]No graph snapshots recorded yet[/yellow]")
            return

        # Limit results if requested
        if limit is not None:
            snapshots_df = snapshots_df.head(limit)

        # Create table
        table = Table(title="Graph Snapshot History")
        table.add_column("Snapshot ID", style="cyan", no_wrap=False, overflow="fold")
        table.add_column("Recorded At", style="green", no_wrap=False)
        table.add_column(
            "Feature Count", style="yellow", justify="right", no_wrap=False
        )

        # Add rows
        for row in snapshots_df.iter_rows(named=True):
            snapshot_id = row["snapshot_id"]
            recorded_at = row["recorded_at"].strftime("%Y-%m-%d %H:%M:%S")
            feature_count = str(row["feature_count"])

            table.add_row(snapshot_id, recorded_at, feature_count)

        console.print(table)
        console.print(f"\nTotal snapshots: {snapshots_df.height}")


@app.command()
def describe(
    snapshot: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--snapshot"],
            help="Snapshot ID to describe (defaults to current graph from code)",
        ),
    ] = None,
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (defaults to configured default store)",
        ),
    ] = None,
):
    """Describe a graph snapshot.

    Shows detailed information about a graph snapshot including:
    - Feature count
    - Graph depth (longest dependency chain)
    - Root features (features with no dependencies)

    Example:
        $ metaxy graph describe

        Graph Snapshot: abc123def456...
        ┌─────────────────────┬────────┐
        │ Metric              │ Value  │
        ├─────────────────────┼────────┤
        │ Feature Count       │ 42     │
        │ Graph Depth         │ 5      │
        │ Root Features       │ 8      │
        └─────────────────────┴────────┘

        Root Features:
        • user__profile
        • transaction__history
        ...
    """
    from metaxy.cli.context import get_store
    from metaxy.entrypoints import load_features
    from metaxy.models.feature import FeatureGraph

    # Load features from entrypoints
    load_features()

    metadata_store = get_store(store)

    with metadata_store:
        # Determine which snapshot to describe
        if snapshot is None:
            # Use current graph from code
            graph = FeatureGraph.get_active()
            snapshot_id = graph.snapshot_id
            console.print("[cyan]Describing current graph from code[/cyan]")
        else:
            # Use specified snapshot
            snapshot_id = snapshot
            console.print(f"[cyan]Describing snapshot: {snapshot_id}[/cyan]")

            # Load graph from snapshot
            features_df = metadata_store.read_features(
                current=False, snapshot_id=snapshot_id
            )

            if features_df.height == 0:
                console.print(
                    f"[red]✗[/red] No features found for snapshot {snapshot_id}"
                )
                return

            # For historical snapshots, we'll use the current graph structure
            # but report on the features that were in that snapshot
            graph = FeatureGraph.get_active()

        # Get graph metrics
        feature_count = len(graph.features_by_key)

        # Calculate graph depth (longest dependency chain)
        def get_feature_depth(feature_key, visited=None):
            if visited is None:
                visited = set()

            if feature_key in visited:
                return 0  # Avoid cycles

            visited.add(feature_key)

            feature_cls = graph.features_by_key.get(feature_key)
            if feature_cls is None or not feature_cls.spec.deps:
                return 1

            max_dep_depth = 0
            for dep in feature_cls.spec.deps:
                dep_depth = get_feature_depth(dep.key, visited.copy())
                max_dep_depth = max(max_dep_depth, dep_depth)

            return max_dep_depth + 1

        max_depth = 0
        for feature_key in graph.features_by_key:
            depth = get_feature_depth(feature_key)
            max_depth = max(max_depth, depth)

        # Find root features (no dependencies)
        root_features = [
            feature_key
            for feature_key, feature_cls in graph.features_by_key.items()
            if not feature_cls.spec.deps
        ]

        # Display summary table
        console.print()
        summary_table = Table(title=f"Graph Snapshot: {snapshot_id}")
        summary_table.add_column("Metric", style="cyan", no_wrap=False)
        summary_table.add_column(
            "Value", style="yellow", justify="right", no_wrap=False
        )

        summary_table.add_row("Feature Count", str(feature_count))
        summary_table.add_row("Graph Depth", str(max_depth))
        summary_table.add_row("Root Features", str(len(root_features)))

        console.print(summary_table)

        # Display root features
        if root_features:
            console.print("\n[bold]Root Features:[/bold]")
            for feature_key in sorted(root_features, key=lambda k: k.to_string()):
                console.print(f"  • {feature_key.to_string()}")
