"""Graph management commands for Metaxy CLI."""

from typing import Annotated, Literal

import cyclopts
from rich.console import Console
from rich.table import Table

from metaxy.graph import RenderConfig

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
    with a deterministic snapshot version. This should be run after deploying
    new feature definitions.

    Example:
        $ metaxy graph push

        ✓ Recorded feature graph
          Snapshot version: abc123def456...

        # Or if already recorded:
        ℹ Snapshot already recorded (skipped)
          Snapshot version: abc123def456...
    """
    from metaxy.cli.context import get_store
    from metaxy.entrypoints import load_features

    # Load features from entrypoints
    load_features()

    metadata_store = get_store(store)

    with metadata_store:
        snapshot_version, was_already_recorded = (
            metadata_store.record_feature_graph_snapshot()
        )

        if was_already_recorded:
            console.print("[blue]ℹ[/blue] Snapshot already recorded (skipped)")
            console.print(f"  Snapshot version: {snapshot_version}")
        else:
            console.print("[green]✓[/green] Recorded feature graph")
            console.print(f"  Snapshot version: {snapshot_version}")


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
    showing snapshot versions, when they were recorded, and feature counts.

    Example:
        $ metaxy graph history

        Graph Snapshot History
        ┌──────────────┬─────────────────────┬───────────────┐
        │ Snapshot version  │ Recorded At         │ Feature Count │
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
        table.add_column(
            "Snapshot version", style="cyan", no_wrap=False, overflow="fold"
        )
        table.add_column("Recorded At", style="green", no_wrap=False)
        table.add_column(
            "Feature Count", style="yellow", justify="right", no_wrap=False
        )

        # Add rows
        for row in snapshots_df.iter_rows(named=True):
            snapshot_version = row["snapshot_version"]
            recorded_at = row["recorded_at"].strftime("%Y-%m-%d %H:%M:%S")
            feature_count = str(row["feature_count"])

            table.add_row(snapshot_version, recorded_at, feature_count)

        console.print(table)
        console.print(f"\nTotal snapshots: {snapshots_df.height}")


@app.command()
def describe(
    snapshot: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--snapshot"],
            help="Snapshot version to describe (defaults to current graph from code)",
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
            snapshot_version = graph.snapshot_version
            console.print("[cyan]Describing current graph from code[/cyan]")
        else:
            # Use specified snapshot
            snapshot_version = snapshot
            console.print(f"[cyan]Describing snapshot: {snapshot_version}[/cyan]")

            # Load graph from snapshot
            features_df = metadata_store.read_features(
                current=False, snapshot_version=snapshot_version
            )

            if features_df.height == 0:
                console.print(
                    f"[red]✗[/red] No features found for snapshot {snapshot_version}"
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
        summary_table = Table(title=f"Graph Snapshot: {snapshot_version}")
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


@app.command()
def render(
    config: Annotated[
        RenderConfig | None, cyclopts.Parameter(name="*", help="Render configuration")
    ] = None,
    format: Annotated[
        str,
        cyclopts.Parameter(
            name=["--format", "-f"],
            help="Output format: terminal, mermaid, or graphviz",
        ),
    ] = "terminal",
    type: Annotated[
        Literal["graph", "cards"],
        cyclopts.Parameter(
            name=["--type", "-t"],
            help="Terminal rendering type: graph or cards (only for --format terminal)",
        ),
    ] = "graph",
    output: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--output", "-o"],
            help="Output file path (default: stdout)",
        ),
    ] = None,
    snapshot: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--snapshot"],
            help="Snapshot version to render (default: current graph from code)",
        ),
    ] = None,
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (for loading historical snapshots)",
        ),
    ] = None,
    # Preset modes
    minimal: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--minimal"],
            help="Minimal output: only feature keys and dependencies",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--verbose"],
            help="Verbose output: show all available information",
        ),
    ] = False,
):
    """Render feature graph visualization.

    Visualize the feature graph in different formats:
    - terminal: Terminal rendering with two types:
      - graph (default): Hierarchical tree view
      - cards: Panel/card-based view with dependency edges
    - mermaid: Mermaid flowchart markup
    - graphviz: Graphviz DOT format

    Examples:
        # Render to terminal (default graph view)
        $ metaxy graph render

        # Render as cards with dependency edges
        $ metaxy graph render --type cards

        # Minimal view
        $ metaxy graph render --minimal

        # Everything
        $ metaxy graph render --verbose

        # Save Mermaid diagram to file
        $ metaxy graph render --format mermaid --output graph.mmd

        # Graphviz DOT format (pipe to dot command)
        $ metaxy graph render --format graphviz | dot -Tpng -o graph.png

        # Custom: show only structure with short hashes
        $ metaxy graph render --no-show-fields --hash-length 6

        # Focus on a specific feature and its dependencies
        $ metaxy graph render --feature video/processing --up 2

        # Show a feature and its downstream dependents
        $ metaxy graph render --feature video/files --down 1

        # Render historical snapshot
        $ metaxy graph render --snapshot abc123... --store prod
    """
    from metaxy.cli.context import get_store
    from metaxy.entrypoints import load_features
    from metaxy.graph import (
        CardsRenderer,
        GraphvizRenderer,
        MermaidRenderer,
        TerminalRenderer,
    )
    from metaxy.models.feature import FeatureGraph

    # Validate format
    valid_formats = ["terminal", "mermaid", "graphviz"]
    if format not in valid_formats:
        console.print(
            f"[red]Error:[/red] Invalid format '{format}'. Must be one of: {', '.join(valid_formats)}"
        )
        raise SystemExit(1)

    # Validate type (only applies to terminal format)
    valid_types = ["graph", "cards"]
    if type not in valid_types:
        console.print(
            f"[red]Error:[/red] Invalid type '{type}'. Must be one of: {', '.join(valid_types)}"
        )
        raise SystemExit(1)

    # Validate type is only used with terminal format
    if type != "graph" and format != "terminal":
        console.print(
            "[red]Error:[/red] --type can only be used with --format terminal"
        )
        raise SystemExit(1)

    # Resolve configuration from presets
    if minimal and verbose:
        console.print("[red]Error:[/red] Cannot specify both --minimal and --verbose")
        raise SystemExit(1)

    # If config is None, create a default instance
    if config is None:
        config = RenderConfig()

    # Apply presets if specified (overrides display settings but preserves filtering)
    if minimal:
        preset = RenderConfig.minimal()
        # Preserve filtering parameters from original config
        preset.feature = config.feature
        preset.up = config.up
        preset.down = config.down
        config = preset
    elif verbose:
        preset = RenderConfig.verbose()
        # Preserve filtering parameters from original config
        preset.feature = config.feature
        preset.up = config.up
        preset.down = config.down
        config = preset

    # Validate direction
    if config.direction not in ["TB", "LR"]:
        console.print(
            f"[red]Error:[/red] Invalid direction '{config.direction}'. Must be TB or LR."
        )
        raise SystemExit(1)

    # Validate filtering options
    if (config.up is not None or config.down is not None) and config.feature is None:
        console.print(
            "[red]Error:[/red] --up and --down require --feature to be specified"
        )
        raise SystemExit(1)

    # Auto-disable field versions if fields are disabled
    if not config.show_fields and config.show_field_versions:
        config.show_field_versions = False

    # Load features from entrypoints
    load_features()

    # Validate feature exists if specified
    if config.feature is not None:
        focus_key = config.get_feature_key()
        graph = FeatureGraph.get_active()
        if focus_key not in graph.features_by_key:
            console.print(
                f"[red]Error:[/red] Feature '{config.feature}' not found in graph"
            )
            console.print("\nAvailable features:")
            for key in sorted(
                graph.features_by_key.keys(), key=lambda k: k.to_string()
            ):
                console.print(f"  • {key.to_string()}")
            raise SystemExit(1)

    # Determine which graph to render
    if snapshot is None:
        # Use current graph from code
        graph = FeatureGraph.get_active()

        if len(graph.features_by_key) == 0:
            console.print(
                "[yellow]Warning:[/yellow] Graph is empty (no features found)"
            )
            if output:
                # Write empty output to file
                with open(output, "w") as f:
                    f.write("")
            return
    else:
        # Load historical snapshot from store
        metadata_store = get_store(store)

        with metadata_store:
            # Read features for this snapshot
            features_df = metadata_store.read_features(
                current=False, snapshot_version=snapshot
            )

            if features_df.height == 0:
                console.print(f"[red]✗[/red] No features found for snapshot {snapshot}")
                raise SystemExit(1)

            # Convert DataFrame to snapshot_data format expected by from_snapshot
            import json

            snapshot_data = {}
            for row in features_df.iter_rows(named=True):
                feature_key_str = row["feature_key"]
                snapshot_data[feature_key_str] = {
                    "feature_spec": json.loads(row["feature_spec"]),
                    "feature_class_path": row["feature_class_path"],
                    "feature_version": row["feature_version"],
                }

            # Reconstruct graph from snapshot
            try:
                graph = FeatureGraph.from_snapshot(snapshot_data)
                console.print(
                    f"[green]✓[/green] Loaded {len(graph.features_by_key)} features from snapshot {snapshot}"
                )
            except ImportError as e:
                console.print(f"[red]✗[/red] Failed to load snapshot: {e}")
                console.print(
                    "[yellow]Hint:[/yellow] Feature classes may have been moved or deleted."
                )
                console.print(
                    "[yellow]Hint:[/yellow] Use --store to ensure feature code is available at recorded paths."
                )
                raise SystemExit(1)
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load snapshot: {e}")
                raise SystemExit(1)

    # Instantiate renderer based on format and type
    if format == "terminal":
        if type == "graph":
            renderer = TerminalRenderer(graph, config)
        elif type == "cards":
            renderer = CardsRenderer(graph, config)
        else:
            # Should not reach here due to validation above
            console.print(f"[red]Error:[/red] Unknown type: {type}")
            raise SystemExit(1)
    elif format == "mermaid":
        renderer = MermaidRenderer(graph, config)
    elif format == "graphviz":
        try:
            renderer = GraphvizRenderer(graph, config)
        except ImportError as e:
            console.print(f"[red]✗[/red] {e}")
            raise SystemExit(1)
    else:
        # Should not reach here due to validation above
        console.print(f"[red]Error:[/red] Unknown format: {format}")
        raise SystemExit(1)

    # Render graph
    try:
        rendered = renderer.render()
    except Exception as e:
        console.print(f"[red]✗[/red] Rendering failed: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

    # Output to stdout or file
    if output:
        try:
            with open(output, "w") as f:
                f.write(rendered)
            console.print(f"[green]✓[/green] Rendered graph saved to: {output}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to write to file: {e}")
            raise SystemExit(1)
    else:
        # Print to stdout
        # For terminal/dag formats, the output already contains ANSI codes from Rich
        # so we print directly to avoid double-escaping
        if format in ("terminal", "dag"):
            print(rendered)
        else:
            console.print(rendered)
