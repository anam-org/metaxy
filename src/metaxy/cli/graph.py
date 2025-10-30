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
        ℹ Snapshot already recorded (no changes)
          Snapshot version: abc123def456...

        # Or if metadata-only changes:
        ℹ Updated feature graph metadata (no topological changes)
          Features with metadata changes:
            - video/processing
            - user/profile
          Snapshot version: abc123def456...
    """
    from metaxy.cli.context import AppContext

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    metadata_store = context.get_store(store)

    with metadata_store:
        result = metadata_store.record_feature_graph_snapshot()

        # Scenario 1: New snapshot (computational changes)
        if not result.already_recorded:
            console.print("[green]✓[/green] Recorded feature graph")
            console.print(f"  Snapshot version: {result.snapshot_version}")

        # Scenario 2: Metadata-only changes
        elif result.metadata_changed:
            console.print(
                "[blue]ℹ[/blue] Updated feature graph metadata (no topological changes)"
            )
            if result.features_with_spec_changes:
                console.print("  Features with metadata changes:")
                for feature_key in result.features_with_spec_changes:
                    console.print(f"    - {feature_key}")
            console.print(f"  Snapshot version: {result.snapshot_version}")

        # Scenario 3: No changes
        else:
            console.print("[blue]ℹ[/blue] Snapshot already recorded (no changes)")
            console.print(f"  Snapshot version: {result.snapshot_version}")


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
    from metaxy.cli.context import AppContext

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store:
        # Read snapshot history
        snapshots_df = metadata_store.read_graph_snapshots(project=context.project)

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
    - Feature count (optionally filtered by project)
    - Graph depth (longest dependency chain)
    - Root features (features with no dependencies)
    - Leaf features (features with no dependents)
    - Project breakdown (if multi-project)

    Example:
        $ metaxy graph describe

        Graph Snapshot: abc123def456...
        ┌─────────────────────┬────────┐
        │ Metric              │ Value  │
        ├─────────────────────┼────────┤
        │ Feature Count       │ 42     │
        │ Graph Depth         │ 5      │
        │ Root Features       │ 8      │
        │ Leaf Features       │ 12     │
        └─────────────────────┴────────┘

        Root Features:
        • user__profile
        • transaction__history
        ...

        $ metaxy graph describe --project my_project
        Shows metrics filtered to my_project features
    """
    from metaxy.cli.context import AppContext
    from metaxy.graph.describe import describe_graph
    from metaxy.models.feature import FeatureGraph

    context = AppContext.get()
    metadata_store = context.get_store(store)

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
                current=False,
                snapshot_version=snapshot_version,
                project=context.project,
            )

            if features_df.height == 0:
                console.print(
                    f"[red]✗[/red] No features found for snapshot {snapshot_version}"
                )
                if context.project:
                    console.print(f"  (filtered by project: {context.project})")
                return

            # For historical snapshots, we'll use the current graph structure
            # but report on the features that were in that snapshot
            graph = FeatureGraph.get_active()

        # Get graph description with optional project filter
        info = describe_graph(graph, project=context.project)

        # Display summary table
        console.print()
        table_title = f"Graph Snapshot: {info['snapshot_version']}"
        if context.project:
            table_title += f" (Project: {context.project})"

        summary_table = Table(title=table_title)
        summary_table.add_column("Metric", style="cyan", no_wrap=False)
        summary_table.add_column(
            "Value", style="yellow", justify="right", no_wrap=False
        )

        # Only show filtered view if filtering actually reduces the feature count
        if (
            "filtered_features" in info
            and info["filtered_features"] < info["total_features"]
        ):
            # Show both total and filtered counts when there's actual filtering
            summary_table.add_row("Total Features", str(info["total_features"]))
            summary_table.add_row(
                f"Features in {info['filter_project']}", str(info["filtered_features"])
            )
        else:
            # Show simple count when no filtering or all features are in the project
            if "filtered_features" in info:
                # Use filtered count if available (all features are in the project)
                summary_table.add_row("Total Features", str(info["filtered_features"]))
            else:
                # Use total count
                summary_table.add_row("Total Features", str(info["total_features"]))

        summary_table.add_row("Graph Depth", str(info["graph_depth"]))
        summary_table.add_row("Root Features", str(len(info["root_features"])))
        summary_table.add_row("Leaf Features", str(len(info["leaf_features"])))

        console.print(summary_table)

        # Display project breakdown if multi-project
        if len(info["projects"]) > 1:
            console.print("\n[bold]Features by Project:[/bold]")
            for proj, count in sorted(info["projects"].items()):
                console.print(f"  • {proj}: {count} features")

        # Display root features
        if info["root_features"]:
            console.print("\n[bold]Root Features:[/bold]")
            for feature_key_str in info["root_features"][:10]:  # Limit to 10
                console.print(f"  • {feature_key_str}")
            if len(info["root_features"]) > 10:
                console.print(f"  ... and {len(info['root_features']) - 10} more")

        # Display leaf features
        if info["leaf_features"]:
            console.print("\n[bold]Leaf Features:[/bold]")
            for feature_key_str in info["leaf_features"][:10]:  # Limit to 10
                console.print(f"  • {feature_key_str}")
            if len(info["leaf_features"]) > 10:
                console.print(f"  ... and {len(info['leaf_features']) - 10} more")


@app.command()
def render(
    render_config: Annotated[
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
    if render_config is None:
        render_config = RenderConfig()

    # Apply presets if specified (overrides display settings but preserves filtering)
    if minimal:
        preset = RenderConfig.minimal()
        # Preserve filtering parameters from original config
        preset.feature = render_config.feature
        preset.up = render_config.up
        preset.down = render_config.down
        render_config = preset
    elif verbose:
        preset = RenderConfig.verbose()
        # Preserve filtering parameters from original config
        preset.feature = render_config.feature
        preset.up = render_config.up
        preset.down = render_config.down
        render_config = preset

    # Validate direction
    if render_config.direction not in ["TB", "LR"]:
        console.print(
            f"[red]Error:[/red] Invalid direction '{render_config.direction}'. Must be TB or LR."
        )
        raise SystemExit(1)

    # Validate filtering options
    if (
        render_config.up is not None or render_config.down is not None
    ) and render_config.feature is None:
        console.print(
            "[red]Error:[/red] --up and --down require --feature to be specified"
        )
        raise SystemExit(1)

    # Auto-disable field versions if fields are disabled
    if not render_config.show_fields and render_config.show_field_versions:
        render_config.show_field_versions = False

    from metaxy.cli.context import AppContext

    context = AppContext.get()

    # Apply project filter from context if not specified in config
    if render_config.project is None and context.project is not None:
        render_config.project = context.project

    # Validate feature exists if specified
    if render_config.feature is not None:
        focus_key = render_config.get_feature_key()
        graph = context.graph
        if focus_key not in graph.features_by_key:
            console.print(
                f"[red]Error:[/red] Feature '{render_config.feature}' not found in graph"
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
        metadata_store = context.get_store(store)

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
            renderer = TerminalRenderer(graph, render_config)
        elif type == "cards":
            renderer = CardsRenderer(graph, render_config)
        else:
            # Should not reach here due to validation above
            console.print(f"[red]Error:[/red] Unknown type: {type}")
            raise SystemExit(1)
    elif format == "mermaid":
        renderer = MermaidRenderer(graph, render_config)
    elif format == "graphviz":
        try:
            renderer = GraphvizRenderer(graph, render_config)
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
