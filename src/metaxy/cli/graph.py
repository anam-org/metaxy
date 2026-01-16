"""Graph management commands for Metaxy CLI."""

from typing import TYPE_CHECKING, Annotated, Literal

import cyclopts
from rich.table import Table

from metaxy.cli.console import console, data_console, error_console
from metaxy.graph import RenderConfig

if TYPE_CHECKING:
    from metaxy.graph import (
        CardsRenderer,
        GraphvizRenderer,
        MermaidRenderer,
        TerminalRenderer,
    )
    from metaxy.models.feature import FeatureGraph

# Graph subcommand app
app = cyclopts.App(
    name="graph",  # pyrefly: ignore[unexpected-keyword]
    help="Manage feature graphs",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    error_console=error_console,  # pyrefly: ignore[unexpected-keyword]
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
    *,
    tags: Annotated[
        dict[str, str] | None,
        cyclopts.Parameter(
            name=["--tags", "-t"],
            help="Arbitrary key-value pairs to attach to the pushed snapshot. Example: `--tags.git_commit abc123def`.",
        ),
    ] = None,
):
    """Serialize all Metaxy features to the metadata store.

    This is intended to be invoked in a CD pipeline **before** running Metaxy code in production.
    """
    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.system.models import METAXY_TAG
    from metaxy.metadata_store.system.storage import SystemTableStorage

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    metadata_store = context.get_store(store)

    tags = tags or {}

    assert METAXY_TAG not in tags, "`metaxy` tag is reserved for internal use"

    with metadata_store.open("write"):
        result = SystemTableStorage(metadata_store).push_graph_snapshot(tags=tags)

        # Log store metadata for the system table
        from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

        store_metadata = metadata_store.get_store_metadata(FEATURE_VERSIONS_KEY)
        if store_metadata:
            console.print(f"[dim]Recorded at: {store_metadata}[/dim]")

        # Scenario 1: New snapshot (computational changes)
        if not result.already_pushed:
            console.print("[green]✓[/green] Recorded feature graph")

        # Scenario 2: Feature info updates to existing snapshot
        elif result.updated_features:
            console.print(
                "[blue]ℹ[/blue] [cyan]Updated feature information[/cyan] (no topological changes)"
            )
            console.print("  [dim]Updated features:[/dim]")
            for feature_key in result.updated_features:
                console.print(f"    [yellow]- {feature_key}[/yellow]")

        # Scenario 3: No changes
        else:
            console.print(
                "[green]✓[/green] [green]Snapshot already recorded[/green] [dim](no changes)[/dim]"
            )

        # Always output the snapshot version to stdout (for scripting)
        # Note: snapshot_version is "empty" when graph has no features
        data_console.print(result.snapshot_version)


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

    from metaxy.metadata_store.system.storage import SystemTableStorage

    with metadata_store:
        # Read snapshot history
        storage = SystemTableStorage(metadata_store)
        snapshots_df = storage.read_graph_snapshots(project=context.project)

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
            snapshot_version = row["metaxy_snapshot_version"]
            recorded_at = row["recorded_at"].strftime("%Y-%m-%d %H:%M:%S")
            feature_count = str(row["feature_count"])

            table.add_row(snapshot_version, recorded_at, feature_count)

        console.print(table)
        console.print(f"\nTotal snapshots: {snapshots_df.height}")


def _get_feature_count_for_table(info: dict) -> str:
    """Get the appropriate feature count to display in the summary table."""
    if "filtered_features" in info:
        return str(info["filtered_features"])
    return str(info["total_features"])


def _display_describe_summary(info: dict, project: str | None) -> None:
    """Display the summary table for graph describe."""
    table_title = f"Graph Snapshot: {info['metaxy_snapshot_version']}"
    if project:
        table_title += f" (Project: {project})"

    summary_table = Table(title=table_title)
    summary_table.add_column("Metric", style="cyan", no_wrap=False)
    summary_table.add_column("Value", style="yellow", justify="right", no_wrap=False)

    has_filtering = (
        "filtered_features" in info
        and info["filtered_features"] < info["total_features"]
    )
    if has_filtering:
        summary_table.add_row("Total Features", str(info["total_features"]))
        summary_table.add_row(
            f"Features in {info['filter_project']}", str(info["filtered_features"])
        )
    else:
        summary_table.add_row("Total Features", _get_feature_count_for_table(info))

    summary_table.add_row("Graph Depth", str(info["graph_depth"]))
    summary_table.add_row("Root Features", str(len(info["root_features"])))
    summary_table.add_row("Leaf Features", str(len(info["leaf_features"])))
    console.print(summary_table)


def _display_feature_list(title: str, features: list[str], limit: int = 10) -> None:
    """Display a list of features with optional truncation."""
    if not features:
        return
    console.print(f"\n[bold]{title}:[/bold]")
    for feature_key_str in features[:limit]:
        console.print(f"  • {feature_key_str}")
    if len(features) > limit:
        console.print(f"  ... and {len(features) - limit} more")


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
        if snapshot is None:
            graph = FeatureGraph.get_active()
            console.print("[cyan]Describing current graph from code[/cyan]")
        else:
            console.print(f"[cyan]Describing snapshot: {snapshot}[/cyan]")
            from metaxy.metadata_store.system.storage import SystemTableStorage

            storage = SystemTableStorage(metadata_store)
            features_df = storage.read_features(
                current=False,
                snapshot_version=snapshot,
                project=context.project,
            )

            if features_df.height == 0:
                console.print(f"[red]✗[/red] No features found for snapshot {snapshot}")
                if context.project:
                    console.print(f"  (filtered by project: {context.project})")
                return

            graph = FeatureGraph.get_active()

        info = describe_graph(graph, project=context.project)
        console.print()
        _display_describe_summary(info, context.project)

        if len(info["projects"]) > 1:
            console.print("\n[bold]Features by Project:[/bold]")
            for proj, count in sorted(info["projects"].items()):
                console.print(f"  • {proj}: {count} features")

        _display_feature_list("Root Features", info["root_features"])
        _display_feature_list("Leaf Features", info["leaf_features"])


def _validate_render_args(
    format: str, type: str, minimal: bool, verbose: bool, render_config: RenderConfig
) -> None:
    """Validate render command arguments."""
    valid_formats = ["terminal", "mermaid", "graphviz"]
    if format not in valid_formats:
        console.print(
            f"[red]Error:[/red] Invalid format '{format}'. Must be one of: {', '.join(valid_formats)}"
        )
        raise SystemExit(1)

    valid_types = ["graph", "cards"]
    if type not in valid_types:
        console.print(
            f"[red]Error:[/red] Invalid type '{type}'. Must be one of: {', '.join(valid_types)}"
        )
        raise SystemExit(1)

    if type != "graph" and format != "terminal":
        console.print(
            "[red]Error:[/red] --type can only be used with --format terminal"
        )
        raise SystemExit(1)

    if minimal and verbose:
        console.print("[red]Error:[/red] Cannot specify both --minimal and --verbose")
        raise SystemExit(1)

    if render_config.direction not in ["TB", "LR"]:
        console.print(
            f"[red]Error:[/red] Invalid direction '{render_config.direction}'. Must be TB or LR."
        )
        raise SystemExit(1)

    if (
        render_config.up is not None or render_config.down is not None
    ) and render_config.feature is None:
        console.print(
            "[red]Error:[/red] --up and --down require --feature to be specified"
        )
        raise SystemExit(1)


def _apply_render_preset(
    render_config: RenderConfig, minimal: bool, verbose: bool
) -> RenderConfig:
    """Apply preset configuration while preserving filtering parameters."""
    if not minimal and not verbose:
        return render_config

    preset = (
        RenderConfig.minimal(show_projects=render_config.show_projects)
        if minimal
        else RenderConfig.verbose(show_projects=render_config.show_projects)
    )
    preset.feature = render_config.feature
    preset.up = render_config.up
    preset.down = render_config.down
    return preset


def _load_graph_from_snapshot(snapshot: str, store: str | None) -> "FeatureGraph":
    """Load a graph from a historical snapshot."""
    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.system.storage import SystemTableStorage

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store:
        storage = SystemTableStorage(metadata_store)
        try:
            graph = storage.load_graph_from_snapshot(snapshot_version=snapshot)
        except ValueError as e:
            from metaxy.cli.utils import print_error

            print_error(console, "Snapshot error", e)
            raise SystemExit(1)
        except ImportError as e:
            from metaxy.cli.utils import print_error

            print_error(console, "Failed to load snapshot", e)
            console.print(
                "[yellow]Hint:[/yellow] Feature classes may have been moved or deleted."
            )
            raise SystemExit(1) from e
        except Exception as e:
            from metaxy.cli.utils import print_error

            print_error(console, "Failed to load snapshot", e)
            raise SystemExit(1) from e

        console.print(
            f"[green]✓[/green] Loaded {len(graph.features_by_key)} features from snapshot {snapshot}"
        )
        return graph


def _create_renderer(
    format: str, type: str, graph: "FeatureGraph", render_config: RenderConfig
) -> "TerminalRenderer | CardsRenderer | MermaidRenderer | GraphvizRenderer":
    """Create the appropriate renderer for the given format and type."""
    from metaxy.graph import (
        CardsRenderer,
        GraphvizRenderer,
        MermaidRenderer,
        TerminalRenderer,
    )

    if format == "terminal":
        if type == "cards":
            return CardsRenderer(graph, render_config)
        return TerminalRenderer(graph, render_config)
    if format == "mermaid":
        return MermaidRenderer(graph, render_config)
    if format == "graphviz":
        try:
            return GraphvizRenderer(graph, render_config)
        except ImportError as e:
            console.print(f"[red]✗[/red] {e}")
            raise SystemExit(1)
    console.print(f"[red]Error:[/red] Unknown format: {format}")
    raise SystemExit(1)


def _get_graph_for_render(
    snapshot: str | None,
    store: str | None,
    render_config: RenderConfig,
    output: str | None,
) -> "FeatureGraph | None":
    """Get the feature graph for rendering, either from code or a snapshot."""
    from metaxy.models.feature import FeatureGraph

    if snapshot is not None:
        return _load_graph_from_snapshot(snapshot, store)

    graph = FeatureGraph.get_active()

    if render_config.feature is not None:
        focus_key = render_config.get_feature_key()
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

    if len(graph.features_by_key) == 0:
        console.print("[yellow]Warning:[/yellow] Graph is empty (no features found)")
        if output:
            with open(output, "w") as f:
                f.write("")
        return None

    return graph


def _output_rendered(rendered: str, output: str | None) -> None:
    """Output the rendered graph to file or stdout."""
    if output:
        try:
            with open(output, "w") as f:
                f.write(rendered)
            console.print(f"[green]✓[/green] Rendered graph saved to: {output}")
        except Exception as e:
            from metaxy.cli.utils import print_error

            print_error(console, "Failed to write to file", e)
            raise SystemExit(1)
    else:
        data_console.print(rendered)


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
    from metaxy.cli.context import AppContext

    if render_config is None:
        render_config = RenderConfig()

    _validate_render_args(format, type, minimal, verbose, render_config)
    render_config = _apply_render_preset(render_config, minimal, verbose)

    if not render_config.show_fields and render_config.show_field_versions:
        render_config.show_field_versions = False

    context = AppContext.get()
    if render_config.project is None and context.project is not None:
        render_config.project = context.project

    graph = _get_graph_for_render(snapshot, store, render_config, output)
    if graph is None:
        return

    renderer = _create_renderer(format, type, graph, render_config)

    try:
        rendered = renderer.render()
    except Exception as e:
        from metaxy.cli.utils import print_error

        print_error(console, "Rendering failed", e)
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

    _output_rendered(rendered, output)
