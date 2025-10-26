"""Graph diff commands for Metaxy CLI."""

from typing import Annotated, Literal

import cyclopts
from rich.console import Console

from metaxy.graph import RenderConfig

# Rich console for formatted output
console = Console()

# Graph-diff subcommand app
app = cyclopts.App(
    name="graph-diff",  # pyrefly: ignore[unexpected-keyword]
    help="Compare and visualize graph snapshots",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
)


@app.command()
def render(
    from_snapshot: Annotated[
        str,
        cyclopts.Parameter(
            help='First snapshot to compare (can be "latest", "current", or snapshot hash)',
        ),
    ],
    to_snapshot: Annotated[
        str,
        cyclopts.Parameter(
            help='Second snapshot to compare (can be "latest", "current", or snapshot hash)',
        ),
    ] = "current",
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (defaults to configured default store)",
        ),
    ] = None,
    format: Annotated[
        Literal["terminal", "cards", "mermaid", "graphviz", "json", "yaml"],
        cyclopts.Parameter(
            name=["--format", "-f"],
            help="Output format: terminal, cards, mermaid, graphviz, json, or yaml",
        ),
    ] = "terminal",
    output: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--output", "-o"],
            help="Output file path (default: stdout)",
        ),
    ] = None,
    config: Annotated[
        RenderConfig | None,
        cyclopts.Parameter(name="*", help="Render configuration"),
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
    """Render merged graph visualization comparing two snapshots.

    Shows all features color-coded by status (added/removed/changed/unchanged).
    Uses the unified rendering system - same renderers as 'metaxy graph render'.

    Special snapshot literals:
    - "latest": Most recent snapshot in the store
    - "current": Current graph state from code

    Output formats:
    - terminal: Hierarchical tree view (default)
    - cards: Panel/card-based view
    - mermaid: Mermaid flowchart diagram
    - graphviz: Graphviz DOT format

    Examples:
        # Show merged graph with default terminal renderer
        $ metaxy graph-diff render latest current

        # Cards view
        $ metaxy graph-diff render latest current --format cards

        # Focus on specific feature with 2 levels up and 1 level down
        $ metaxy graph-diff render latest current --feature user/profile --up 2 --down 1

        # Show only changed fields (hide unchanged)
        $ metaxy graph-diff render latest current --show-changed-fields-only

        # Save Mermaid diagram to file
        $ metaxy graph-diff render latest current --format mermaid --output diff.mmd

        # Graphviz DOT format
        $ metaxy graph-diff render latest current --format graphviz --output diff.dot

        # Minimal view
        $ metaxy graph-diff render latest current --minimal

        # Everything
        $ metaxy graph-diff render latest current --verbose
    """
    from metaxy.cli.context import get_store
    from metaxy.entrypoints import load_features
    from metaxy.graph import (
        CardsRenderer,
        GraphData,
        GraphvizRenderer,
    )
    from metaxy.graph_diff import GraphDiffer, SnapshotResolver
    from metaxy.models.feature import FeatureGraph

    # Validate format
    valid_formats = ["terminal", "cards", "mermaid", "graphviz", "json", "yaml"]
    if format not in valid_formats:
        console.print(
            f"[red]Error:[/red] Invalid format '{format}'. Must be one of: {', '.join(valid_formats)}"
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

    # Validate filtering options
    if (config.up is not None or config.down is not None) and config.feature is None:
        console.print(
            "[red]Error:[/red] --up and --down require --feature to be specified"
        )
        raise SystemExit(1)

    # Load features from entrypoints (needed for "current" literal)
    load_features()
    graph = FeatureGraph.get_active()

    metadata_store = get_store(store)

    with metadata_store:
        # Resolve snapshot versions
        resolver = SnapshotResolver()
        try:
            from_snapshot_version = resolver.resolve_snapshot(
                from_snapshot, metadata_store, graph
            )
            to_snapshot_version = resolver.resolve_snapshot(
                to_snapshot, metadata_store, graph
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)

        # Load snapshot data
        differ = GraphDiffer()
        try:
            from_snapshot_data = differ.load_snapshot_data(
                metadata_store, from_snapshot_version
            )
            to_snapshot_data = differ.load_snapshot_data(
                metadata_store, to_snapshot_version
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)

        # Compute diff
        graph_diff = differ.diff(from_snapshot_data, to_snapshot_data)

        # Create merged graph data
        merged_data = differ.create_merged_graph_data(
            from_snapshot_data, to_snapshot_data, graph_diff
        )

        # Apply graph slicing if requested
        if config.feature is not None:
            try:
                merged_data = differ.filter_merged_graph(
                    merged_data,
                    focus_feature=config.feature,
                    up=config.up,
                    down=config.down,
                )
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise SystemExit(1)

        # Render the diff
        # Use DiffFormatter for terminal/json/yaml/mermaid (has proper diff visualization)
        # Use unified renderers for cards/graphviz (DiffFormatter doesn't support these)
        if format in ("terminal", "mermaid", "json", "yaml"):
            from metaxy.graph.diff.rendering.formatter import DiffFormatter

            formatter = DiffFormatter(console)

            # Determine show_all_fields based on config
            # TODO: add show_changed_fields_only to config
            show_all_fields = True  # Default: show all fields

            try:
                rendered = formatter.format(
                    merged_data=merged_data,
                    format=format,
                    verbose=verbose,
                    diff_only=False,  # Always use merged view for graph-diff render
                    show_all_fields=show_all_fields,
                )
            except Exception as e:
                console.print(f"[red]Error:[/red] Rendering failed: {e}")
                import traceback

                traceback.print_exc()
                raise SystemExit(1)
        else:
            # Use unified renderers for cards/graphviz formats
            from metaxy.graph.diff.rendering.theme import Theme

            theme = Theme.default()
            graph_data = GraphData.from_merged_diff(merged_data)

            if format == "cards":
                renderer = CardsRenderer(
                    graph_data=graph_data, config=config, theme=theme
                )
            elif format == "graphviz":
                renderer = GraphvizRenderer(
                    graph_data=graph_data, config=config, theme=theme
                )
            else:
                console.print(f"[red]Error:[/red] Unknown format: {format}")
                raise SystemExit(1)

            try:
                rendered = renderer.render()
            except Exception as e:
                console.print(f"[red]Error:[/red] Rendering failed: {e}")
                import traceback

                traceback.print_exc()
                raise SystemExit(1)

        # Output to file or stdout
        if output:
            try:
                with open(output, "w") as f:
                    f.write(rendered)
                console.print(f"[green]Success:[/green] Diff rendered to: {output}")
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to write to file: {e}")
                raise SystemExit(1)
        else:
            # Print to stdout
            if format in ("terminal", "cards"):
                # Use plain print for terminal formats (they have ANSI codes)
                print(rendered)
            else:
                # Use Rich console for non-terminal formats
                console.print(rendered)
