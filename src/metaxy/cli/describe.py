"""Describe commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts
from rich.table import Table

from metaxy.cli.console import console, data_console, error_console
from metaxy.cli.utils import FeatureSelector, OutputFormat

if TYPE_CHECKING:
    from metaxy.models.feature import FeatureGraph
    from metaxy.models.types import FeatureKey

# Describe subcommand app
app = cyclopts.App(
    name="describe",
    help="Describe Metaxy entities in detail",
    console=console,
    error_console=error_console,
)


@app.command()
def graph(
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
    - Project breakdown (if there some features are defined in different projects)

    Example:
        $ metaxy describe graph

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
    """
    from metaxy.cli.context import AppContext
    from metaxy.graph.describe import describe_graph
    from metaxy.models.feature import FeatureGraph

    context = AppContext.get()

    # Determine which snapshot to describe
    if snapshot is None:
        # Use current graph from code
        feature_graph = FeatureGraph.get_active()
        console.print("[cyan]Describing current feature graph...[/cyan]")
    else:
        # Use specified snapshot - requires a metadata store
        console.print(f"[cyan]Describing feature graph snapshot: {snapshot}[/cyan]")

        metadata_store = context.get_store(store)

        # Load graph from snapshot
        from metaxy.metadata_store.system.storage import SystemTableStorage

        with metadata_store:
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

        # For historical snapshots, we'll use the current graph structure
        # but report on the features that were in that snapshot
        feature_graph = FeatureGraph.get_active()

    # Get graph description with optional project filter
    info = describe_graph(feature_graph, project=context.project)

    # Display summary table
    console.print()
    table_title = f"Graph Snapshot: {info['metaxy_snapshot_version']}"
    if context.project:
        table_title += f" (Project: {context.project})"

    summary_table = Table(title=table_title)
    summary_table.add_column("Metric", style="cyan", no_wrap=False)
    summary_table.add_column("Value", style="yellow", justify="right", no_wrap=False)

    # Only show filtered view if filtering actually reduces the feature count
    if "filtered_features" in info and info["filtered_features"] < info["total_features"]:
        # Show both total and filtered counts when there's actual filtering
        summary_table.add_row("Total Features", str(info["total_features"]))
        summary_table.add_row(f"Features in {info['filter_project']}", str(info["filtered_features"]))
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


@app.command(name="features")
@app.command(name="feature", show=False)
def feature(
    selector: FeatureSelector = FeatureSelector(),
    *,
    format: Annotated[
        OutputFormat,
        cyclopts.Parameter(
            name=["-f", "--format"],
            help="Output format: 'plain' (default) or 'json'.",
        ),
    ] = "plain",
) -> None:
    """Describe one or more features in detail.

    Shows comprehensive information about features including project, key,
    version, description, fields with their versions and dependencies.

    Examples:
        $ metaxy describe feature my_feature
        $ metaxy describe feature namespace/feature another/feature
        $ metaxy describe feature my_feature --format json
        $ metaxy describe feature --all-features
    """
    from metaxy.cli.context import AppContext

    context = AppContext.get()
    graph = context.graph

    selector.resolve(format, graph=graph, error_missing=True)

    if not selector:
        return

    features_data = [_collect_feature_data(graph, feature_key) for feature_key in selector]

    if format == "json":
        if len(features_data) == 1:
            print(json.dumps(features_data[0], indent=2))
        else:
            print(json.dumps({"features": features_data}, indent=2))
    else:
        _output_features_plain(features_data)


def _collect_feature_data(graph: FeatureGraph, feature_key: FeatureKey) -> dict[str, Any]:
    """Collect all data for a single feature."""
    from metaxy.models.plan import FQFieldKey

    definition = graph.feature_definitions_by_key[feature_key]
    feature_spec = definition.spec
    version = graph.get_feature_version(feature_key)
    is_root = not feature_spec.deps

    # Get the feature plan for resolved field dependencies
    feature_plan = graph.get_feature_plan(feature_key) if not is_root else None

    # Build fields info with dependencies
    fields_info: list[dict[str, Any]] = []
    for field_key, field_spec in feature_spec.fields_by_key.items():
        field_version = graph.get_field_version(FQFieldKey(feature=feature_key, field=field_key))
        field_data: dict[str, Any] = {
            "key": field_spec.key.to_string(),
            "code_version": field_spec.code_version,
            "version": field_version,
        }

        # Get resolved field dependencies from the plan
        if feature_plan:
            resolved_deps = feature_plan.field_dependencies.get(field_key, {})
            if resolved_deps:
                deps_list = []
                for upstream_feature, upstream_fields in resolved_deps.items():
                    deps_list.append(
                        {
                            "feature": upstream_feature.to_string(),
                            "fields": [f.to_string() for f in upstream_fields],
                        }
                    )
                field_data["deps"] = deps_list

        fields_info.append(field_data)

    # Build dependencies info
    deps_info: list[dict[str, Any]] = []
    for dep in feature_spec.deps:
        dep_data: dict[str, Any] = {
            "feature": dep.feature.to_string(),
        }
        if dep.columns is not None:
            dep_data["columns"] = list(dep.columns)
        if dep.rename:
            dep_data["rename"] = dep.rename
        if dep.optional:
            dep_data["optional"] = True
        deps_info.append(dep_data)

    feature_data: dict[str, Any] = {
        "project": definition.project,
        "key": feature_key.to_string(),
        "version": version,
        "description": feature_spec.description,
        "is_external": definition.is_external,
        "source": definition.source,
        "id_columns": list(feature_spec.id_columns),
        "fields": fields_info,
        "dependencies": deps_info,
    }

    if feature_spec.metadata:
        feature_data["metadata"] = feature_spec.metadata

    return feature_data


def _output_features_plain(features_data: list[dict[str, Any]]) -> None:
    """Output feature descriptions in human-readable block format."""
    for i, feature in enumerate(features_data):
        if i > 0:
            # Horizontal separator between features
            data_console.print()
            data_console.print("[dim]" + "─" * 60 + "[/dim]")
            data_console.print()

        _output_single_feature(feature)


def _output_single_feature(feature: dict[str, Any]) -> None:
    """Output a single feature in human-readable block format."""
    key = feature["key"]
    project = feature["project"]
    version = feature["version"][:8]
    source = feature.get("source", "?")
    id_columns = feature.get("id_columns", [])
    external_marker = " [yellow](external)[/yellow]" if feature.get("is_external") else ""

    # Compact header: key (version) project
    header = f"[bold cyan]{key}[/bold cyan]{external_marker} [dim]({version})[/dim] [green]{project}[/green]"
    data_console.print(header)

    # Source line
    data_console.print(f"[bold]Source:[/bold] {source}")

    # Description block
    description = feature.get("description")
    if description:
        import inspect

        from rich.markdown import Markdown
        from rich.padding import Padding

        data_console.print()
        # Clean up docstring-style indentation
        cleaned = inspect.cleandoc(description)
        md = Markdown(cleaned)
        data_console.print(Padding(md, (0, 0, 0, 2)))

    # ID columns (after description)
    if id_columns:
        data_console.print()
        data_console.print(f"[bold]ID columns:[/bold] {', '.join(id_columns)}")

    # Fields block
    fields = feature.get("fields", [])
    data_console.print()
    data_console.print("[bold]Fields:[/bold]")
    if fields:
        for field in fields:
            field_version = field["version"][:8]
            field_line = f"  [cyan]{field['key']}[/cyan] [dim]({field_version})[/dim]"

            # Show field dependencies
            if "deps" in field and field["deps"]:
                deps_strs = []
                for dep in field["deps"]:
                    if dep["fields"]:
                        fields_str = ", ".join(dep["fields"])
                        deps_strs.append(f"{dep['feature']}:({fields_str})")
                    else:
                        deps_strs.append(dep["feature"])
                field_line += f" [dim]← {', '.join(deps_strs)}[/dim]"

            data_console.print(field_line)
    else:
        data_console.print("  [dim]none[/dim]")

    # Dependencies block
    deps = feature.get("dependencies", [])
    data_console.print()
    data_console.print("[bold]Dependencies:[/bold]")
    if deps:
        for dep in deps:
            dep_line = f"  [cyan]{dep['feature']}[/cyan]"

            options = []
            if dep.get("columns") is not None:
                columns = dep["columns"]
                if columns:
                    options.append(f"columns: {', '.join(columns)}")
                else:
                    options.append("columns: none")

            if dep.get("optional"):
                options.append("[yellow]optional[/yellow]")

            if dep.get("rename"):
                renames = [f"{k} → {v}" for k, v in dep["rename"].items()]
                options.append(f"rename: {', '.join(renames)}")

            if options:
                dep_line += f" [dim]| {' | '.join(options)}[/dim]"

            data_console.print(dep_line)
    else:
        data_console.print("  [dim]none (root feature)[/dim]")

    # Metadata block
    metadata = feature.get("metadata")
    if metadata:
        data_console.print()
        data_console.print("[bold]Metadata:[/bold]")
        for meta_key, meta_value in metadata.items():
            data_console.print(f"  {meta_key}: {meta_value}")
