"""List commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts
from rich.table import Table

from metaxy.cli.console import console, data_console, error_console
from metaxy.cli.utils import OutputFormat

if TYPE_CHECKING:
    pass

# List subcommand app
app = cyclopts.App(
    name="list",  # pyrefly: ignore[unexpected-keyword]
    help="List Metaxy entities",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    error_console=error_console,  # pyrefly: ignore[unexpected-keyword]
)


@app.command()
def features(
    *,
    verbose: Annotated[
        bool,
        cyclopts.Parameter(
            name=["-v", "--verbose"],
            help="Show detailed information including field dependencies and versions.",
        ),
    ] = False,
    format: Annotated[
        OutputFormat,
        cyclopts.Parameter(
            name=["-f", "--format"],
            help="Output format: 'plain' (default) or 'json'.",
        ),
    ] = "plain",
) -> None:
    """List Metaxy features in the current project.

    Examples:
        $ metaxy list features
        $ metaxy list features --verbose
        $ metaxy list features --format json
    """
    from metaxy import get_feature_by_key
    from metaxy.cli.context import AppContext
    from metaxy.models.plan import FQFieldKey

    context = AppContext.get()
    graph = context.graph

    # Collect feature data
    features_data: list[dict[str, Any]] = []

    for feature_key, feature_spec in graph.feature_specs_by_key.items():
        if (
            context.project
            and get_feature_by_key(feature_key).project != context.project
        ):
            continue

        version = graph.get_feature_version(feature_key)

        # Determine if it's a root feature (no deps)
        is_root = not feature_spec.deps

        # Get the feature plan for resolved field dependencies
        feature_plan = graph.get_feature_plan(feature_key) if verbose else None

        # Build field info
        fields_info: list[dict[str, Any]] = []
        for field_key, field_spec in feature_spec.fields_by_key.items():
            field_version = graph.get_field_version(
                FQFieldKey(feature=feature_key, field=field_key)
            )
            field_data: dict[str, Any] = {
                "key": field_spec.key.to_string(),
                "code_version": field_spec.code_version,
                "version": field_version,
            }

            # In verbose mode, get resolved field dependencies from the plan
            if verbose and feature_plan and not is_root:
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

        # Build feature info
        feature_data: dict[str, Any] = {
            "key": feature_key.to_string(),
            "version": version,
            "is_root": is_root,
            "field_count": len(fields_info),
            "fields": fields_info,
        }

        if verbose and feature_spec.deps:
            feature_data["deps"] = [
                dep.feature.to_string() for dep in feature_spec.deps
            ]

        features_data.append(feature_data)

    # Output based on format
    if format == "json":
        output: dict[str, Any] = {
            "feature_count": len(features_data),
            "features": features_data,
        }
        print(json.dumps(output, indent=2))
    else:
        _output_features_plain(features_data, verbose)


def _output_features_plain(features_data: list[dict[str, Any]], verbose: bool) -> None:
    """Output features in plain format using rich tables."""
    if not features_data:
        data_console.print("[yellow]No features found in the current project.[/yellow]")
        return

    # Create main table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Type", justify="center", no_wrap=True)
    table.add_column("Feature", no_wrap=True)
    table.add_column("Fields", justify="right", no_wrap=True)
    table.add_column("Version", no_wrap=True)

    if verbose:
        table.add_column("Dependencies", no_wrap=False)

    for feature in features_data:
        # Root features get ○, dependent features get ◆
        type_icon = "[cyan]○[/cyan]" if feature["is_root"] else "[blue]◆[/blue]"

        # Truncate version hash for display (first 12 chars)
        version_display = (
            feature["version"][:12] + "..."
            if len(feature["version"]) > 12
            else feature["version"]
        )

        if verbose:
            deps_display = ", ".join(feature.get("deps", [])) or "-"
            table.add_row(
                type_icon,
                feature["key"],
                str(feature["field_count"]),
                version_display,
                deps_display,
            )
        else:
            table.add_row(
                type_icon,
                feature["key"],
                str(feature["field_count"]),
                version_display,
            )

    data_console.print(table)

    # Summary
    root_count = sum(1 for f in features_data if f["is_root"])
    dependent_count = len(features_data) - root_count
    data_console.print()
    data_console.print(
        f"[dim]Total: {len(features_data)} feature(s) "
        f"([cyan]○[/cyan] {root_count} root, [blue]◆[/blue] {dependent_count} dependent)[/dim]"
    )

    # Verbose: show field details for each feature
    if verbose:
        data_console.print()
        for feature in features_data:
            data_console.print(f"[bold cyan]`{feature['key']}` fields[/bold cyan]")

            field_table = Table(show_header=True, header_style="bold dim")
            field_table.add_column("Field", no_wrap=True)
            field_table.add_column("Code Version", no_wrap=True)
            field_table.add_column("Version", no_wrap=True)
            field_table.add_column("Dependencies", no_wrap=False)

            for field in feature["fields"]:
                field_version_display = (
                    field["version"][:12] + "..."
                    if len(field["version"]) > 12
                    else field["version"]
                )
                deps_str = "-"
                if "deps" in field:
                    deps_list = []
                    for dep in field["deps"]:
                        if "special" in dep:
                            # SpecialFieldDep.ALL
                            deps_list.append("[all upstream]")
                        else:
                            for dep_field in dep["fields"]:
                                if dep_field == "*":
                                    deps_list.append(f"{dep['feature']}.*")
                                else:
                                    deps_list.append(f"{dep['feature']}.{dep_field}")
                    deps_str = ", ".join(deps_list) if deps_list else "-"

                field_table.add_row(
                    field["key"],
                    field["code_version"],
                    field_version_display,
                    deps_str,
                )

            data_console.print(field_table)
            data_console.print()
