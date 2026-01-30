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
    name="list",
    help="List Metaxy entities",
    console=console,
    error_console=error_console,
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
    from metaxy.cli.context import AppContext
    from metaxy.models.plan import FQFieldKey

    context = AppContext.get()
    graph = context.graph

    # Collect feature data
    features_data: list[dict[str, Any]] = []

    for feature_key, definition in graph.feature_definitions_by_key.items():
        if context.project and definition.project != context.project:
            continue

        feature_spec = definition.spec
        version = graph.get_feature_version(feature_key)

        # Determine if it's a root feature (no deps)
        is_root = not feature_spec.deps

        # Get import path from definition
        import_path = definition.feature_class_path

        # Get the feature plan for resolved field dependencies
        feature_plan = graph.get_feature_plan(feature_key) if verbose else None

        # Build field info
        fields_info: list[dict[str, Any]] = []
        for field_key, field_spec in feature_spec.fields_by_key.items():
            field_version = graph.get_field_version(FQFieldKey(feature=feature_key, field=field_key))
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
            "is_external": definition.is_external,
            "project": definition.project,
            "import_path": import_path,
            "field_count": len(fields_info),
            "fields": fields_info,
        }

        if verbose and feature_spec.deps:
            feature_data["deps"] = [dep.feature.to_string() for dep in feature_spec.deps]

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

    # Group features by project

    features_by_project: dict[str, list[dict[str, Any]]] = {}
    for feature in features_data:
        project = feature["project"]
        if project not in features_by_project:
            features_by_project[project] = []
        features_by_project[project].append(feature)

    # Output each project group
    for project, project_features in features_by_project.items():
        table = Table(
            title=f"[bold]{project}[/bold]",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Feature", no_wrap=True)
        table.add_column("Version", no_wrap=True)
        table.add_column("Import Path", no_wrap=True)

        if verbose:
            table.add_column("Dependencies")

        for feature in project_features:
            feature_key_display = feature["key"]

            # Show import path or <external> marker
            if feature["is_external"]:
                import_path_display = "[dim]<external>[/dim]"
            else:
                import_path_display = feature["import_path"] or "[dim]-[/dim]"

            if verbose:
                deps_display = ", ".join(feature.get("deps", [])) or "-"
                table.add_row(
                    feature_key_display,
                    feature["version"],
                    import_path_display,
                    deps_display,
                )
            else:
                table.add_row(
                    feature_key_display,
                    feature["version"],
                    import_path_display,
                )

        data_console.print(table)
        data_console.print()

    # Summary
    root_count = sum(1 for f in features_data if f["is_root"])
    dependent_count = len(features_data) - root_count
    external_count = sum(1 for f in features_data if f["is_external"])
    external_suffix = f", {external_count} external" if external_count else ""
    data_console.print(
        f"[dim]Total: {len(features_data)} feature(s) ({root_count} root, {dependent_count} dependent{external_suffix})[/dim]"
    )

    # Verbose: show field details for each feature
    if verbose:
        data_console.print()
        for feature in features_data:
            if feature["is_external"]:
                import_path = "[dim]<external>[/dim]"
            else:
                import_path = feature["import_path"] or "-"
            data_console.print(
                f"[bold cyan]{feature['key']}[/bold cyan] [dim]({feature['project']})[/dim] {import_path}"
            )

            field_table = Table(show_header=True, header_style="bold dim")
            field_table.add_column("Field", no_wrap=True)
            field_table.add_column("Code Version", no_wrap=True)
            field_table.add_column("Version", no_wrap=True)
            field_table.add_column("Dependencies", no_wrap=False)

            for field in feature["fields"]:
                field_version_display = (
                    field["version"][:12] + "..." if len(field["version"]) > 12 else field["version"]
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


@app.command()
def stores(
    *,
    format: Annotated[
        OutputFormat,
        cyclopts.Parameter(
            name=["-f", "--format"],
            help="Output format: 'plain' (default) or 'json'.",
        ),
    ] = "plain",
) -> None:
    """List configured metadata stores.

    Examples:
        $ metaxy list stores
        $ metaxy list stores --format json
    """
    from metaxy.cli.context import AppContext

    context = AppContext.get()
    config = context.config

    # Collect store data
    stores_data: list[dict[str, Any]] = []

    for name, store_config in config.stores.items():
        store = config.get_store(name)
        fallback_names = store_config.config.get("fallback_stores", [])
        store_data: dict[str, Any] = {
            "name": name,
            "is_default": name == config.store,
            "info": store.display(),
            "fallbacks": fallback_names,
        }
        stores_data.append(store_data)

    # Sort: default store first, then alphabetically
    stores_data.sort(key=lambda s: (not s["is_default"], s["name"]))

    # Output based on format
    if format == "json":
        output: dict[str, Any] = {
            "default_store": config.store,
            "store_count": len(stores_data),
            "stores": stores_data,
        }
        print(json.dumps(output, indent=2))
    else:
        _output_stores_plain(stores_data, config.store)


def _output_stores_plain(stores_data: list[dict[str, Any]], default_store: str) -> None:
    """Output stores in plain format using rich tables."""
    if not stores_data:
        data_console.print("[yellow]No stores configured.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", no_wrap=True)
    table.add_column("Info", no_wrap=False)
    table.add_column("Fallbacks", no_wrap=False)

    for store in stores_data:
        name = store["name"]
        if store["is_default"]:
            name = f"[bold cyan]{name}[/bold cyan] [dim](default)[/dim]"

        fallbacks = ", ".join(store["fallbacks"]) if store["fallbacks"] else "[dim]-[/dim]"

        table.add_row(name, store["info"], fallbacks)

    data_console.print(table)
    data_console.print()
    data_console.print(f"[dim]Total: {len(stores_data)} store(s)[/dim]")
