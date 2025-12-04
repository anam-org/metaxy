"""Metadata management commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts

from metaxy.cli.console import console, data_console, error_console
from metaxy.cli.utils import FeatureSelector, FilterArgs, OutputFormat

if TYPE_CHECKING:
    pass


# Metadata subcommand app
app = cyclopts.App(
    name="metadata",  # pyrefly: ignore[unexpected-keyword]
    help="Manage Metaxy metadata",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    error_console=error_console,  # pyrefly: ignore[unexpected-keyword]
)


@app.command()
def status(
    *,
    selector: FeatureSelector = FeatureSelector(),
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store name (defaults to configured default store).",
        ),
    ] = None,
    filters: FilterArgs | None = None,
    snapshot_version: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--snapshot-id"],
            help="Check metadata against a specific snapshot version.",
        ),
    ] = None,
    assert_in_sync: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--assert-in-sync"],
            help="Exit with error if any feature needs updates or metadata is missing.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--verbose"],
            help="Show additional details about samples needing updates.",
        ),
    ] = False,
    format: Annotated[
        OutputFormat,
        cyclopts.Parameter(
            name=["--format"],
        ),
    ] = "plain",
) -> None:
    """Check metadata completeness and freshness for specified features.

    Examples:
        $ metaxy metadata status --feature user_features
        $ metaxy metadata status --feature feat1 --feature feat2
        $ metaxy metadata status --all-features
        $ metaxy metadata status --store dev --all-features
        $ metaxy metadata status --all-features --filter "status = 'active'"
    """
    from metaxy.cli.context import AppContext
    from metaxy.cli.utils import CLIError, exit_with_error, load_graph_for_command
    from metaxy.graph.status import get_feature_metadata_status

    filters = filters or []

    # Validate feature selection
    selector.validate(format)

    # Filters are already parsed by the converter
    global_filters = filters if filters else None

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store:
        # Load graph (from snapshot or current)
        graph = load_graph_for_command(
            context, snapshot_version, metadata_store, format
        )

        # Resolve feature keys
        valid_keys, missing_keys = selector.resolve_keys(graph, format)

        # Handle empty result for --all-features
        if selector.all_features and not valid_keys:
            _output_no_features_warning(format, snapshot_version)
            return

        # Handle missing features
        if missing_keys:
            if assert_in_sync:
                exit_with_error(
                    CLIError(
                        code="FEATURES_NOT_FOUND",
                        message="Feature(s) not found in graph",
                        details={"features": [k.to_string() for k in missing_keys]},
                    ),
                    format,
                )
            elif format == "plain":
                formatted = ", ".join(k.to_string() for k in missing_keys)
                data_console.print(
                    f"[yellow]Warning:[/yellow] Feature(s) not found in graph: {formatted}"
                )

        # If no valid features remain
        if not valid_keys:
            _output_no_features_warning(format, snapshot_version)
            return

        # Print header for plain format
        if format == "plain":
            header = (
                f"Metadata status (snapshot {snapshot_version})"
                if snapshot_version
                else "Metadata status"
            )
            data_console.print(f"\n[bold]{header}[/bold]")

        # Collect status for all features
        needs_update = False
        feature_reps: dict[str, Any] = {}

        for feature_key in valid_keys:
            feature_cls = graph.features_by_key[feature_key]
            status_with_increment = get_feature_metadata_status(
                feature_cls, metadata_store, global_filters=global_filters
            )

            if status_with_increment.status.needs_update:
                needs_update = True

            if format == "json":
                feature_reps[feature_key.to_string()] = (
                    status_with_increment.to_representation(
                        feature_cls=feature_cls, verbose=verbose
                    )
                )
            else:
                data_console.print(status_with_increment.status.format_status_line())
                # Print store metadata (table_name, uri, etc.)
                store_metadata = status_with_increment.status.store_metadata
                if store_metadata:
                    data_console.print("    ", store_metadata)
                if verbose:
                    for line in status_with_increment.sample_details(feature_cls):
                        data_console.print(line)

        # Output JSON result
        if format == "json":
            from pydantic import TypeAdapter

            from metaxy.graph.status import FullFeatureMetadataRepresentation

            adapter = TypeAdapter(dict[str, FullFeatureMetadataRepresentation])
            output: dict[str, Any] = {
                "snapshot_version": snapshot_version,
                "features": json.loads(
                    adapter.dump_json(feature_reps, exclude_none=True)
                ),
                "needs_update": needs_update,
            }
            if missing_keys:
                output["warnings"] = {
                    "missing_in_graph": [k.to_string() for k in missing_keys]
                }
            print(json.dumps(output, indent=2))

        # Exit with error if assert_in_sync and updates needed
        if assert_in_sync and needs_update:
            raise SystemExit(1)


def _output_no_features_warning(
    format: OutputFormat, snapshot_version: str | None
) -> None:
    """Output warning when no features are found to check."""
    if format == "json":
        print(
            json.dumps(
                {
                    "warning": "No valid features to check",
                    "features": {},
                    "snapshot_version": snapshot_version,
                    "needs_update": False,
                },
                indent=2,
            )
        )
    else:
        data_console.print("[yellow]Warning:[/yellow] No valid features to check.")


@app.command()
def drop(
    *,
    selector: FeatureSelector = FeatureSelector(),
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Store name to drop metadata from (defaults to configured default store).",
        ),
    ] = None,
    confirm: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--confirm"],
            help="Confirm the drop operation (required to prevent accidental deletion).",
        ),
    ] = False,
    format: Annotated[
        OutputFormat,
        cyclopts.Parameter(
            name=["--format"],
            help="Output format: 'plain' (default) or 'json'.",
        ),
    ] = "plain",
) -> None:
    """Drop metadata from a store.

    Removes metadata for specified features. This is destructive and requires --confirm.

    Examples:
        $ metaxy metadata drop --feature user_features --confirm
        $ metaxy metadata drop --feature feat1 --feature feat2 --confirm
        $ metaxy metadata drop --store dev --all-features --confirm
    """
    from metaxy.cli.context import AppContext
    from metaxy.cli.utils import CLIError, exit_with_error

    # Validate feature selection
    selector.validate(format)

    # Require confirmation
    if not confirm:
        exit_with_error(
            CLIError(
                code="MISSING_CONFIRMATION",
                message="This is a destructive operation. Must specify --confirm flag.",
                details={"required_flag": "--confirm"},
            ),
            format,
        )

    context = AppContext.get()
    context.raise_command_cannot_override_project()
    metadata_store = context.get_store(store)

    with metadata_store.open("write"):
        graph = context.graph

        # Resolve feature keys
        valid_keys, _ = selector.resolve_keys(graph, format)

        # Handle no features
        if not valid_keys:
            if format == "json":
                print(
                    json.dumps(
                        {
                            "warning": "NO_FEATURES_FOUND",
                            "message": "No features found in active graph.",
                            "features_dropped": 0,
                        }
                    )
                )
            else:
                console.print(
                    "[yellow]Warning:[/yellow] No features found in active graph."
                )
            return

        if format == "plain":
            console.print(
                f"\n[bold]Dropping metadata for {len(valid_keys)} feature(s)...[/bold]\n"
            )

        # Drop each feature
        dropped: list[str] = []
        failed: list[dict[str, str]] = []

        for feature_key in valid_keys:
            key_str = feature_key.to_string()
            try:
                metadata_store.drop_feature_metadata(feature_key)
                dropped.append(key_str)
                if format == "plain":
                    console.print(f"[green]✓[/green] Dropped: {key_str}")
            except Exception as e:
                failed.append({"feature": key_str, "error": str(e)})
                if format == "plain":
                    from metaxy.cli.utils import print_error_item

                    print_error_item(
                        console, key_str, e, prefix="[red]✗[/red] Failed to drop"
                    )

        # Output result
        if format == "json":
            result: dict[str, Any] = {
                "success": True,
                "features_dropped": len(dropped),
                "dropped": dropped,
            }
            if failed:
                result["failed"] = failed
            print(json.dumps(result, indent=2))
        else:
            console.print(
                f"\n[green]✓[/green] Drop complete: {len(dropped)} feature(s) dropped"
            )
