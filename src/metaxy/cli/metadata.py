"""Metadata management commands for Metaxy CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import cyclopts
from pydantic import TypeAdapter

from metaxy.cli.console import console, data_console, error_console
from metaxy.graph.status import FullFeatureMetadataRepresentation
from metaxy.models.types import FeatureKey

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
    feature_keys: Annotated[
        list[str],
        cyclopts.Parameter(
            help="Feature keys to inspect (e.g., namespace/feature). Provide at least one.",
        ),
    ],
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store name (defaults to configured default store).",
        ),
    ] = None,
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
        Literal["plain", "json"],
        cyclopts.Parameter(
            name=["--format"],
            help="Output format: 'plain' (default, human-readable) or 'json' (machine-parseable).",
        ),
    ] = "plain",
) -> None:
    """Check metadata completeness and freshness for specified features."""
    import json

    from metaxy.cli.context import AppContext
    from metaxy.graph.status import get_feature_metadata_status

    # Write human-readable output to stdout so tests can capture it
    text_console = data_console

    parsed_keys: list[FeatureKey] = []
    for raw_key in feature_keys:
        try:
            parsed_keys.append(FeatureKey(raw_key))
        except ValueError as exc:
            if format == "json":
                error_data = {
                    "error": "Invalid feature key",
                    "key": raw_key,
                    "message": str(exc),
                }
                print(json.dumps(error_data))
            else:
                text_console.print(
                    f"[red]✗[/red] Invalid feature key '{raw_key}': {exc}"
                )
            raise SystemExit(1)

    context = AppContext.get()
    metadata_store = context.get_store(store)
    as_json = format == "json"

    with metadata_store:
        # If snapshot_version provided, reconstruct graph from storage
        if snapshot_version:
            from metaxy.metadata_store.system.storage import SystemTableStorage

            storage = SystemTableStorage(metadata_store)
            try:
                graph = storage.load_graph_from_snapshot(
                    snapshot_version=snapshot_version,
                    project=context.project,
                )
            except ValueError as e:
                if as_json:
                    error_data = {"error": "Snapshot error", "message": str(e)}
                    print(json.dumps(error_data))
                else:
                    text_console.print(f"[red]✗[/red] {e}")
                raise SystemExit(1)
            except ImportError as e:
                if as_json:
                    error_data = {"error": "Failed to load snapshot", "message": str(e)}
                    print(json.dumps(error_data))
                else:
                    text_console.print(f"[red]✗[/red] Failed to load snapshot: {e}")
                    text_console.print(
                        "[yellow]Hint:[/yellow] Feature classes may have been moved or deleted."
                    )
                raise SystemExit(1)
        else:
            # Use current graph from context
            graph = context.graph

        # Verify all requested features exist in the graph
        missing_in_graph = [
            key for key in parsed_keys if key not in graph.features_by_key
        ]
        if missing_in_graph:
            formatted = ", ".join(key.to_string() for key in missing_in_graph)
            if assert_in_sync:
                # Only exit if assert_in_sync is enabled
                if as_json:
                    error_data = {
                        "error": "Features not found in graph",
                        "features": [k.to_string() for k in missing_in_graph],
                    }
                    print(json.dumps(error_data))
                else:
                    text_console.print(
                        f"[red]✗[/red] Feature(s) not found in graph: {formatted}"
                    )
                raise SystemExit(1)
            else:
                # Otherwise, warn and continue with what we have
                if not as_json:
                    text_console.print(
                        f"[yellow]Warning:[/yellow] Feature(s) not found in graph: {formatted}"
                    )
                # Filter to only existing features
                parsed_keys = [
                    key for key in parsed_keys if key in graph.features_by_key
                ]
                if not parsed_keys:
                    if as_json:
                        print(
                            json.dumps(
                                {
                                    "features": {},
                                    "warning": "No valid features to check",
                                    "snapshot_version": snapshot_version,
                                    "needs_update": False,
                                },
                                indent=2,
                            )
                        )
                    else:
                        text_console.print(
                            "[yellow]Warning:[/yellow] No valid features to check."
                        )
                    return

        needs_update = False

        # Print header for text format
        if not as_json:
            header = (
                f"Metadata status (snapshot {snapshot_version})"
                if snapshot_version
                else "Metadata status"
            )
            text_console.print(f"\n[bold]{header}[/bold]")

        feature_representations: dict[str, FullFeatureMetadataRepresentation] = {}

        for feature_key in parsed_keys:
            feature_cls = graph.features_by_key[feature_key]

            # Use SDK function to get status (returns NamedTuple with status + lazy_increment)
            status_with_increment = get_feature_metadata_status(
                feature_cls, metadata_store
            )

            if status_with_increment.status.needs_update:
                needs_update = True

            if as_json:
                feature_representations[feature_key.to_string()] = (
                    status_with_increment.to_representation(
                        feature_cls=feature_cls, verbose=verbose
                    )
                )
            else:
                # Plain text output
                text_console.print(status_with_increment.status.format_status_line())

                # Verbose output with sample previews
                if verbose:
                    for line in status_with_increment.sample_details(feature_cls):
                        text_console.print(line)

        if as_json:
            adapter = TypeAdapter(dict[str, FullFeatureMetadataRepresentation])
            output = {
                "snapshot_version": snapshot_version,
                "features": json.loads(
                    adapter.dump_json(feature_representations, exclude_none=True)
                ),
                "needs_update": needs_update,
            }
            if missing_in_graph:
                output["warnings"] = {
                    "missing_in_graph": [k.to_string() for k in missing_in_graph]
                }
            print(json.dumps(output, indent=2))

        # Only fail if updates needed AND --assert-in-sync flag is set
        if assert_in_sync and needs_update:
            raise SystemExit(1)


@app.command()
def drop(
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Store name to drop metadata from (defaults to configured default store)",
        ),
    ] = None,
    features: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name=["--feature"],
            help="Feature key to drop (e.g., 'my_feature' or 'group/my_feature'). Can be repeated multiple times. If not specified, uses --all-features.",
        ),
    ] = None,
    all_features: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--all-features"],
            help="Drop metadata for all features defined in the current project's feature graph",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--confirm"],
            help="Confirm the drop operation (required to prevent accidental deletion)",
        ),
    ] = False,
    format: Annotated[
        str,
        cyclopts.Parameter(
            name=["--format"],
            help="Output format: 'text' (default, human-readable) or 'json' (machine-parseable).",
        ),
    ] = "text",
):
    """Drop metadata from a store.

    Removes metadata for specified features from the store.
    This is a destructive operation and requires --confirm flag.

    When using --all-features, drops metadata for all features defined in the
    current project's feature graph.

    Useful for:
    - Cleaning up test data
    - Re-computing feature metadata from scratch
    - Removing obsolete features

    Examples:
        # Drop specific feature (requires confirmation)
        $ metaxy metadata drop --feature user_features --confirm

        # Drop multiple features
        $ metaxy metadata drop --feature user_features --feature customer_features --confirm

        # Drop all features defined in current project
        $ metaxy metadata drop --store dev --all-features --confirm
    """
    import json

    from metaxy.cli.context import AppContext

    # Validate format
    if format not in ("text", "json"):
        error_msg = f"Invalid format '{format}'. Must be 'text' or 'json'."
        if format == "json":
            print(json.dumps({"error": "INVALID_FORMAT", "message": error_msg}))
        else:
            console.print(f"[red]✗[/red] {error_msg}")
        raise SystemExit(1)

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    # Validate arguments
    if not all_features and not features:
        if format == "json":
            print(
                json.dumps(
                    {
                        "error": "MISSING_REQUIRED_FLAG",
                        "message": "Must specify either --all-features or --feature",
                        "required_flags": ["--all-features", "--feature"],
                    }
                )
            )
        else:
            console.print(
                "[red]Error:[/red] Must specify either --all-features or --feature"
            )
        raise SystemExit(1)

    if all_features and features:
        if format == "json":
            print(
                json.dumps(
                    {
                        "error": "CONFLICTING_FLAGS",
                        "message": "Cannot specify both --all-features and --feature",
                        "conflicting_flags": ["--all-features", "--feature"],
                    }
                )
            )
        else:
            console.print(
                "[red]Error:[/red] Cannot specify both --all-features and --feature"
            )
        raise SystemExit(1)

    if not confirm:
        if format == "json":
            print(
                json.dumps(
                    {
                        "error": "MISSING_CONFIRMATION",
                        "message": "This is a destructive operation. Must specify --confirm flag.",
                        "required_flag": "--confirm",
                    }
                )
            )
        else:
            console.print(
                "[red]Error:[/red] This is a destructive operation. Must specify --confirm flag."
            )
        raise SystemExit(1)

    # Parse feature keys
    feature_keys: list[FeatureKey] = []
    if features:
        for feature_str in features:
            # Parse feature key (supports both "feature" and "part1/part2/..." formats)
            if "/" in feature_str:
                parts = feature_str.split("/")
                feature_keys.append(FeatureKey(parts))
            else:
                # Single-part key
                feature_keys.append(FeatureKey([feature_str]))

    # Get store
    metadata_store = context.get_store(store)

    with metadata_store.open("write"):
        # If all_features, get all feature keys from the active feature graph
        if all_features:
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
            # Get all feature keys from the graph (features defined in code for current project)
            feature_keys = graph.list_features(only_current_project=True)

            if not feature_keys:
                if format == "json":
                    print(
                        json.dumps(
                            {
                                "warning": "NO_FEATURES_FOUND",
                                "message": "No features found in active graph. Make sure your features are imported.",
                                "features_dropped": 0,
                            }
                        )
                    )
                else:
                    console.print(
                        "[yellow]Warning:[/yellow] No features found in active graph. "
                        "Make sure your features are imported."
                    )
                return

        if format == "text":
            console.print(
                f"\n[bold]Dropping metadata for {len(feature_keys)} feature(s)...[/bold]\n"
            )

        dropped_features: list[str] = []
        failed_features: list[dict[str, str]] = []

        for feature_key in feature_keys:
            feature_key_str = feature_key.to_string()
            try:
                metadata_store.drop_feature_metadata(feature_key)
                dropped_features.append(feature_key_str)
                if format == "text":
                    console.print(f"[green]✓[/green] Dropped: {feature_key_str}")
            except Exception as e:
                failed_features.append({"feature": feature_key_str, "error": str(e)})
                if format == "text":
                    console.print(f"[red]✗[/red] Failed to drop {feature_key_str}: {e}")

        if format == "json":
            result: dict[str, Any] = {
                "success": True,
                "features_dropped": len(dropped_features),
                "dropped": dropped_features,
            }
            if failed_features:
                result["failed"] = failed_features
            print(json.dumps(result, indent=2))
        else:
            console.print(
                f"\n[green]✓[/green] Drop complete: {len(dropped_features)} feature(s) dropped"
            )
