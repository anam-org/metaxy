"""Metadata management commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts
from rich.table import Table

from metaxy.cli.console import console, data_console, error_console
from metaxy.cli.utils import FeatureSelector, FilterArgs, OutputFormat

if TYPE_CHECKING:
    from metaxy import BaseFeature
    from metaxy.graph.status import FeatureMetadataStatusWithIncrement


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
            help="Whether to display sample slices of dataframes.",
        ),
    ] = False,
    progress: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--progress"],
            help="Display progress percentage showing how many input units have been processed at least once. Stale samples are counted as processed.",
        ),
    ] = False,
    allow_fallback_stores: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--allow-fallback-stores"],
            help="Whether to read metadata from fallback stores.",
        ),
    ] = True,
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

        # Print header for plain format only when using a snapshot
        if format == "plain" and snapshot_version:
            data_console.print(
                f"\n[bold]Metadata status (snapshot {snapshot_version})[/bold]"
            )

        # Collect status for all features
        needs_update = False
        feature_reps: dict[str, Any] = {}
        # Each item is (feature_cls, status_with_increment, error_msg)
        # error_msg is None for success, or a string for errors
        collected_statuses: list[
            tuple[
                type[BaseFeature], FeatureMetadataStatusWithIncrement | None, str | None
            ]
        ] = []

        errors: list[tuple[str, str, Exception]] = []

        # Enable progress calculation if --progress or --verbose is set
        compute_progress = progress or verbose

        for feature_key in valid_keys:
            feature_cls = graph.features_by_key[feature_key]
            try:
                status_with_increment = get_feature_metadata_status(
                    feature_cls,
                    metadata_store,
                    use_fallback=allow_fallback_stores,
                    global_filters=global_filters,
                    compute_progress=compute_progress,
                )

                if status_with_increment.status.needs_update:
                    needs_update = True

                if format == "json":
                    feature_reps[feature_key.to_string()] = (
                        status_with_increment.to_representation(verbose=verbose)
                    )
                else:
                    collected_statuses.append(
                        (feature_cls, status_with_increment, None)
                    )
            except Exception as e:
                # Log the error and continue with other features
                error_msg = str(e)
                errors.append((feature_key.to_string(), error_msg, e))

                if format == "json":
                    from metaxy.graph.status import FullFeatureMetadataRepresentation

                    feature_reps[feature_key.to_string()] = (
                        FullFeatureMetadataRepresentation(
                            feature_key=feature_key.to_string(),
                            status="error",
                            needs_update=False,
                            metadata_exists=False,
                            store_rows=0,
                            missing=None,
                            stale=None,
                            orphaned=None,
                            target_version="",
                            is_root_feature=False,
                            error_message=error_msg,
                        )
                    )
                else:
                    # For plain format, store (feature_cls, None, error_msg)
                    collected_statuses.append((feature_cls, None, error_msg))

        # Output plain format as Rich Table
        if format == "plain":
            from rich.traceback import Traceback

            from metaxy.graph.status import _STATUS_ICONS

            # Print rich tracebacks for any errors before the table
            if errors:
                for feat, _, exc in errors:
                    error_console.print(
                        f"\n[bold red]Error processing feature:[/bold red] {feat}"
                    )
                    error_console.print(
                        Traceback.from_exception(type(exc), exc, exc.__traceback__)
                    )

            table = Table(show_header=True, header_style="bold")
            table.add_column("Status", justify="center", no_wrap=True)
            table.add_column("Feature", no_wrap=True)
            table.add_column("Materialized", justify="right", no_wrap=True)
            table.add_column("Missing", justify="right", no_wrap=True)
            table.add_column("Stale", justify="right", no_wrap=True)
            table.add_column("Orphaned", justify="right", no_wrap=True)
            table.add_column("Info", no_wrap=False)

            verbose_details: list[tuple[str, list[str]]] = []

            for feature_cls, status_with_increment, error_msg in collected_statuses:
                # Handle error case
                if error_msg is not None:
                    icon = _STATUS_ICONS["error"]
                    feature_key_str = feature_cls.spec().key.to_string()
                    # Truncate error message for display
                    truncated_error = (
                        error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
                    )
                    table.add_row(
                        icon,
                        feature_key_str,
                        "-",
                        "-",
                        "-",
                        "-",
                        f"[red]{truncated_error}[/red]",
                    )
                    continue

                # status_with_increment is not None at this point
                assert status_with_increment is not None
                status = status_with_increment.status
                icon = _STATUS_ICONS[status.status_category]
                feature_key_str = status.feature_key.to_string()

                # Determine if this is a "no input" case
                no_input = (
                    compute_progress
                    and not status.is_root_feature
                    and status.progress_percentage is None
                )

                # Format status column with progress if available
                if status.progress_percentage is not None:
                    status_str = f"{icon} ({status.progress_percentage:.0f}%)"
                elif no_input:
                    # Progress was requested but None means no upstream input available
                    # Show warning icon since we can't determine status without input
                    warning_icon = _STATUS_ICONS["needs_update"]
                    status_str = f"{warning_icon} [dim](no input)[/dim]"
                else:
                    status_str = icon

                # Format store metadata for Info column
                info_str = str(status.store_metadata) if status.store_metadata else ""

                # For root features or no-input cases, show "-" for Missing/Stale/Orphaned
                if status.is_root_feature or no_input:
                    table.add_row(
                        status_str,
                        feature_key_str,
                        str(status.store_row_count),
                        "-",
                        "-",
                        "-",
                        info_str,
                    )
                else:
                    table.add_row(
                        status_str,
                        feature_key_str,
                        str(status.store_row_count),
                        str(status.missing_count),
                        str(status.stale_count),
                        str(status.orphaned_count),
                        info_str,
                    )

                # Collect verbose details for later
                if verbose:
                    sample_details = status_with_increment.sample_details()
                    if sample_details:
                        verbose_details.append((feature_key_str, sample_details))

            data_console.print(table)

            # Print verbose sample details after the table
            if verbose and verbose_details:
                for feature_key_str, details in verbose_details:
                    data_console.print()  # Blank line before each feature
                    data_console.print(
                        f"[bold cyan]`{feature_key_str}` preview[/bold cyan]"
                    )
                    data_console.print()  # Blank line after title
                    for line in details:
                        data_console.print(line)
                        data_console.print()  # Blank line after each section

            # Print error summary (tracebacks already printed above)
            if errors:
                data_console.print()
                data_console.print(
                    f"[yellow]Warning:[/yellow] {len(errors)} feature(s) had errors (see tracebacks above)"
                )

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
            warnings_dict: dict[str, Any] = {}
            if missing_keys:
                warnings_dict["missing_in_graph"] = [
                    k.to_string() for k in missing_keys
                ]
            if errors:
                warnings_dict["errors"] = [
                    {"feature": feat, "error": err} for feat, err, _ in errors
                ]
            if warnings_dict:
                output["warnings"] = warnings_dict
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


@app.command()
def copy(
    from_store: Annotated[
        str,
        cyclopts.Parameter(
            name=["--from"],
            help="Source store name to copy metadata from.",
        ),
    ],
    to_store: Annotated[
        str,
        cyclopts.Parameter(
            name=["--to"],
            help="Destination store name to copy metadata to.",
        ),
    ],
    *features: Annotated[
        str,
        cyclopts.Parameter(
            help="One or more feature keys to copy, separated by whitespaces.",
        ),
    ],
    filters: FilterArgs | None = None,
    current_only: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--current-only"],
            help="Only copy rows with the current feature_version (as defined in loaded feature graph).",
        ),
    ] = False,
    latest_only: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--latest-only"],
            help="Deduplicate samples by keeping only the latest row per id_columns group.",
        ),
    ] = True,
) -> None:
    """Copy metadata from one store to another.

    Copies metadata for specified features from source to destination store.
    By default, copies all versions (--no-current-only) and deduplicates by
    keeping only the latest row per sample (--latest-only).

    Examples:
        $ metaxy metadata copy user_features --from prod --to dev
        $ metaxy metadata copy feat1 feat2 --from prod --to dev
        $ metaxy metadata copy feat1 --from prod --to dev --current-only
        $ metaxy metadata copy feat1 --from prod --to dev --filter "sample_uid IN (1, 2)"
    """
    from pydantic import ValidationError
    from rich.status import Status

    from metaxy import coerce_to_feature_key
    from metaxy.cli.context import AppContext
    from metaxy.models.types import FeatureKey

    filters = filters or []

    # Require at least one feature
    if not features:
        data_console.print("[red]Error:[/red] At least one feature must be specified.")
        raise SystemExit(1)

    context = AppContext.get()

    # Get source and destination stores
    source_store = context.get_store(from_store)
    dest_store = context.get_store(to_store)

    # Parse and resolve feature keys
    graph = context.graph
    valid_keys: list[FeatureKey] = []
    missing_keys: list[FeatureKey] = []

    for raw_key in features:
        try:
            key = coerce_to_feature_key(raw_key)
            if key in graph.features_by_key:
                valid_keys.append(key)
            else:
                missing_keys.append(key)
        except ValidationError as exc:
            error_console.print(
                f"[red]Error:[/red] Invalid feature key '{raw_key}': {exc}"
            )
            raise SystemExit(1)

    # Handle missing features
    if missing_keys:
        formatted = ", ".join(k.to_string() for k in missing_keys)
        data_console.print(
            f"[yellow]Warning:[/yellow] Feature(s) not found in graph: {formatted}"
        )

    # Handle no valid features
    if not valid_keys:
        data_console.print("[yellow]Warning:[/yellow] No valid features to copy.")
        return

    # Convert global filters list to the format expected by copy_metadata
    global_filters = filters if filters else None

    with Status(
        f"Copying metadata for {len(valid_keys)} feature(s) from '{from_store}' to '{to_store}'...",
        console=data_console,
        spinner="dots",
    ):
        with source_store.open("read"), dest_store.open("write"):
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=list(valid_keys),
                global_filters=global_filters,
                current_only=current_only,
                latest_only=latest_only,
            )

    data_console.print(
        f"[green]✓[/green] Copy complete: {stats['features_copied']} feature(s), {stats['rows_copied']} row(s) copied"
    )
