"""Metadata management commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts
import narwhals as nw
from rich.table import Table

from metaxy.cli.console import console, data_console, error_console
from metaxy.cli.utils import FeatureSelector, FilterArgs, GlobalFilterArgs, OutputFormat

if TYPE_CHECKING:
    from metaxy import FeatureDefinition
    from metaxy.graph.status import FeatureMetadataStatusWithIncrement
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.types import FeatureKey


# Metadata subcommand app
app = cyclopts.App(
    name="metadata",
    help="Manage Metaxy metadata",
    console=console,
    error_console=error_console,
)


@app.command()
def status(
    selector: FeatureSelector = FeatureSelector(),
    *,
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store name (defaults to configured default store).",
        ),
    ] = None,
    filters: FilterArgs | None = None,
    global_filters: GlobalFilterArgs | None = None,
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
        $ metaxy metadata status my/feature
        $ metaxy metadata status feat1 feat2
        $ metaxy metadata status --all-features
        $ metaxy metadata status --store dev --all-features
        $ metaxy metadata status --all-features --filter "status = 'active'"
    """
    from metaxy.cli.context import AppContext
    from metaxy.cli.utils import load_graph_for_command
    from metaxy.graph.status import get_feature_metadata_status

    # Normalize filter arguments
    target_filters_list = filters if filters else None
    global_filters_list = global_filters if global_filters else None

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store:
        # Load graph (from snapshot or current)
        graph = load_graph_for_command(context, snapshot_version, metadata_store, format)

        # Resolve feature keys
        selector.resolve(format, graph=graph, error_missing=assert_in_sync)
        missing_keys = selector.missing_keys

        # Handle empty result for --all-features
        if selector.all_features and not selector:
            _output_no_features_warning(format, snapshot_version)
            return

        # If no valid features remain
        if not selector:
            _output_no_features_warning(format, snapshot_version)
            return

        # Print header for plain format only when using a snapshot
        if format == "plain" and snapshot_version:
            data_console.print(f"\n[bold]Metadata status (snapshot {snapshot_version})[/bold]")

        # Collect status for all features
        needs_update = False
        feature_reps: dict[str, Any] = {}
        # Each item is (definition, status_with_increment, error_msg)
        # error_msg is None for success, or a string for errors
        collected_statuses: list[tuple[FeatureDefinition, FeatureMetadataStatusWithIncrement | None, str | None]] = []

        errors: list[tuple[str, str, Exception]] = []

        # Enable progress calculation if --progress or --verbose is set
        compute_progress = progress or verbose

        for feature_key in selector:
            definition = graph.get_feature_definition(feature_key)
            try:
                status_with_increment = get_feature_metadata_status(
                    definition,
                    metadata_store,
                    use_fallback=allow_fallback_stores,
                    global_filters=global_filters_list,
                    target_filters=target_filters_list,
                    compute_progress=compute_progress,
                )

                if status_with_increment.status.needs_update:
                    needs_update = True

                if format == "json":
                    feature_reps[feature_key.to_string()] = status_with_increment.to_representation(verbose=verbose)
                else:
                    collected_statuses.append((definition, status_with_increment, None))
            except Exception as e:
                # Log the error and continue with other features
                error_msg = str(e)
                errors.append((feature_key.to_string(), error_msg, e))

                if format == "json":
                    from metaxy.graph.status import FullFeatureMetadataRepresentation

                    feature_reps[feature_key.to_string()] = FullFeatureMetadataRepresentation(
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
                else:
                    # For plain format, store (definition, None, error_msg)
                    collected_statuses.append((definition, None, error_msg))

        # Output plain format as Rich Table
        if format == "plain":
            from rich.traceback import Traceback

            from metaxy.graph.status import _STATUS_ICONS

            # Print rich tracebacks for any errors before the table
            if errors:
                for feat, _, exc in errors:
                    error_console.print(f"\n[bold red]Error processing feature:[/bold red] {feat}")
                    error_console.print(Traceback.from_exception(type(exc), exc, exc.__traceback__))

            table = Table(show_header=True, header_style="bold")
            table.add_column("Status", justify="center", no_wrap=True)
            table.add_column("Feature", no_wrap=True)
            table.add_column("Materialized", justify="right", no_wrap=True)
            table.add_column("Missing", justify="right", no_wrap=True)
            table.add_column("Stale", justify="right", no_wrap=True)
            table.add_column("Orphaned", justify="right", no_wrap=True)
            table.add_column("Info", no_wrap=False)

            verbose_details: list[tuple[str, list[str]]] = []

            for definition, status_with_increment, error_msg in collected_statuses:
                # Handle error case
                if error_msg is not None:
                    icon = _STATUS_ICONS["error"]
                    feature_key_str = definition.key.to_string()
                    # Truncate error message for display
                    truncated_error = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
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
                no_input = compute_progress and not status.is_root_feature and status.progress_percentage is None

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
                    data_console.print(f"[bold cyan]`{feature_key_str}` preview[/bold cyan]")
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
                "features": json.loads(adapter.dump_json(feature_reps, exclude_none=True)),
                "needs_update": needs_update,
            }
            warnings_dict: dict[str, Any] = {}
            if missing_keys:
                warnings_dict["missing_in_graph"] = [k.to_string() for k in missing_keys]
            if errors:
                warnings_dict["errors"] = [{"feature": feat, "error": err} for feat, err, _ in errors]
            if warnings_dict:
                output["warnings"] = warnings_dict
            print(json.dumps(output, indent=2))

        # Exit with error if assert_in_sync and updates needed
        if assert_in_sync and needs_update:
            raise SystemExit(1)


@app.command()
def delete(
    selector: FeatureSelector = FeatureSelector(),
    *,
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store name (defaults to configured default store).",
        ),
    ] = None,
    filters: FilterArgs | None = None,
    soft: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--soft"],
            negative="--hard",
            help="Whether to mark records with deletion timestamps vs physically remove them.",
        ),
    ] = True,
    current_only: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--current-only"],
            help="Only delete rows with the current feature_version (as defined in loaded feature graph).",
        ),
    ] = True,
    yes: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--yes"],
            help="Confirm deletion without prompting (required for hard deletes without filters).",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--dry-run"],
            help="Preview deletion: show features, filters, and row counts without executing.",
        ),
    ] = False,
) -> None:
    """Delete metadata rows matching filters.

    Examples:
        $ metaxy metadata delete my/feature --filter "status = 'inactive'"
        $ metaxy metadata delete --all-features --filter "created_at < '2024-01-01'" --hard
        $ metaxy metadata delete test_feature --filter "1=1" --yes  # Delete all rows
    """
    from metaxy.cli.context import AppContext
    from metaxy.cli.utils import CLIError, CLIErrorCode, exit_with_error

    filters = filters or []

    # Require --yes confirmation for hard deletes when no filters provided
    if not soft and not filters and not yes:
        exit_with_error(
            CLIError(
                code=CLIErrorCode.MISSING_CONFIRMATION,
                message="Hard deleting all metadata requires --yes flag to prevent accidental deletion.",
                hint="Use --yes to confirm, or provide --filter to restrict deletion.",
            ),
            "plain",
        )

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store:
        # Resolve feature keys
        selector.resolve("plain")

        if not selector:
            exit_with_error(
                CLIError(
                    code=CLIErrorCode.NO_FEATURES,
                    message="No valid features selected for deletion.",
                ),
                "plain",
            )

        # Dry-run mode: count rows, print features and filters, then exit
        if dry_run:
            store_name = store if store else context.config.store
            row_counts = _count_rows_to_delete(metadata_store, selector, filters, soft, current_only)
            _print_dry_run_info(selector, filters, soft, current_only, store_name, row_counts)
            return

        errors: dict[str, str] = {}

        with metadata_store.open("write"):
            for feature_key in selector:
                try:
                    metadata_store.delete_metadata(
                        feature_key,
                        filters=filters,
                        soft=soft,
                        current_only=current_only,
                    )
                except Exception as e:  # pragma: no cover - CLI surface
                    error_msg = str(e)
                    errors[feature_key.to_string()] = error_msg
                    error_console.print(f"[red]Error deleting {feature_key.to_string()}:[/red] {error_msg}")

        mode_str = "soft" if soft else "hard"
        console.print(f"[green]✓[/green] Deletion complete ({mode_str} delete)")

        if errors:
            raise SystemExit(1)


def _output_no_features_warning(format: OutputFormat, snapshot_version: str | None) -> None:
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


def _count_rows_to_delete(
    store: MetadataStore,
    selector: FeatureSelector,
    filters: list[nw.Expr],
    soft: bool,
    current_only: bool,
) -> dict[FeatureKey, int | str]:
    """Count rows that would be deleted for each feature.

    Returns a dict mapping FeatureKey to row counts.
    If counting fails for a feature, the value will be an error message string.

    The reason this isn't implemented on the MetadataStore class is because this is an estimation,
    we don't actually get a number of deleted rows from the storage backend.
    """

    row_counts: dict[FeatureKey, int | str] = {}
    count_frames: list[tuple[FeatureKey, nw.LazyFrame]] = []

    with store.open("read"):
        # Build count lazy frames for all features
        for feature_key in selector:
            try:
                # Match the same parameters as delete_metadata uses for soft deletes
                # For hard deletes, this is an approximation since hard delete uses
                # _delete_metadata_impl directly, but this gives a reasonable count
                lazy = store.read_metadata(
                    feature_key,
                    filters=filters,
                    include_soft_deleted=False,
                    current_only=current_only,
                    latest_only=True,  # matches delete_metadata default
                    allow_fallback=soft,  # soft deletes use fallback, hard deletes don't
                )
                # Pre-aggregate to count per feature
                count_frame = lazy.select(nw.len().alias("count"))
                count_frames.append((feature_key, count_frame))
            except Exception as e:
                row_counts[feature_key] = f"error: {e}"

        # Concat and collect all at once for parallel execution
        if count_frames:
            combined = nw.concat([cf for _, cf in count_frames])
            counts_dicts = combined.collect().to_polars().to_dicts()

            # Map counts back to feature keys (order preserved from concat)
            for (feature_key, _), row in zip(count_frames, counts_dicts):
                row_counts[feature_key] = row["count"]

    return row_counts


def _print_dry_run_info(
    selector: FeatureSelector,
    filters: list[nw.Expr],
    soft: bool,
    current_only: bool,
    store_name: str,
    row_counts: dict[FeatureKey, int | str],
) -> None:
    """Print dry-run information for delete command."""
    mode_str = "[yellow]soft[/yellow]" if soft else "[red]hard[/red]"
    current_only_str = "[green]yes[/green]" if current_only else "[yellow]no[/yellow]"
    console.print(f"[bold cyan]Dry run[/bold cyan] ({mode_str} delete, current_only={current_only_str})")
    console.print()

    console.print(f"[bold]Store:[/bold] [blue]{store_name}[/blue]")
    console.print()

    # Calculate total rows
    total_rows = sum(c for c in row_counts.values() if isinstance(c, int))

    console.print(f"[bold]Features[/bold] [dim]({len(selector)})[/dim]:")
    for feature_key in selector:
        count = row_counts.get(feature_key, "?")
        if isinstance(count, int):
            count_str = f"[dim]({count} rows)[/dim]"
        else:
            count_str = f"[red]{count}[/red]"
        console.print(f"  [cyan]•[/cyan] {feature_key.to_string()} {count_str}")
    console.print()

    console.print(f"[bold]Total rows to delete:[/bold] [yellow]{total_rows}[/yellow]")
    console.print()

    console.print("[bold]Filters:[/bold]")
    if filters:
        for f in filters:
            console.print(f"  [magenta]•[/magenta] {f!r}")
    else:
        console.print("  [dim](none)[/dim]")


@app.command()
def copy(
    selector: FeatureSelector = FeatureSelector(),
    *,
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
        $ metaxy metadata copy my/feature --from prod --to dev
        $ metaxy metadata copy feat1 feat2 --from prod --to dev
        $ metaxy metadata copy --all-features --from prod --to dev
        $ metaxy metadata copy feat1 --from prod --to dev --current-only
        $ metaxy metadata copy feat1 --from prod --to dev --filter "sample_uid IN (1, 2)"
    """
    from rich.status import Status

    from metaxy.cli.context import AppContext

    filters = filters or []

    context = AppContext.get()

    # Get source and destination stores
    source_store = context.get_store(from_store)
    dest_store = context.get_store(to_store)

    # Resolve feature keys
    selector.resolve("plain")

    # Handle no valid features
    if not selector:
        data_console.print("[yellow]Warning:[/yellow] No valid features to copy.")
        return

    # Convert global filters list to the format expected by copy_metadata
    global_filters = filters if filters else None

    with Status(
        f"Copying metadata for {len(selector)} feature(s) from '{from_store}' to '{to_store}'...",
        console=data_console,
        spinner="dots",
    ):
        with source_store.open("read"), dest_store.open("write"):
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=list(selector),
                global_filters=global_filters,
                current_only=current_only,
                latest_only=latest_only,
            )

    data_console.print(
        f"[green]✓[/green] Copy complete: {stats['features_copied']} feature(s), {stats['rows_copied']} row(s) copied"
    )
