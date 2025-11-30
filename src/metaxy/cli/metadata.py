"""Metadata management commands for Metaxy CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts
import narwhals as nw

from metaxy.cli.console import console, data_console, error_console
from metaxy.cli.context import AppContext
from metaxy.cli.utils import (
    CLIError,
    FeatureSelector,
    FilterArgs,
    OutputFormat,
    exit_with_error,
    load_graph_for_command,
    print_error_item,
)
from metaxy.graph.status import (
    get_feature_metadata_status,
)

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
def cleanup(
    *,
    selector: FeatureSelector = FeatureSelector(),
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Store to clean up (defaults to configured default store).",
        ),
    ] = None,
    # Deletion criteria
    ids: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name=["--id"],
            help="Delete records by primary ID column. Use multiple --id flags to delete several IDs.",
        ),
    ] = None,
    retention_days: Annotated[
        int | None,
        cyclopts.Parameter(
            name=["--retention-days"],
            help="Delete records older than this many days.",
        ),
    ] = None,
    timestamp_column: Annotated[
        str,
        cyclopts.Parameter(
            name=["--timestamp-column"],
            help="Column to use for retention calculation (default: metaxy_created_at).",
        ),
    ] = "metaxy_created_at",
    primary_key: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--primary-key"],
            help="GDPR: Delete all records for this feature primary key (id column) across selected features.",
        ),
    ] = None,
    column_filter: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name=["--filter"],
            help="Delete records matching column filter (e.g., 'status=archived'). Can be specified multiple times.",
        ),
    ] = None,
    hard_delete: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--hard"],
            help="Perform physical deletion (default is soft delete).",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--confirm"],
            help="Confirm the cleanup operation.",
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
    """Clean up old or unwanted metadata from the store.

    Supports deletion by IDs, retention policies, feature primary keys (GDPR), or general filters.

    Examples:
        # Soft delete old records (marks as deleted, filters in queries)
        $ metaxy metadata cleanup --all-features --retention-days 90 --confirm

        # Hard delete (physically remove)
        $ metaxy metadata cleanup --feature logs --retention-days 30 --hard --confirm

        # GDPR: Delete records for a specific primary key across all features
        $ metaxy metadata cleanup --primary-key user_123 --hard --confirm

        # Delete by specific IDs
        $ metaxy metadata cleanup --feature events --id event_1 --id event_2 --confirm

        # Delete with custom filter
        $ metaxy metadata cleanup --feature orders --filter 'status=cancelled' --confirm
    """
    # Validate feature selection
    selector.validate(format)

    _require_cleanup_confirmation(confirm, format)
    _validate_cleanup_criteria(ids, retention_days, primary_key, column_filter, format)

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store.open("write"):
        graph = context.graph

        features_to_cleanup = _resolve_cleanup_features(selector, graph, format)
        if not features_to_cleanup:
            _output_no_cleanup_features(format)
            return

        feature_filters = {
            feature_key: _build_cleanup_filter_for_feature(
                feature_key=feature_key,
                graph=graph,
                ids=ids,
                retention_days=retention_days,
                timestamp_column=timestamp_column,
                primary_key=primary_key,
                column_filter=column_filter,
                format=format,
            )
            for feature_key in features_to_cleanup
        }

        deletion_timestamp = datetime.now(timezone.utc)
        deletion_mode_str = "hard" if hard_delete else "soft"

        features_affected, total_rows_deleted = _execute_cleanup(
            metadata_store=metadata_store,
            features_to_cleanup=features_to_cleanup,
            feature_filters=feature_filters,
            hard_delete=hard_delete,
            format=format,
            deletion_mode_str=deletion_mode_str,
        )

        _output_cleanup_result(
            format=format,
            deletion_mode_str=deletion_mode_str,
            deletion_timestamp=deletion_timestamp,
            total_rows_deleted=total_rows_deleted,
            features_affected=features_affected,
        )


def _require_cleanup_confirmation(confirm: bool, format: OutputFormat) -> None:
    """Ensure destructive cleanup is explicitly confirmed."""
    if confirm:
        return

    exit_with_error(
        CLIError(
            code="MISSING_CONFIRMATION",
            message="This is a destructive operation. Use --confirm to proceed.",
            details={"required_flags": ["--confirm"]},
        ),
        format,
    )


def _validate_cleanup_criteria(
    ids: list[str] | None,
    retention_days: int | None,
    primary_key: str | None,
    column_filter: list[str] | None,
    format: OutputFormat,
) -> None:
    """Ensure at least one cleanup criterion is provided."""
    if any([ids, retention_days, primary_key, column_filter]):
        return

    exit_with_error(
        CLIError(
            code="MISSING_CRITERIA",
            message="Must specify at least one deletion criterion",
            details={
                "available_flags": [
                    "--id",
                    "--retention-days",
                    "--primary-key",
                    "--filter",
                ]
            },
        ),
        format,
    )


def _resolve_cleanup_features(selector: FeatureSelector, graph, format: OutputFormat):
    """Resolve which features should be cleaned up."""
    if selector.all_features:
        return list(graph.list_features(only_current_project=True))

    valid_keys, _ = selector.resolve_keys(graph, format)
    return valid_keys if valid_keys else []


def _output_no_cleanup_features(format: OutputFormat) -> None:
    """Output warning when no features are available for cleanup."""
    if format == "json":
        print(
            json.dumps(
                {
                    "warning": "NO_FEATURES_FOUND",
                    "message": "No features found to clean up.",
                    "total_rows_deleted": 0,
                }
            )
        )
    else:
        console.print("[yellow]Warning:[/yellow] No features found to clean up.")


def _get_single_id_column(feature_key, graph, format: OutputFormat) -> str:
    """Return the single ID column for a feature or exit with an error."""
    feature_cls = graph.features_by_key[feature_key]
    id_columns = feature_cls.spec().id_columns

    if len(id_columns) == 1:
        return id_columns[0]

    exit_with_error(
        CLIError(
            code="MULTI_COLUMN_ID_NOT_SUPPORTED",
            message=f"Feature {feature_key.to_string()} has multiple ID columns {id_columns}. "
            "ID-based filters only support single-column IDs. Use Python API for multi-column IDs.",
            details={"id_columns": id_columns},
        ),
        format,
    )


def _build_cleanup_filter_for_feature(
    *,
    feature_key,
    graph,
    ids: list[str] | None,
    retention_days: int | None,
    timestamp_column: str,
    primary_key: str | None,
    column_filter: list[str] | None,
    format: OutputFormat,
) -> nw.Expr:
    """Combine cleanup criteria into a Narwhals expression for a specific feature."""

    filter_expr: nw.Expr | None = None

    if ids:
        id_col = _get_single_id_column(feature_key, graph, format)
        filter_expr = nw.col(id_col).is_in(ids)

    if retention_days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        retention_filter = nw.col(timestamp_column) < nw.lit(cutoff)
        filter_expr = (
            retention_filter if filter_expr is None else filter_expr & retention_filter
        )

    if primary_key:
        id_col = _get_single_id_column(feature_key, graph, format)
        pk_filter = nw.col(id_col) == primary_key
        filter_expr = pk_filter if filter_expr is None else filter_expr & pk_filter

    if column_filter:
        for filt in column_filter:
            if "=" not in filt:
                exit_with_error(
                    CLIError(
                        code="INVALID_FILTER",
                        message=f"Invalid filter format: '{filt}'. Must be 'column=value'.",
                        details={"example": "--filter 'status=archived'"},
                    ),
                    format,
                )
            key, val = filt.split("=", 1)
            col_name = key.strip()
            col_val = val.strip()
            col_filter = nw.col(col_name) == col_val
            filter_expr = (
                col_filter if filter_expr is None else filter_expr & col_filter
            )

    if filter_expr is None:
        exit_with_error(
            CLIError(
                code="NO_FILTER_BUILT",
                message=f"Failed to build filter expression for feature {feature_key.to_string()} from deletion criteria",
                details={},
            ),
            format,
        )

    return filter_expr


def _execute_cleanup(
    *,
    metadata_store,
    features_to_cleanup,
    feature_filters,
    hard_delete: bool,
    format: OutputFormat,
    deletion_mode_str: str,
):
    """Run the cleanup for each feature and collect results."""
    total_rows_deleted = 0
    features_affected: list[dict[str, Any]] = []

    for feature_key in features_to_cleanup:
        try:
            feature_filter = feature_filters[feature_key]
            if hard_delete:
                result = metadata_store.delete_metadata(
                    feature_key,
                    filter=feature_filter,
                )
            else:
                result = metadata_store.soft_delete_metadata(
                    feature_key,
                    filter=feature_filter,
                )

            total_rows_deleted += result.rows_affected
            features_affected.append(
                {
                    "feature_key": feature_key.to_string(),
                    "rows_deleted": result.rows_affected,
                    "deletion_mode": deletion_mode_str,
                    "error": result.error,
                }
            )
        except NotImplementedError as e:
            exit_with_error(
                CLIError(
                    code="NOT_IMPLEMENTED",
                    message=str(e),
                    details={
                        "hint": "This feature may not be implemented for your metadata store backend."
                    },
                ),
                format,
            )
        except Exception as e:  # pragma: no cover - defensive fallback
            features_affected.append(
                {
                    "feature_key": feature_key.to_string(),
                    "rows_deleted": 0,
                    "deletion_mode": deletion_mode_str,
                    "error": str(e),
                }
            )

    return features_affected, total_rows_deleted


def _output_cleanup_result(
    *,
    format: OutputFormat,
    deletion_mode_str: str,
    deletion_timestamp,
    total_rows_deleted: int,
    features_affected: list[dict[str, Any]],
) -> None:
    """Render cleanup results in the requested format."""
    if format == "json":
        output_data: dict[str, Any] = {
            "total_rows_deleted": total_rows_deleted,
            "timestamp": deletion_timestamp.isoformat(),
            "deletion_mode": deletion_mode_str,
            "features_affected": features_affected,
        }
        print(json.dumps(output_data, indent=2))
        return

    console.print("\n[bold]Cleanup Summary[/bold]")
    console.print(f"  Deletion mode: {deletion_mode_str}")
    console.print(f"  Features affected: {len(features_affected)}")
    console.print(f"  Total rows deleted: {total_rows_deleted}")

    if features_affected:
        console.print("\n[bold]Features:[/bold]")
        for feat in features_affected:
            if feat["error"]:
                console.print(f"  [red]✗[/red] {feat['feature_key']}: {feat['error']}")
            else:
                console.print(
                    f"  [green]✓[/green] {feat['feature_key']}: {feat['rows_deleted']} rows"
                )

    mode_str = "hard deleted" if deletion_mode_str == "hard" else "soft deleted"
    console.print(
        f"\n[green]✓[/green] Cleanup complete: {total_rows_deleted} rows {mode_str}"
    )
