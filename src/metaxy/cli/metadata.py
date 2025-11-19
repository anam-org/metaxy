"""Metadata management commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

import cyclopts

from metaxy.cli.console import console, error_console
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey

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
) -> None:
    """Check metadata completeness and freshness for specified features."""
    from metaxy.cli.context import AppContext
    from metaxy.graph.status import (
        get_feature_metadata_status,
        preview_samples,
    )
    from metaxy.models.feature import FeatureGraph



    parsed_keys: list[FeatureKey] = []
    for raw_key in feature_keys:
        try:
            parsed_keys.append(FeatureKey(raw_key))
        except ValueError as exc:
            console.print(f"[red]✗[/red] Invalid feature key '{raw_key}': {exc}")
            raise SystemExit(1)

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store:
        # If snapshot_version provided, reconstruct graph from storage
        if snapshot_version:
            features_df = metadata_store.read_features(
                current=False,
                snapshot_version=snapshot_version,
                project=context.project,
            )

            if features_df.height == 0:
                console.print(
                    f"[red]✗[/red] No features recorded for snapshot {snapshot_version}."
                )
                raise SystemExit(1)

            # Build snapshot data dict for FeatureGraph.from_snapshot()
            snapshot_data = {
                row["feature_key"]: {
                    "feature_spec": json.loads(row["feature_spec"])
                    if isinstance(row["feature_spec"], str)
                    else row["feature_spec"],
                    "feature_class_path": row["feature_class_path"],
                    "metaxy_feature_version": row["feature_version"],
                }
                for row in features_df.iter_rows(named=True)
            }

            # Reconstruct graph from snapshot
            try:
                graph = FeatureGraph.from_snapshot(snapshot_data)
            except ImportError as e:
                console.print(f"[red]✗[/red] Failed to load snapshot: {e}")
                console.print(
                    "[yellow]Hint:[/yellow] Feature classes may have been moved or deleted."
                )
                raise SystemExit(1)
        else:
            # Use current graph from context
            graph = context.graph

        # Verify all requested features exist in the graph
        # This consolidates the check - we only check against the graph (whether current or reconstructed)
        missing_in_graph = [
            key for key in parsed_keys if key not in graph.features_by_key
        ]
        if missing_in_graph:
            formatted = ", ".join(key.to_string() for key in missing_in_graph)
            if assert_in_sync:
                # Only exit if assert_in_sync is enabled
                console.print(
                    f"[red]✗[/red] Feature(s) not found in graph: {formatted}"
                )
                raise SystemExit(1)
            else:
                # Otherwise, warn and continue with what we have
                console.print(
                    f"[yellow]Warning:[/yellow] Feature(s) not found in graph: {formatted}"
                )
                # Filter to only existing features
                parsed_keys = [
                    key for key in parsed_keys if key in graph.features_by_key
                ]
                if not parsed_keys:
                    console.print(
                        "[yellow]Warning:[/yellow] No valid features to check."
                    )
                    return

        needs_update = False
        header = (
            f"Metadata status (snapshot {snapshot_version})"
            if snapshot_version
            else "Metadata status"
        )
        console.print(f"\n[bold]{header}[/bold]")

        for feature_key in parsed_keys:
            feature_cls = graph.features_by_key[feature_key]

            # Use SDK function to get status
            status_info = get_feature_metadata_status(feature_cls, metadata_store)

            added_count = status_info["added_count"]
            changed_count = status_info["changed_count"]
            row_count = status_info["row_count"]
            lazy_increment = status_info["lazy_increment"]

            if status_info["needs_update"]:
                needs_update = True

            # Determine status display
            if not status_info["metadata_exists"]:
                status_icon = "[red]✗[/red]"
                status_text = "missing metadata"
            elif status_info["needs_update"]:
                status_icon = "[yellow]⚠[/yellow]"
                status_text = "needs update"
            else:
                status_icon = "[green]✓[/green]"
                status_text = "up-to-date"

            console.print(
                f"{status_icon} {feature_key.to_string()} "
                f"(rows: {row_count}, added: {added_count}, changed: {changed_count}) — {status_text}"
            )

            # Verbose output with sample previews
            if verbose and lazy_increment is not None:
                id_columns_spec = feature_cls.spec().id_columns  # type: ignore[attr-defined]
                id_columns_seq = (
                    tuple(id_columns_spec) if id_columns_spec is not None else None
                )

                if added_count > 0:
                    added_preview_df = preview_samples(
                        lazy_increment.added,
                        id_columns_seq,
                    )
                    if added_preview_df.height > 0:
                        preview_lines = [
                            ", ".join(
                                f"{col}={row[col]}" for col in added_preview_df.columns
                            )
                            for row in added_preview_df.to_dicts()
                        ]
                        console.print("    Added samples: " + "; ".join(preview_lines))

                if changed_count > 0:
                    changed_preview_df = preview_samples(
                        lazy_increment.changed,
                        id_columns_seq,
                    )
                    if changed_preview_df.height > 0:
                        preview_lines = [
                            ", ".join(
                                f"{col}={row[col]}"
                                for col in changed_preview_df.columns
                            )
                            for row in changed_preview_df.to_dicts()
                        ]
                        console.print(
                            "    Changed samples: " + "; ".join(preview_lines)
                        )

        # Only fail if updates needed AND --assert-in-sync flag is set
        if assert_in_sync and needs_update:
            raise SystemExit(1)


@app.command()
def copy(
    from_store: Annotated[
        str,
        cyclopts.Parameter(
            name=["--from", "FROM"],
            help="Source store name (must be configured in metaxy.toml)",
        ),
    ],
    to_store: Annotated[
        str,
        cyclopts.Parameter(
            name=["--to", "TO"],
            help="Destination store name (must be configured in metaxy.toml)",
        ),
    ],
    features: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name=["--feature"],
            help="Feature key to copy (e.g., 'my_feature' or 'group/my_feature'). Can be repeated multiple times. If not specified, uses --all-features.",
        ),
    ] = None,
    all_features: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--all-features"],
            help="Copy all features from source store",
        ),
    ] = False,
    from_snapshot: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--snapshot"],
            help="Snapshot version to copy (defaults to latest in source store). The snapshot_version is preserved in the destination.",
        ),
    ] = None,
    incremental: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--incremental"],
            help="Use incremental copy (compare provenance_by_field to skip existing rows). Disable for better performance if destination is empty or uses deduplication.",
        ),
    ] = True,
):
    """Copy metadata between stores.

    Copies metadata for specified features from one store to another,
    optionally using a historical version. Useful for:
    - Migrating data between environments
    - Backfilling metadata
    - Copying specific feature versions

    Incremental Mode (default):
        By default, performs an anti-join on sample_uid to skip rows that already exist
        in the destination for the same snapshot_version. This prevents duplicate writes.

        Disabling incremental (--no-incremental) may improve performance when:
        - The destination store is empty or has no overlap with source
        - The destination store has eventual deduplication

    Examples:
        # Copy all features from latest snapshot in dev to staging
        $ metaxy metadata copy --from dev --to staging --all-features

        # Copy specific features (repeatable flag)
        $ metaxy metadata copy --from dev --to staging --feature user_features --feature customer_features

        # Copy specific snapshot
        $ metaxy metadata copy --from prod --to staging --all-features --snapshot abc123

        # Non-incremental copy (faster, but may create duplicates)
        $ metaxy metadata copy --from dev --to staging --all-features --no-incremental
    """

    from metaxy.cli.context import AppContext

    context = AppContext.get()
    config = context.config

    # Validate arguments
    if not all_features and not features:
        console.print(
            "[red]Error:[/red] Must specify either --all-features or --feature"
        )
        raise SystemExit(1)

    if all_features and features:
        console.print(
            "[red]Error:[/red] Cannot specify both --all-features and --feature"
        )
        raise SystemExit(1)

    # Parse feature keys
    feature_keys: list[CoercibleToFeatureKey] | None = None
    if features:
        feature_keys = []
        for feature_str in features:
            # Parse feature key (supports both "feature" and "part1/part2/..." formats)
            if "/" in feature_str:
                parts = feature_str.split("/")
                feature_keys.append(FeatureKey(parts))
            else:
                # Single-part key
                feature_keys.append(FeatureKey([feature_str]))

    # Get stores
    console.print(f"[cyan]Source store:[/cyan] {from_store}")
    console.print(f"[cyan]Destination store:[/cyan] {to_store}")

    source_store = config.get_store(from_store)
    dest_store = config.get_store(to_store)

    # Open both stores and copy
    with source_store.open(), dest_store.open("write"):
        console.print("\n[bold]Starting copy operation...[/bold]\n")

        try:
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=feature_keys,
                from_snapshot=from_snapshot,
                incremental=incremental,
            )

            console.print(
                f"\n[green]✓[/green] Copy complete: {stats['features_copied']} features, {stats['rows_copied']} rows"
            )

        except Exception as e:
            console.print(f"\n[red]✗[/red] Copy failed:\n{e}")
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
    from metaxy.cli.context import AppContext

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    # Validate arguments
    if not all_features and not features:
        console.print(
            "[red]Error:[/red] Must specify either --all-features or --feature"
        )
        raise SystemExit(1)

    if all_features and features:
        console.print(
            "[red]Error:[/red] Cannot specify both --all-features and --feature"
        )
        raise SystemExit(1)

    if not confirm:
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
                console.print(
                    "[yellow]Warning:[/yellow] No features found in active graph. "
                    "Make sure your features are imported."
                )
                return

        console.print(
            f"\n[bold]Dropping metadata for {len(feature_keys)} feature(s)...[/bold]\n"
        )

        dropped_count = 0
        for feature_key in feature_keys:
            try:
                metadata_store.drop_feature_metadata(feature_key)
                console.print(f"[green]✓[/green] Dropped: {feature_key.to_string()}")
                dropped_count += 1
            except Exception as e:
                console.print(
                    f"[red]✗[/red] Failed to drop {feature_key.to_string()}: {e}"
                )

        console.print(
            f"\n[green]✓[/green] Drop complete: {dropped_count} feature(s) dropped"
        )
