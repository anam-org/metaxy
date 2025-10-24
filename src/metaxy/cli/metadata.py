"""Metadata management commands for Metaxy CLI."""

from typing import TYPE_CHECKING, Annotated

import cyclopts
from rich.console import Console

from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store import FilteredFeature
    from metaxy.models.feature import Feature

# Rich console for formatted output
console = Console()

# Metadata subcommand app
app = cyclopts.App(
    name="metadata",  # pyrefly: ignore[unexpected-keyword]
    help="Manage Metaxy metadata",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
)


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
            help="Feature key to copy (e.g., 'my_feature' or 'namespace__my_feature'). Can be repeated multiple times. If not specified, uses --all-features.",
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
            help="Snapshot ID to copy (defaults to latest in source store). The snapshot_id is preserved in the destination.",
        ),
    ] = None,
):
    """Copy metadata between stores.

    Copies metadata for specified features from one store to another,
    optionally using a historical version. Useful for:
    - Migrating data between environments
    - Backfilling metadata
    - Copying specific feature versions

    Examples:
        # Copy all features from latest snapshot in dev to staging
        $ metaxy metadata copy --from dev --to staging --all-features

        # Copy specific features (repeatable flag)
        $ metaxy metadata copy --from dev --to staging --feature user_features --feature customer_features

        # Copy specific snapshot
        $ metaxy metadata copy --from prod --to staging --all-features --snapshot abc123
    """
    import logging

    from metaxy.cli.context import get_config

    # Enable logging to show progress
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = get_config()

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
    feature_keys: list[FeatureKey | type[Feature] | FilteredFeature] | None = None
    if features:
        feature_keys = []
        for feature_str in features:
            # Parse feature key (supports both "feature" and "namespace__feature" formats)
            if "__" in feature_str:
                namespace_parts = feature_str.split("__")
                feature_keys.append(FeatureKey(namespace_parts))
            else:
                # Single-part key
                feature_keys.append(FeatureKey([feature_str]))

    # Get stores
    console.print(f"[cyan]Source store:[/cyan] {from_store}")
    console.print(f"[cyan]Destination store:[/cyan] {to_store}")

    source_store = config.get_store(from_store)
    dest_store = config.get_store(to_store)

    # Open both stores and copy
    with source_store, dest_store:
        console.print("\n[bold]Starting copy operation...[/bold]\n")

        try:
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=feature_keys,
                from_snapshot=from_snapshot,
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
            help="Feature key to drop (e.g., 'my_feature' or 'namespace__my_feature'). Can be repeated multiple times. If not specified, uses --all-features.",
        ),
    ] = None,
    all_features: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--all-features"],
            help="Drop metadata for all features in the store",
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

    Useful for:
    - Cleaning up test data
    - Re-computing feature metadata from scratch
    - Removing obsolete features

    Examples:
        # Drop specific feature (requires confirmation)
        $ metaxy metadata drop --feature user_features --confirm

        # Drop multiple features
        $ metaxy metadata drop --feature user_features --feature customer_features --confirm

        # Drop all features from specific store
        $ metaxy metadata drop --store dev --all-features --confirm
    """
    from metaxy.cli.context import get_store

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
            # Parse feature key (supports both "feature" and "namespace__feature" formats)
            if "__" in feature_str:
                namespace_parts = feature_str.split("__")
                feature_keys.append(FeatureKey(namespace_parts))
            else:
                # Single-part key
                feature_keys.append(FeatureKey([feature_str]))

    # Get store
    metadata_store = get_store(store)

    with metadata_store:
        # If all_features, get all feature keys from store
        if all_features:
            # Get all features that have metadata in the store
            feature_keys = metadata_store.list_features(include_fallback=False)

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
