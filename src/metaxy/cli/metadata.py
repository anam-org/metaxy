"""Metadata management commands for Metaxy CLI."""

from typing import TYPE_CHECKING, Annotated

import cyclopts
from rich.console import Console

from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature

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
    feature_keys: list[FeatureKey | type[BaseFeature]] | None = None
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
    with source_store, dest_store:
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
