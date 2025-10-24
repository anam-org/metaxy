"""Migration CLI commands."""

from pathlib import Path
from typing import Annotated

import cyclopts
from rich.console import Console

# Rich console for formatted output
console = Console()

# Migrations subcommand app
app = cyclopts.App(
    name="migrations",  # pyrefly: ignore[unexpected-keyword]
    help="Metadata migration commands",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
)


@app.command
def generate(
    *,
    migrations_dir: Annotated[
        Path | None,
        cyclopts.Parameter(
            help="Directory for migration files (uses config if not specified)"
        ),
    ] = None,
    from_snapshot: Annotated[
        str | None,
        cyclopts.Parameter(help="Compare from this historical snapshot ID (optional)"),
    ] = None,
    to_snapshot: Annotated[
        str | None,
        cyclopts.Parameter(help="Compare to this historical snapshot ID (optional)"),
    ] = None,
):
    """Generate migration file from detected feature changes.

    Two modes:
    1. Default (no snapshots): Compare store's latest snapshot vs current code
    2. Historical (both snapshots): Compare two historical snapshots

    Automatically detects features that need migration and generates
    explicit operations for ALL affected features.

    Example (default):
        $ metaxy migrations generate

        Detected 1 root feature change(s):
          ✓ video_processing: abc12345 → def67890

        Generating explicit operations for 3 downstream features:
          ✓ feature_c (current: xyz111)
          ✓ feature_d (current: aaa222)
          ✓ feature_e (current: bbb333)

        Generated 4 total operations (1 root + 3 downstream)

    Example (historical):
        $ metaxy migrations generate --from-snapshot abc123... --to-snapshot def456...
    """
    from metaxy.cli.context import get_config, get_store
    from metaxy.entrypoints import load_features
    from metaxy.migrations import generate_migration

    # Load features from entrypoints
    load_features()

    # Get migrations_dir from config if not specified
    if migrations_dir is None:
        config = get_config()
        migrations_dir = Path(config.migrations_dir)

    # Get store and open it only for this command
    metadata_store = get_store()
    with metadata_store:
        # Generate migration (returns Migration object)
        migration = generate_migration(
            metadata_store,
            from_snapshot_id=from_snapshot,
            to_snapshot_id=to_snapshot,
        )

        if migration is None:
            # No changes detected - message already printed by generator
            return

        # Build filename from migration metadata
        timestamp_str = migration.created_at.strftime("%Y%m%d_%H%M%S")

        # Get operation feature keys for filename
        ops = migration.get_operations()
        root_keys = [op.feature_key for op in ops[:2]]  # First 2 operations
        feature_names = "_".join("_".join(key) for key in root_keys)
        if len(ops) > 2:
            feature_names += f"_and_{len(ops) - 2}_more"

        filename = migrations_dir / f"{timestamp_str}_update_{feature_names}.yaml"

        # Write YAML file
        migrations_dir.mkdir(parents=True, exist_ok=True)
        migration.to_yaml(str(filename))

        # Print next steps
        app.console.print(f"\n[green]Generated:[/green] {filename}")
        app.console.print("\n[bold]NEXT STEPS:[/bold]")
        app.console.print("1. Review the migration file")
        app.console.print("2. Edit the 'reason' fields for root changes")
        app.console.print("3. Run 'metaxy migrations apply --dry-run' to preview")
        app.console.print("4. Run 'metaxy migrations apply' to execute")
        app.console.print("5. Commit the migration file to git")


@app.command
def scaffold(
    *,
    migrations_dir: Annotated[
        Path | None,
        cyclopts.Parameter(
            help="Directory for migration files (uses config if not specified)"
        ),
    ] = None,
    description: Annotated[
        str | None,
        cyclopts.Parameter(help="Migration description (optional)"),
    ] = None,
    from_snapshot: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Use this as from_snapshot_id (defaults to latest in store)"
        ),
    ] = None,
    to_snapshot: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Use this as to_snapshot_id (defaults to current graph)"
        ),
    ] = None,
):
    """Create an empty migration scaffold for user-defined operations.

    Generates a migration file template with:
    - Snapshot IDs from current store state
    - Empty operations list for manual editing
    - Proper structure and metadata

    Use this when you need to write custom migration operations that can't
    be auto-generated (e.g., complex data transformations, backfills).

    Example:
        $ metaxy migrations scaffold --description "Backfill video metadata"

        Created: migrations/20250113_103000_manual_migration.yaml

        Edit the file to add custom operations, then apply with:
        $ metaxy migrations apply migrations/20250113_103000_manual_migration.yaml
    """
    from datetime import datetime

    from metaxy.cli.context import get_config, get_store
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.migrations import Migration
    from metaxy.migrations.executor import MIGRATIONS_KEY
    from metaxy.models.feature import FeatureGraph

    # Get migrations_dir from config if not specified
    if migrations_dir is None:
        config = get_config()
        migrations_dir = Path(config.migrations_dir)

    import narwhals as nw

    metadata_store = get_store()
    with metadata_store:
        # Get from_snapshot_id (use provided or default to latest in store)
        if from_snapshot is None:
            try:
                feature_versions = metadata_store.read_metadata(
                    FEATURE_VERSIONS_KEY, current_only=False
                )
                # Check if any snapshots exist
                fv_sample = nw.from_native(feature_versions.head(1).collect())
                if fv_sample.shape[0] > 0:
                    latest_snapshot = (
                        feature_versions.sort("recorded_at", descending=True)
                        .head(1)
                        .collect()
                    )
                    # Convert to Polars for indexing

                    latest_native = nw.from_native(latest_snapshot).to_polars()
                    from_snapshot_id = latest_native["snapshot_id"].item()
                else:
                    app.error_console.print(
                        "[red]✗[/red] No feature snapshots found in store."
                    )
                    app.error_console.print(
                        "Run 'metaxy push' first to record the feature graph."
                    )
                    raise SystemExit(1)
            except FeatureNotFoundError:
                app.error_console.print(
                    "[red]✗[/red] No feature snapshots found in store."
                )
                app.error_console.print(
                    "Run 'metaxy push' first to record the feature graph."
                )
                raise SystemExit(1)
        else:
            from_snapshot_id = from_snapshot

        # Get to_snapshot_id (use provided or default to current graph)
        if to_snapshot is None:
            graph = FeatureGraph.get_active()
            to_snapshot_id = graph.snapshot_id
        else:
            to_snapshot_id = to_snapshot

        # Get parent migration ID
        parent_migration_id = None
        try:
            existing_migrations = metadata_store.read_metadata(
                MIGRATIONS_KEY, current_only=False
            )
            # Check if any migrations exist
            mig_sample = nw.from_native(existing_migrations.head(1).collect())
            if mig_sample.shape[0] > 0:
                latest = (
                    existing_migrations.sort("created_at", descending=True)
                    .head(1)
                    .collect()
                )
                # Convert to Polars for indexing

                latest_native = nw.from_native(latest).to_polars()
                parent_migration_id = latest_native["migration_id"].item()
        except FeatureNotFoundError:
            pass  # No migrations yet

        # Generate migration ID
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        migration_id = f"migration_{timestamp_str}"

        # Create empty migration
        migration = Migration(
            version=1,
            id=migration_id,
            parent_migration_id=parent_migration_id,
            from_snapshot_id=from_snapshot_id,
            to_snapshot_id=to_snapshot_id,
            description=description or "User-defined migration",
            created_at=timestamp,
            operations=[],  # Empty - user will add custom operations
        )

        # Write YAML file
        filename = migrations_dir / f"{timestamp_str}_manual_migration.yaml"
        migrations_dir.mkdir(parents=True, exist_ok=True)
        migration.to_yaml(str(filename))

        app.console.print(f"\n[green]Created:[/green] {filename}")
        app.console.print("\n[bold]NEXT STEPS:[/bold]")
        app.console.print("1. Edit the file to add custom operations")
        app.console.print("2. Update the description field")
        app.console.print("3. Run 'metaxy migrations apply --dry-run' to preview")
        app.console.print("4. Run 'metaxy migrations apply' to execute")
        app.console.print("5. Commit the migration file to git")


@app.command
def apply(
    revision: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Migration ID to apply up to (applies all if not specified)"
        ),
    ] = None,
    *,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(help="Preview changes without executing"),
    ] = False,
    force: Annotated[
        bool,
        cyclopts.Parameter(help="Re-apply even if already completed"),
    ] = False,
    migrations_dir: Annotated[
        Path | None,
        cyclopts.Parameter(
            help="Directory containing migration files (uses config if not specified)"
        ),
    ] = None,
):
    """Apply migration(s) up to specified revision.

    Applies all migrations in dependency order up to the target revision.
    If no revision specified, applies all migrations.

    Migrations are applied with parent validation - parent migrations
    must be completed before applying child migrations. Already-completed
    migrations are skipped.

    Errors if there are multiple heads (migrations with no children) and
    no revision is specified.

    Examples:
        # Apply all migrations
        $ metaxy migrations apply

        # Apply up to specific revision
        $ metaxy migrations apply migration_20250113_103000

        # Dry run
        $ metaxy migrations apply --dry-run
    """
    from pathlib import Path as PathlibPath

    from metaxy.cli.context import get_config, get_store
    from metaxy.entrypoints import load_features
    from metaxy.migrations import Migration, apply_migration

    # Load features from entrypoints
    load_features()

    # Get migrations_dir from config if not specified
    if migrations_dir is None:
        config = get_config()
        migrations_dir = PathlibPath(config.migrations_dir)

    metadata_store = get_store()

    with metadata_store:
        # Load all migration files from directory
        migration_files = sorted(migrations_dir.glob("*.yaml"))

        if not migration_files:
            console.print(
                f"[yellow]No migration files found in {migrations_dir}[/yellow]"
            )
            return

        # Parse all migrations
        migrations_by_id = {}
        for file in migration_files:
            mig = Migration.from_yaml(str(file))
            migrations_by_id[mig.id] = mig

        # Build dependency graph
        children_by_parent = {}  # parent_id -> [child_ids]
        for mig_id, mig in migrations_by_id.items():
            if mig.parent_migration_id:
                if mig.parent_migration_id not in children_by_parent:
                    children_by_parent[mig.parent_migration_id] = []
                children_by_parent[mig.parent_migration_id].append(mig_id)

        # Find head migrations (migrations with no children)
        heads = [
            mig_id
            for mig_id in migrations_by_id.keys()
            if mig_id not in children_by_parent
        ]

        # Determine target migration
        if revision:
            if revision not in migrations_by_id:
                console.print(f"[red]✗[/red] Migration '{revision}' not found")
                raise SystemExit(1)
            target_id = revision
        else:
            # No revision specified - apply to latest head
            if len(heads) > 1:
                console.print(f"[red]✗[/red] Multiple migration heads found: {heads}")
                console.print("Please specify which revision to apply:")
                for head in heads:
                    console.print(f"  $ metaxy migrations apply {head}")
                raise SystemExit(1)
            elif len(heads) == 0:
                console.print("[yellow]No migrations to apply.[/yellow]")
                return
            target_id = heads[0]

        # Build path from root to target (topological order)
        def build_path_to_target(target: str) -> list[str]:
            path = []
            current = target
            while current is not None:
                path.append(current)
                mig = migrations_by_id[current]
                current = mig.parent_migration_id
                if current and current not in migrations_by_id:
                    console.print(
                        f"[red]✗[/red] Parent migration '{current}' not found"
                    )
                    raise SystemExit(1)
            return list(reversed(path))  # Root first

        migration_path = build_path_to_target(target_id)

        if dry_run:
            console.print("[yellow]=== DRY RUN MODE ===[/yellow]\n")

        console.print(
            f"Applying {len(migration_path)} migration(s) to reach '{target_id}':\n"
        )

        # Apply each migration in order
        applied_count = 0
        skipped_count = 0
        failed_count = 0

        for mig_id in migration_path:
            migration = migrations_by_id[mig_id]

            result = apply_migration(
                metadata_store, migration, dry_run=dry_run, force=force
            )

            # Print result
            if result.status == "completed":
                console.print(f"[green]✓[/green] {migration.id}")
                console.print(f"  Operations: {result.operations_applied}")
                console.print(f"  Features: {len(result.affected_features)}")
                applied_count += 1
            elif result.status == "skipped":
                console.print(f"[yellow]⊘[/yellow] {migration.id} (already applied)")
                skipped_count += 1
            else:
                console.print(f"[red]✗[/red] {migration.id} FAILED")
                for op_id, error in result.errors.items():
                    console.print(f"  [red]✗[/red] {op_id}: {error}")
                failed_count += 1
                if not force:
                    console.print("\n[red]Stopping due to failure.[/red]")
                    raise SystemExit(1)

        # Summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Applied: {applied_count}")
        console.print(f"  Skipped: {skipped_count}")
        console.print(f"  Failed: {failed_count}")

        if failed_count > 0:
            raise SystemExit(1)


@app.command
def status():
    """Show migration status.

    Displays all registered migrations and their completion status.
    Status is derived from system tables (migrations, ops, steps).

    Example:
        $ metaxy migrations status

        Migration Status:
        ────────────────────────────────────────────
        ✓ migration_20250110_120000
          Status: COMPLETED
          Applied: 2025-01-10 12:05:00
          Operations: 2/2 completed

        ⚠ migration_20250113_103000
          Status: PARTIAL
          Applied: 2025-01-13 10:35:00
          Operations: 3/5 completed
    """
    from metaxy.cli.context import get_store
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.migrations import MigrationStatus
    from metaxy.migrations.executor import MIGRATIONS_KEY

    metadata_store = get_store()

    with metadata_store:
        status_checker = MigrationStatus(metadata_store)

        # Query all migrations
        try:
            migrations = metadata_store.read_metadata(
                MIGRATIONS_KEY,
                current_only=False,
            ).sort("created_at", descending=True)
        except FeatureNotFoundError:
            app.console.print("[yellow]No migrations found.[/yellow]")
            return

        app.console.print("\n[bold]Migration Status:[/bold]")
        app.console.print("─" * 60)

        # Collect LazyFrame to iterate
        import narwhals as nw

        migrations_eager = nw.from_native(migrations.collect())

        for row in migrations_eager.iter_rows(named=True):
            migration_id = row["migration_id"]
            created_at = row["created_at"]
            description = row["description"]

            # Derive status
            is_complete = status_checker.is_migration_complete(migration_id)

            # Parse operation_ids
            operation_ids_raw = row["operation_ids"]
            if isinstance(operation_ids_raw, str):
                import json

                operation_ids = json.loads(operation_ids_raw)
            elif hasattr(operation_ids_raw, "to_list"):
                operation_ids = operation_ids_raw.to_list()
            else:
                operation_ids = list(operation_ids_raw)

            # Count completed operations
            completed_ops = sum(
                1
                for op_id in operation_ids
                if status_checker.is_operation_complete(migration_id, op_id)
            )

            if is_complete:
                app.console.print(f"[green]✓[/green] {migration_id}")
                app.console.print("  Status: [green]COMPLETED[/green]")
            else:
                app.console.print(f"[yellow]⚠[/yellow] {migration_id}")
                app.console.print("  Status: [yellow]IN PROGRESS[/yellow]")

            app.console.print(f"  Created: {created_at}")
            app.console.print(f"  Description: {description}")
            app.console.print(
                f"  Operations: {completed_ops}/{len(operation_ids)} completed"
            )
            app.console.print()


if __name__ == "__main__":
    app()
