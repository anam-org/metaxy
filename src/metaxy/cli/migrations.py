"""New Migration CLI commands using event-based system."""

from __future__ import annotations

from typing import Annotated, Literal

import cyclopts

from metaxy.cli.console import console, data_console, error_console

# Migrations subcommand app
app = cyclopts.App(
    name="migrations",
    help="Metadata migration commands",
    console=console,
    error_console=error_console,
)


@app.command
def generate(
    *,
    name: Annotated[
        str | None,
        cyclopts.Parameter(help="Migration name (creates {timestamp}_{name} ID)"),
    ] = None,
    store: Annotated[
        str | None,
        cyclopts.Parameter(help="Store name (defaults to default)"),
    ] = None,
    from_snapshot: Annotated[
        str | None,
        cyclopts.Parameter(help="Compare from this historical snapshot version (defaults to latest)"),
    ] = None,
    op: Annotated[
        list[str],
        cyclopts.Parameter(
            help="Operation class path to use (can be repeated). Example: metaxy.migrations.ops.DataVersionReconciliation"
        ),
    ],
    type: Annotated[
        Literal["diff", "full"],
        cyclopts.Parameter(
            help="Migration type: 'diff' (compare different graph snapshots) or 'full' (operates on a single graph snapshot)"
        ),
    ] = "diff",
):
    """Generate migration from detected feature changes.

    Two migration types are supported:

    - **diff** : Compares the latest snapshot in the store (or specified
      from_snapshot) with the current active graph to detect changes. Only affected
      features are included.

    - **full**: Creates a migration that includes ALL features in the current graph.
      Each operation will have a 'features' list with all feature keys.

    Examples:
        # Generate diff migration with DataVersionReconciliation operation
        $ metaxy migrations generate --op metaxy.migrations.ops.DataVersionReconciliation

        # Generate full graph migration (all features)
        $ metaxy migrations generate --migration-type full --op myproject.ops.CustomBackfill

        # Custom operation type
        $ metaxy migrations generate --op myproject.ops.CustomReconciliation

        # Multiple operations
        $ metaxy migrations generate \
            --op metaxy.migrations.ops.DataVersionReconciliation \
            --op myproject.ops.CustomBackfill
    """
    import shlex
    import sys
    from pathlib import Path

    from metaxy.cli.context import AppContext
    from metaxy.migrations.detector import (
        detect_diff_migration,
        generate_full_graph_migration,
    )

    context = AppContext.get()
    context.raise_command_cannot_override_project()
    config = context.config

    # Convert op_type list to ops format
    if len(op) == 0:
        app.console.print(
            "[red]✗[/red] --op is required. Example: --op metaxy.migrations.ops.DataVersionReconciliation"
        )
        raise SystemExit(1)

    ops = [{"type": o} for o in op]

    # Get store and project from config
    metadata_store = context.get_store(store)
    migrations_dir = Path(config.migrations_dir)
    project = context.get_required_project()  # This command needs a specific project

    # Reconstruct CLI command for YAML comment
    cli_command = shlex.join(sys.argv)

    with metadata_store.open("write"):
        # Load feature definitions from the store to replace any external feature placeholders
        # This ensures version hashes are computed correctly against actual stored definitions
        from metaxy.metadata_store.system.storage import SystemTableStorage

        SystemTableStorage(metadata_store)._load_feature_definitions()

        if type == "diff":
            # Detect migration and write YAML
            migration = detect_diff_migration(
                metadata_store,
                project=project,
                from_snapshot_version=from_snapshot,
                ops=ops,
                migrations_dir=migrations_dir,
                name=name,
                command=cli_command,
            )

            if migration is None:
                app.console.print("[yellow]No changes detected[/yellow]")
                app.console.print("  Current graph matches latest snapshot")
                return

            # Print summary for DiffMigration
            yaml_path = migrations_dir / f"{migration.migration_id}.yaml"
            app.console.print("\n[green]✓[/green] DiffMigration generated")
            app.console.print(f"  Migration ID: {migration.migration_id}")
            app.console.print(f"  YAML file: {yaml_path}")
            app.console.print(f"  From snapshot: {migration.from_snapshot_version}")
            app.console.print(f"  To snapshot: {migration.to_snapshot_version}")

        else:  # type == "full"
            if from_snapshot is not None:
                app.console.print("[yellow]Warning:[/yellow] --from-snapshot is ignored for full graph migrations")

            migration = generate_full_graph_migration(
                metadata_store,
                project=project,
                ops=ops,
                migrations_dir=migrations_dir,
                name=name,
                command=cli_command,
            )

            # Print summary for FullGraphMigration
            yaml_path = migrations_dir / f"{migration.migration_id}.yaml"
            app.console.print("\n[green]✓[/green] FullGraphMigration generated")
            app.console.print(f"  Migration ID: {migration.migration_id}")
            app.console.print(f"  YAML file: {yaml_path}")
            app.console.print(f"  Snapshot: {migration.snapshot_version}")

        # Output migration ID to stdout for scripting
        data_console.print(migration.migration_id)

        # Get affected features (computed on-demand)
        affected_features = migration.get_affected_features(metadata_store, project)
        app.console.print(f"\n  Affected features ({len(affected_features)}):")
        for feature_key in affected_features[:5]:
            app.console.print(f"    ✓ {feature_key}")
        if len(affected_features) > 5:
            app.console.print(f"    ... and {len(affected_features) - 5} more")

        app.console.print("\n[bold]NEXT STEPS:[/bold]")
        app.console.print(f"1. Review migration YAML: {yaml_path}")
        app.console.print(
            f"2. Run 'metaxy migrations apply {migration.migration_id}' to execute. Use CD for production!"
        )


@app.command
def apply(
    migration_id: Annotated[
        str | None,
        cyclopts.Parameter(help="Migration ID to apply (applies all unapplied if not specified)"),
    ] = None,
    store: Annotated[
        str | None,
        cyclopts.Parameter(help="Metadata store to use."),
    ] = None,
    *,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(help="Preview changes without executing"),
    ] = False,
    rerun: Annotated[
        bool,
        cyclopts.Parameter(help="Re-run all steps, including already completed ones"),
    ] = False,
):
    """Apply migration(s) from YAML files.

    Reads migration definitions from .metaxy/migrations/ directory (git).
    Follows parent chain to ensure correct order.
    Tracks execution state in database (events).

    Examples:
        # Apply all unapplied migrations in chain order
        $ metaxy migrations apply

        # Apply specific migration (and all its unapplied predecessors)
        $ metaxy migrations apply 20250113_103000

        # Dry run
        $ metaxy migrations apply --dry-run

        # Re-run all migrations, including already completed ones
        $ metaxy migrations apply --rerun
    """
    from pathlib import Path

    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.system import SystemTableStorage
    from metaxy.migrations.executor import MigrationExecutor
    from metaxy.migrations.loader import build_migration_chain

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    # Get context and project from config
    project = context.get_required_project()  # This command needs a specific project
    metadata_store = context.get_store(store)
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store.open("write"):
        storage = SystemTableStorage(metadata_store)

        # Load feature definitions from the store to replace any external feature placeholders
        # This ensures version hashes are computed correctly against actual stored definitions
        storage._load_feature_definitions()

        # Build migration chain
        try:
            chain = build_migration_chain(migrations_dir)
        except ValueError as e:
            from metaxy.cli.utils import print_error

            print_error(app.console, "Invalid migration chain", e)
            raise SystemExit(1)

        if not chain:
            app.console.print("[yellow]No migrations found[/yellow]")
            app.console.print("Run 'metaxy migrations generate' first")
            return

        # Get completed migrations from events
        from metaxy.metadata_store.system import MigrationStatus

        completed_ids = set()
        for m in chain:
            # Get expected features from migration to verify all operations completed
            expected_features = m.get_affected_features(metadata_store, project)
            if (
                storage.get_migration_status(m.migration_id, project, expected_features=expected_features)
                == MigrationStatus.COMPLETED
            ):
                completed_ids.add(m.migration_id)

        # Filter to migrations to apply
        if migration_id is None:
            if rerun:
                # Re-run all migrations
                to_apply = list(chain)
            else:
                # Apply all unapplied
                to_apply = [m for m in chain if m.migration_id not in completed_ids]
        else:
            # Apply specific migration and its predecessors
            target_index = None
            for i, m in enumerate(chain):
                if m.migration_id == migration_id:
                    target_index = i
                    break

            if target_index is None:
                app.console.print(f"[red]✗[/red] Migration '{migration_id}' not found in chain")
                raise SystemExit(1)

            if rerun:
                # Include all migrations up to and including target
                to_apply = list(chain[: target_index + 1])
            else:
                # Include all unapplied migrations up to and including target
                to_apply = [m for m in chain[: target_index + 1] if m.migration_id not in completed_ids]

        if not to_apply:
            app.console.print("[blue]ℹ[/blue] All migrations already completed")
            return

        # Execute migrations in order
        if dry_run:
            app.console.print("[yellow]=== DRY RUN MODE ===[/yellow]\n")
        if rerun:
            app.console.print("[yellow]=== RERUN MODE ===[/yellow]\n")

        app.console.print(f"Applying {len(to_apply)} migration(s) in chain order:")
        for m in to_apply:
            app.console.print(f"  • {m.migration_id}")
        app.console.print()

        executor = MigrationExecutor(storage)

        for migration in to_apply:
            # Get status info to show accurate progress
            status_info = migration.get_status_info(metadata_store, project)

            app.console.print(f"[bold]Applying: {migration.migration_id}[/bold]")
            if rerun:
                # In rerun mode, all features from YAML will be reprocessed
                app.console.print(f"  Reprocessing all {status_info.features_total} feature(s)")
            elif status_info.features_remaining > 0:
                app.console.print(
                    f"  Processing {status_info.features_remaining} feature(s) "
                    f"({len(status_info.completed_features)} already completed)"
                )
            else:
                app.console.print(f"  All {status_info.features_total} feature(s) already completed")

            result = executor.execute(migration, metadata_store, project, dry_run=dry_run, rerun=rerun)

            # Print result
            if result.status == "completed":
                app.console.print("[green]✓[/green] Migration completed")
            elif result.status == "skipped":
                app.console.print("[yellow]⊘[/yellow] Migration skipped (dry run)")
            else:
                app.console.print("[red]✗[/red] Migration failed")

            app.console.print(f"  Features completed: {result.features_completed}")
            app.console.print(f"  Features failed: {result.features_failed}")
            app.console.print(f"  Rows affected: {result.rows_affected}")
            app.console.print(f"  Duration: {result.duration_seconds:.2f}s")

            if result.errors:
                from metaxy.cli.utils import print_error_list

                print_error_list(
                    app.console,
                    result.errors,
                    header="\n[red]Errors:[/red]",
                )

            if result.status == "failed":
                app.console.print(f"\n[red]Migration {migration.migration_id} failed. Stopping chain execution.[/red]")
                raise SystemExit(1)

            app.console.print()  # Blank line between migrations


@app.command
def status():
    """Show migrations and execution status.

    Reads migration definitions from YAML files (git).
    Shows execution status from database events.
    Displays the parent chain in order.

    Example:
        $ metaxy migrations status

        Migration:
        ────────────────────────────────────────────
        ✓ 20250110_120000 (parent: initial)
          Status: COMPLETED
          Features: 5/5 completed

        ○ 20250113_103000 (parent: 20250110_120000)
          Status: NOT STARTED
          Features: 3 affected

        ⚠ Multiple heads detected: [20250110_120000_a, 20250110_120000_b]
    """
    from pathlib import Path

    from metaxy.cli.context import AppContext
    from metaxy.migrations.loader import build_migration_chain

    context = AppContext.get()

    # Get context and project from config
    project = context.get_required_project()  # This command needs a specific project
    metadata_store = context.get_store()
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store:
        # Try to build migration chain
        try:
            chain = build_migration_chain(migrations_dir)
        except ValueError as e:
            from metaxy.cli.utils import print_error

            print_error(app.console, "Invalid migrations", e)
            return

        if not chain:
            app.console.print("[yellow]No migrations found.[/yellow]")
            app.console.print(f"  Migrations directory: {migrations_dir.resolve()}")
            return

        app.console.print("\n[bold]Migration:[/bold]")
        app.console.print("─" * 60)

        for migration in chain:
            migration_id = migration.migration_id

            # Get comprehensive status info from migration
            from metaxy.metadata_store.system import MigrationStatus

            status_info = migration.get_status_info(metadata_store, project)
            migration_status = status_info.status
            completed_features = status_info.completed_features
            failed_features = status_info.failed_features
            total_affected = status_info.features_total

            # Print status icon
            if migration_status == MigrationStatus.COMPLETED:
                app.console.print(f"[green]✓[/green] {migration_id} (parent: {migration.parent})")
                app.console.print("  Status: [green]COMPLETED[/green]")
            elif migration_status == MigrationStatus.FAILED:
                app.console.print(f"[red]✗[/red] {migration_id} (parent: {migration.parent})")
                app.console.print("  Status: [red]FAILED[/red]")
            elif migration_status == MigrationStatus.IN_PROGRESS:
                app.console.print(f"[yellow]⚠[/yellow] {migration_id} (parent: {migration.parent})")
                app.console.print("  Status: [yellow]IN PROGRESS[/yellow]")
            else:
                app.console.print(f"[blue]○[/blue] {migration_id} (parent: {migration.parent})")
                app.console.print("  Status: [blue]NOT STARTED[/blue]")

            # Show snapshot info for DiffMigration
            from metaxy.migrations.models import DiffMigration

            if isinstance(migration, DiffMigration):
                app.console.print("  Snapshots:")
                app.console.print(f"    From: {migration.from_snapshot_version}")
                app.console.print(f"    To:   {migration.to_snapshot_version}")

            # Show operation-level progress for FullGraphMigration
            from metaxy.migrations.models import FullGraphMigration, OperationConfig

            if isinstance(migration, FullGraphMigration) and migration.ops:
                app.console.print("  Operations:")
                completed_set = set(status_info.completed_features)

                for i, op_dict in enumerate(migration.ops, 1):
                    op_config = OperationConfig.model_validate(op_dict)
                    op_type_short = op_config.type.split(".")[-1]  # Extract class name
                    op_features = set(op_config.features)
                    op_completed = op_features & completed_set

                    # Determine status icon
                    if len(op_completed) == 0:
                        icon = "[blue]○[/blue]"  # Not started
                    elif len(op_completed) == len(op_features):
                        icon = "[green]✓[/green]"  # Completed
                    else:
                        icon = "[yellow]⚠[/yellow]"  # In progress

                    app.console.print(
                        f"    {icon} {i}. {op_type_short} ({len(op_completed)}/{len(op_features)} features)"
                    )

            app.console.print(f"  Features: {len(completed_features)}/{total_affected} completed")

            if failed_features:
                from metaxy.cli.utils import print_error_list

                print_error_list(
                    app.console,
                    failed_features,
                    header=f"  [red]Failed features ({len(failed_features)}):[/red]",
                    indent="  ",
                    max_items=3,
                )

            app.console.print()


@app.command(name="list")
def list_migrations():
    """List all migrations in chain order as defined in code.

    Displays a simple table showing migration ID, creation time, and operations.

    Example:
        $ metaxy migrations list

        20250110_120000  2025-01-10 12:00  DataVersionReconciliation
        20250113_103000  2025-01-13 10:30  DataVersionReconciliation
    """
    from pathlib import Path

    from rich.table import Table

    from metaxy.cli.context import AppContext
    from metaxy.migrations.loader import build_migration_chain

    AppContext.get()
    migrations_dir = Path(".metaxy/migrations")

    # Build migration chain
    try:
        chain = build_migration_chain(migrations_dir)
    except ValueError as e:
        from metaxy.cli.utils import print_error

        print_error(app.console, "Invalid migration", e)
        return

    if not chain:
        app.console.print("[yellow]No migrations found.[/yellow]")
        app.console.print(f"  Migrations directory: {migrations_dir.resolve()}")
        return

    # Create borderless table with blue headers (no truncation)
    table = Table(
        show_header=True,
        show_edge=False,
        box=None,
        padding=(0, 2),
        header_style="bold blue",
    )
    table.add_column("ID", style="bold", no_wrap=False, overflow="fold")
    table.add_column("Created", style="dim", no_wrap=False, overflow="fold")
    table.add_column("Operations", no_wrap=False, overflow="fold")

    for migration in chain:
        # Format created_at - simpler format without seconds
        created_str = migration.created_at.strftime("%Y-%m-%d %H:%M")

        # Format operations - extract short names from raw ops dicts
        # Use .ops (raw dicts) instead of .operations (instantiated) to avoid
        # importing operation classes that might not exist
        op_names = []
        ops = getattr(migration, "ops", [])
        for op_dict in ops:
            op_type = op_dict.get("type", "unknown")
            # Extract just the class name (last part after final dot)
            op_short = op_type.split(".")[-1]
            op_names.append(op_short)

        ops_str = ", ".join(op_names)

        table.add_row(migration.migration_id, created_str, ops_str)

    app.console.print()
    app.console.print(table)
    app.console.print()


@app.command
def explain(
    migration_id: Annotated[
        str | None,
        cyclopts.Parameter(help="Migration ID to explain (explains latest if not specified)"),
    ] = None,
):
    """Show detailed diff for a migration.

    Reads migration from YAML file.
    Computes and displays the GraphDiff between the two snapshots on-demand.

    Examples:
        # Explain latest migration (head of chain)
        $ metaxy migrations explain

        # Explain specific migration
        $ metaxy migrations explain 20250113_103000
    """
    from pathlib import Path

    from metaxy.cli.context import AppContext
    from metaxy.migrations.loader import (
        find_latest_migration,
        find_migration_yaml,
        load_migration_from_yaml,
    )

    context = AppContext.get()
    # Get context and project from config
    project = context.get_required_project()  # This command needs a specific project
    metadata_store = context.get_store(None)
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store:
        # Get migration ID
        if migration_id is None:
            # Get latest migration (head)
            try:
                migration_id = find_latest_migration(migrations_dir)
            except ValueError as e:
                app.console.print(f"[red]✗[/red] {e}")
                raise SystemExit(1)

            if migration_id is None:
                app.console.print("[yellow]No migrations found[/yellow]")
                app.console.print("Run 'metaxy migrations generate' first")
                return

        # Load migration from YAML
        try:
            yaml_path = find_migration_yaml(migration_id, migrations_dir)
            migration = load_migration_from_yaml(yaml_path)
        except FileNotFoundError as e:
            app.console.print(f"[red]✗[/red] {e}")
            raise SystemExit(1)

        # Type narrow to DiffMigration for explain command
        from metaxy.migrations.models import DiffMigration

        if not isinstance(migration, DiffMigration):
            app.console.print(f"[red]✗[/red] Migration '{migration_id}' is not a DiffMigration")
            app.console.print(f"  Type: {type(migration).__name__} (explain only supports DiffMigration)")
            raise SystemExit(1)

        # Print header
        app.console.print(f"\n[bold]Migration: {migration_id}[/bold]")
        app.console.print(f"From: {migration.from_snapshot_version}")
        app.console.print(f"To:   {migration.to_snapshot_version}")
        app.console.print()

        # Compute diff on-demand
        try:
            graph_diff = migration.compute_graph_diff(metadata_store, project)
        except Exception as e:
            from metaxy.cli.utils import print_error

            print_error(app.console, "Failed to compute diff", e)
            raise SystemExit(1)

        # Display detailed diff
        if not graph_diff.has_changes:
            app.console.print("[yellow]No changes detected[/yellow]")
            return

        # Added nodes
        if graph_diff.added_nodes:
            app.console.print(f"[green]Added Features ({len(graph_diff.added_nodes)}):[/green]")
            for node in graph_diff.added_nodes:
                app.console.print(f"  ✓ {node.feature_key}")
                if node.fields:
                    app.console.print(f"    Fields ({len(node.fields)}):")
                    for field in node.fields[:3]:
                        app.console.print(f"      - {field['key']} (cv={field.get('code_version', '?')})")
                    if len(node.fields) > 3:
                        app.console.print(f"      ... and {len(node.fields) - 3} more")
            app.console.print()

        # Removed nodes
        if graph_diff.removed_nodes:
            app.console.print(f"[red]Removed Features ({len(graph_diff.removed_nodes)}):[/red]")
            for node in graph_diff.removed_nodes:
                app.console.print(f"  ✗ {node.feature_key}")
                if node.fields:
                    app.console.print(f"    Fields ({len(node.fields)}):")
                    for field in node.fields[:3]:
                        app.console.print(f"      - {field['key']} (cv={field.get('code_version', '?')})")
                    if len(node.fields) > 3:
                        app.console.print(f"      ... and {len(node.fields) - 3} more")
            app.console.print()

        # Changed nodes
        if graph_diff.changed_nodes:
            app.console.print(f"[yellow]Changed Features ({len(graph_diff.changed_nodes)}):[/yellow]")
            for node in graph_diff.changed_nodes:
                app.console.print(f"  ⚠ {node.feature_key}")
                old_ver = node.old_version if node.old_version else "None"
                new_ver = node.new_version if node.new_version else "None"
                app.console.print(f"    Version: {old_ver} → {new_ver}")

                if node.old_code_version is not None or node.new_code_version is not None:
                    app.console.print(f"    Code version: {node.old_code_version} → {node.new_code_version}")

                # Show field changes
                total_field_changes = len(node.added_fields) + len(node.removed_fields) + len(node.changed_fields)
                if total_field_changes > 0:
                    app.console.print(f"    Field changes ({total_field_changes}):")

                    if node.added_fields:
                        app.console.print(f"      [green]Added ({len(node.added_fields)}):[/green]")
                        for field in node.added_fields[:2]:
                            app.console.print(f"        + {field.field_key} (cv={field.new_code_version})")
                        if len(node.added_fields) > 2:
                            app.console.print(f"        ... and {len(node.added_fields) - 2} more")

                    if node.removed_fields:
                        app.console.print(f"      [red]Removed ({len(node.removed_fields)}):[/red]")
                        for field in node.removed_fields[:2]:
                            app.console.print(f"        - {field.field_key} (cv={field.old_code_version})")
                        if len(node.removed_fields) > 2:
                            app.console.print(f"        ... and {len(node.removed_fields) - 2} more")

                    if node.changed_fields:
                        app.console.print(f"      [yellow]Changed ({len(node.changed_fields)}):[/yellow]")
                        for field in node.changed_fields[:2]:
                            app.console.print(
                                f"        ~ {field.field_key} (cv={field.old_code_version}→{field.new_code_version})"
                            )
                        if len(node.changed_fields) > 2:
                            app.console.print(f"        ... and {len(node.changed_fields) - 2} more")

            app.console.print()

        # Print affected features summary (computed on-demand)
        affected_features = migration.get_affected_features(metadata_store, project)
        app.console.print(f"[bold]Affected Features ({len(affected_features)}):[/bold]")
        for feature_key in affected_features[:10]:
            app.console.print(f"  • {feature_key}")
        if len(affected_features) > 10:
            app.console.print(f"  ... and {len(affected_features) - 10} more")


@app.command
def describe(
    migration_ids: Annotated[
        list[str],
        cyclopts.Parameter(help="Migration IDs to describe (default: all migrations in order)"),
    ] = [],
    store: Annotated[
        str | None,
        cyclopts.Parameter(None, help="Metadata store to use."),
    ] = None,
):
    """Show verbose description of migration(s).

    Displays detailed information about what the migration will do:
    - Migration metadata (ID, parent, snapshots, created timestamp)
    - Operations to execute
    - Affected features with row counts
    - Execution status if already run

    Examples:
        # Describe all migrations in chain order
        $ metaxy migrations describe

        # Describe specific migration
        $ metaxy migrations describe 20250127_120000

        # Describe multiple migrations
        $ metaxy migrations describe 20250101_120000 20250102_090000
    """
    from pathlib import Path

    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.system import SystemTableStorage
    from metaxy.migrations.loader import (
        build_migration_chain,
        find_migration_yaml,
        load_migration_from_yaml,
    )

    context = AppContext.get()
    # Get context and project from config
    project = context.get_required_project()  # This command needs a specific project
    metadata_store = context.get_store(store)
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store:
        storage = SystemTableStorage(metadata_store)

        # Determine which migrations to describe
        if not migration_ids:
            # Default: describe all migrations in chain order
            try:
                chain = build_migration_chain(migrations_dir)
            except ValueError as e:
                from metaxy.cli.utils import print_error

                print_error(app.console, "Invalid migration chain", e)
                raise SystemExit(1)

            if not chain:
                app.console.print("[yellow]No migrations found.[/yellow]")
                return

            migrations_to_describe = chain
        else:
            # Load specific migrations
            migrations_to_describe = []
            for migration_id in migration_ids:
                try:
                    yaml_path = find_migration_yaml(migration_id, migrations_dir)
                    migration_obj = load_migration_from_yaml(yaml_path)
                    migrations_to_describe.append(migration_obj)
                except FileNotFoundError as e:
                    app.console.print(f"[red]✗[/red] {e}")
                    raise SystemExit(1)

        # Describe each migration
        for i, migration_obj in enumerate(migrations_to_describe):
            if i > 0:
                app.console.print("\n")  # Separator between migrations

            # Find YAML path for display
            yaml_path = find_migration_yaml(migration_obj.migration_id, migrations_dir)

            # Print header
            app.console.print("\n[bold]Migration Description[/bold]")
            app.console.print("─" * 60)
            app.console.print(f"[bold]ID:[/bold] {migration_obj.migration_id}")
            app.console.print(f"[bold]Created:[/bold] {migration_obj.created_at.isoformat()}")
            app.console.print(f"[bold]Parent:[/bold] {migration_obj.parent}")
            app.console.print(f"[bold]YAML:[/bold] {yaml_path}")
            app.console.print()

            # Snapshots (for DiffMigration)
            from metaxy.migrations.models import DiffMigration, FullGraphMigration

            if isinstance(migration_obj, DiffMigration):
                app.console.print("[bold]Snapshots:[/bold]")
                app.console.print(f"  From: {migration_obj.from_snapshot_version}")
                app.console.print(f"  To:   {migration_obj.to_snapshot_version}")
                app.console.print()

            # Operations (for FullGraphMigration)
            if isinstance(migration_obj, FullGraphMigration):
                app.console.print("[bold]Operations:[/bold]")
                for j, op in enumerate(migration_obj.ops, 1):
                    op_type = op.get("type", "unknown")
                    app.console.print(f"  {j}. {op_type}")
                app.console.print()

            # Get affected features
            app.console.print("[bold]Computing affected features...[/bold]")
            try:
                affected_features = migration_obj.get_affected_features(metadata_store, project)
            except Exception as e:
                app.console.print(f"[red]✗[/red] Failed to compute affected features: {e}")
                continue  # Skip to next migration

            app.console.print(f"\n[bold]Affected Features ({len(affected_features)}):[/bold]")
            app.console.print("─" * 60)

            from metaxy.models.feature import FeatureGraph
            from metaxy.models.types import FeatureKey

            graph = FeatureGraph.get_active()

            # Get events for this migration to extract rows_affected per feature
            events_df = storage.get_migration_events(migration_obj.migration_id, project)

            for feature_key_str in affected_features:
                feature_key_obj = FeatureKey(feature_key_str.split("/"))

                # Check feature exists in graph
                if feature_key_obj not in graph.feature_definitions_by_key:
                    app.console.print(f"[yellow]⚠[/yellow] {feature_key_str}")
                    app.console.print("    [yellow]Feature not in current graph[/yellow]")
                    continue

                # Get rows affected from events (sum of all completed events for this feature)
                import polars as pl

                feature_events = events_df.filter(
                    (pl.col("feature_key") == feature_key_str) & (pl.col("event_type") == "feature_migration_completed")
                )

                if feature_events.height > 0:
                    # Extract rows_affected from JSON payload
                    feature_events = feature_events.with_columns(
                        pl.col("payload")
                        .str.json_path_match("$.rows_affected")
                        .cast(pl.Int64, strict=False)
                        .fill_null(0)
                        .alias("rows_affected")
                    )
                    rows_affected = int(feature_events["rows_affected"].sum())

                    # Count number of attempts (feature_migration_started events)
                    attempts = events_df.filter(
                        (pl.col("feature_key") == feature_key_str)
                        & (pl.col("event_type") == "feature_migration_started")
                    ).height
                else:
                    rows_affected = 0
                    attempts = 0

                # Check if feature has upstream dependencies
                plan = graph.get_feature_plan(feature_key_obj)
                has_upstream = plan.deps is not None and len(plan.deps) > 0

                app.console.print(f"[bold]{feature_key_str}[/bold]")
                app.console.print(f"    Rows Affected: {rows_affected}")
                app.console.print(f"    Attempts: {attempts}")
                app.console.print(f"    Has upstream: {has_upstream}")

            app.console.print()

            # Execution status
            from metaxy.metadata_store.system import MigrationStatus

            summary = storage.get_migration_summary(
                migration_obj.migration_id,
                project,
                expected_features=affected_features,
            )
            migration_status = summary["status"]
            completed_features = summary["completed_features"]
            failed_features = summary["failed_features"]

            app.console.print("[bold]Execution Status:[/bold]")
            if migration_status == MigrationStatus.COMPLETED:
                app.console.print("  [green]✓ COMPLETED[/green]")
                app.console.print(f"    Features processed: {len(completed_features)}/{len(affected_features)}")
            elif migration_status == MigrationStatus.FAILED:
                app.console.print("  [red]✗ FAILED[/red]")
                app.console.print(f"    Features completed: {len(completed_features)}/{len(affected_features)}")
                app.console.print(f"    Features failed: {len(failed_features)}")
                if failed_features:
                    from metaxy.cli.utils import print_error_list

                    print_error_list(
                        app.console,
                        failed_features,
                        header="    Failed features:",
                        prefix="    •",
                        indent="  ",
                        max_items=5,
                    )
            elif migration_status == MigrationStatus.IN_PROGRESS:
                app.console.print("  [yellow]⚠ IN PROGRESS[/yellow]")
                app.console.print(f"    Features completed: {len(completed_features)}/{len(affected_features)}")
            else:
                app.console.print("  [blue]○ NOT STARTED[/blue]")
                app.console.print(f"    Features to process: {len(affected_features)}")


if __name__ == "__main__":
    app()
