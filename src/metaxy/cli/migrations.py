"""New Migration CLI commands using event-based system."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import cyclopts

from metaxy.cli.console import console, data_console, error_console

# Migrations subcommand app
app = cyclopts.App(
    name="migrations",  # pyrefly: ignore[unexpected-keyword]
    help="Metadata migration commands",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    error_console=error_console,  # pyrefly: ignore[unexpected-keyword]
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
        cyclopts.Parameter(
            help="Compare from this historical snapshot version (defaults to latest)"
        ),
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
            "[red]✗[/red] --op is required. "
            "Example: --op metaxy.migrations.ops.DataVersionReconciliation"
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
                app.console.print(
                    "[yellow]Warning:[/yellow] --from-snapshot is ignored for full graph migrations"
                )

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


def _get_completed_migration_ids(chain, metadata_store, project, storage) -> set[str]:
    """Get IDs of all completed migrations in the chain."""
    from metaxy.metadata_store.system import MigrationStatus

    completed_ids: set[str] = set()
    for m in chain:
        expected_features = m.get_affected_features(metadata_store, project)
        if (
            storage.get_migration_status(
                m.migration_id, project, expected_features=expected_features
            )
            == MigrationStatus.COMPLETED
        ):
            completed_ids.add(m.migration_id)
    return completed_ids


def _find_target_migration_index(chain, migration_id: str) -> int:
    """Find the index of the target migration in the chain. Raises SystemExit if not found."""
    for i, m in enumerate(chain):
        if m.migration_id == migration_id:
            return i
    app.console.print(f"[red]✗[/red] Migration '{migration_id}' not found in chain")
    raise SystemExit(1)


def _filter_migrations_to_apply(
    chain, migration_id: str | None, rerun: bool, completed_ids: set[str]
) -> list:
    """Filter the chain to get migrations that need to be applied."""
    if migration_id is None:
        if rerun:
            return list(chain)
        return [m for m in chain if m.migration_id not in completed_ids]

    target_index = _find_target_migration_index(chain, migration_id)
    if rerun:
        return list(chain[: target_index + 1])
    return [m for m in chain[: target_index + 1] if m.migration_id not in completed_ids]


def _print_migration_progress(migration, status_info, rerun: bool) -> None:
    """Print progress information for a migration about to be applied."""
    app.console.print(f"[bold]Applying: {migration.migration_id}[/bold]")
    if rerun:
        app.console.print(f"  Reprocessing all {status_info.features_total} feature(s)")
    elif status_info.features_remaining > 0:
        app.console.print(
            f"  Processing {status_info.features_remaining} feature(s) "
            f"({len(status_info.completed_features)} already completed)"
        )
    else:
        app.console.print(
            f"  All {status_info.features_total} feature(s) already completed"
        )


def _print_migration_result(result) -> None:
    """Print the result of executing a migration."""
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


@app.command
def apply(
    migration_id: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Migration ID to apply (applies all unapplied if not specified)"
        ),
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

    project = context.get_required_project()
    metadata_store = context.get_store(store)
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store.open("write"):
        storage = SystemTableStorage(metadata_store)

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

        completed_ids = _get_completed_migration_ids(
            chain, metadata_store, project, storage
        )
        to_apply = _filter_migrations_to_apply(
            chain, migration_id, rerun, completed_ids
        )

        if not to_apply:
            app.console.print("[blue]ℹ[/blue] All migrations already completed")
            return

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
            status_info = migration.get_status_info(metadata_store, project)
            _print_migration_progress(migration, status_info, rerun)

            result = executor.execute(
                migration, metadata_store, project, dry_run=dry_run, rerun=rerun
            )
            _print_migration_result(result)

            if result.status == "failed":
                app.console.print(
                    f"\n[red]Migration {migration.migration_id} failed. "
                    "Stopping chain execution.[/red]"
                )
                raise SystemExit(1)

            app.console.print()


def _print_migration_status_header(migration, migration_status) -> None:
    """Print the status header line for a migration."""
    from metaxy.metadata_store.system import MigrationStatus

    migration_id = migration.migration_id
    parent = migration.parent

    status_map = {
        MigrationStatus.COMPLETED: ("[green]✓[/green]", "[green]COMPLETED[/green]"),
        MigrationStatus.FAILED: ("[red]✗[/red]", "[red]FAILED[/red]"),
        MigrationStatus.IN_PROGRESS: (
            "[yellow]⚠[/yellow]",
            "[yellow]IN PROGRESS[/yellow]",
        ),
    }

    icon, status_text = status_map.get(
        migration_status, ("[blue]○[/blue]", "[blue]NOT STARTED[/blue]")
    )
    app.console.print(f"{icon} {migration_id} (parent: {parent})")
    app.console.print(f"  Status: {status_text}")


def _print_operation_progress(migration, completed_features: list[str]) -> None:
    """Print operation-level progress for a FullGraphMigration."""
    from metaxy.migrations.models import FullGraphMigration, OperationConfig

    if not isinstance(migration, FullGraphMigration) or not migration.ops:
        return

    app.console.print("  Operations:")
    completed_set = set(completed_features)

    for i, op_dict in enumerate(migration.ops, 1):
        op_config = OperationConfig.model_validate(op_dict)
        op_type_short = op_config.type.split(".")[-1]
        op_features = set(op_config.features)
        op_completed = op_features & completed_set

        if len(op_completed) == 0:
            icon = "[blue]○[/blue]"
        elif len(op_completed) == len(op_features):
            icon = "[green]✓[/green]"
        else:
            icon = "[yellow]⚠[/yellow]"

        app.console.print(
            f"    {icon} {i}. {op_type_short} "
            f"({len(op_completed)}/{len(op_features)} features)"
        )


def _print_single_migration_status(migration, metadata_store, project) -> None:
    """Print status details for a single migration."""
    from metaxy.migrations.models import DiffMigration

    status_info = migration.get_status_info(metadata_store, project)

    _print_migration_status_header(migration, status_info.status)

    if isinstance(migration, DiffMigration):
        app.console.print("  Snapshots:")
        app.console.print(f"    From: {migration.from_snapshot_version}")
        app.console.print(f"    To:   {migration.to_snapshot_version}")

    _print_operation_progress(migration, status_info.completed_features)

    app.console.print(
        f"  Features: {len(status_info.completed_features)}/{status_info.features_total} completed"
    )

    if status_info.failed_features:
        from metaxy.cli.utils import print_error_list

        print_error_list(
            app.console,
            status_info.failed_features,
            header=f"  [red]Failed features ({len(status_info.failed_features)}):[/red]",
            indent="  ",
            max_items=3,
        )

    app.console.print()


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

    project = context.get_required_project()
    metadata_store = context.get_store()
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store:
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
            _print_single_migration_status(migration, metadata_store, project)


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


def _display_node_fields(node, max_fields: int = 3) -> None:
    """Display fields for a node with truncation."""
    if not node.fields:
        return
    app.console.print(f"    Fields ({len(node.fields)}):")
    for field in node.fields[:max_fields]:
        app.console.print(
            f"      - {field['key']} (cv={field.get('code_version', '?')})"
        )
    if len(node.fields) > max_fields:
        app.console.print(f"      ... and {len(node.fields) - max_fields} more")


def _display_field_list(
    fields: list, header: str, prefix: str, format_fn: Any, max_items: int = 2
) -> None:
    """Display a list of fields with truncation."""
    if not fields:
        return
    app.console.print(f"      {header}")
    for field in fields[:max_items]:
        app.console.print(f"        {prefix} {format_fn(field)}")
    if len(fields) > max_items:
        app.console.print(f"        ... and {len(fields) - max_items} more")


def _display_field_changes(node) -> None:
    """Display field changes for a changed node."""
    total = len(node.added_fields) + len(node.removed_fields) + len(node.changed_fields)
    if total == 0:
        return

    app.console.print(f"    Field changes ({total}):")

    _display_field_list(
        node.added_fields,
        f"[green]Added ({len(node.added_fields)}):[/green]",
        "+",
        lambda f: f"{f.field_key} (cv={f.new_code_version})",
    )
    _display_field_list(
        node.removed_fields,
        f"[red]Removed ({len(node.removed_fields)}):[/red]",
        "-",
        lambda f: f"{f.field_key} (cv={f.old_code_version})",
    )
    _display_field_list(
        node.changed_fields,
        f"[yellow]Changed ({len(node.changed_fields)}):[/yellow]",
        "~",
        lambda f: f"{f.field_key} (cv={f.old_code_version}→{f.new_code_version})",
    )


def _display_graph_diff(graph_diff) -> None:
    """Display the graph diff details."""
    if graph_diff.added_nodes:
        app.console.print(
            f"[green]Added Features ({len(graph_diff.added_nodes)}):[/green]"
        )
        for node in graph_diff.added_nodes:
            app.console.print(f"  ✓ {node.feature_key}")
            _display_node_fields(node)
        app.console.print()

    if graph_diff.removed_nodes:
        app.console.print(
            f"[red]Removed Features ({len(graph_diff.removed_nodes)}):[/red]"
        )
        for node in graph_diff.removed_nodes:
            app.console.print(f"  ✗ {node.feature_key}")
            _display_node_fields(node)
        app.console.print()

    if graph_diff.changed_nodes:
        app.console.print(
            f"[yellow]Changed Features ({len(graph_diff.changed_nodes)}):[/yellow]"
        )
        for node in graph_diff.changed_nodes:
            app.console.print(f"  ⚠ {node.feature_key}")
            old_ver = node.old_version if node.old_version else "None"
            new_ver = node.new_version if node.new_version else "None"
            app.console.print(f"    Version: {old_ver} → {new_ver}")

            if node.old_code_version is not None or node.new_code_version is not None:
                app.console.print(
                    f"    Code version: {node.old_code_version} → {node.new_code_version}"
                )
            _display_field_changes(node)
        app.console.print()


@app.command
def explain(
    migration_id: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Migration ID to explain (explains latest if not specified)"
        ),
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
    from metaxy.migrations.models import DiffMigration

    context = AppContext.get()
    project = context.get_required_project()
    metadata_store = context.get_store(None)
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store:
        if migration_id is None:
            try:
                migration_id = find_latest_migration(migrations_dir)
            except ValueError as e:
                app.console.print(f"[red]✗[/red] {e}")
                raise SystemExit(1)

            if migration_id is None:
                app.console.print("[yellow]No migrations found[/yellow]")
                app.console.print("Run 'metaxy migrations generate' first")
                return

        try:
            yaml_path = find_migration_yaml(migration_id, migrations_dir)
            migration = load_migration_from_yaml(yaml_path)
        except FileNotFoundError as e:
            app.console.print(f"[red]✗[/red] {e}")
            raise SystemExit(1)

        if not isinstance(migration, DiffMigration):
            app.console.print(
                f"[red]✗[/red] Migration '{migration_id}' is not a DiffMigration"
            )
            app.console.print(
                f"  Type: {type(migration).__name__} (explain only supports DiffMigration)"
            )
            raise SystemExit(1)

        app.console.print(f"\n[bold]Migration: {migration_id}[/bold]")
        app.console.print(f"From: {migration.from_snapshot_version}")
        app.console.print(f"To:   {migration.to_snapshot_version}")
        app.console.print()

        try:
            graph_diff = migration.compute_graph_diff(metadata_store, project)
        except Exception as e:
            from metaxy.cli.utils import print_error

            print_error(app.console, "Failed to compute diff", e)
            raise SystemExit(1)

        if not graph_diff.has_changes:
            app.console.print("[yellow]No changes detected[/yellow]")
            return

        _display_graph_diff(graph_diff)

        affected_features = migration.get_affected_features(metadata_store, project)
        app.console.print(f"[bold]Affected Features ({len(affected_features)}):[/bold]")
        for feature_key in affected_features[:10]:
            app.console.print(f"  • {feature_key}")
        if len(affected_features) > 10:
            app.console.print(f"  ... and {len(affected_features) - 10} more")


def _load_migrations_to_describe(migration_ids: list[str], migrations_dir) -> list:
    """Load migrations to describe, either from chain or specific IDs."""
    from metaxy.migrations.loader import (
        build_migration_chain,
        find_migration_yaml,
        load_migration_from_yaml,
    )

    if not migration_ids:
        try:
            chain = build_migration_chain(migrations_dir)
        except ValueError as e:
            from metaxy.cli.utils import print_error

            print_error(app.console, "Invalid migration chain", e)
            raise SystemExit(1)

        if not chain:
            app.console.print("[yellow]No migrations found.[/yellow]")
            return []

        return chain

    migrations_to_describe = []
    for migration_id in migration_ids:
        try:
            yaml_path = find_migration_yaml(migration_id, migrations_dir)
            migration_obj = load_migration_from_yaml(yaml_path)
            migrations_to_describe.append(migration_obj)
        except FileNotFoundError as e:
            app.console.print(f"[red]✗[/red] {e}")
            raise SystemExit(1)
    return migrations_to_describe


def _print_describe_header(migration_obj, yaml_path) -> None:
    """Print the header section of a migration description."""
    from metaxy.migrations.models import DiffMigration, FullGraphMigration

    app.console.print("\n[bold]Migration Description[/bold]")
    app.console.print("─" * 60)
    app.console.print(f"[bold]ID:[/bold] {migration_obj.migration_id}")
    app.console.print(f"[bold]Created:[/bold] {migration_obj.created_at.isoformat()}")
    app.console.print(f"[bold]Parent:[/bold] {migration_obj.parent}")
    app.console.print(f"[bold]YAML:[/bold] {yaml_path}")
    app.console.print()

    if isinstance(migration_obj, DiffMigration):
        app.console.print("[bold]Snapshots:[/bold]")
        app.console.print(f"  From: {migration_obj.from_snapshot_version}")
        app.console.print(f"  To:   {migration_obj.to_snapshot_version}")
        app.console.print()

    if isinstance(migration_obj, FullGraphMigration):
        app.console.print("[bold]Operations:[/bold]")
        for j, op in enumerate(migration_obj.ops, 1):
            op_type = op.get("type", "unknown")
            app.console.print(f"  {j}. {op_type}")
        app.console.print()


def _get_feature_stats(feature_key_str: str, events_df, graph) -> tuple[int, int, bool]:
    """Get rows_affected, attempts, and has_upstream for a feature."""
    import polars as pl

    from metaxy.models.types import FeatureKey

    feature_key_obj = FeatureKey(feature_key_str.split("/"))

    feature_events = events_df.filter(
        (pl.col("feature_key") == feature_key_str)
        & (pl.col("event_type") == "feature_migration_completed")
    )

    if feature_events.height > 0:
        feature_events = feature_events.with_columns(
            pl.col("payload")
            .str.json_path_match("$.rows_affected")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("rows_affected")
        )
        rows_affected = int(feature_events["rows_affected"].sum())

        attempts = events_df.filter(
            (pl.col("feature_key") == feature_key_str)
            & (pl.col("event_type") == "feature_migration_started")
        ).height
    else:
        rows_affected = 0
        attempts = 0

    plan = graph.get_feature_plan(feature_key_obj)
    has_upstream = plan.deps is not None and len(plan.deps) > 0

    return rows_affected, attempts, has_upstream


def _print_affected_features(affected_features: list[str], events_df, graph) -> None:
    """Print the affected features section of a migration description."""
    from metaxy.models.types import FeatureKey

    app.console.print(f"\n[bold]Affected Features ({len(affected_features)}):[/bold]")
    app.console.print("─" * 60)

    for feature_key_str in affected_features:
        feature_key_obj = FeatureKey(feature_key_str.split("/"))

        if feature_key_obj not in graph.features_by_key:
            app.console.print(f"[yellow]⚠[/yellow] {feature_key_str}")
            app.console.print("    [yellow]Feature not in current graph[/yellow]")
            continue

        rows_affected, attempts, has_upstream = _get_feature_stats(
            feature_key_str, events_df, graph
        )

        app.console.print(f"[bold]{feature_key_str}[/bold]")
        app.console.print(f"    Rows Affected: {rows_affected}")
        app.console.print(f"    Attempts: {attempts}")
        app.console.print(f"    Has upstream: {has_upstream}")

    app.console.print()


def _print_execution_status(
    migration_status,
    completed_features: list[str],
    failed_features: dict[str, str],
    affected_features: list[str],
) -> None:
    """Print the execution status section of a migration description."""
    from metaxy.metadata_store.system import MigrationStatus

    app.console.print("[bold]Execution Status:[/bold]")

    if migration_status == MigrationStatus.COMPLETED:
        app.console.print("  [green]✓ COMPLETED[/green]")
        app.console.print(
            f"    Features processed: {len(completed_features)}/{len(affected_features)}"
        )
    elif migration_status == MigrationStatus.FAILED:
        app.console.print("  [red]✗ FAILED[/red]")
        app.console.print(
            f"    Features completed: {len(completed_features)}/{len(affected_features)}"
        )
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
        app.console.print(
            f"    Features completed: {len(completed_features)}/{len(affected_features)}"
        )
    else:
        app.console.print("  [blue]○ NOT STARTED[/blue]")
        app.console.print(f"    Features to process: {len(affected_features)}")


def _describe_single_migration(
    migration_obj, migrations_dir, metadata_store, project, storage
) -> None:
    """Describe a single migration with all its details."""
    from metaxy.migrations.loader import find_migration_yaml
    from metaxy.models.feature import FeatureGraph

    yaml_path = find_migration_yaml(migration_obj.migration_id, migrations_dir)
    _print_describe_header(migration_obj, yaml_path)

    app.console.print("[bold]Computing affected features...[/bold]")
    try:
        affected_features = migration_obj.get_affected_features(metadata_store, project)
    except Exception as e:
        app.console.print(f"[red]✗[/red] Failed to compute affected features: {e}")
        return

    graph = FeatureGraph.get_active()
    events_df = storage.get_migration_events(migration_obj.migration_id, project)
    _print_affected_features(affected_features, events_df, graph)

    summary = storage.get_migration_summary(
        migration_obj.migration_id,
        project,
        expected_features=affected_features,
    )
    _print_execution_status(
        summary["status"],
        summary["completed_features"],
        summary["failed_features"],
        affected_features,
    )


@app.command
def describe(
    migration_ids: Annotated[
        list[str],
        cyclopts.Parameter(
            help="Migration IDs to describe (default: all migrations in order)"
        ),
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

    context = AppContext.get()
    project = context.get_required_project()
    metadata_store = context.get_store(store)
    migrations_dir = Path(".metaxy/migrations")

    with metadata_store:
        storage = SystemTableStorage(metadata_store)
        migrations_to_describe = _load_migrations_to_describe(
            migration_ids, migrations_dir
        )

        if not migrations_to_describe:
            return

        for i, migration_obj in enumerate(migrations_to_describe):
            if i > 0:
                app.console.print("\n")

            _describe_single_migration(
                migration_obj, migrations_dir, metadata_store, project, storage
            )


if __name__ == "__main__":
    app()
