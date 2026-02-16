"""Load migrations from YAML files."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaxy.migrations.models import Migration


def load_migration_from_yaml(yaml_path: Path) -> "Migration":
    """Load migration from YAML file.

    Uses Pydantic's discriminated unions for automatic polymorphic deserialization
    based on the migration_type field.

    Args:
        yaml_path: Path to migration YAML file

    Returns:
        Migration instance (DiffMigration or FullGraphMigration)

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML is invalid or migration type is not supported
    """
    import yaml

    from metaxy.migrations.models import MigrationAdapter

    if not yaml_path.exists():
        raise FileNotFoundError(f"Migration YAML not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Use Pydantic's discriminated union to automatically deserialize
    try:
        migration = MigrationAdapter.validate_python(data)
    except Exception as e:
        raise ValueError(f"Failed to load migration from {yaml_path}: {e}") from e

    return migration


def find_migration_yaml(migration_id: str, migrations_dir: Path | None = None) -> Path:
    """Find YAML file for a migration ID by searching all YAML files.

    Args:
        migration_id: Migration ID (e.g., "20250127_120000" or "20250127_120000_feature_update")
        migrations_dir: Directory containing migrations (defaults to .metaxy/migrations/)

    Returns:
        Path to migration YAML file

    Raises:
        FileNotFoundError: If migration YAML not found
    """
    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        raise FileNotFoundError(
            f"Migration '{migration_id}' not found. Migrations directory does not exist: {migrations_dir}"
        )

    # Search through all YAML files to find the one with matching ID
    for yaml_file in migrations_dir.glob("*.yaml"):
        try:
            migration = load_migration_from_yaml(yaml_file)
            if migration.migration_id == migration_id:
                return yaml_file
        except Exception:
            # Skip files that can't be loaded
            continue

    # Not found - list available migrations
    available = []
    for yaml_file in migrations_dir.glob("*.yaml"):
        try:
            migration = load_migration_from_yaml(yaml_file)
            available.append(migration.migration_id)
        except Exception:
            continue

    raise FileNotFoundError(
        f"Migration '{migration_id}' not found in {migrations_dir}.\nAvailable migrations: {available}"
    )


def list_migrations(migrations_dir: Path | None = None) -> list[str]:
    """List all available migration IDs.

    Args:
        migrations_dir: Directory containing migrations (defaults to .metaxy/migrations/)

    Returns:
        List of migration IDs sorted by creation time
    """
    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        return []

    yaml_files = sorted(migrations_dir.glob("*.yaml"))
    return [f.stem for f in yaml_files]


def find_latest_migration(migrations_dir: Path | None = None) -> str | None:
    """Find the latest migration ID (head of the chain).

    Args:
        migrations_dir: Directory containing migrations (defaults to .metaxy/migrations/)

    Returns:
        Migration ID of the head, or None if no migrations exist

    Raises:
        ValueError: If multiple heads detected (conflict)
    """
    from metaxy.migrations.models import Migration

    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        return None

    # Load all migrations - all migrations form chains via parent IDs
    migrations: dict[str, Migration] = {}
    for yaml_file in migrations_dir.glob("*.yaml"):
        migration = load_migration_from_yaml(yaml_file)
        migrations[migration.migration_id] = migration

    if not migrations:
        return None

    # Find migrations that are parents of others
    all_parents = {m.parent for m in migrations.values() if m.parent != "initial"}

    # Find heads (migrations that are not parents of any other migration)
    heads = [mid for mid in migrations.keys() if mid not in all_parents]

    if len(heads) == 0:
        # This means there's a cycle or orphaned migrations
        raise ValueError(
            f"No head migration found - possible cycle in migration chain. All migrations: {list(migrations.keys())}"
        )

    if len(heads) > 1:
        raise ValueError(
            f"Multiple migration heads detected: {heads}. "
            "This usually means two migrations were created in parallel. "
            "Please merge them by creating a new migration that depends on one head, "
            "or delete one of the conflicting migrations."
        )

    return heads[0]


def build_migration_chain(
    migrations_dir: Path | None = None,
) -> list["Migration"]:
    """Build ordered migration chain from parent IDs.

    Args:
        migrations_dir: Directory containing migrations (defaults to .metaxy/migrations/)

    Returns:
        List of migrations in order from oldest to newest

    Raises:
        ValueError: If chain is invalid (cycles, orphans, multiple heads)
    """
    from metaxy.migrations.models import Migration

    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        return []

    # Load all migrations - all migrations form chains via parent IDs
    migrations: dict[str, Migration] = {}
    for yaml_file in sorted(migrations_dir.glob("*.yaml")):
        migration = load_migration_from_yaml(yaml_file)
        migrations[migration.migration_id] = migration

    if not migrations:
        return []

    # Validate single head
    head_id = find_latest_migration(migrations_dir)
    if head_id is None:
        return []

    # Build chain by following parent links backwards
    chain = []
    current_id: str | None = head_id

    visited = set()
    while current_id is not None and current_id != "initial":
        if current_id in visited:
            raise ValueError(f"Cycle detected in migration chain at: {current_id}")

        if current_id not in migrations:
            raise ValueError(
                f"Migration '{current_id}' referenced as parent but YAML not found. "
                f"Available migrations: {list(migrations.keys())}"
            )

        visited.add(current_id)
        migration = migrations[current_id]
        chain.append(migration)
        current_id = migration.parent

    # Reverse to get oldest-first order
    chain.reverse()

    # Validate all migrations are in the chain (no orphans)
    if len(chain) != len(migrations):
        orphans = set(migrations.keys()) - set(m.migration_id for m in chain)
        raise ValueError(
            f"Orphaned migrations detected (not in main chain): {orphans}. "
            "Each migration must have parent pointing to previous migration or 'initial'."
        )

    return chain
