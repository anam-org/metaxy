"""Load migrations from YAML and Python files."""

import importlib.util
import inspect
import sys
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

    # Translate YAML field names to model field names
    if "id" in data and "migration_id" not in data:
        data["migration_id"] = data.pop("id")

    # Infer migration_type if not present (default to DiffMigration for backwards compatibility)
    if "migration_type" not in data:
        # Check if it's a DiffMigration (has parent, from_snapshot_version, to_snapshot_version)
        if "parent" in data and "from_snapshot_version" in data:
            data["migration_type"] = "metaxy.migrations.models.DiffMigration"
        else:
            data["migration_type"] = "metaxy.migrations.models.FullGraphMigration"

    # Use Pydantic's discriminated union to automatically deserialize
    try:
        migration = MigrationAdapter.validate_python(data)
    except Exception as e:
        raise ValueError(f"Failed to load migration from {yaml_path}: {e}") from e

    return migration


def load_migration_from_python(py_path: Path) -> "Migration":
    """Load migration from Python file.

    Args:
        py_path: Path to migration Python file

    Returns:
        Migration instance (subclass of Migration)

    Raises:
        FileNotFoundError: If Python file doesn't exist
        ValueError: If Python file is invalid
        ImportError: If Python file has import errors
    """
    from metaxy.migrations.models import Migration

    if not py_path.exists():
        raise FileNotFoundError(f"Migration Python file not found: {py_path}")

    # Dynamic import using importlib
    try:
        # Create a unique module name based on file path
        module_name = f"metaxy_migration_{py_path.stem}"

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec from {py_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    except SyntaxError as e:
        raise ValueError(
            f"Syntax error in migration Python file {py_path}:\n"
            f"  Line {e.lineno}: {e.msg}\n"
            f"  {e.text}"
        ) from e
    except ImportError as e:
        raise ImportError(
            f"Import error in migration Python file {py_path}:\n"
            f"  {e}\n"
            f"  Make sure all dependencies are available."
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to load migration Python file {py_path}: {e}") from e

    # Find Migration subclass in module
    migration_classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if it's a Migration subclass (but not Migration itself)
        if issubclass(obj, Migration) and obj is not Migration:
            # Only include classes defined in this module (not imported)
            if obj.__module__ == module_name:
                migration_classes.append((name, obj))

    if len(migration_classes) == 0:
        raise ValueError(
            f"No Migration subclass found in {py_path}.\n"
            f"  Python migration files must define exactly one class inheriting from Migration.\n"
            f"  Example:\n"
            f"    from metaxy.migrations.models import DiffMigration\n"
            f"    \n"
            f"    class MyMigration(DiffMigration):\n"
            f"        migration_id = '20250101_120000'\n"
            f"        ..."
        )

    if len(migration_classes) > 1:
        class_names = [name for name, _ in migration_classes]
        raise ValueError(
            f"Multiple Migration subclasses found in {py_path}: {class_names}\n"
            f"  Python migration files must define exactly one Migration subclass."
        )

    # Instantiate the migration class
    migration_class_name, migration_class = migration_classes[0]

    try:
        # Try to instantiate without arguments (class should define all fields as class attributes)
        migration_instance = migration_class()
        return migration_instance
    except Exception as e:
        raise ValueError(
            f"Failed to instantiate {migration_class_name} from {py_path}:\n"
            f"  {e}\n"
            f"  Migration classes should define all required fields as class attributes or provide defaults."
        ) from e


def load_migration_from_file(migration_path: Path) -> "Migration":
    """Load migration from YAML or Python file based on extension.

    Args:
        migration_path: Path to migration file (.yaml or .py)

    Returns:
        Migration instance

    Raises:
        ValueError: If file extension is not .yaml or .py
        FileNotFoundError: If file doesn't exist
    """
    if migration_path.suffix == ".yaml":
        return load_migration_from_yaml(migration_path)
    elif migration_path.suffix == ".py":
        return load_migration_from_python(migration_path)
    else:
        raise ValueError(
            f"Unsupported migration file type: {migration_path.suffix}\n"
            f"  Migration files must be .yaml or .py"
        )


def find_migration_file(migration_id: str, migrations_dir: Path | None = None) -> Path:
    """Find migration file (YAML or Python) for a migration ID.

    Args:
        migration_id: Migration ID (e.g., "20250127_120000" or "20250127_120000_feature_update")
        migrations_dir: Directory containing migrations (defaults to .metaxy/migrations/)

    Returns:
        Path to migration file (.yaml or .py)

    Raises:
        FileNotFoundError: If migration file not found
    """
    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        raise FileNotFoundError(
            f"Migration '{migration_id}' not found. "
            f"Migrations directory does not exist: {migrations_dir}"
        )

    # Search through all migration files (.yaml and .py) to find the one with matching ID
    for pattern in ["*.yaml", "*.py"]:
        for migration_file in migrations_dir.glob(pattern):
            try:
                migration = load_migration_from_file(migration_file)
                if migration.migration_id == migration_id:
                    return migration_file
            except Exception:
                # Skip files that can't be loaded
                continue

    # Not found - list available migrations
    available = []
    for pattern in ["*.yaml", "*.py"]:
        for migration_file in migrations_dir.glob(pattern):
            try:
                migration = load_migration_from_file(migration_file)
                available.append(migration.migration_id)
            except Exception:
                continue

    raise FileNotFoundError(
        f"Migration '{migration_id}' not found in {migrations_dir}.\n"
        f"Available migrations: {available}"
    )


def find_migration_yaml(migration_id: str, migrations_dir: Path | None = None) -> Path:
    """Find YAML file for a migration ID by searching all YAML files.

    DEPRECATED: Use find_migration_file() instead to support both YAML and Python migrations.

    Args:
        migration_id: Migration ID (e.g., "20250127_120000" or "20250127_120000_feature_update")
        migrations_dir: Directory containing migrations (defaults to .metaxy/migrations/)

    Returns:
        Path to migration YAML file

    Raises:
        FileNotFoundError: If migration YAML not found
    """
    # For backward compatibility, first try to find YAML specifically
    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        raise FileNotFoundError(
            f"Migration '{migration_id}' not found. "
            f"Migrations directory does not exist: {migrations_dir}"
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
        f"Migration '{migration_id}' not found in {migrations_dir}.\n"
        f"Available migrations: {available}"
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

    # Collect both YAML and Python migration files
    migration_files = []
    for pattern in ["*.yaml", "*.py"]:
        migration_files.extend(migrations_dir.glob(pattern))

    migration_files = sorted(migration_files)
    return [f.stem for f in migration_files]


def find_latest_migration(migrations_dir: Path | None = None) -> str | None:
    """Find the latest migration ID (head of the chain).

    Args:
        migrations_dir: Directory containing migrations (defaults to .metaxy/migrations/)

    Returns:
        Migration ID of the head, or None if no migrations exist

    Raises:
        ValueError: If multiple heads detected (conflict)
    """
    from metaxy.migrations.models import DiffMigration

    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        return None

    # Load all migrations (both YAML and Python)
    migrations: dict[str, DiffMigration] = {}
    for pattern in ["*.yaml", "*.py"]:
        for migration_file in migrations_dir.glob(pattern):
            try:
                migration = load_migration_from_file(migration_file)
                # Only include DiffMigration instances (which have parent chains)
                if isinstance(migration, DiffMigration):
                    migrations[migration.migration_id] = migration
            except Exception:
                # Skip files that can't be loaded
                continue

    if not migrations:
        return None

    # Find migrations that are parents of others
    all_parents = {m.parent for m in migrations.values() if m.parent != "initial"}

    # Find heads (migrations that are not parents of any other migration)
    heads = [mid for mid in migrations.keys() if mid not in all_parents]

    if len(heads) == 0:
        # This means there's a cycle or orphaned migrations
        raise ValueError(
            "No head migration found - possible cycle in migration chain. "
            f"All migrations: {list(migrations.keys())}"
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
        ValueError: If chain is invalid (cycles, orphans, multiple heads, duplicate IDs)
    """
    from metaxy.migrations.models import DiffMigration

    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    if not migrations_dir.exists():
        return []

    # Load all migrations (both YAML and Python)
    # Track file paths for better error messages
    migrations: dict[str, DiffMigration] = {}
    migration_id_to_path: dict[str, Path] = {}
    migration_files = []
    for pattern in ["*.yaml", "*.py"]:
        migration_files.extend(migrations_dir.glob(pattern))

    for migration_file in sorted(migration_files):
        try:
            migration = load_migration_from_file(migration_file)
            # Only include DiffMigration instances (which have parent chains)
            if isinstance(migration, DiffMigration):
                # Check for duplicate migration IDs across different file types
                if migration.migration_id in migrations:
                    existing_path = migration_id_to_path[migration.migration_id]
                    existing_type = (
                        "Python" if existing_path.suffix == ".py" else "YAML"
                    )
                    new_type = "Python" if migration_file.suffix == ".py" else "YAML"
                    raise ValueError(
                        f"Duplicate migration ID '{migration.migration_id}' found:\n"
                        f"  {existing_type}: {existing_path}\n"
                        f"  {new_type}: {migration_file}\n"
                        f"Please ensure migration IDs are unique across all migration files."
                    )
                migrations[migration.migration_id] = migration
                migration_id_to_path[migration.migration_id] = migration_file
        except ValueError:
            # Re-raise ValueError (including duplicate ID errors)
            raise
        except Exception:
            # Skip files that can't be loaded (syntax errors, etc.)
            continue

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
            # Provide cycle path for debugging
            cycle_path = " -> ".join(list(visited) + [current_id])
            raise ValueError(
                f"Cycle detected in migration chain:\n"
                f"  Path: {cycle_path}\n"
                f"Each migration's parent must reference an earlier migration or 'initial'."
            )

        if current_id not in migrations:
            # Enhanced error message with file type info
            current_path = migration_id_to_path.get(current_id)
            if current_path:
                file_type = "Python" if current_path.suffix == ".py" else "YAML"
                raise ValueError(
                    f"Migration '{current_id}' referenced as parent but not found.\n"
                    f"  Referenced in: {file_type} file {current_path}\n"
                    f"  Available migrations: {list(migrations.keys())}"
                )
            else:
                raise ValueError(
                    f"Migration '{current_id}' referenced as parent but file not found.\n"
                    f"  Available migrations: {list(migrations.keys())}\n"
                    f"Ensure the parent migration file exists in {migrations_dir}"
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
        # Provide detailed info about orphaned migrations with file types
        orphan_details = []
        for orphan_id in orphans:
            orphan_path = migration_id_to_path[orphan_id]
            file_type = "Python" if orphan_path.suffix == ".py" else "YAML"
            orphan_parent = migrations[orphan_id].parent
            orphan_details.append(
                f"  - {orphan_id} ({file_type} file: {orphan_path.name})\n"
                f"    Parent: {orphan_parent}"
            )

        raise ValueError(
            f"Orphaned migrations detected (not in main chain):\n"
            f"{''.join(orphan_details)}\n"
            f"Each migration must have parent pointing to previous migration or 'initial'.\n"
            f"Current head: {head_id}"
        )

    return chain
