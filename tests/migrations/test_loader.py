"""Tests for migration loader validation with mixed Python/YAML migrations."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from metaxy.migrations.loader import (
    build_migration_chain,
    find_latest_migration,
    load_migration_from_file,
)
from metaxy.migrations.models import DiffMigration


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """Create temporary migrations directory."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    return migrations_dir


def write_yaml_migration(
    migrations_dir: Path,
    migration_id: str,
    parent: str = "initial",
    from_snapshot: str = "from_snap",
    to_snapshot: str = "to_snap",
) -> Path:
    """Helper to write a YAML migration file."""
    yaml_path = migrations_dir / f"{migration_id}.yaml"
    migration_data = {
        "id": migration_id,
        "parent": parent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "from_snapshot_version": from_snapshot,
        "to_snapshot_version": to_snapshot,
        "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    }

    with open(yaml_path, "w") as f:
        yaml.dump(migration_data, f)

    return yaml_path


def write_python_migration(
    migrations_dir: Path,
    migration_id: str,
    parent: str = "initial",
    from_snapshot: str = "from_snap",
    to_snapshot: str = "to_snap",
) -> Path:
    """Helper to write a Python migration file."""
    py_path = migrations_dir / f"{migration_id}.py"

    code = f'''"""Migration {migration_id}."""

from datetime import datetime, timezone
from metaxy.migrations.models import DiffMigration

class Migration(DiffMigration):
    migration_id: str = "{migration_id}"
    parent: str = "{parent}"
    created_at: datetime = datetime.now(timezone.utc)
    from_snapshot_version: str = "{from_snapshot}"
    to_snapshot_version: str = "{to_snapshot}"
    ops: list = [{{"type": "metaxy.migrations.ops.DataVersionReconciliation"}}]
'''

    with open(py_path, "w") as f:
        f.write(code)

    return py_path


def write_python_builder_migration(
    migrations_dir: Path,
    migration_id: str,
    parent: str = "initial",
    from_snapshot: str = "from_snap",
    to_snapshot: str = "to_snap",
) -> Path:
    """Helper to write a PythonMigration that defines operations()."""
    py_path = migrations_dir / f"{migration_id}.py"

    code = f'''"""PythonMigration {migration_id} with operations() builder."""

from datetime import datetime, timezone
from metaxy.migrations.models import PythonMigration
from metaxy.migrations.ops import DataVersionReconciliation


class Migration(PythonMigration):
    migration_id: str = "{migration_id}"
    parent: str = "{parent}"
    created_at: datetime = datetime.now(timezone.utc)
    from_snapshot_version: str = "{from_snapshot}"
    to_snapshot_version: str = "{to_snapshot}"

    def operations(self):
        return [
            DataVersionReconciliation(),
        ]
'''

    with open(py_path, "w") as f:
        f.write(code)

    return py_path


class TestBasicLoading:
    """Test basic loading of YAML and Python migrations."""

    def test_load_yaml_migration(self, migrations_dir: Path):
        """Test loading a YAML migration."""
        yaml_path = write_yaml_migration(migrations_dir, "20250101_120000")

        migration = load_migration_from_file(yaml_path)
        assert isinstance(migration, DiffMigration)
        assert migration.migration_id == "20250101_120000"
        assert migration.parent == "initial"

    def test_load_python_migration(self, migrations_dir: Path):
        """Test loading a Python migration."""
        py_path = write_python_migration(migrations_dir, "20250101_120000")

        migration = load_migration_from_file(py_path)
        assert isinstance(migration, DiffMigration)
        assert migration.migration_id == "20250101_120000"
        assert migration.parent == "initial"
        assert migration.ops == [
            {"type": "metaxy.migrations.ops.DataVersionReconciliation"}
        ]

    def test_load_python_migration_with_operations_builder(self, migrations_dir: Path):
        """PythonMigration subclasses should serialize operations() automatically."""
        py_path = write_python_builder_migration(migrations_dir, "20250101_120500")

        migration = load_migration_from_file(py_path)
        assert isinstance(migration, DiffMigration)
        assert migration.migration_id == "20250101_120500"
        assert migration.ops == [
            {"type": "metaxy.migrations.ops.DataVersionReconciliation"}
        ]

    def test_empty_directory(self, migrations_dir: Path):
        """Test empty migrations directory."""
        chain = build_migration_chain(migrations_dir)
        assert chain == []

    def test_nonexistent_directory(self, tmp_path: Path):
        """Test nonexistent migrations directory."""
        nonexistent = tmp_path / "does_not_exist"
        chain = build_migration_chain(nonexistent)
        assert chain == []


class TestMixedChains:
    """Test building chains with mixed Python/YAML migrations."""

    def test_yaml_only_chain(self, migrations_dir: Path):
        """Test chain with only YAML migrations."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_140000", parent="20250101_130000"
        )

        chain = build_migration_chain(migrations_dir)
        assert len(chain) == 3
        assert chain[0].migration_id == "20250101_120000"
        assert chain[1].migration_id == "20250101_130000"
        assert chain[2].migration_id == "20250101_140000"

    def test_python_only_chain(self, migrations_dir: Path):
        """Test chain with only Python migrations."""
        write_python_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )
        write_python_migration(
            migrations_dir, "20250101_140000", parent="20250101_130000"
        )

        chain = build_migration_chain(migrations_dir)
        assert len(chain) == 3
        assert chain[0].migration_id == "20250101_120000"
        assert chain[1].migration_id == "20250101_130000"
        assert chain[2].migration_id == "20250101_140000"

    def test_mixed_yaml_python_chain(self, migrations_dir: Path):
        """Test chain with mixed YAML and Python migrations."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_140000", parent="20250101_130000"
        )
        write_python_migration(
            migrations_dir, "20250101_150000", parent="20250101_140000"
        )

        chain = build_migration_chain(migrations_dir)
        assert len(chain) == 4
        assert chain[0].migration_id == "20250101_120000"
        assert chain[1].migration_id == "20250101_130000"
        assert chain[2].migration_id == "20250101_140000"
        assert chain[3].migration_id == "20250101_150000"

    def test_find_latest_in_mixed_chain(self, migrations_dir: Path):
        """Test finding latest migration in mixed chain."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_140000", parent="20250101_130000"
        )

        latest = find_latest_migration(migrations_dir)
        assert latest == "20250101_140000"


class TestDuplicateIDDetection:
    """Test detection of duplicate migration IDs across file types."""

    def test_duplicate_id_yaml_and_python(self, migrations_dir: Path):
        """Test error when same ID exists in both YAML and Python."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(migrations_dir, "20250101_120000", parent="initial")

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Duplicate migration ID '20250101_120000' found" in error_msg
        assert "YAML:" in error_msg
        assert "Python:" in error_msg
        assert "20250101_120000.yaml" in error_msg
        assert "20250101_120000.py" in error_msg

    def test_duplicate_id_two_yaml_files(self, migrations_dir: Path):
        """Test error when same ID exists in two YAML files with different names."""
        # Create first YAML with standard name
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")

        # Create second YAML with different filename but same ID
        yaml_path2 = migrations_dir / "migration_other_name.yaml"
        migration_data = {
            "id": "20250101_120000",  # Same ID!
            "parent": "initial",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "from_snapshot_version": "from_snap",
            "to_snapshot_version": "to_snap",
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }
        with open(yaml_path2, "w") as f:
            yaml.dump(migration_data, f)

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Duplicate migration ID '20250101_120000' found" in error_msg

    def test_no_error_when_ids_unique(self, migrations_dir: Path):
        """Test no error when all IDs are unique across file types."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_140000", parent="20250101_130000"
        )

        # Should not raise
        chain = build_migration_chain(migrations_dir)
        assert len(chain) == 3


class TestParentReferenceValidation:
    """Test validation of parent references across file types."""

    def test_python_references_yaml_parent(self, migrations_dir: Path):
        """Test Python migration can reference YAML parent."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        chain = build_migration_chain(migrations_dir)
        assert len(chain) == 2
        assert chain[0].migration_id == "20250101_120000"
        assert chain[1].migration_id == "20250101_130000"

    def test_yaml_references_python_parent(self, migrations_dir: Path):
        """Test YAML migration can reference Python parent."""
        write_python_migration(migrations_dir, "20250101_120000", parent="initial")
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        chain = build_migration_chain(migrations_dir)
        assert len(chain) == 2
        assert chain[0].migration_id == "20250101_120000"
        assert chain[1].migration_id == "20250101_130000"

    def test_missing_parent_reference(self, migrations_dir: Path):
        """Test error when parent migration doesn't exist."""
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )  # Parent doesn't exist

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Migration '20250101_120000' referenced as parent but" in error_msg
        assert "Available migrations:" in error_msg

    def test_cross_type_parent_chain(self, migrations_dir: Path):
        """Test complex chain alternating between file types."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_140000", parent="20250101_130000"
        )
        write_python_migration(
            migrations_dir, "20250101_150000", parent="20250101_140000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_160000", parent="20250101_150000"
        )

        chain = build_migration_chain(migrations_dir)
        assert len(chain) == 5
        # Verify order
        for i, expected_id in enumerate(
            [
                "20250101_120000",
                "20250101_130000",
                "20250101_140000",
                "20250101_150000",
                "20250101_160000",
            ]
        ):
            assert chain[i].migration_id == expected_id


class TestCycleDetection:
    """Test detection of cycles in migration chains."""

    def test_cycle_yaml_only(self, migrations_dir: Path):
        """Test cycle detection in YAML-only chain."""
        write_yaml_migration(
            migrations_dir, "20250101_120000", parent="20250101_130000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        # Cycle is detected by find_latest_migration() as "no head"
        assert "No head migration found" in error_msg or "Cycle detected" in error_msg

    def test_cycle_python_only(self, migrations_dir: Path):
        """Test cycle detection in Python-only chain."""
        write_python_migration(
            migrations_dir, "20250101_120000", parent="20250101_130000"
        )
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        assert "No head migration found" in error_msg or "Cycle detected" in error_msg

    def test_cycle_mixed_types(self, migrations_dir: Path):
        """Test cycle detection across YAML and Python migrations."""
        write_yaml_migration(
            migrations_dir, "20250101_120000", parent="20250101_130000"
        )
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        assert "No head migration found" in error_msg or "Cycle detected" in error_msg

    def test_self_reference_cycle(self, migrations_dir: Path):
        """Test detection of migration referencing itself."""
        write_yaml_migration(
            migrations_dir, "20250101_120000", parent="20250101_120000"
        )

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        # Self-reference will be caught during chain building
        assert "Cycle detected" in error_msg or "No head migration found" in error_msg


class TestOrphanDetection:
    """Test detection of orphaned migrations."""

    def test_orphan_yaml_migration(self, migrations_dir: Path):
        """Test detection of orphaned YAML migration."""
        # Create valid chain
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        # Create orphan (points to non-existent parent, breaking the chain)
        write_yaml_migration(
            migrations_dir, "20250101_125000", parent="20250101_110000"
        )  # Parent doesn't exist

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        # This creates multiple heads scenario
        assert "Multiple migration heads" in error_msg or "Orphaned" in error_msg

    def test_orphan_python_migration(self, migrations_dir: Path):
        """Test detection of orphaned Python migration."""
        # Create valid chain
        write_python_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        # Create orphan pointing to non-existent parent
        write_python_migration(
            migrations_dir, "20250101_125000", parent="20250101_110000"
        )

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Multiple migration heads" in error_msg or "Orphaned" in error_msg

    def test_orphan_in_broken_chain(self, migrations_dir: Path):
        """Test actual orphan scenario where parent exists but chain is broken."""
        # Create chain: A -> B -> C
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )
        write_yaml_migration(
            migrations_dir, "20250101_140000", parent="20250101_130000"
        )

        # Create orphan that references A but isn't part of main chain
        write_python_migration(
            migrations_dir, "20250101_125000", parent="20250101_120000"
        )  # Creates branch

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        # Multiple heads will be detected
        assert "Multiple migration heads" in error_msg

    def test_orphan_error_with_true_orphan(self, migrations_dir: Path):
        """Test error when migration's parent doesn't exist at all."""
        # Single migration with non-existent parent
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        # Should complain about missing parent
        assert "20250101_120000" in error_msg
        assert "referenced as parent" in error_msg or "not found" in error_msg


class TestMultipleHeadsDetection:
    """Test detection of multiple heads in migration chain."""

    def test_multiple_heads_yaml(self, migrations_dir: Path):
        """Test detection of multiple heads with YAML migrations."""
        # Two separate chains
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_yaml_migration(migrations_dir, "20250101_130000", parent="initial")

        with pytest.raises(ValueError) as exc_info:
            find_latest_migration(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Multiple migration heads detected" in error_msg

    def test_multiple_heads_python(self, migrations_dir: Path):
        """Test detection of multiple heads with Python migrations."""
        write_python_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(migrations_dir, "20250101_130000", parent="initial")

        with pytest.raises(ValueError) as exc_info:
            find_latest_migration(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Multiple migration heads detected" in error_msg

    def test_multiple_heads_mixed_types(self, migrations_dir: Path):
        """Test detection of multiple heads across YAML and Python."""
        # Chain 1: YAML -> Python
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )

        # Chain 2: Python -> YAML
        write_python_migration(migrations_dir, "20250101_140000", parent="initial")
        write_yaml_migration(
            migrations_dir, "20250101_150000", parent="20250101_140000"
        )

        with pytest.raises(ValueError) as exc_info:
            find_latest_migration(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Multiple migration heads detected" in error_msg


class TestErrorMessages:
    """Test quality and clarity of error messages."""

    def test_duplicate_id_shows_both_paths(self, migrations_dir: Path):
        """Test duplicate ID error shows both file paths."""
        write_yaml_migration(migrations_dir, "20250101_120000")
        write_python_migration(migrations_dir, "20250101_120000")

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        # Check that both filenames are mentioned
        assert "20250101_120000.yaml" in error_msg
        assert "20250101_120000.py" in error_msg
        assert "Duplicate migration ID" in error_msg

    def test_missing_parent_shows_clear_message(self, migrations_dir: Path):
        """Test missing parent reference shows clear error."""
        write_yaml_migration(
            migrations_dir, "20250101_130000", parent="20250101_120000"
        )  # Parent missing

        with pytest.raises(ValueError) as exc_info:
            build_migration_chain(migrations_dir)

        error_msg = str(exc_info.value)
        assert "20250101_120000" in error_msg
        assert "referenced as parent" in error_msg or "not found" in error_msg

    def test_multiple_heads_shows_head_list(self, migrations_dir: Path):
        """Test multiple heads error shows all heads."""
        write_yaml_migration(migrations_dir, "20250101_120000", parent="initial")
        write_python_migration(migrations_dir, "20250101_130000", parent="initial")

        with pytest.raises(ValueError) as exc_info:
            find_latest_migration(migrations_dir)

        error_msg = str(exc_info.value)
        assert "Multiple migration heads" in error_msg
        assert "20250101_120000" in error_msg
        assert "20250101_130000" in error_msg
