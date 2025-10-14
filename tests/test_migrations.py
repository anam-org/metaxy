"""Tests for migration system."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    ContainerDep,
    ContainerKey,
    ContainerSpec,
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    InMemoryMetadataStore,
)
from metaxy.migrations import (
    FeatureVersionMigration,
    Migration,
    apply_migration,
    detect_feature_changes,
    generate_migration,
)
from metaxy.models.feature import FeatureRegistry


def migrate_store_to_registry(
    source_store: InMemoryMetadataStore,
    target_registry: FeatureRegistry,
) -> InMemoryMetadataStore:
    """Create new store with target registry but source store's data.

    Helper for testing migrations - simulates code changing while data stays the same.
    """
    new_store = InMemoryMetadataStore(registry=target_registry)
    new_store._storage = source_store._storage.copy()
    return new_store


@pytest.fixture
def registry_v1() -> FeatureRegistry:
    """Registry with v1 features."""
    registry = FeatureRegistry()

    # Define features in this registry
    class UpstreamV1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_migrations", "upstream"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
        registry=registry,
    ):
        pass

    class DownstreamV1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_migrations", "downstream"]),
            deps=[FeatureDep(key=FeatureKey(["test_migrations", "upstream"]))],
            containers=[
                ContainerSpec(
                    key=ContainerKey(["default"]),
                    code_version=1,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["test_migrations", "upstream"]),
                            containers=[ContainerKey(["default"])],
                        )
                    ],
                ),
            ],
        ),
        registry=registry,
    ):
        pass

    return registry


@pytest.fixture
def registry_v2() -> FeatureRegistry:
    """Registry with v2 features (upstream code_version changed)."""
    registry = FeatureRegistry()

    # UpstreamV2 with code_version=2
    class UpstreamV2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_migrations", "upstream"]),
            deps=None,
            containers=[
                ContainerSpec(
                    key=ContainerKey(["default"]), code_version=2
                ),  # Changed!
            ],
        ),
        registry=registry,
    ):
        pass

    # Downstream unchanged
    class DownstreamV2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_migrations", "downstream"]),
            deps=[FeatureDep(key=FeatureKey(["test_migrations", "upstream"]))],
            containers=[
                ContainerSpec(
                    key=ContainerKey(["default"]),
                    code_version=1,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["test_migrations", "upstream"]),
                            containers=[ContainerKey(["default"])],
                        )
                    ],
                ),
            ],
        ),
        registry=registry,
    ):
        pass

    return registry


@pytest.fixture
def store_with_v1_data(registry_v1: FeatureRegistry) -> InMemoryMetadataStore:
    """Store with v1 upstream and downstream data."""
    store = InMemoryMetadataStore(registry=registry_v1)

    # Get feature classes
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Write upstream
    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [
                {"default": "hash1"},
                {"default": "hash2"},
                {"default": "hash3"},
            ],
        }
    )
    store.write_metadata(UpstreamV1, upstream_data)
    # Explicitly record feature version
    store.record_feature_version(UpstreamV1)

    # Write downstream
    downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
    store.calculate_and_write_data_versions(DownstreamV1, downstream_data)
    # Explicitly record feature version
    store.record_feature_version(DownstreamV1)

    return store


# Tests


def test_detect_no_changes(
    store_with_v1_data: InMemoryMetadataStore,
) -> None:
    """Test detection when no features changed (registry matches data)."""
    changes = detect_feature_changes(store_with_v1_data)

    # Should detect no changes (v1 registry, v1 data)
    assert len(changes) == 0


def test_detect_single_change(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test detection of single feature change."""
    # MigrateMigrate store fromto v2 registry to(simulates code changesimulates code change)
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    # Get features for version comparison
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Detect changes

    # Detect changes
    changes = detect_feature_changes(store_v2)

    assert len(changes) == 1
    assert changes[0].feature_key == FeatureKey(["test_migrations", "upstream"])
    assert changes[0].from_version == UpstreamV1.feature_version()
    assert changes[0].to_version == UpstreamV2.feature_version()


def test_generate_migration_no_changes(
    store_with_v1_data: InMemoryMetadataStore,
    tmp_path: Path,
) -> None:
    """Test generation when no changes detected."""
    result = generate_migration(store_with_v1_data, output_dir=str(tmp_path))

    # No changes (v1 registry, v1 data)
    assert result is None


def test_generate_migration_with_changes(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    tmp_path: Path,
) -> None:
    """Test migration file generation."""
    # Migrate to v2 registry
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    # Get features
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Generate migration
    migration_file = generate_migration(store_v2, output_dir=str(tmp_path))

    assert migration_file is not None
    assert Path(migration_file).exists()

    # Load and verify
    migration = Migration.from_yaml(migration_file)

    assert migration.version == 1
    assert len(migration.operations) == 1
    assert migration.operations[0].feature_key == ["test_migrations", "upstream"]
    assert migration.operations[0].from_ == UpstreamV1.feature_version()
    assert migration.operations[0].to == UpstreamV2.feature_version()


def test_apply_migration_recalculates_source(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test that migration recalculates source feature data_versions."""
    # Migrate to v2 registry
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    # Get features
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Create migration
    migration = Migration(
        version=1,
        id="migration_test_recalc",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                change_type="code_version",
                reason="Test",
            )
        ],
    )

    # Apply migration
    result = apply_migration(store_v2, migration)

    # Debug output
    if result.status != "completed":
        print(f"Migration failed with errors: {result.errors}")
        print(f"Summary:\n{result.summary()}")

    assert result.status == "completed", f"Migration failed: {result.errors}"
    assert "test_migrations_upstream" in result.affected_features


def test_apply_migration_idempotent(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test that migrations are idempotent."""
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    migration = Migration(
        version=1,
        id="migration_test_idempotent",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                change_type="code_version",
                reason="Test",
            )
        ],
    )

    # First application
    result1 = apply_migration(store_v2, migration)
    assert result1.status == "completed"

    # Second application (should skip)
    result2 = apply_migration(store_v2, migration)
    assert result2.status == "skipped"


def test_apply_migration_dry_run(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test dry-run mode doesn't modify data."""
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    migration = Migration(
        version=1,
        id="migration_test_dryrun",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                change_type="code_version",
                reason="Test",
            )
        ],
    )

    # Dry-run
    result = apply_migration(store_v2, migration, dry_run=True)

    assert result.status == "skipped"
    assert len(result.affected_features) > 0

    # Verify data unchanged - should still only have v1 feature_version
    all_data = store_v2.read_metadata(
        UpstreamV2,  # Reading through v2 registry
        current_only=False,  # Get all versions
    )
    # All rows should have v1 feature_version (not v2)
    assert all(
        fv == UpstreamV1.feature_version()
        for fv in all_data["feature_version"].to_list()
    )


def test_apply_migration_propagates_downstream(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test that migrations automatically propagate to downstream features."""
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Get initial downstream data_versions
    initial_downstream = store_v2.read_metadata(DownstreamV2, current_only=False)
    initial_data_versions = initial_downstream["data_version"].to_list()

    # Create migration
    migration = Migration(
        version=1,
        id="migration_test_propagate",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                change_type="code_version",
                reason="Test",
            )
        ],
    )

    # Apply migration
    result = apply_migration(store_v2, migration)

    assert result.status == "completed"
    # Should affect both upstream and downstream
    assert "test_migrations_upstream" in result.affected_features
    assert "test_migrations_downstream" in result.affected_features

    # Verify downstream was recalculated
    new_downstream = store_v2.read_metadata(DownstreamV2, current_only=True)
    new_data_versions = new_downstream["data_version"].to_list()

    # Data versions should have changed (based on new upstream)
    assert new_data_versions != initial_data_versions


def test_migration_yaml_roundtrip(tmp_path: Path) -> None:
    """Test saving and loading migration from YAML."""
    migration = Migration(
        version=1,
        id="migration_test_yaml",
        description="Test YAML",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["my", "feature"],
                from_="abc12345",
                to="def67890",
                change_type="code_version",
                reason="Updated algorithm",
            )
        ],
    )

    # Save to YAML
    yaml_path = tmp_path / "test_migration.yaml"
    migration.to_yaml(str(yaml_path))

    assert yaml_path.exists()

    # Load back
    loaded = Migration.from_yaml(str(yaml_path))

    assert loaded.version == migration.version
    assert loaded.id == migration.id
    assert loaded.description == migration.description
    assert len(loaded.operations) == 1
    assert loaded.operations[0].feature_key == ["my", "feature"]
    assert loaded.operations[0].from_ == "abc12345"
    assert loaded.operations[0].to == "def67890"


def test_feature_version_in_metadata(registry_v1: FeatureRegistry) -> None:
    """Test that feature_version column is automatically added."""
    store = InMemoryMetadataStore(registry=registry_v1)
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Write data without feature_version
    data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )

    # Should not have feature_version before write
    assert "feature_version" not in data.columns

    store.write_metadata(UpstreamV1, data)

    # Read back
    result = store.read_metadata(UpstreamV1, current_only=False)

    # Should have feature_version after write
    assert "feature_version" in result.columns
    assert all(
        fv == UpstreamV1.feature_version() for fv in result["feature_version"].to_list()
    )


def test_current_only_filtering(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test current_only parameter filters by feature_version."""
    # Write v1 data with v1 registry
    store_v1 = InMemoryMetadataStore(registry=registry_v1)
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    data_v1 = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )
    store_v1.write_metadata(UpstreamV1, data_v1)

    # Migrate to v2 registry and write v2 data
    store_v2 = migrate_store_to_registry(store_v1, registry_v2)
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    data_v2 = pl.DataFrame(
        {
            "sample_id": [3, 4],
            "data_version": [{"default": "h3"}, {"default": "h4"}],
        }
    )
    store_v2.write_metadata(UpstreamV2, data_v2)

    # Read current only (should get v2)
    current = store_v2.read_metadata(UpstreamV2, current_only=True)
    assert len(current) == 2
    assert set(current["sample_id"].to_list()) == {3, 4}

    # Read all versions
    all_data = store_v2.read_metadata(UpstreamV2, current_only=False)
    assert len(all_data) == 4
    assert set(all_data["sample_id"].to_list()) == {1, 2, 3, 4}

    # Verify feature_version values
    v1_rows = all_data.filter(pl.col("feature_version") == UpstreamV1.feature_version())
    v2_rows = all_data.filter(pl.col("feature_version") == UpstreamV2.feature_version())

    assert len(v1_rows) == 2
    assert len(v2_rows) == 2


def test_system_tables_created(registry_v1: FeatureRegistry) -> None:
    """Test that system tables are created when explicitly recording."""
    store = InMemoryMetadataStore(registry=registry_v1)
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Write some data
    data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )
    store.write_metadata(UpstreamV1, data)

    # Explicitly record feature version
    store.record_feature_version(UpstreamV1)

    # Feature version history should be recorded
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    version_history = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)

    assert len(version_history) > 0
    assert "feature_key" in version_history.columns
    assert "feature_version" in version_history.columns
    assert "recorded_at" in version_history.columns
    assert "snapshot_id" in version_history.columns

    # Should have recorded upstream
    assert "test_migrations_upstream" in version_history["feature_key"].to_list()


def test_registry_rejects_duplicate_keys() -> None:
    """Test that FeatureRegistry raises error on duplicate feature keys."""
    registry = FeatureRegistry()

    # Define first feature
    class Feature1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["duplicate", "key"]),
            deps=None,
            containers=[ContainerSpec(key=ContainerKey(["default"]), code_version=1)],
        ),
        registry=registry,
    ):
        pass

    # Try to define second feature with same key
    with pytest.raises(ValueError, match="already registered"):

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["duplicate", "key"]),  # Same key!
                deps=None,
                containers=[
                    ContainerSpec(key=ContainerKey(["default"]), code_version=2)
                ],
            ),
            registry=registry,
        ):
            pass


def test_detect_uses_latest_version_from_multiple_materializations(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test that detection uses the latest materialization when 10 versions exist."""
    from datetime import datetime, timedelta

    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    # Create store and manually insert 10 version history entries
    store = InMemoryMetadataStore(registry=registry_v2)

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Simulate 10 historical materializations
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    versions = [f"version{i:02d}" for i in range(10)]  # version00 through version09
    # The last one is v1 (latest)
    versions[-1] = UpstreamV1.feature_version()

    # Insert version history in random order to test sorting
    # Mix up the order to ensure we're sorting correctly
    indices = [0, 5, 2, 8, 1, 9, 3, 7, 4, 6]  # Random permutation

    version_history = pl.DataFrame(
        {
            "feature_key": ["test_migrations_upstream"] * 10,
            "feature_version": [versions[i] for i in indices],
            "recorded_at": [base_time + timedelta(hours=i) for i in indices],
            "containers": [["default"]] * 10,
            "snapshot_id": [None] * 10,  # No snapshot_id for historical data
        }
    )

    store._write_metadata_impl(FEATURE_VERSIONS_KEY, version_history)

    # Write some actual data with v1 feature_version
    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
            "feature_version": [
                UpstreamV1.feature_version(),
                UpstreamV1.feature_version(),
            ],
        }
    )
    store._write_metadata_impl(
        FeatureKey(["test_migrations", "upstream"]), upstream_data
    )

    # Now detect changes (code is at v2, latest materialized is v1)
    changes = detect_feature_changes(store)

    assert len(changes) == 1
    assert changes[0].feature_key == FeatureKey(["test_migrations", "upstream"])

    # Should use v1 (latest by timestamp, index 9), not any earlier version
    assert changes[0].from_version == UpstreamV1.feature_version()
    assert changes[0].to_version == UpstreamV2.feature_version()

    # Verify it's not using any of the older versions
    for i in range(9):  # versions 0-8
        assert changes[0].from_version != f"version{i:02d}"

    # Snapshot the detected change
    change_dict = {
        "feature_key": changes[0].feature_key.to_string(),
        "from_version": changes[0].from_version,
        "to_version": changes[0].to_version,
        "change_type": changes[0].change_type,
    }
    assert change_dict == snapshot


def test_migration_result_snapshots(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test migration execution with snapshot of affected features."""
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    migration = Migration(
        version=1,
        id="migration_snapshot_test",
        description="Test for snapshots",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                change_type="code_version",
                reason="Test snapshot",
            )
        ],
    )

    # Apply migration
    result = apply_migration(store_v2, migration)

    # Snapshot the result
    result_dict = {
        "status": result.status,
        "operations_applied": result.operations_applied,
        "operations_failed": result.operations_failed,
        "affected_features": sorted(result.affected_features),
        "has_errors": len(result.errors) > 0,
    }
    assert result_dict == snapshot


def test_feature_versions_snapshot(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test that feature_version hashes are stable for v1 and v2."""
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Snapshot all feature versions
    versions = {
        "upstream_v1": UpstreamV1.feature_version(),
        "upstream_v2": UpstreamV2.feature_version(),
        "downstream_v1": DownstreamV1.feature_version(),
        "downstream_v2": DownstreamV2.feature_version(),
    }

    assert versions == snapshot

    # Verify v1 and v2 differ for upstream (code_version changed)
    assert versions["upstream_v1"] != versions["upstream_v2"]

    # Verify downstream versions are the same (code didn't change, only upstream dependency)
    assert versions["downstream_v1"] == versions["downstream_v2"]


def test_generated_migration_yaml_snapshot(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    tmp_path: Path,
    snapshot: SnapshotAssertion,
) -> None:
    """Test generated YAML migration file structure with snapshot."""
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    # Generate migration
    migration_file = generate_migration(store_v2, output_dir=str(tmp_path))

    assert migration_file is not None

    # Load migration
    migration = Migration.from_yaml(migration_file)

    # Snapshot the migration structure (excluding timestamp-based fields)
    migration_dict = {
        "version": migration.version,
        "operations": [
            {
                "feature_key": op.feature_key,
                "from_version": op.from_,
                "to_version": op.to,
                "change_detected": op.change_type,
            }
            for op in migration.operations
        ],
    }

    assert migration_dict == snapshot


def test_record_all_feature_versions(
    registry_v1: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test recording all features with deterministic snapshot_id."""
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    store = InMemoryMetadataStore(registry=registry_v1)

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Write data for both features
    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [{"default": "h1"}, {"default": "h2"}, {"default": "h3"}],
        }
    )
    store.write_metadata(UpstreamV1, upstream_data)

    downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
    store.calculate_and_write_data_versions(DownstreamV1, downstream_data)

    # Record all features at once
    snapshot_id = store.record_all_feature_versions()

    # Verify snapshot_id is deterministic (8-char hash, no timestamp)
    assert len(snapshot_id) == 8
    assert all(c in "0123456789abcdef" for c in snapshot_id)

    # Verify both features were recorded
    version_history = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)

    assert len(version_history) == 2  # Both features

    # Both should have the same snapshot_id
    snapshot_ids = version_history["snapshot_id"].unique().to_list()
    assert len(snapshot_ids) == 1
    assert snapshot_ids[0] == snapshot_id

    # Verify correct features recorded
    feature_keys = set(version_history["feature_key"].to_list())
    assert feature_keys == {"test_migrations_upstream", "test_migrations_downstream"}

    # Verify feature_versions match
    upstream_row = version_history.filter(
        pl.col("feature_key") == "test_migrations_upstream"
    )
    downstream_row = version_history.filter(
        pl.col("feature_key") == "test_migrations_downstream"
    )

    assert upstream_row["feature_version"][0] == UpstreamV1.feature_version()
    assert downstream_row["feature_version"][0] == DownstreamV1.feature_version()

    # Snapshot the snapshot_id (should be deterministic)
    assert snapshot_id == snapshot


def test_record_all_feature_versions_is_idempotent(
    registry_v1: FeatureRegistry,
) -> None:
    """Test that snapshot_id is deterministic and recording is idempotent."""
    store = InMemoryMetadataStore(registry=registry_v1)

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Write data
    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )
    store.write_metadata(UpstreamV1, upstream_data)

    downstream_data = pl.DataFrame({"sample_id": [1, 2]})
    store.calculate_and_write_data_versions(DownstreamV1, downstream_data)

    # Record twice
    import time

    snapshot_id1 = store.record_all_feature_versions()
    time.sleep(0.01)  # Small delay
    snapshot_id2 = store.record_all_feature_versions()

    # snapshot_id should be identical (deterministic, no timestamp)
    assert snapshot_id1 == snapshot_id2

    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    version_history = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)

    # Should only have 2 records (idempotent - doesn't re-record same version+snapshot)
    # The idempotency check prevents duplicate records
    assert len(version_history) == 2

    # Both should have the same snapshot_id
    assert all(sid == snapshot_id1 for sid in version_history["snapshot_id"].to_list())


def test_snapshot_workflow_without_migrations(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test standard workflow: compute v2 data, record snapshot (no migration needed)."""
    # Step 1: Materialize v1 features and record
    store_v1 = InMemoryMetadataStore(registry=registry_v1)
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    upstream_data_v1 = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [{"default": "h1"}, {"default": "h2"}, {"default": "h3"}],
        }
    )
    store_v1.write_metadata(UpstreamV1, upstream_data_v1)

    downstream_data_v1 = pl.DataFrame({"sample_id": [1, 2, 3]})
    store_v1.calculate_and_write_data_versions(DownstreamV1, downstream_data_v1)

    # Record v1 graph snapshot
    snapshot_id_v1 = store_v1.record_all_feature_versions()

    # Step 2: Code changes (v1 -> v2), migrate store to v2 registry
    store_v2 = migrate_store_to_registry(store_v1, registry_v2)
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Step 3: RECOMPUTE v2 data (no migration needed!)
    # This is the standard workflow when algorithm actually changed
    upstream_data_v2 = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [{"default": "h4"}, {"default": "h5"}, {"default": "h6"}],
        }
    )
    store_v2.write_metadata(UpstreamV2, upstream_data_v2)

    downstream_data_v2 = pl.DataFrame({"sample_id": [1, 2, 3]})
    store_v2.calculate_and_write_data_versions(DownstreamV2, downstream_data_v2)

    # Record v2 graph snapshot
    snapshot_id_v2 = store_v2.record_all_feature_versions()

    # Verify both snapshots recorded
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    version_history = store_v2.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)

    # Should have 3 records:
    # - upstream v1 (feature_version changed)
    # - downstream v1 (same feature_version as v2, so only recorded once)
    # - upstream v2 (new feature_version)
    # Downstream v2 has same feature_version as v1 (code didn't change), so idempotency skips it
    assert len(version_history) == 3

    # Verify snapshot_ids are different (different graph states)
    snapshot_ids = version_history["snapshot_id"].unique().to_list()
    assert len(snapshot_ids) == 2
    assert snapshot_id_v1 in snapshot_ids
    assert snapshot_id_v2 in snapshot_ids
    assert snapshot_id_v1 != snapshot_id_v2

    # Verify downstream only recorded once (same feature_version for v1 and v2)
    downstream_records = version_history.filter(
        pl.col("feature_key") == "test_migrations_downstream"
    )
    assert (
        len(downstream_records) == 1
    )  # Only one record despite two snapshot recordings

    # Verify we can read v2 data (current)
    current_upstream = store_v2.read_metadata(UpstreamV2, current_only=True)
    assert len(current_upstream) == 3
    assert set(current_upstream["sample_id"].to_list()) == {1, 2, 3}

    # Verify v1 data still exists (immutable)
    v1_upstream = store_v2.read_metadata(
        UpstreamV2,
        current_only=False,
        filters=pl.col("feature_version") == UpstreamV1.feature_version(),
    )
    assert len(v1_upstream) == 3

    # Snapshot the snapshot_ids (should be deterministic)
    assert {"v1": snapshot_id_v1, "v2": snapshot_id_v2} == snapshot


def test_migrations_preserve_immutability(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test that migrations preserve old data (immutable)."""
    # Setup v1 data
    store_v1 = InMemoryMetadataStore(registry=registry_v1)
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "path": [
                "/data/1.mp4",
                "/data/2.mp4",
                "/data/3.mp4",
            ],  # User metadata column
            "data_version": [{"default": "h1"}, {"default": "h2"}, {"default": "h3"}],
        }
    )
    store_v1.write_metadata(UpstreamV1, upstream_data)
    store_v1.record_feature_version(UpstreamV1)

    # Get original data for comparison
    original_data = store_v1.read_metadata(UpstreamV1, current_only=False)
    original_data_versions = original_data["data_version"].to_list()

    # Migrate to v2
    store_v2 = migrate_store_to_registry(store_v1, registry_v2)
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Apply migration
    migration = Migration(
        version=1,
        id="test_immutability",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                change_type="code_version",
                reason="Test",
            )
        ],
    )

    apply_migration(store_v2, migration)

    # Verify old data still exists unchanged
    all_data = store_v2.read_metadata(UpstreamV2, current_only=False)

    # Should have both v1 and v2 rows
    assert len(all_data) == 6  # 3 original + 3 migrated

    # Old rows should still exist with old feature_version
    v1_rows = all_data.filter(pl.col("feature_version") == UpstreamV1.feature_version())
    assert len(v1_rows) == 3

    # Old rows should have same data as before migration
    assert v1_rows["data_version"].to_list() == original_data_versions
    assert v1_rows["path"].to_list() == ["/data/1.mp4", "/data/2.mp4", "/data/3.mp4"]

    # New rows should exist with new feature_version
    v2_rows = all_data.filter(pl.col("feature_version") == UpstreamV2.feature_version())
    assert len(v2_rows) == 3

    # New rows should have same sample_ids and user columns (path)
    assert set(v2_rows["sample_id"].to_list()) == {1, 2, 3}
    assert v2_rows["path"].to_list() == ["/data/1.mp4", "/data/2.mp4", "/data/3.mp4"]

    # But different data_versions (recalculated)
    v2_data_versions = v2_rows["data_version"].to_list()
    assert v2_data_versions != original_data_versions  # Should be different!


def test_migration_vs_recompute_comparison(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Compare migration (no recompute) vs standard workflow (with recompute).

    This test demonstrates when to use migrations vs when to just recompute.
    """
    # Setup: v1 data exists
    store_v1 = InMemoryMetadataStore(registry=registry_v1)
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [{"default": "h1"}, {"default": "h2"}, {"default": "h3"}],
        }
    )
    store_v1.write_metadata(UpstreamV1, upstream_data)

    downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
    store_v1.calculate_and_write_data_versions(DownstreamV1, downstream_data)
    store_v1.record_all_feature_versions()

    # Get initial downstream data_versions
    initial_downstream_data_versions = store_v1.read_metadata(DownstreamV1)[
        "data_version"
    ].to_list()

    # Scenario A: Migration (skip recompute, just update hashes)
    store_migration = migrate_store_to_registry(store_v1, registry_v2)
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    migration = Migration(
        version=1,
        id="migration_test_comparison",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            FeatureVersionMigration(
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                change_type="code_version",
                reason="Test",
            )
        ],
    )

    result = apply_migration(store_migration, migration)
    assert result.status == "completed"

    # Migration: downstream data_versions CHANGED (recalculated hashes based on new upstream)
    migrated_downstream = store_migration.read_metadata(DownstreamV2, current_only=True)
    migrated_data_versions = migrated_downstream["data_version"].to_list()

    # Data versions should be different because upstream feature_version changed
    # Even though underlying data is the same, the hash includes feature_version
    assert migrated_data_versions != initial_downstream_data_versions

    # Scenario B: Standard workflow (recompute)
    # Would compute new data and write, data_versions would also change
    # Both approaches update data_versions, but migration skips expensive computation

    # Snapshot comparison
    comparison = {
        "migration_updated_hashes": True,
        "migration_skipped_computation": True,
        "initial_data_versions_count": len(initial_downstream_data_versions),
        "migrated_data_versions_count": len(migrated_data_versions),
    }
    assert comparison == snapshot
