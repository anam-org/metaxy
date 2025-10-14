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
    DataVersionReconciliation,
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
    """Create new store with target registry context but source store's data.

    Helper for testing migrations - simulates code changing while data stays the same.
    """
    # Create new store and copy data - will be used within target_registry context
    new_store = InMemoryMetadataStore()
    new_store._storage = source_store._storage.copy()
    return new_store


@pytest.fixture
def registry_v1():
    """Registry with v1 features."""
    registry = FeatureRegistry()

    with registry.use():
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
        ):
            pass

        yield registry


@pytest.fixture
def registry_v2():
    """Registry with v2 features (upstream code_version changed)."""
    registry = FeatureRegistry()

    with registry.use():
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
        ):
            pass

        yield registry


@pytest.fixture
def store_with_v1_data(registry_v1: FeatureRegistry) -> InMemoryMetadataStore:
    """Store with v1 upstream and downstream data."""
    store = InMemoryMetadataStore()

    with store:
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
        store.record_feature_graph_snapshot(UpstreamV1)

        # Write downstream using new API
        downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store.resolve_update(DownstreamV1, sample_df=downstream_data)
        if len(diff_result.added) > 0:
            store.write_metadata(DownstreamV1, diff_result.added)
        # Explicitly record feature version
        store.record_feature_graph_snapshot(DownstreamV1)

    return store


# Tests


def test_detect_no_changes(
    store_with_v1_data: InMemoryMetadataStore,
) -> None:
    """Test detection when no features changed (registry matches data)."""
    with store_with_v1_data:
        changes = detect_feature_changes(store_with_v1_data)

    # Should detect no changes (v1 registry, v1 data)
    assert len(changes) == 0


def test_detect_single_change(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test detection of single feature change."""
    # Migrate store to v2 registry (simulates code change)
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    # Get features for version comparison
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Detect changes within v2 registry context
    with registry_v2.use(), store_v2:
        operations = detect_feature_changes(store_v2)

        assert len(operations) == 1
        assert FeatureKey(operations[0].feature_key) == FeatureKey(
            ["test_migrations", "upstream"]
        )
        assert operations[0].from_ == UpstreamV1.feature_version()
        assert operations[0].to == UpstreamV2.feature_version()
        assert operations[0].id.startswith("reconcile_")


def test_generate_migration_no_changes(
    store_with_v1_data: InMemoryMetadataStore,
    tmp_path: Path,
) -> None:
    """Test generation when no changes detected."""
    with store_with_v1_data:
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
    with store_v2:
        migration_file = generate_migration(store_v2, output_dir=str(tmp_path))

        assert migration_file is not None
        assert Path(migration_file).exists()

        # Load and verify
        migration = Migration.from_yaml(migration_file)

        assert migration.version == 1
        # Generator now creates explicit operations for root + downstream
        assert len(migration.operations) == 2  # 1 root + 1 downstream

        # Parse operations
        ops = migration.get_operations()

        # First operation is root change
        assert ops[0].feature_key == ["test_migrations", "upstream"]
        assert ops[0].from_ == UpstreamV1.feature_version()  # type: ignore[attr-defined]
        assert ops[0].to == UpstreamV2.feature_version()  # type: ignore[attr-defined]

        # Second operation is downstream reconciliation
        assert ops[1].feature_key == ["test_migrations", "downstream"]
        assert "Reconcile data_versions" in ops[1].reason


def test_apply_migration_rejects_root_features(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test that DataVersionReconciliation rejects root features (no upstream).

    Root features have user-defined data_versions that cannot be automatically
    reconciled. User must re-run their computation pipeline.
    """
    # Migrate to v2 registry
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    # Get features
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    # Create migration attempting to reconcile a root feature
    migration = Migration(
        version=1,
        id="migration_test_recalc",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            DataVersionReconciliation(
                id="reconcile_upstream",
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                reason="Test",
            ).model_dump(by_alias=True)
        ],
    )

    # Apply migration - should fail with clear error
    with store_v2:
        result = apply_migration(store_v2, migration)

        # Should fail because upstream is a root feature
        assert result.status == "failed"
        assert "reconcile_upstream" in result.errors
        assert (
            "Root features have user-defined data_versions"
            in result.errors["reconcile_upstream"]
        )


def test_apply_migration_idempotent(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test that migrations are idempotent."""
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    registry_v2.features_by_key[FeatureKey(["test_migrations", "downstream"])]

    # Create migration for downstream feature (has upstream)
    migration = Migration(
        version=1,
        id="migration_test_idempotent",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            DataVersionReconciliation(
                id="reconcile_downstream",
                feature_key=["test_migrations", "downstream"],
                from_=DownstreamV1.feature_version(),
                to=DownstreamV1.feature_version(),  # Same version, just reconciling
                reason="Test",
            ).model_dump(by_alias=True)
        ],
    )

    # First application
    with store_v2:
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

    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Create migration for downstream feature
    migration = Migration(
        version=1,
        id="migration_test_dryrun",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            DataVersionReconciliation(
                id="reconcile_downstream",
                feature_key=["test_migrations", "downstream"],
                from_=DownstreamV1.feature_version(),
                to=DownstreamV1.feature_version(),  # Same version
                reason="Test",
            ).model_dump(by_alias=True)
        ],
    )

    # Dry-run
    with store_v2:
        result = apply_migration(store_v2, migration, dry_run=True)

        assert result.status == "skipped"
        assert len(result.affected_features) > 0

        # Verify data unchanged - should still only have v1 feature_version
        all_data = store_v2.read_metadata(
            DownstreamV2,  # Reading through v2 registry
            current_only=False,  # Get all versions
        )
        # All rows should have v1 feature_version (not v2 - because it's dry-run)
        assert all(
            fv == DownstreamV1.feature_version()
            for fv in all_data["feature_version"].to_list()
        )


def test_apply_migration_propagates_downstream(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test that downstream reconciliation works when upstream changes.

    Scenario:
    1. User manually updates upstream (root feature) with new data
    2. Migration reconciles downstream data_versions based on new upstream
    """
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    with store_v2:
        # Get initial downstream data_versions
        initial_downstream = store_v2.read_metadata(DownstreamV2, current_only=False)
        initial_data_versions = initial_downstream["data_version"].to_list()

        # Simulate user manually updating upstream (root feature) with new data
        # This is what user would do when root feature changes - re-run computation
        import polars as pl

        new_upstream_data = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"default": "new_h1"},
                    {"default": "new_h2"},
                    {"default": "new_h3"},
                ],
            }
        )
        store_v2.write_metadata(UpstreamV2, new_upstream_data)

        # Create migration with only downstream operation
        # (upstream was manually updated by user)
        migration = Migration(
            version=1,
            id="migration_test_propagate",
            description="Test",
            created_at=datetime(2025, 1, 1, 0, 0, 0),
            operations=[
                # Only downstream reconciliation (upstream already updated manually)
                DataVersionReconciliation(
                    id="reconcile_downstream",
                    feature_key=["test_migrations", "downstream"],
                    from_=DownstreamV1.feature_version(),
                    to=DownstreamV1.feature_version(),  # Same version, just reconciling data_versions
                    reason="Reconcile data_versions due to upstream change",
                ).model_dump(by_alias=True),
            ],
        )

        # Apply migration
        result = apply_migration(store_v2, migration)

        # Debug output
        if result.status != "completed":
            print(f"Migration failed with errors: {result.errors}")
            print(f"Summary:\n{result.summary()}")

        assert result.status == "completed"
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
            DataVersionReconciliation(
                id="test_op_id",
                feature_key=["my", "feature"],
                from_="abc12345",
                to="def67890",
                reason="Updated algorithm",
            ).model_dump(by_alias=True)
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

    # Parse operations from dicts
    parsed_ops = loaded.get_operations()
    assert len(parsed_ops) == 1
    assert parsed_ops[0].feature_key == ["my", "feature"]
    assert parsed_ops[0].from_ == "abc12345"  # type: ignore[attr-defined]
    assert parsed_ops[0].to == "def67890"  # type: ignore[attr-defined]


def test_feature_version_in_metadata(registry_v1: FeatureRegistry) -> None:
    """Test that feature_version column is automatically added."""
    store = InMemoryMetadataStore()
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

    with store:
        store.write_metadata(UpstreamV1, data)

        # Read back
        result = store.read_metadata(UpstreamV1, current_only=False)

        # Should have feature_version after write
        assert "feature_version" in result.columns
        assert all(
            fv == UpstreamV1.feature_version()
            for fv in result["feature_version"].to_list()
        )


def test_current_only_filtering(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
) -> None:
    """Test current_only parameter filters by feature_version."""
    # Write v1 data with v1 registry
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    data_v1 = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )
    with store_v1:
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
    with store_v2:
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
        v1_rows = all_data.filter(
            pl.col("feature_version") == UpstreamV1.feature_version()
        )
        v2_rows = all_data.filter(
            pl.col("feature_version") == UpstreamV2.feature_version()
        )

        assert len(v1_rows) == 2
        assert len(v2_rows) == 2


def test_system_tables_created(registry_v1: FeatureRegistry) -> None:
    """Test that system tables are created when explicitly recording."""
    store = InMemoryMetadataStore()
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
    with store:
        store.write_metadata(UpstreamV1, data)

        # Explicitly record feature version
        store.record_feature_graph_snapshot(UpstreamV1)

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
    store = InMemoryMetadataStore()

    UpstreamV1 = registry_v1.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]

    with store:
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
        assert changes[0].from_ == UpstreamV1.feature_version()
        assert changes[0].to == UpstreamV2.feature_version()

        # Verify it's not using any of the older versions
        for i in range(9):  # versions 0-8
            assert changes[0].from_ != f"version{i:02d}"

        # Snapshot the detected change
        change_dict = {
            "feature_key": FeatureKey(changes[0].feature_key).to_string(),
            "from_version": changes[0].from_,
            "to_version": changes[0].to,
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
            DataVersionReconciliation(
                id="test_op_id",
                feature_key=["test_migrations", "upstream"],
                from_=UpstreamV1.feature_version(),
                to=UpstreamV2.feature_version(),
                reason="Test snapshot",
            ).model_dump(by_alias=True)
        ],
    )

    # Apply migration
    with store_v2:
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
    with store_v2:
        migration_file = generate_migration(store_v2, output_dir=str(tmp_path))

        assert migration_file is not None

        # Load migration
        migration = Migration.from_yaml(migration_file)

        # Parse operations for snapshot
        ops = migration.get_operations()

        # Snapshot the migration structure (excluding timestamp-based fields)
        migration_dict = {
            "version": migration.version,
            "operations": [
                {
                    "feature_key": op.feature_key,
                    "from_version": op.from_,  # type: ignore[attr-defined]
                    "to_version": op.to,  # type: ignore[attr-defined]
                }
                for op in ops
            ],
        }

        assert migration_dict == snapshot


def test_serialize_feature_graph(
    registry_v1: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test recording all features with deterministic snapshot_id."""
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    store = InMemoryMetadataStore()

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
    with store:
        store.write_metadata(UpstreamV1, upstream_data)

        downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store.resolve_update(DownstreamV1, sample_df=downstream_data)
        if len(diff_result.added) > 0:
            store.write_metadata(DownstreamV1, diff_result.added)

        # Record all features at once
        snapshot_id = store.serialize_feature_graph()

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
        assert feature_keys == {
            "test_migrations_upstream",
            "test_migrations_downstream",
        }

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


def test_serialize_feature_graph_is_idempotent(
    registry_v1: FeatureRegistry,
) -> None:
    """Test that snapshot_id is deterministic and recording is idempotent."""
    store = InMemoryMetadataStore()

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
    with store:
        store.write_metadata(UpstreamV1, upstream_data)

        downstream_data = pl.DataFrame({"sample_id": [1, 2]})
        diff_result = store.resolve_update(DownstreamV1, sample_df=downstream_data)
        if len(diff_result.added) > 0:
            store.write_metadata(DownstreamV1, diff_result.added)

        # Record twice
        import time

        snapshot_id1 = store.serialize_feature_graph()
        time.sleep(0.01)  # Small delay
        snapshot_id2 = store.serialize_feature_graph()

        # snapshot_id should be identical (deterministic, no timestamp)
        assert snapshot_id1 == snapshot_id2

        from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

        version_history = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)

        # Should only have 2 records (idempotent - doesn't re-record same version+snapshot)
        # The idempotency check prevents duplicate records
        assert len(version_history) == 2

        # Both should have the same snapshot_id
        assert all(
            sid == snapshot_id1 for sid in version_history["snapshot_id"].to_list()
        )


def test_snapshot_workflow_without_migrations(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test standard workflow: compute v2 data, record snapshot (no migration needed)."""
    # Step 1: Materialize v1 features and record
    store_v1 = InMemoryMetadataStore()
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
    with registry_v1.use(), store_v1:
        store_v1.write_metadata(UpstreamV1, upstream_data_v1)

        downstream_data_v1 = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store_v1.resolve_update(
            DownstreamV1, sample_df=downstream_data_v1
        )
        if len(diff_result.added) > 0:
            store_v1.write_metadata(DownstreamV1, diff_result.added)

        # Record v1 graph snapshot
        snapshot_id_v1 = store_v1.serialize_feature_graph()

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
    with registry_v2.use(), store_v2:
        store_v2.write_metadata(UpstreamV2, upstream_data_v2)

        downstream_data_v2 = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store_v2.resolve_update(
            DownstreamV2, sample_df=downstream_data_v2
        )
        if len(diff_result.added) > 0:
            store_v2.write_metadata(DownstreamV2, diff_result.added)

        # Record v2 graph snapshot
        snapshot_id_v2 = store_v2.serialize_feature_graph()

        # Verify both snapshots recorded
        from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

        version_history = store_v2.read_metadata(
            FEATURE_VERSIONS_KEY, current_only=False
        )

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
    store_v1 = InMemoryMetadataStore()
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
    downstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "path": ["/data/1.mp4", "/data/2.mp4", "/data/3.mp4"],  # User metadata
            "data_version": [{"default": "d1"}, {"default": "d2"}, {"default": "d3"}],
        }
    )

    with store_v1:
        store_v1.write_metadata(UpstreamV1, upstream_data)
        store_v1.write_metadata(DownstreamV1, downstream_data)
        store_v1.record_feature_graph_snapshot(UpstreamV1)
        store_v1.record_feature_graph_snapshot(DownstreamV1)

        # Get original downstream data for comparison
        original_data = store_v1.read_metadata(DownstreamV1, current_only=False)
        original_data_versions = original_data["data_version"].to_list()

    # Migrate to v2
    store_v2 = migrate_store_to_registry(store_v1, registry_v2)
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Apply migration to downstream (has upstream dependencies)
    migration = Migration(
        version=1,
        id="test_immutability",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        operations=[
            DataVersionReconciliation(
                id="reconcile_downstream",
                feature_key=["test_migrations", "downstream"],
                from_=DownstreamV1.feature_version(),
                to=DownstreamV1.feature_version(),  # Same version, reconciling data_versions
                reason="Test",
            ).model_dump(by_alias=True)
        ],
    )

    with store_v2:
        apply_migration(store_v2, migration)

        # Verify old data still exists unchanged (immutability)
        all_data = store_v2.read_metadata(DownstreamV2, current_only=False)

        # Should have both original and migrated rows (6 total)
        assert len(all_data) == 6  # 3 original + 3 migrated

        # All rows have same feature_version (from_ == to in reconciliation)
        assert all(all_data["feature_version"] == DownstreamV1.feature_version())

        # But two different sets of data_versions (old and recalculated)
        all_data_versions = all_data["data_version"].to_list()
        unique_data_versions = set(str(dv) for dv in all_data_versions)

        # Should have more than 3 unique data_versions (old + new)
        assert len(unique_data_versions) > 3

        # Old data_versions should still exist (immutability check)
        original_dv_set = set(str(dv) for dv in original_data_versions)
        assert original_dv_set.issubset(unique_data_versions)

        # User metadata preserved across both sets
        assert set(all_data["path"].to_list()) == {
            "/data/1.mp4",
            "/data/2.mp4",
            "/data/3.mp4",
        }
        assert set(all_data["sample_id"].to_list()) == {1, 2, 3}


def test_metadata_backfill_operation() -> None:
    """Test MetadataBackfill operation with custom user logic."""
    from metaxy.migrations.ops import MetadataBackfill

    # Create test backfill subclass for a feature WITH upstream dependencies
    class TestBackfill(MetadataBackfill):
        type: str = "tests.test_migrations.TestBackfill"
        test_data: list[dict]

        def execute(self, store, *, dry_run=False):
            import polars as pl

            from metaxy.models.feature import FeatureRegistry
            from metaxy.models.types import FeatureKey

            # Load external data (from test_data)
            external_df = pl.DataFrame(self.test_data)

            if dry_run:
                return len(external_df)

            # Get feature
            feature_key = FeatureKey(self.feature_key)
            registry = FeatureRegistry.get_active()
            feature_cls = registry.features_by_key[feature_key]

            # For features with upstream: use resolve_update
            # For root features: user provides data_version directly or computes it
            plan = registry.get_feature_plan(feature_key)
            has_upstream = plan.deps is not None and len(plan.deps) > 0

            if has_upstream:
                # Calculate data_versions via resolve_update
                diff = store.resolve_update(feature_cls, sample_df=external_df)
                if len(diff.added) > 0:
                    to_write = external_df.join(
                        diff.added.select(["sample_id", "data_version"]), on="sample_id"
                    )
                    store.write_metadata(feature_cls, to_write)
                    return len(to_write)
            else:
                # Root feature: user provides complete data (including data_version)
                # For this test, use dummy data_versions
                to_write = external_df.with_columns(
                    pl.lit({"default": "user_defined_hash"}).alias("data_version")
                )
                store.write_metadata(feature_cls, to_write)
                return len(to_write)

            return 0

    # Create registry with root feature
    registry = FeatureRegistry()
    with registry.use():

        class RootFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["backfill", "test"]),
                deps=None,
                containers=[
                    ContainerSpec(key=ContainerKey(["default"]), code_version=1)
                ],
            ),
        ):
            pass

        # Create store and backfill
        store = InMemoryMetadataStore()
        with store:
            # Create backfill operation
            backfill = TestBackfill(
                id="test_backfill",
                feature_key=["backfill", "test"],
                reason="Test backfill",
                test_data=[
                    {"sample_id": "s1", "path": "/data/1.txt", "size": 100},
                    {"sample_id": "s2", "path": "/data/2.txt", "size": 200},
                ],
            )

            # Test dry run
            rows = backfill.execute(store, dry_run=True)
            assert rows == 2

            # Execute for real
            rows = backfill.execute(store)
            assert rows == 2

            # Verify data was written
            result = store.read_metadata(RootFeature)
            assert len(result) == 2
            assert set(result["sample_id"].to_list()) == {"s1", "s2"}
            assert set(result["path"].to_list()) == {"/data/1.txt", "/data/2.txt"}
            assert set(result["size"].to_list()) == {100, 200}
            assert "data_version" in result.columns


def test_migration_chaining_validates_parent() -> None:
    """Test that migrations validate parent completion before applying."""
    from metaxy.migrations import Migration, apply_migration

    registry = FeatureRegistry()
    with registry.use():

        class UpFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["up"]),
                deps=None,
                containers=[ContainerSpec(key=ContainerKey(["d"]), code_version=1)],
            ),
        ):
            pass

        class DownFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["down"]),
                deps=[FeatureDep(key=FeatureKey(["up"]))],
                containers=[
                    ContainerSpec(
                        key=ContainerKey(["d"]),
                        code_version=1,
                        deps=[
                            ContainerDep(
                                feature_key=FeatureKey(["up"]),
                                containers=[ContainerKey(["d"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

        store = InMemoryMetadataStore()
        with store:
            # Write initial data
            store.write_metadata(
                UpFeature,
                pl.DataFrame({"sample_id": [1], "data_version": [{"d": "h1"}]}),
            )
            store.write_metadata(
                DownFeature,
                pl.DataFrame({"sample_id": [1], "data_version": [{"d": "d1"}]}),
            )

            # Create migration 1 (no parent) - empty operations but registers in system
            migration1 = Migration(
                version=1,
                id="migration_001",
                parent_migration_id=None,
                description="First migration",
                created_at=datetime(2025, 1, 1),
                operations=[],
            )

            # Create migration 2 (parent = migration1)
            migration2 = Migration(
                version=1,
                id="migration_002",
                parent_migration_id="migration_001",
                description="Second migration",
                created_at=datetime(2025, 1, 2),
                operations=[],
            )

            # Try to apply migration2 before migration1 - should fail
            with pytest.raises(ValueError, match="Parent migration.*must be completed"):
                apply_migration(store, migration2)

            # Apply migration1 first (even with no operations, registers as complete)
            result1 = apply_migration(store, migration1)
            # Empty operations = completed immediately
            assert result1.status in ("completed", "skipped")

            # Now migration2 should work
            result2 = apply_migration(store, migration2)
            assert result2.status in ("completed", "skipped")


def test_generator_sets_parent_migration_id(
    store_with_v1_data: InMemoryMetadataStore,
    registry_v2: FeatureRegistry,
    tmp_path: Path,
) -> None:
    """Test that generator automatically sets parent_migration_id."""
    store_v2 = migrate_store_to_registry(store_with_v1_data, registry_v2)

    with store_v2:
        # Generate first migration
        file1 = generate_migration(store_v2, output_dir=str(tmp_path))
        assert file1 is not None

        migration1 = Migration.from_yaml(file1)
        assert migration1.parent_migration_id is None  # First migration has no parent

        # Apply it
        apply_migration(store_v2, migration1)

        # Generate second migration (should reference first)
        file2 = generate_migration(store_v2, output_dir=str(tmp_path))

        if file2 is not None:
            # If there are more changes, second migration should reference first
            migration2 = Migration.from_yaml(file2)
            assert migration2.parent_migration_id == migration1.id


def test_migration_vs_recompute_comparison(
    registry_v1: FeatureRegistry,
    registry_v2: FeatureRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Compare migration (no recompute) vs standard workflow (with recompute).

    This test demonstrates when to use migrations vs when to just recompute.
    """
    # Setup: v1 data exists
    store_v1 = InMemoryMetadataStore()
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
    with store_v1:
        store_v1.write_metadata(UpstreamV1, upstream_data)

        downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store_v1.resolve_update(DownstreamV1, sample_df=downstream_data)
        if len(diff_result.added) > 0:
            store_v1.write_metadata(DownstreamV1, diff_result.added)
        store_v1.serialize_feature_graph()

        # Get initial downstream data_versions
        initial_downstream_data_versions = store_v1.read_metadata(DownstreamV1)[
            "data_version"
        ].to_list()

    # Scenario A: Migration (user manually updates upstream, then reconciles downstream)
    store_migration = migrate_store_to_registry(store_v1, registry_v2)
    UpstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "upstream"])
    ]
    DownstreamV2 = registry_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    with store_migration:
        # User manually writes new upstream data (root feature changed)
        new_upstream_data = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"default": "new_h1"},
                    {"default": "new_h2"},
                    {"default": "new_h3"},
                ],
            }
        )
        store_migration.write_metadata(UpstreamV2, new_upstream_data)

        # Now reconcile downstream to reflect new upstream
        migration = Migration(
            version=1,
            id="migration_test_comparison",
            description="Test",
            created_at=datetime(2025, 1, 1, 0, 0, 0),
            operations=[
                DataVersionReconciliation(
                    id="reconcile_downstream",
                    feature_key=["test_migrations", "downstream"],
                    from_=DownstreamV1.feature_version(),
                    to=DownstreamV1.feature_version(),  # Same version, reconciling
                    reason="Test",
                ).model_dump(by_alias=True)
            ],
        )

        result = apply_migration(store_migration, migration)
        assert result.status == "completed"

        # Migration: downstream data_versions CHANGED (recalculated based on new upstream)
        migrated_downstream = store_migration.read_metadata(
            DownstreamV2, current_only=True
        )
        migrated_data_versions = migrated_downstream["data_version"].to_list()

        # Data versions should be different because upstream data changed
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
