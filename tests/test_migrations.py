"""Tests for migration system."""

import importlib
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
    InMemoryMetadataStore,
)
from metaxy.migrations import (
    DataVersionReconciliation,
    Migration,
    apply_migration,
    detect_feature_changes,
    generate_migration,
)
from metaxy.models.feature import FeatureGraph


class TempFeatureModule:
    """Helper to create temporary Python modules with feature definitions.

    This allows features to be importable by historical graph reconstruction.
    The same import path (e.g., 'temp_features.Upstream') can be used across
    different feature versions by overwriting the module file.
    """

    def __init__(self, module_name: str = "temp_test_features"):
        self.temp_dir = tempfile.mkdtemp(prefix="metaxy_test_")
        self.module_name = module_name
        self.module_path = Path(self.temp_dir) / f"{module_name}.py"

        # Add to sys.path so module can be imported
        sys.path.insert(0, self.temp_dir)

    def write_features(self, feature_specs: dict[str, FeatureSpec]):
        """Write feature classes to the module file.

        Args:
            feature_specs: Dict mapping class names to FeatureSpec objects
        """
        code_lines = [
            "# Auto-generated test feature module",
            "from metaxy import Feature, FeatureSpec, FieldSpec, FieldKey, FeatureDep, FeatureKey, FieldDep, SpecialFieldDep",
            "from metaxy.models.feature import FeatureGraph",
            "",
            "# Use a dedicated graph for this temp module",
            "_graph = FeatureGraph()",
            "",
        ]

        for class_name, spec in feature_specs.items():
            # Generate the spec definition
            spec_dict = spec.model_dump(mode="python")
            spec_repr = self._generate_spec_repr(spec_dict)

            code_lines.extend(
                [
                    f"# Define {class_name} in the temp graph context",
                    "with _graph.use():",
                    f"    class {class_name}(",
                    "        Feature,",
                    f"        spec={spec_repr}",
                    "    ):",
                    "        pass",
                    "",
                ]
            )

        # Write the file
        self.module_path.write_text("\n".join(code_lines))

        # Reload module if it was already imported
        if self.module_name in sys.modules:
            importlib.reload(sys.modules[self.module_name])

    def _generate_spec_repr(self, spec_dict: dict) -> str:
        """Generate FeatureSpec constructor call from dict."""
        # This is a simple representation - could be made more robust
        parts = []

        # key
        key = spec_dict["key"]
        parts.append(f"key=FeatureKey({key!r})")

        # deps
        deps = spec_dict.get("deps")
        if deps is None:
            parts.append("deps=None")
        else:
            deps_repr = [f"FeatureDep(key=FeatureKey({d['key']!r}))" for d in deps]
            parts.append(f"deps=[{', '.join(deps_repr)}]")

        # fields
        fields = spec_dict.get("fields", [])
        if fields:
            field_reprs = []
            for c in fields:
                c_parts = [
                    f"key=FieldKey({c['key']!r})",
                    f"code_version={c['code_version']}",
                ]

                # Handle deps
                deps_val = c.get("deps")
                if deps_val == "__METAXY_ALL_DEP__":
                    c_parts.append("deps=SpecialFieldDep.ALL")
                elif isinstance(deps_val, list) and deps_val:
                    # Field deps (list of FieldDep)
                    cdeps: list[str] = []  # type: ignore[misc]
                    for cd in deps_val:
                        fields_val = cd.get("fields")
                        if fields_val == "__METAXY_ALL_DEP__":
                            cdeps.append(  # type: ignore[arg-type]
                                f"FieldDep(feature_key=FeatureKey({cd['feature_key']!r}), fields=SpecialFieldDep.ALL)"
                            )
                        else:
                            # Build list of FieldKey objects
                            field_keys = [f"FieldKey({k!r})" for k in fields_val]
                            cdeps.append(
                                f"FieldDep(feature_key=FeatureKey({cd['feature_key']!r}), fields=[{', '.join(field_keys)}])"
                            )
                    c_parts.append(f"deps=[{', '.join(cdeps)}]")

                field_reprs.append(f"FieldSpec({', '.join(c_parts)})")  # type: ignore[arg-type]

            parts.append(f"fields=[{', '.join(field_reprs)}]")

        return f"FeatureSpec({', '.join(parts)})"

    def get_graph(self) -> FeatureGraph:
        """Get the graph from the temp module.

        Creates a new graph and re-registers the features from the module.
        This ensures we get fresh features after module reloading.
        """
        # Force reload to get latest class definitions
        if self.module_name in sys.modules:
            module = importlib.reload(sys.modules[self.module_name])
        else:
            module = importlib.import_module(self.module_name)

        # Create fresh graph and register features from module
        fresh_graph = FeatureGraph()

        # Find and register all Feature subclasses in the module
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Feature)
                and obj is not Feature
                and hasattr(obj, "spec")
            ):
                fresh_graph.add_feature(obj)

        return fresh_graph

    def cleanup(self):
        """Remove temp directory and module from sys.path.

        NOTE: Don't call this until the test session is completely done,
        as historical graph loading may need to import from these modules.
        """
        if self.temp_dir in sys.path:
            sys.path.remove(self.temp_dir)

        # Remove from sys.modules
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]

        # Delete temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


def get_latest_snapshot_id(store: InMemoryMetadataStore) -> str:
    """Get the latest snapshot_id from store's feature_versions table."""
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    with store:
        versions = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
        if len(versions) == 0:
            raise ValueError("No snapshots recorded in store")
        latest = versions.sort("recorded_at", descending=True).head(1)
        return latest["snapshot_id"][0]


# Additional test feature classes for specific tests (must be importable)
_temp_graph_chaining = FeatureGraph()

with _temp_graph_chaining.use():

    class UpFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["up"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["d"]), code_version=1)],
        ),
    ):
        pass

    class DownFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["down"]),
            deps=[FeatureDep(key=FeatureKey(["up"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["d"]),
                    code_version=1,
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["up"]),
                            fields=[FieldKey(["d"])],
                        )
                    ],
                )
            ],
        ),
    ):
        pass


def migrate_store_to_graph(
    source_store: InMemoryMetadataStore,
    target_graph: FeatureGraph,
) -> InMemoryMetadataStore:
    """Create new store with target graph context but source store's data.

    Helper for testing migrations - simulates code changing while data stays the same.
    """
    # Create new store and copy data - will be used within target_graph context
    new_store = InMemoryMetadataStore()
    new_store._storage = source_store._storage.copy()
    return new_store


@pytest.fixture
def graph_v1():
    """Registry with v1 features using temporary module."""
    temp_module = TempFeatureModule("test_migrations_features_v1")

    # Define v1 specs
    upstream_spec_v1 = FeatureSpec(
        key=FeatureKey(["test_migrations", "upstream"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    downstream_spec_v1 = FeatureSpec(
        key=FeatureKey(["test_migrations", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test_migrations", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test_migrations", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    # Write to temp module
    temp_module.write_features(
        {
            "Upstream": upstream_spec_v1,
            "Downstream": downstream_spec_v1,
        }
    )

    # Get graph from module
    graph = temp_module.get_graph()

    yield graph

    # Cleanup after test completes
    temp_module.cleanup()


@pytest.fixture
def graph_v2():
    """Registry with v2 features (upstream code_version changed) using temporary module."""
    temp_module = TempFeatureModule("test_migrations_features_v2")

    # Define v2 specs (upstream version changed)
    upstream_spec_v2 = FeatureSpec(
        key=FeatureKey(["test_migrations", "upstream"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=2),  # Changed!
        ],
    )

    downstream_spec_v2 = FeatureSpec(
        key=FeatureKey(["test_migrations", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test_migrations", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test_migrations", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    # Write to temp module
    temp_module.write_features(
        {
            "Upstream": upstream_spec_v2,
            "Downstream": downstream_spec_v2,
        }
    )

    # Get graph from module
    graph = temp_module.get_graph()

    yield graph

    # Cleanup after test completes
    temp_module.cleanup()


@pytest.fixture
def store_with_v1_data(graph_v1: FeatureGraph) -> InMemoryMetadataStore:
    """Store with v1 upstream and downstream data."""
    store = InMemoryMetadataStore()

    with graph_v1.use(), store:
        # Get feature classes
        UpstreamV1 = graph_v1.features_by_key[
            FeatureKey(["test_migrations", "upstream"])
        ]
        DownstreamV1 = graph_v1.features_by_key[
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
        # Explicitly record feature version (must be within graph context!)
        store.record_feature_graph_snapshot()

        # Write downstream using new API
        downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store.resolve_update(DownstreamV1, sample_df=downstream_data)
        if len(diff_result.added) > 0:
            store.write_metadata(DownstreamV1, diff_result.added)

        # Verify downstream was written
        assert ("test_migrations", "downstream") in store._storage

    return store


# Tests


def test_detect_no_changes(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
) -> None:
    """Test detection when no features changed (graph matches data)."""
    with store_with_v1_data:
        # Compare v1 graph with itself (no changes)
        changes = detect_feature_changes(
            store_with_v1_data,
            graph_v1,
            graph_v1,
        )

    # Should detect no changes (v1 graph, v1 data)
    assert len(changes) == 0


def test_detect_single_change(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
) -> None:
    """Test detection of upstream code version change.

    Since downstream's feature_version depends on upstream, both are detected as changed.
    """
    # Migrate store to v2 graph (simulates code change)
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    # Get features for version comparison
    graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    # Record v2 snapshot so detector can query it
    with graph_v2.use(), store_v2:
        store_v2.record_feature_graph_snapshot()

    # Detect changes by comparing v1 graph (in store) with v2 graph (new code)
    with store_v2:
        operations = detect_feature_changes(
            store_v2,
            graph_v1,
            graph_v2,
        )

        # Both features changed (downstream version depends on upstream)
        assert len(operations) == 2

        # Find upstream operation (order may vary)
        upstream_op = next(
            op for op in operations if op.feature_key == ["test_migrations", "upstream"]
        )
        # Feature versions are now derived from snapshots, not stored in operations
        assert upstream_op.id.startswith("reconcile_")


def test_generate_migration_no_changes(
    store_with_v1_data: InMemoryMetadataStore,
    tmp_path: Path,
) -> None:
    """Test generation when no changes detected."""
    with store_with_v1_data:
        result = generate_migration(store_with_v1_data)

        # No changes (v1 graph, v1 data)
        assert result is None


def test_generate_migration_with_changes(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
    tmp_path: Path,
) -> None:
    """Test migration file generation."""
    # Migrate to v2 graph
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    # Get features
    graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    # Generate migration
    with graph_v2.use(), store_v2:
        migration = generate_migration(store_v2)

        assert migration is not None

        assert migration.version == 1
        # Both features changed (downstream version depends on upstream now)
        assert len(migration.operations) == 2  # 2 root changes

        # Parse operations
        ops = migration.get_operations()

        # Find upstream operation (order may vary)
        next(op for op in ops if op.feature_key == ["test_migrations", "upstream"])
        # Feature versions are now derived from snapshots, not stored in operations

        # Find downstream operation (version changed due to upstream dep)
        downstream_op = next(
            op for op in ops if op.feature_key == ["test_migrations", "downstream"]
        )
        # Both are detected as root changes, not downstream reconciliation
        assert downstream_op.reason == "TODO: Describe what changed and why"


def test_apply_migration_rejects_root_features(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
) -> None:
    """Test that DataVersionReconciliation rejects root features (no upstream).

    Root features have user-defined data_versions that cannot be automatically
    reconciled. User must re-run their computation pipeline.
    """
    # Migrate to v2 graph
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    # Get features
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    UpstreamV2 = graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    # Get actual snapshot_id from store
    v1_snapshot_id = get_latest_snapshot_id(store_with_v1_data)

    # Create migration attempting to reconcile a root feature
    migration = Migration(
        version=1,
        id="migration_test_recalc",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        from_snapshot_id=v1_snapshot_id,
        to_snapshot_id=v1_snapshot_id,
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
    with graph_v2.use(), store_v2:
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
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
) -> None:
    """Test that migrations are idempotent."""
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    graph_v2.features_by_key[FeatureKey(["test_migrations", "downstream"])]

    # Create migration for downstream feature (has upstream)
    migration = Migration(
        version=1,
        id="migration_test_idempotent",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        from_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
        to_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
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
    with graph_v2.use(), store_v2:
        result1 = apply_migration(store_v2, migration)
        assert result1.status == "completed"

        # Second application (should skip)
        result2 = apply_migration(store_v2, migration)
        assert result2.status == "skipped"


def test_apply_migration_dry_run(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
) -> None:
    """Test dry-run mode doesn't modify data."""
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    DownstreamV2 = graph_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Create migration for downstream feature
    migration = Migration(
        version=1,
        id="migration_test_dryrun",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        from_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
        to_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
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
    with graph_v2.use(), store_v2:
        result = apply_migration(store_v2, migration, dry_run=True)

        assert result.status == "skipped"
        assert len(result.affected_features) > 0

        # Verify data unchanged - should still only have v1 feature_version
        all_data = store_v2.read_metadata(
            DownstreamV2,  # Reading through v2 graph
            current_only=False,  # Get all versions
        )
        # All rows should have v1 feature_version (not v2 - because it's dry-run)
        assert all(
            fv == DownstreamV1.feature_version()
            for fv in all_data["feature_version"].to_list()
        )


def test_apply_migration_propagates_downstream(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
) -> None:
    """Test that downstream reconciliation works when upstream changes.

    Scenario:
    1. User manually updates upstream (root feature) with new data
    2. Migration reconciles downstream data_versions based on new upstream
    """
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    UpstreamV2 = graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    DownstreamV2 = graph_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    with graph_v2.use(), store_v2:
        # Get initial downstream data_versions
        # Read directly from storage to bypass read_metadata issues
        downstream_storage_key = ("test_migrations", "downstream")
        if downstream_storage_key in store_v2._storage:
            initial_data_versions = store_v2._storage[downstream_storage_key][
                "data_version"
            ].to_list()
        else:
            raise AssertionError("Downstream data not in storage!")

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
            from_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
            to_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
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
        # Note: migration writes with V1 feature_version (from_=to=V1), so read with current_only=False
        new_downstream = store_v2.read_metadata(DownstreamV2, current_only=False)
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
        from_snapshot_id="abc123",
        to_snapshot_id="def456",
        operations=[
            DataVersionReconciliation(
                id="test_op_id",
                feature_key=["my", "feature"],
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
    # Feature versions are now derived from snapshots
    assert parsed_ops[0].reason == "Updated algorithm"


def test_feature_version_in_metadata(graph_v1: FeatureGraph) -> None:
    """Test that feature_version column is automatically added."""
    store = InMemoryMetadataStore()
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    # Write data without feature_version
    data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )

    # Should not have feature_version before write
    assert "feature_version" not in data.columns

    with graph_v1.use(), store:
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
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
) -> None:
    """Test current_only parameter filters by feature_version."""
    # Write v1 data with v1 graph
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    data_v1 = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )
    with graph_v1.use(), store_v1:
        store_v1.write_metadata(UpstreamV1, data_v1)

    # Migrate to v2 graph and write v2 data
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)
    UpstreamV2 = graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    data_v2 = pl.DataFrame(
        {
            "sample_id": [3, 4],
            "data_version": [{"default": "h3"}, {"default": "h4"}],
        }
    )
    with graph_v2.use(), store_v2:
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


def test_system_tables_created(graph_v1: FeatureGraph) -> None:
    """Test that system tables are created when explicitly recording."""
    store = InMemoryMetadataStore()
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    # Write some data
    data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )
    with graph_v1.use(), store:
        store.record_feature_graph_snapshot()
        store.write_metadata(UpstreamV1, data)

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


def test_graph_rejects_duplicate_keys() -> None:
    """Test that FeatureGraph raises error on duplicate feature keys."""
    graph = FeatureGraph()

    # Define first feature
    class Feature1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["duplicate", "key"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
        graph=graph,
    ):
        pass

    # Try to define second feature with same key
    with pytest.raises(ValueError, match="already registered"):

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["duplicate", "key"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
            ),
            graph=graph,
        ):
            pass


def test_detect_uses_latest_version_from_multiple_materializations(
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
    snapshot: SnapshotAssertion,
) -> None:
    """Test that detector compares feature versions between two snapshots."""
    store = InMemoryMetadataStore()

    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    # Record v1 snapshot and write data
    with graph_v1.use(), store:
        store.record_feature_graph_snapshot()

        upstream_data = pl.DataFrame(
            {
                "sample_id": [1, 2],
                "data_version": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        store.write_metadata(UpstreamV1, upstream_data)

    # Record v2 snapshot
    with graph_v2.use(), store:
        store.record_feature_graph_snapshot()

    # Detect changes by comparing v1 and v2 snapshots
    with store:
        changes = detect_feature_changes(
            store,
            graph_v1,
            graph_v2,
        )

        # Both upstream and downstream changed (downstream version depends on upstream)
        assert len(changes) == 2

        # Verify both features detected
        feature_keys = {FeatureKey(op.feature_key).to_string() for op in changes}
        assert feature_keys == {
            "test_migrations_upstream",
            "test_migrations_downstream",
        }

        # Snapshot the detected changes
        changes_dict = {
            "feature_keys": sorted(feature_keys),
            "operation_count": len(changes),
        }
        assert changes_dict == snapshot


def test_migration_result_snapshots(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
    snapshot: SnapshotAssertion,
) -> None:
    """Test migration execution with snapshot of affected features."""
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    UpstreamV2 = graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    migration = Migration(
        version=1,
        id="migration_snapshot_test",
        description="Test for snapshots",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        from_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
        to_snapshot_id=get_latest_snapshot_id(store_with_v1_data),
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
    with graph_v2.use(), store_v2:
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
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
    snapshot: SnapshotAssertion,
) -> None:
    """Test that feature_version hashes are stable for v1 and v2."""
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    UpstreamV2 = graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]
    DownstreamV2 = graph_v2.features_by_key[
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

    # Verify downstream versions also differ (depends on upstream version now)
    assert versions["downstream_v1"] != versions["downstream_v2"]


def test_generated_migration_yaml_snapshot(
    store_with_v1_data: InMemoryMetadataStore,
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
    tmp_path: Path,
    snapshot: SnapshotAssertion,
) -> None:
    """Test generated YAML migration file structure with snapshot."""
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    # Generate migration
    with graph_v2.use(), store_v2:
        migration = generate_migration(store_v2)

        assert migration is not None

        # Parse operations for snapshot
        ops = migration.get_operations()

        # Snapshot the migration structure (excluding timestamp-based fields)
        # Feature versions are now derived from snapshots, not stored in operations
        migration_dict = {
            "version": migration.version,
            "operations": [
                {
                    "feature_key": op.feature_key,
                    "operation_id": op.id,
                    "reason": op.reason,
                }
                for op in ops
            ],
        }

        assert migration_dict == snapshot


def test_serialize_feature_graph(
    graph_v1: FeatureGraph,
    snapshot: SnapshotAssertion,
) -> None:
    """Test recording all features with deterministic snapshot_id."""
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    store = InMemoryMetadataStore()

    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Write data for both features
    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [{"default": "h1"}, {"default": "h2"}, {"default": "h3"}],
        }
    )
    with graph_v1.use(), store:
        store.write_metadata(UpstreamV1, upstream_data)

        downstream_data = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store.resolve_update(DownstreamV1, sample_df=downstream_data)
        if len(diff_result.added) > 0:
            store.write_metadata(DownstreamV1, diff_result.added)

        # Record all features at once
        snapshot_id = store.serialize_feature_graph()

        # Verify snapshot_id is deterministic
        assert len(snapshot_id) == 64
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
    graph_v1: FeatureGraph,
) -> None:
    """Test that snapshot_id is deterministic and recording is idempotent."""
    store = InMemoryMetadataStore()

    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Write data
    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [{"default": "h1"}, {"default": "h2"}],
        }
    )
    with graph_v1.use(), store:
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
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
    snapshot: SnapshotAssertion,
) -> None:
    """Test standard workflow: compute v2 data, record snapshot (no migration needed)."""
    # Step 1: Materialize v1 features and record
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    upstream_data_v1 = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [{"default": "h1"}, {"default": "h2"}, {"default": "h3"}],
        }
    )
    with graph_v1.use(), store_v1:
        store_v1.write_metadata(UpstreamV1, upstream_data_v1)

        downstream_data_v1 = pl.DataFrame({"sample_id": [1, 2, 3]})
        diff_result = store_v1.resolve_update(
            DownstreamV1, sample_df=downstream_data_v1
        )
        if len(diff_result.added) > 0:
            store_v1.write_metadata(DownstreamV1, diff_result.added)

        # Record v1 graph snapshot
        snapshot_id_v1 = store_v1.serialize_feature_graph()

    # Step 2: Code changes (v1 -> v2), migrate store to v2 graph
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)
    UpstreamV2 = graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV2 = graph_v2.features_by_key[
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
    with graph_v2.use(), store_v2:
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

        # Should have 4 records:
        # - upstream v1 (feature_version changed)
        # - downstream v1 (feature_version depends on upstream, so also changed)
        # - upstream v2 (new feature_version)
        # - downstream v2 (new feature_version because upstream changed)
        assert len(version_history) == 4

        # Verify snapshot_ids are different (different graph states)
        snapshot_ids = version_history["snapshot_id"].unique().to_list()
        assert len(snapshot_ids) == 2
        assert snapshot_id_v1 in snapshot_ids
        assert snapshot_id_v2 in snapshot_ids
        assert snapshot_id_v1 != snapshot_id_v2

        # Verify downstream recorded twice (feature_version changed due to upstream dependency)
        downstream_records = version_history.filter(
            pl.col("feature_key") == "test_migrations_downstream"
        )
        assert (
            len(downstream_records) == 2
        )  # Two records because downstream feature_version depends on upstream

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
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
) -> None:
    """Test that migrations preserve old data (immutable)."""
    # Setup v1 data
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[
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

    with graph_v1.use(), store_v1:
        store_v1.record_feature_graph_snapshot()
        store_v1.write_metadata(UpstreamV1, upstream_data)
        store_v1.write_metadata(DownstreamV1, downstream_data)

        # Get original downstream data for comparison
        original_data = store_v1.read_metadata(DownstreamV1, current_only=False)
        original_data_versions = original_data["data_version"].to_list()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)
    DownstreamV2 = graph_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    # Apply migration to downstream (has upstream dependencies)
    snapshot_id_v1 = get_latest_snapshot_id(store_v1)

    migration = Migration(
        version=1,
        id="test_immutability",
        description="Test",
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        from_snapshot_id=snapshot_id_v1,
        to_snapshot_id=snapshot_id_v1,
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

    with graph_v2.use(), store_v2:
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

        def execute(self, store, *, dry_run=False):  # pyrefly: ignore[bad-override]
            import polars as pl

            from metaxy.models.feature import FeatureGraph
            from metaxy.models.types import FeatureKey

            # Load external data (from test_data)
            external_df = pl.DataFrame(self.test_data)

            if dry_run:
                return len(external_df)

            # Get feature
            feature_key = FeatureKey(self.feature_key)
            graph = FeatureGraph.get_active()
            feature_cls = graph.features_by_key[feature_key]

            # For features with upstream: use resolve_update
            # For root features: user provides data_version directly or computes it
            plan = graph.get_feature_plan(feature_key)
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

    # Create graph with root feature
    graph = FeatureGraph()
    with graph.use():

        class RootFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["backfill", "test"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
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

    graph = FeatureGraph()
    with graph.use():
        # Use module-level classes for this test
        graph.add_feature(UpFeature)
        graph.add_feature(DownFeature)

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

            # Record snapshot
            snapshot_id = store.serialize_feature_graph()

            # Create migration 1 (no parent) - empty operations but registers in system
            migration1 = Migration(
                version=1,
                id="migration_001",
                parent_migration_id=None,
                description="First migration",
                created_at=datetime(2025, 1, 1),
                from_snapshot_id=snapshot_id,
                to_snapshot_id=snapshot_id,
                operations=[],
            )

            # Create migration 2 (parent = migration1)
            migration2 = Migration(
                version=1,
                id="migration_002",
                parent_migration_id="migration_001",
                description="Second migration",
                created_at=datetime(2025, 1, 2),
                from_snapshot_id=snapshot_id,
                to_snapshot_id=snapshot_id,
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
    graph_v2: FeatureGraph,
    tmp_path: Path,
) -> None:
    """Test that generator automatically sets parent_migration_id."""
    store_v2 = migrate_store_to_graph(store_with_v1_data, graph_v2)

    with graph_v2.use(), store_v2:
        # Generate first migration
        migration1 = generate_migration(store_v2)
        assert migration1 is not None

        assert migration1.parent_migration_id is None  # First migration has no parent

        # Apply it
        apply_migration(store_v2, migration1)

        # Generate second migration (should reference first)
        migration2 = generate_migration(store_v2)

        if migration2 is not None:
            # If there are more changes, second migration should reference first
            assert migration2.parent_migration_id == migration1.id


def test_migration_ignores_new_features(
    graph_v1: FeatureGraph,
) -> None:
    """Test that adding a new feature to the graph doesn't trigger migrations.

    New features have no existing data, so they should be ignored by detect_feature_changes.
    """
    # Create graph with v1 + new feature
    temp_module = TempFeatureModule("test_new_feature")

    # V1 upstream spec
    upstream_spec = FeatureSpec(
        key=FeatureKey(["test_migrations", "upstream"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    # NEW feature that didn't exist in v1
    new_feature_spec = FeatureSpec(
        key=FeatureKey(["test_migrations", "new_feature"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    temp_module.write_features(
        {
            "Upstream": upstream_spec,
            "NewFeature": new_feature_spec,
        }
    )

    graph_with_new = temp_module.get_graph()

    # Create store with only v1 upstream data (no new_feature data)
    store = InMemoryMetadataStore()
    upstream_v1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]

    with graph_v1.use(), store:
        upstream_data = pl.DataFrame(
            {
                "sample_id": [1, 2],
                "data_version": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        store.write_metadata(upstream_v1, upstream_data)
        store.serialize_feature_graph()

    # Migrate store to graph with new feature
    store_new = migrate_store_to_graph(store, graph_with_new)

    # Record new graph snapshot
    with graph_with_new.use(), store_new:
        store_new.record_feature_graph_snapshot()

    with store_new:
        # Detect changes - should ignore new_feature (no existing data)
        operations = detect_feature_changes(
            store_new,
            graph_v1,
            graph_with_new,
        )

        # Should only detect 0 operations (upstream unchanged, new_feature has no data)
        assert len(operations) == 0

        # Verify new_feature is in graph but not detected as needing migration
        assert (
            FeatureKey(["test_migrations", "new_feature"])
            in graph_with_new.features_by_key
        )

    temp_module.cleanup()


def test_migration_with_dependency_change() -> None:
    """Test migration when feature-level dependencies change.

    Changing a feature's dependencies changes its feature_version,
    triggering a migration even if code_version didn't change.
    """
    # Create v1: Downstream depends on UpstreamA
    temp_v1 = TempFeatureModule("test_dep_change_v1")

    upstream_a_spec = FeatureSpec(
        key=FeatureKey(["test", "upstream_a"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    upstream_b_spec = FeatureSpec(
        key=FeatureKey(["test", "upstream_b"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    downstream_v1_spec = FeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream_a"]))],  # Depends on A
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream_a"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    temp_v1.write_features(
        {
            "UpstreamA": upstream_a_spec,
            "UpstreamB": upstream_b_spec,
            "Downstream": downstream_v1_spec,
        }
    )

    graph_v1 = temp_v1.get_graph()

    # Create v2: Downstream now depends on UpstreamB instead
    temp_v2 = TempFeatureModule("test_dep_change_v2")

    downstream_v2_spec = FeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream_b"]))],  # Changed to B!
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,  # Same code_version
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream_b"]),  # Changed!
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    temp_v2.write_features(
        {
            "UpstreamA": upstream_a_spec,
            "UpstreamB": upstream_b_spec,
            "Downstream": downstream_v2_spec,
        }
    )

    graph_v2 = temp_v2.get_graph()

    # Get feature classes
    down_v1 = graph_v1.features_by_key[FeatureKey(["test", "downstream"])]
    down_v2 = graph_v2.features_by_key[FeatureKey(["test", "downstream"])]

    # Verify feature_versions are different (dependency changed)
    assert down_v1.feature_version() != down_v2.feature_version()

    # Create store with v1 data
    store = InMemoryMetadataStore()
    upstream_a = graph_v1.features_by_key[FeatureKey(["test", "upstream_a"])]
    upstream_b = graph_v1.features_by_key[FeatureKey(["test", "upstream_b"])]

    with graph_v1.use(), store:
        # Write root features with user-defined data_versions
        store.write_metadata(
            upstream_a,
            pl.DataFrame({"sample_id": [1], "data_version": [{"default": "ha"}]}),
        )
        store.write_metadata(
            upstream_b,
            pl.DataFrame({"sample_id": [1], "data_version": [{"default": "hb"}]}),
        )

        # Downstream is not a root feature - use resolve_update to get correct data_version
        downstream_samples = pl.DataFrame({"sample_id": [1]})
        diff = store.resolve_update(down_v1, sample_df=downstream_samples)
        if len(diff.added) > 0:
            store.write_metadata(down_v1, diff.added)

        store.serialize_feature_graph()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store, graph_v2)

    # Record v2 snapshot
    with graph_v2.use(), store_v2:
        store_v2.record_feature_graph_snapshot()

    with store_v2:
        # Should detect downstream as changed (dependencies changed)
        operations = detect_feature_changes(
            store_v2,
            graph_v1,
            graph_v2,
        )

        # Downstream should be detected as changed
        assert len(operations) == 1
        assert operations[0].feature_key == ["test", "downstream"]
        # Feature versions are now derived from snapshots, not stored in operations

    temp_v1.cleanup()
    temp_v2.cleanup()


def test_migration_with_field_dependency_change() -> None:
    """Test migration when field-level dependencies change.

    Changing which fields a feature depends on changes its field version,
    which changes the feature_version.
    """
    # Create v1: Downstream depends on both upstream fields
    temp_v1 = TempFeatureModule("test_field_dep_v1")

    upstream_spec = FeatureSpec(
        key=FeatureKey(["test", "upstream"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["frames"]), code_version=1),
            FieldSpec(key=FieldKey(["audio"]), code_version=1),
        ],
    )

    downstream_v1_spec = FeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream"]),
                        fields=[
                            FieldKey(["frames"]),
                            FieldKey(["audio"]),
                        ],  # Both
                    )
                ],
            )
        ],
    )

    temp_v1.write_features(
        {
            "Upstream": upstream_spec,
            "Downstream": downstream_v1_spec,
        }
    )

    graph_v1 = temp_v1.get_graph()

    # Create v2: Downstream now only depends on frames field
    temp_v2 = TempFeatureModule("test_field_dep_v2")

    downstream_v2_spec = FeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,  # Same code_version
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream"]),
                        fields=[FieldKey(["frames"])],  # Only frames now!
                    )
                ],
            )
        ],
    )

    temp_v2.write_features(
        {
            "Upstream": upstream_spec,
            "Downstream": downstream_v2_spec,
        }
    )

    graph_v2 = temp_v2.get_graph()

    # Get feature classes
    down_v1 = graph_v1.features_by_key[FeatureKey(["test", "downstream"])]
    down_v2 = graph_v2.features_by_key[FeatureKey(["test", "downstream"])]

    # Verify feature_versions are different (field deps changed)
    assert down_v1.feature_version() != down_v2.feature_version()

    # Create store with v1 data
    store = InMemoryMetadataStore()
    upstream_v1 = graph_v1.features_by_key[FeatureKey(["test", "upstream"])]

    with graph_v1.use(), store:
        # Write root feature with user-defined data_version
        store.write_metadata(
            upstream_v1,
            pl.DataFrame(
                {"sample_id": [1], "data_version": [{"frames": "hf", "audio": "ha"}]}
            ),
        )

        # Downstream is not a root feature - use resolve_update to get correct data_version
        downstream_samples = pl.DataFrame({"sample_id": [1]})
        diff = store.resolve_update(down_v1, sample_df=downstream_samples)
        if len(diff.added) > 0:
            store.write_metadata(down_v1, diff.added)

        store.serialize_feature_graph()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store, graph_v2)

    # Record v2 snapshot
    with graph_v2.use(), store_v2:
        store_v2.record_feature_graph_snapshot()

    with store_v2:
        # Should detect downstream as changed (field deps changed)
        operations = detect_feature_changes(
            store_v2,
            graph_v1,
            graph_v2,
        )

        # Downstream should be detected as changed
        assert len(operations) == 1
        assert operations[0].feature_key == ["test", "downstream"]
        # Feature versions are now derived from snapshots, not stored in operations

    temp_v1.cleanup()
    temp_v2.cleanup()


def test_sequential_migration_application():
    """Test applying multiple migrations in sequence.

    Creates 3 migrations with parent dependencies and verifies they're applied
    in the correct order with proper skipping of already-applied migrations.
    """

    # Create a simple graph
    temp_module = TempFeatureModule("test_sequential")
    spec = FeatureSpec(
        key=FeatureKey(["test", "feature"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )
    temp_module.write_features({"TestFeature": spec})
    graph = temp_module.get_graph()

    # Create store and record snapshot
    store = InMemoryMetadataStore()
    test_feature = graph.features_by_key[FeatureKey(["test", "feature"])]

    with graph.use(), store:
        store.write_metadata(
            test_feature,
            pl.DataFrame(
                {
                    "sample_id": [1, 2, 3],
                    "data_version": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                    ],
                }
            ),
        )
        snapshot_id = store.serialize_feature_graph()

    # Create 3 migrations with side effects (using MetadataBackfill with unique markers)
    from datetime import datetime

    # Migration 1 (root)
    migration1 = Migration(
        version=1,
        id="migration_001",
        parent_migration_id=None,
        from_snapshot_id=snapshot_id,
        to_snapshot_id=snapshot_id,
        description="First migration",
        created_at=datetime(2025, 1, 1),
        operations=[],
    )

    # Migration 2 (child of 1)
    migration2 = Migration(
        version=1,
        id="migration_002",
        parent_migration_id="migration_001",
        from_snapshot_id=snapshot_id,
        to_snapshot_id=snapshot_id,
        description="Second migration",
        created_at=datetime(2025, 1, 2),
        operations=[],
    )

    # Migration 3 (child of 2)
    migration3 = Migration(
        version=1,
        id="migration_003",
        parent_migration_id="migration_002",
        from_snapshot_id=snapshot_id,
        to_snapshot_id=snapshot_id,
        description="Third migration",
        created_at=datetime(2025, 1, 3),
        operations=[],
    )

    with graph.use(), store:
        # Apply migration 1 (empty operations = skipped/completed immediately)
        result1 = apply_migration(store, migration1)
        assert result1.status in ("completed", "skipped")

        # Apply migration 2
        result2 = apply_migration(store, migration2)
        assert result2.status in ("completed", "skipped")

        # Apply migration 3
        result3 = apply_migration(store, migration3)
        assert result3.status in ("completed", "skipped")

        # Try to re-apply migration 2 (should skip)
        result2_again = apply_migration(store, migration2)
        assert result2_again.status == "skipped"

        # Verify all migrations are registered
        from metaxy.migrations.executor import MIGRATIONS_KEY

        migrations_table = store.read_metadata(MIGRATIONS_KEY, current_only=False)
        assert len(migrations_table) == 3
        assert set(migrations_table["migration_id"].to_list()) == {
            "migration_001",
            "migration_002",
            "migration_003",
        }

    temp_module.cleanup()


def test_multiple_migration_heads_detection():
    """Test that multiple heads are detected correctly.

    Creates two independent migration chains and verifies the system can
    detect multiple heads.
    """
    # Create graph
    temp_module = TempFeatureModule("test_multiple_heads")
    spec = FeatureSpec(
        key=FeatureKey(["test", "feature"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )
    temp_module.write_features({"TestFeature": spec})
    graph = temp_module.get_graph()

    store = InMemoryMetadataStore()
    test_feature = graph.features_by_key[FeatureKey(["test", "feature"])]

    with graph.use(), store:
        store.write_metadata(
            test_feature,
            pl.DataFrame({"sample_id": [1], "data_version": [{"default": "h1"}]}),
        )
        snapshot_id = store.serialize_feature_graph()

    from datetime import datetime

    # Create two independent chains
    # Chain 1: migration_a1 -> migration_a2
    migration_a1 = Migration(
        version=1,
        id="migration_a1",
        parent_migration_id=None,
        from_snapshot_id=snapshot_id,
        to_snapshot_id=snapshot_id,
        description="Chain A - step 1",
        created_at=datetime(2025, 1, 1),
        operations=[],
    )

    migration_a2 = Migration(
        version=1,
        id="migration_a2",
        parent_migration_id="migration_a1",
        from_snapshot_id=snapshot_id,
        to_snapshot_id=snapshot_id,
        description="Chain A - step 2",
        created_at=datetime(2025, 1, 2),
        operations=[],
    )

    # Chain 2: migration_b1 -> migration_b2
    migration_b1 = Migration(
        version=1,
        id="migration_b1",
        parent_migration_id=None,
        from_snapshot_id=snapshot_id,
        to_snapshot_id=snapshot_id,
        description="Chain B - step 1",
        created_at=datetime(2025, 1, 3),
        operations=[],
    )

    migration_b2 = Migration(
        version=1,
        id="migration_b2",
        parent_migration_id="migration_b1",
        from_snapshot_id=snapshot_id,
        to_snapshot_id=snapshot_id,
        description="Chain B - step 2",
        created_at=datetime(2025, 1, 4),
        operations=[],
    )

    with graph.use(), store:
        # Apply all migrations
        apply_migration(store, migration_a1)
        apply_migration(store, migration_a2)
        apply_migration(store, migration_b1)
        apply_migration(store, migration_b2)

        # Verify both heads exist
        from metaxy.migrations.executor import MIGRATIONS_KEY

        migrations_table = store.read_metadata(MIGRATIONS_KEY, current_only=False)
        assert len(migrations_table) == 4

        # Build dependency graph to find heads
        children_by_parent = {}
        for row in migrations_table.iter_rows(named=True):
            mig_id = row["migration_id"]
            # Parse migration to get parent
            import json

            mig_yaml = row["migration_yaml"]
            if isinstance(mig_yaml, str):
                mig_data = json.loads(mig_yaml)
            else:
                mig_data = mig_yaml

            parent_id = mig_data.get("parent_migration_id")
            if parent_id:
                if parent_id not in children_by_parent:
                    children_by_parent[parent_id] = []
                children_by_parent[parent_id].append(mig_id)

        # Find heads (migrations with no children)
        all_mig_ids = set(migrations_table["migration_id"].to_list())
        heads = [mig_id for mig_id in all_mig_ids if mig_id not in children_by_parent]

        # Should have 2 heads
        assert len(heads) == 2
        assert set(heads) == {"migration_a2", "migration_b2"}

    temp_module.cleanup()


def test_migration_vs_recompute_comparison(
    graph_v1: FeatureGraph,
    graph_v2: FeatureGraph,
    snapshot: SnapshotAssertion,
) -> None:
    """Compare migration (no recompute) vs standard workflow (with recompute).

    This test demonstrates when to use migrations vs when to just recompute.
    """
    # Setup: v1 data exists
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [{"default": "h1"}, {"default": "h2"}, {"default": "h3"}],
        }
    )
    with graph_v1.use(), store_v1:
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
    store_migration = migrate_store_to_graph(store_v1, graph_v2)
    UpstreamV2 = graph_v2.features_by_key[FeatureKey(["test_migrations", "upstream"])]
    DownstreamV2 = graph_v2.features_by_key[
        FeatureKey(["test_migrations", "downstream"])
    ]

    with graph_v2.use(), store_migration:
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
        snapshot_id_v1 = get_latest_snapshot_id(store_v1)

        migration = Migration(
            version=1,
            id="migration_test_comparison",
            description="Test",
            created_at=datetime(2025, 1, 1, 0, 0, 0),
            from_snapshot_id=snapshot_id_v1,
            to_snapshot_id=snapshot_id_v1,
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
        # Note: migration writes with V1 feature_version, so use current_only=False
        migrated_downstream = store_migration.read_metadata(
            DownstreamV2, current_only=False
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
