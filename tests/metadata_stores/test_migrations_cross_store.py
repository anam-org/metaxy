"""Test migrations work correctly across all metadata store backends."""

from datetime import datetime
from typing import Any

import polars as pl
from pytest_cases import parametrize_with_cases
from syrupy.assertion import SnapshotAssertion

from metaxy.metadata_store.base import MetadataStore
from metaxy.migrations import DataVersionReconciliation, Migration, apply_migration
from metaxy.models.feature import FeatureGraph

from .conftest import StoreCases  # type: ignore[import-not-found]


@parametrize_with_cases("store_config", cases=StoreCases)
def test_migration_system_tables_serialize_cross_store(
    store_config: tuple[type[MetadataStore], dict[str, Any]],
    test_graph: tuple[FeatureGraph, dict[str, type]],
    snapshot: SnapshotAssertion,
) -> None:
    """Test that migration system tables serialize correctly across all backends.

    Verifies:
    - System tables (migrations, ops, steps) can be written and read
    - JSON serialization works for SQLite
    - Native structs/lists work for DuckDB/ClickHouse
    - Migration metadata (operation_ids, expected_steps) survives round-trip
    """
    graph, features = test_graph
    store_type, config = store_config

    UpstreamFeatureA = features["UpstreamFeatureA"]
    DownstreamFeature = features["DownstreamFeature"]

    store = store_type(**config)  # type: ignore[abstract]
    with graph.use(), store:
        # Record feature graph snapshot first (mimics CI/CD workflow)
        # This must be done before any operations that need historical graph
        snapshot_id = store.serialize_feature_graph()

        # Write minimal data for downstream feature (has upstream deps)
        upstream_data = pl.DataFrame(
            {
                "sample_id": [1],
                "data_version": [{"frames": "f1", "audio": "a1"}],
            }
        )
        store.write_metadata(UpstreamFeatureA, upstream_data)

        downstream_data = pl.DataFrame(
            {
                "sample_id": [1],
                "data_version": [{"default": "d1"}],
            }
        )
        store.write_metadata(DownstreamFeature, downstream_data)

        # Create and apply migration
        migration = Migration(
            version=1,
            id="test_serialization",
            description="Test system table serialization",
            created_at=datetime(2025, 1, 1),
            from_snapshot_id=snapshot_id,
            to_snapshot_id=snapshot_id,
            operations=[
                DataVersionReconciliation(
                    id="reconcile_downstream",
                    feature_key=["test_stores", "downstream"],
                    reason="Test",
                ).model_dump(by_alias=True)
            ],
        )

        result = apply_migration(store, migration)
        assert result.status == "completed", f"Migration failed: {result.errors}"

        # Verify system tables were created and are readable
        from metaxy.migrations.executor import (
            MIGRATION_OP_STEPS_KEY,
            MIGRATION_OPS_KEY,
            MIGRATIONS_KEY,
        )

        # Test migrations table
        migrations = store.read_metadata(MIGRATIONS_KEY, current_only=False)
        assert len(migrations) == 1
        assert migrations["migration_id"][0] == "test_serialization"

        # Parse operation_ids (handles both JSON string and native list)
        operation_ids_raw = migrations["operation_ids"][0]
        if isinstance(operation_ids_raw, str):
            import json

            operation_ids = json.loads(operation_ids_raw)
        else:
            # For native list/Series, convert to Python list
            if hasattr(operation_ids_raw, "to_list"):
                operation_ids = operation_ids_raw.to_list()
            else:
                operation_ids = list(operation_ids_raw)
        assert operation_ids == ["reconcile_downstream"]

        # Test ops table
        ops = store.read_metadata(MIGRATION_OPS_KEY, current_only=False)
        assert len(ops) == 1
        assert ops["operation_id"][0] == "reconcile_downstream"
        assert ops["feature_key"][0] == "test_stores_downstream"

        # Parse expected_steps (handles both JSON string and native list)
        expected_steps_raw = ops["expected_steps"][0]
        if isinstance(expected_steps_raw, str):
            import json

            expected_steps = json.loads(expected_steps_raw)
        else:
            # For native list/Series, convert to Python list
            if hasattr(expected_steps_raw, "to_list"):
                expected_steps = expected_steps_raw.to_list()
            else:
                expected_steps = list(expected_steps_raw)
        assert expected_steps == ["test_stores_downstream"]

        # Test steps table
        steps = store.read_metadata(MIGRATION_OP_STEPS_KEY, current_only=False)
        assert len(steps) == 1
        assert steps["migration_id"][0] == "test_serialization"
        assert steps["operation_id"][0] == "reconcile_downstream"
        assert steps["feature_key"][0] == "test_stores_downstream"
        assert steps["error"][0] is None  # Should complete without errors

        # Snapshot system table contents for verification
        system_tables_snapshot = {
            "migration": {
                "migration_id": migrations["migration_id"][0],
                "description": migrations["description"][0],
                "operation_ids": operation_ids,
            },
            "operation": {
                "operation_id": ops["operation_id"][0],
                "operation_type": ops["operation_type"][0],
                "feature_key": ops["feature_key"][0],
                "expected_steps": expected_steps,
            },
            "step": {
                "migration_id": steps["migration_id"][0],
                "operation_id": steps["operation_id"][0],
                "feature_key": steps["feature_key"][0],
                "rows_affected": steps["rows_affected"][0],
                "error": steps["error"][0],
            },
        }
        assert system_tables_snapshot == snapshot

        # Snapshot the reconciled downstream data (sorted for determinism)
        final_downstream = store.read_metadata(
            DownstreamFeature, current_only=False
        ).sort(
            ["sample_id", "feature_version", "snapshot_id"]
        )  # Sort by multiple columns for determinism

        # Convert to snapshot-friendly format (sorted by keys for consistency)
        rows_list = [
            {
                "sample_id": row["sample_id"],
                "feature_version": row["feature_version"],
                # Sort dict keys for deterministic string representation
                "data_version": str(dict(sorted(row["data_version"].items()))),
            }
            for row in final_downstream.iter_rows(named=True)
        ]

        downstream_snapshot = {
            "row_count": len(final_downstream),
            "rows": sorted(
                rows_list,
                key=lambda x: (x["sample_id"], x["feature_version"], x["data_version"]),
            ),
        }
        assert downstream_snapshot == snapshot
