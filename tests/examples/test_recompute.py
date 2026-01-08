"""Test the recompute example using the runbook system."""

from pathlib import Path

from metaxy._testing import RunbookRunner
from metaxy._testing.runbook import GraphPushed, PatchApplied


def test_recompute_runbook(tmp_path):
    """Test recompute example using the .example.yaml runbook.

    This test demonstrates the new runbook system:
    - Loads .example.yaml from the example directory
    - Executes all scenarios in order
    - Automatically applies and reverts patches
    - Handles assertions on command output
    """
    example_dir = Path("examples/example-recompute")
    test_db = tmp_path / "example_recompute.db"

    # Run the runbook with a temporary test database
    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        env_overrides={"METAXY_STORES__DEV__CONFIG__DATABASE": str(test_db)},
    ) as runner:
        runner.run()

        # Verify execution state was captured
        execution_state = runner.runbook.execution_state
        assert execution_state is not None, "Execution state should be captured"

        # Should have events recorded
        assert len(execution_state.events) > 0, "Should have recorded events"

        # Should have at least one GraphPushed event (initial push)
        graph_pushed_events = [
            e for e in execution_state.events if isinstance(e, GraphPushed)
        ]
        assert len(graph_pushed_events) >= 1, (
            "Should have at least one GraphPushed event"
        )

        # Verify snapshot versions are non-empty
        for event in graph_pushed_events:
            assert event.snapshot_version, "Snapshot version should not be empty"

        # The latest snapshot in events should match what's actually in the store
        store_snapshot = runner.get_latest_snapshot_version()
        assert store_snapshot is not None, "Store should have a snapshot"
        assert execution_state.latest_snapshot == store_snapshot

        # Should have PatchApplied events (this example has patches)
        patch_events = [
            e for e in execution_state.events if isinstance(e, PatchApplied)
        ]
        assert len(patch_events) >= 1, "Should have at least one PatchApplied event"

        # Verify patch snapshots are accessible via property
        patch_snapshots = runner.runbook.patch_snapshots
        assert len(patch_snapshots) > 0, "Should have patch snapshots"

        # Each patch should have before/after snapshots
        for patch_path, (before, after) in patch_snapshots.items():
            assert before is not None, f"Patch {patch_path} should have before snapshot"
            assert after is not None, f"Patch {patch_path} should have after snapshot"
            # Before and after should be different (patch changed something)
            assert before != after, f"Patch {patch_path} should change the snapshot"
