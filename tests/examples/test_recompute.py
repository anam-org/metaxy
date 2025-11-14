"""Test the recompute example using the runbook system."""

from pathlib import Path

from metaxy._testing import RunbookRunner


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
        override_db_path=test_db,
    ) as runner:
        runner.run()

    # If we get here, all scenarios passed!
    assert True, "Runbook executed successfully"
