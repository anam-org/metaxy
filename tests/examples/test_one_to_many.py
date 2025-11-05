"""Test the one-to-many expansion example using the runbook system."""

from pathlib import Path

from metaxy._testing import RunbookRunner


def test_one_to_many_runbook(tmp_path):
    """Test one-to-many expansion example using the .example.yaml runbook.

    This test verifies:
    1. Initial run creates videos and splits them into chunks
    2. Second run detects no changes and creates no new chunks (idempotent behavior)

    The key assertion is that expansion relationships correctly detect when
    no upstream changes have occurred, preventing duplicate child records.
    """
    example_dir = Path("examples/example-one-to-many")
    test_db = tmp_path / "example_one_to_many.db"

    # Run the runbook with a temporary test database
    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        override_db_path=test_db,
    ) as runner:
        runner.run()

    # If we get here, all scenarios passed!
    assert True, "Runbook executed successfully - expansion relationships are idempotent"
