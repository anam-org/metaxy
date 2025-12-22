"""Test the aggregation example using the runbook system."""

from pathlib import Path

from metaxy._testing import RunbookRunner


def test_aggregation_runbook(tmp_path):
    """Test N:1 aggregation example using the .example.yaml runbook.

    This test verifies:
    1. Initial run creates audio recordings and speaker embeddings
    2. Second run detects no changes and creates no new embeddings (idempotent behavior)

    The key assertion is that aggregation relationships correctly aggregate
    multiple upstream records into a single downstream record, and detect when
    no upstream changes have occurred.
    """
    example_dir = Path("examples/example-aggregation")
    test_db = tmp_path / "example_aggregation"

    # Run the runbook with a temporary test database
    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        override_db_path=test_db,
    ) as runner:
        runner.run()
