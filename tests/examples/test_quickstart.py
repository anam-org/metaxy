"""Test the quickstart example using the runbook system."""

from pathlib import Path

from metaxy_testing import RunbookRunner


def test_quickstart_runbook(tmp_path):
    """Test quickstart example using the .example.yaml runbook."""
    example_dir = Path("examples/example-quickstart")
    test_storage = tmp_path / "example_quickstart"

    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        env_overrides={"METAXY_STORES__DEV__CONFIG__ROOT_PATH": str(test_storage)},
    ) as runner:
        runner.run()

        execution_state = runner.runbook.execution_state
        assert execution_state is not None, "Execution state should be captured"
        assert len(execution_state.events) > 0, "Should have recorded events"
