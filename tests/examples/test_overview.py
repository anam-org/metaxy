"""Test the overview example using the runbook system."""

from pathlib import Path

from metaxy_testing import RunbookRunner


def test_overview_runbook(tmp_path):
    """Test overview example using the .example.yaml runbook."""
    example_dir = Path("examples/example-overview")
    test_storage = tmp_path / "example_overview"

    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        env_overrides={"METAXY_STORES__DEV__CONFIG__ROOT_PATH": str(test_storage)},
    ) as runner:
        runner.run()

        execution_state = runner.runbook.execution_state
        assert execution_state is not None, "Execution state should be captured"
        assert len(execution_state.events) > 0, "Should have recorded events"
