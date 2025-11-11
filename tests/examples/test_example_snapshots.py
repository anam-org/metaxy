"""Snapshot tests for all examples with .example.yaml files."""

from pathlib import Path

import pytest


def discover_examples() -> list[Path]:
    """Discover all example directories with .example.yaml files."""
    examples_dir = Path("examples")
    examples = []
    for yaml_file in examples_dir.glob("*/.example.yaml"):
        examples.append(yaml_file.parent)
    return sorted(examples)


@pytest.mark.parametrize("example_dir", discover_examples(), ids=lambda p: p.name)
def test_example_snapshot(example_dir: Path, tmp_path: Path, example_snapshot):
    """Execute runbook, save raw results, and verify snapshot matches."""
    from metaxy._testing.runbook import RunbookRunner

    # Setup test database
    test_db = tmp_path / f"{example_dir.name}.db"

    # Run the runbook with deterministic random seed
    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        env_overrides={
            "METAXY_STORES__DEV__CONFIG__DATABASE": str(test_db),
            "RANDOM_SEED": "42",
        },
    ) as runner:
        runner.run()
        runbook = runner.runbook

        # Save raw execution state for docs (with timestamps)
        raw_result_path = example_dir / ".example.result.json"
        runbook.save_execution_state(raw_result_path)

        # Get execution state for snapshot comparison
        execution_state = runbook.execution_state
        assert execution_state is not None, "No execution state captured"

        # Snapshot comparison (timestamps excluded automatically by fixture)
        assert execution_state.model_dump(mode="json") == example_snapshot
