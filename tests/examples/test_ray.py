"""Test the Ray example using the runbook system."""

from pathlib import Path

from metaxy_testing import RunbookRunner
from metaxy_testing.runbook import Scenario

EXAMPLE_DIR = Path("examples/example-ray")


def _env_overrides(tmp_path: Path) -> dict[str, str]:
    return {
        "RAY_DEMO_DB": str(tmp_path / "ray_demo.db"),
        "RAY_META_DB": str(tmp_path / "ray_meta.db"),
        "RAY_STORAGE_PATH": str(tmp_path / "ray_storage"),
    }


def _find_scenario(runner: RunbookRunner, name: str) -> Scenario:
    for scenario in runner.runbook.scenarios:
        if scenario.name == name:
            return scenario
    msg = f"Scenario {name!r} not found in runbook"
    raise ValueError(msg)


def test_ray_basic_pipeline(tmp_path):
    """Test Example 1: basic Ray pipeline with seed, process, idempotent rerun."""
    with RunbookRunner.runner_for_project(
        example_dir=EXAMPLE_DIR,
        env_overrides=_env_overrides(tmp_path),
    ) as runner:
        runner.run_scenario(_find_scenario(runner, "Setup video metadata"))


def test_ray_dagster_assets(tmp_path):
    """Test Example 2: Dagster assets with per-row writes via BufferedMetadataWriter."""
    with RunbookRunner.runner_for_project(
        example_dir=EXAMPLE_DIR,
        env_overrides=_env_overrides(tmp_path),
    ) as runner:
        runner.run_scenario(_find_scenario(runner, "Dagster assets"))


def test_ray_production_assets(tmp_path):
    """Test Example 3: production asset with branching IO, counters, error handling."""
    with RunbookRunner.runner_for_project(
        example_dir=EXAMPLE_DIR,
        env_overrides=_env_overrides(tmp_path),
    ) as runner:
        runner.run_scenario(_find_scenario(runner, "Production assets"))
