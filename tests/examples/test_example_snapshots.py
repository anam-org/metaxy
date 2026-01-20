"""Snapshot tests for all examples with .example.yaml files."""

from pathlib import Path

import pytest
import tomli as tomllib


def discover_examples() -> list[Path]:
    """Discover all example directories with .example.yaml files."""
    examples_dir = Path("examples")
    examples = []
    for yaml_file in examples_dir.glob("*/.example.yaml"):
        examples.append(yaml_file.parent)
    return sorted(examples)


def get_store_config_override(example_dir: Path, tmp_path: Path) -> dict[str, str]:
    """Determine the appropriate env override based on the metadata store type."""
    metaxy_toml = example_dir / "metaxy.toml"
    if not metaxy_toml.exists():
        return {}

    config = tomllib.loads(metaxy_toml.read_text())
    store_type = config.get("stores", {}).get("dev", {}).get("type", "")

    if "DuckDBMetadataStore" in store_type:
        test_db = tmp_path / f"{example_dir.name}.db"
        return {"METAXY_STORES__DEV__CONFIG__DATABASE": str(test_db)}
    elif "DeltaMetadataStore" in store_type:
        test_storage = tmp_path / example_dir.name
        return {"METAXY_STORES__DEV__CONFIG__ROOT_PATH": str(test_storage)}

    return {}


@pytest.mark.parametrize("example_dir", discover_examples(), ids=lambda p: p.name)
def test_example_snapshot(example_dir: Path, tmp_path: Path, example_snapshot):
    """Execute runbook, save raw results, and verify snapshot matches."""
    from metaxy._testing.runbook import RunbookRunner

    # Get store-specific env overrides
    store_overrides = get_store_config_override(example_dir, tmp_path)

    # Run the runbook with deterministic random seed
    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        env_overrides={
            **store_overrides,
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
