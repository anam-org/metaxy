"""Test the DuckLake example using the runbook system."""

from pathlib import Path

from metaxy_testing import RunbookRunner


def test_ducklake_runbook(tmp_path):
    """Test DuckLake example output and table inspection flow."""
    example_dir = Path("examples/example-ducklake")

    with RunbookRunner.runner_for_project(
        example_dir=example_dir,
        env_overrides={
            "DUCKLAKE_DEMO_DB": str(tmp_path / "ducklake_demo.db"),
            "DUCKLAKE_META_DB": str(tmp_path / "ducklake_meta.db"),
            "DUCKLAKE_STORAGE_PATH": str(tmp_path / "ducklake_storage"),
        },
    ) as runner:
        runner.run()
