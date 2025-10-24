"""Tests for example scripts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_pipeline(tmp_path):
    """Test that the recompute example runs successfully."""
    # Set up environment with test-specific database path
    env = os.environ.copy()
    test_db = tmp_path / "example_recompute.db"
    env["METAXY_STORES__DEV__CONFIG__DATABASE"] = str(test_db)

    example_dir = Path("examples/src/examples/recompute")

    # Step 1: Setup upstream data
    result = subprocess.run(
        [sys.executable, "-m", "examples.recompute.setup_data"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Setup failed: {result.stderr}"
    print(result.stdout)

    # Step 2: Run pipeline with STAGE=1
    env["STAGE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "examples.recompute.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Stage 1 failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)
    assert "Pipeline STAGE=1" in result.stdout
    assert "✅ Stage 1 pipeline complete!" in result.stdout

    # Step 3: Run pipeline with STAGE=1 again (demonstrate idempotence)
    result = subprocess.run(
        [sys.executable, "-m", "examples.recompute.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Stage 1 (rerun) failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print("\n--- Stage 1 rerun (should be idempotent) ---")
    print(result.stdout)
    assert "Pipeline STAGE=1" in result.stdout
    # Should not recompute anything
    assert "No changes detected (idempotent)" in result.stdout

    # Step 4: Run pipeline with STAGE=2 (with updated algorithm)
    env["STAGE"] = "2"
    result = subprocess.run(
        [sys.executable, "-m", "examples.recompute.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Stage 2 failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)
    assert "Pipeline STAGE=2" in result.stdout
    assert "✅ Stage 2 pipeline complete!" in result.stdout
    # When feature_version changes, samples are "new" for that version (Materialized), not "changed" (Recomputed)
    assert (
        "Materialized" in result.stdout
        or "Recomputed" in result.stdout
        or "new samples" in result.stdout
        or "changed samples" in result.stdout
    ), "Child feature should recompute when parent algorithm changes"


@pytest.mark.parametrize("stage", ["1", "2"])
def test_list_features(snapshot, stage):
    """Snapshot test for 'metaxy list features' output at different stages.

    This captures the feature specifications at STAGE=1 and STAGE=2,
    demonstrating how feature versions change when code_version changes.
    """
    # Set up environment
    env = os.environ.copy()
    env["STAGE"] = stage

    # Run metaxy list features command
    result = subprocess.run(
        [sys.executable, "-m", "metaxy.cli.app", "list", "features"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        cwd=Path("examples/src/examples/recompute"),
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Snapshot the output
    assert result.stdout == snapshot
