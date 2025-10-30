"""Tests for DuckLake example."""

import subprocess
import sys
from pathlib import Path


def test_ducklake_demo_preview():
    """Test that the DuckLake demo runs and previews attachment SQL."""
    example_dir = Path("examples/src/examples/ducklake")

    result = subprocess.run(
        [sys.executable, "-m", "examples.ducklake.demo"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=example_dir,
    )

    assert result.returncode == 0, (
        f"Demo failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)

    assert "DuckLake store initialised" in result.stdout
    assert "Preview of DuckLake ATTACH SQL:" in result.stdout
    assert "INSTALL" in result.stdout or "LOAD" in result.stdout
    assert "ATTACH" in result.stdout or "ducklake" in result.stdout.lower()
