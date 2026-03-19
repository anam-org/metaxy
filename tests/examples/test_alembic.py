"""Test the Alembic example in an isolated venv with duckdb 1.5.

example-alembic is excluded from the workspace so it can pin duckdb>=1.5.0
without affecting the rest of the project. The test creates its own venv
via ``uv sync`` and runs commands through ``uv run`` so the example's
pinned dependencies are used regardless of PATH or sys.executable.
"""

import shutil
import subprocess
from pathlib import Path

from metaxy_testing import RunbookRunner

_METAXY_ROOT = Path(__file__).resolve().parent.parent.parent


def test_alembic_runbook(tmp_path):
    """Test Alembic migration workflow with DuckLake on duckdb 1.5."""
    example_src = Path("examples/example-alembic")
    work_dir = tmp_path / "example-alembic"

    shutil.copytree(
        example_src,
        work_dir,
        ignore=shutil.ignore_patterns("__pycache__", ".venv", "*.pyc"),
    )

    # Rewrite relative metaxy source path to absolute for the temp copy.
    import tomli as tomllib
    import tomlkit

    pyproject = work_dir / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert config["tool"]["uv"]["sources"]["metaxy"]["path"] == "../..", (
        "Expected relative metaxy path in example pyproject.toml"
    )

    doc = tomlkit.loads(pyproject.read_text(encoding="utf-8"))
    doc["tool"]["uv"]["sources"]["metaxy"]["path"] = _METAXY_ROOT.as_posix()  # ty: ignore[not-subscriptable, invalid-assignment]
    pyproject.write_text(tomlkit.dumps(doc), encoding="utf-8")

    # Create isolated venv with duckdb>=1.5.0.
    subprocess.run(
        ["uv", "sync", "--no-dev"],
        cwd=work_dir,
        check=True,
        capture_output=True,
        timeout=120,
    )

    # Use from_yaml_file directly (not runner_for_project which would re-copy).
    runner = RunbookRunner.from_yaml_file(
        yaml_path=work_dir / ".example.yaml",
        env_overrides={
            "ALEMBIC_DEMO_DB": str(tmp_path / "alembic_demo.db"),
            "ALEMBIC_META_DB": str(tmp_path / "alembic_meta.db"),
            "ALEMBIC_STORAGE_PATH": str(tmp_path / "alembic_storage"),
            "METAXY_AUTO_CREATE_TABLES": "0",
            # Point VIRTUAL_ENV at the example's venv so uv doesn't emit
            # a mismatch warning containing the host's absolute path.
            "VIRTUAL_ENV": str(work_dir / ".venv"),
        },
    )
    runner.run()

    # Save execution state for docs tooling, sanitizing local paths.
    import json
    import re

    result_path = example_src / ".example.result.json"
    runner.runbook.save_execution_state(result_path)
    raw = result_path.read_text(encoding="utf-8")
    sanitized = re.sub(r"[A-Za-z:\\\/][^\s\"]*?[\\\/]example-alembic[\\\/]", "<workdir>/", raw)
    result_path.write_text(
        json.dumps(json.loads(sanitized), indent=2) + "\n",
        encoding="utf-8",
    )
