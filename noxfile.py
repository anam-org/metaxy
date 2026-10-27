import os
import shutil
from pathlib import Path

import nox
import nox_uv

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"

INTEGRATIONS = [
    "duckdb",
    "clickhouse",
    "postgres",
    "delta",
    "iceberg",
    "lancedb",
    "bigquery",
    "dagster",
    "ray",
    "sqlalchemy",
    "sqlmodel",
    "mcp",
]

IBIS_VERSIONS = ["11", "12"]
SETTINGS_VERSIONS = ["2.11.0", "2.13.1", "latest"]

COVERAGE_SUBPROCESS_PTH = Path(".github/coverage_subprocess.pth")


def _install_coverage_subprocess(session: nox.Session) -> None:
    """Install coverage subprocess support in the nox venv when running under CI coverage."""
    if not os.environ.get("COVERAGE_PROCESS_START") or not COVERAGE_SUBPROCESS_PTH.exists():
        return
    site_packages = session.run(
        "python",
        "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))",
        silent=True,
    )
    assert site_packages is not None
    shutil.copy(COVERAGE_SUBPROCESS_PTH, site_packages.strip())


@nox_uv.session(
    python="3.10",
    uv_extras=["duckdb"],
    uv_all_groups=True,
    uv_sync_locked=False,
    tags=["integration"],
)
@nox.parametrize("integration", INTEGRATIONS)
def integration(session: nox.Session, integration: str) -> None:
    """Run tests for a single integration."""
    coverage_enabled = bool(os.environ.get("COVERAGE_PROCESS_START"))
    pytest_cov_enabled = any(arg.startswith("--cov") for arg in session.posargs)
    if coverage_enabled and pytest_cov_enabled:
        session.error("Coverage is enabled by COVERAGE_PROCESS_START; remove --cov arguments.")
    session.install("-e", f".[{integration}]")
    _install_coverage_subprocess(session)
    runner = ("coverage", "run", "-m", "pytest") if coverage_enabled else ("pytest",)
    session.run(*runner, f"tests/ext/{integration}/", "--durations=10", *session.posargs)
    if coverage_enabled or pytest_cov_enabled:
        session.run("coverage", "combine", "--keep", success_codes=[0, 1])
        session.run("coverage", "xml", "-o", "coverage.xml")


@nox_uv.session(
    python="3.10",
    uv_all_extras=True,
    uv_all_groups=True,
    uv_sync_locked=False,
    tags=["compat"],
)
@nox.parametrize("version", IBIS_VERSIONS)
def ibis(session: nox.Session, version: str) -> None:
    """Test ibis compatibility across major versions."""
    session.install(f"ibis-framework[duckdb,clickhouse,bigquery]=={version}.*")
    session.run("pytest", "-m", "ibis", "--durations=10")


@nox_uv.session(
    python="3.12",
    uv_extras=["duckdb", "delta", "sqlalchemy", "map"],
    uv_groups=["dev"],
    uv_sync_locked=False,
    tags=["compat", "settings-compat"],
)
@nox.parametrize("version", SETTINGS_VERSIONS)
def settings(session: nox.Session, version: str) -> None:
    """Test pydantic-settings compatibility at the supported version boundaries."""
    if version == "latest":
        session.install("--upgrade", "pydantic-settings")
    else:
        session.install(f"pydantic-settings=={version}")
    # These tests do not use PostgreSQL fixtures.
    session.run(
        "pytest",
        "-p",
        "no:tach",
        "-p",
        "no:pytest_postgresql",
        "tests/test_config.py",
        "tests/cli/test_cli_list.py",
        *session.posargs,
    )
