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
    session.install("-e", f".[{integration}]")
    _install_coverage_subprocess(session)
    session.run("pytest", f"tests/ext/{integration}/", "--durations=10", *session.posargs)
    if any(arg.startswith("--cov") for arg in session.posargs):
        session.run("coverage", "combine", "--keep", success_codes=[0, 1])
        session.run("coverage", "xml", "-o", "coverage.xml")


@nox_uv.session(
    python="3.10",
    uv_extras=["duckdb"],
    uv_all_groups=True,
    uv_sync_locked=False,
    tags=["compat"],
)
@nox.parametrize("version", IBIS_VERSIONS)
def ibis(session: nox.Session, version: str) -> None:
    """Test ibis compatibility across major versions."""
    session.install(f"ibis-framework[duckdb,clickhouse,bigquery]=={version}.*")
    session.run("pytest", "-m", "ibis", "--durations=10")
