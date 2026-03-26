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
    session.install(f".[{integration}]")
    session.run("pytest", f"tests/ext/{integration}/", "--durations=10", *session.posargs)


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
