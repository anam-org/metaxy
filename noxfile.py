import nox
import nox_uv

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"

IBIS_VERSIONS = ["11", "12"]


@nox_uv.session(python="3.10", uv_all_extras=True, uv_all_groups=True, uv_sync_locked=False)
@nox.parametrize("version", IBIS_VERSIONS)
def ibis(session: nox.Session, version: str) -> None:
    """Test ibis compatibility across major versions."""
    session.install(f"ibis-framework[duckdb,clickhouse,bigquery]=={version}.*")
    session.run("pytest", "-m", "ibis", "--durations=10")
