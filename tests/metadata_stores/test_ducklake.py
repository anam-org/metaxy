"""Tests for the DuckLake integration components."""

from metaxy.data_versioning.calculators.ducklake import DuckLakeDataVersionCalculator
from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.ducklake import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    DuckLakePostgresMetadataBackend,
    DuckLakeS3StorageBackend,
)


class _StubCursor:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False

    def execute(self, command: str) -> None:
        self.commands.append(command.strip())

    def close(self) -> None:
        self.closed = True


class _StubConnection:
    def __init__(self) -> None:
        self._cursor = _StubCursor()

    def cursor(self) -> _StubCursor:
        return self._cursor


def test_ducklake_attachment_sequence() -> None:
    """DuckLakeAttachmentManager issues commands in the expected order."""
    config = DuckLakeAttachmentConfig(
        metadata_backend=DuckLakePostgresMetadataBackend(
            database="ducklake_meta",
            user="ducklake",
            password="secret",
        ),
        storage_backend=DuckLakeS3StorageBackend(
            endpoint_url="https://object-store",
            bucket="ducklake",
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        ),
        alias="lake",
        plugins=["ducklake"],
        attach_options={"api_version": "0.2", "override_data_path": True},
    )

    manager = DuckLakeAttachmentManager(config)
    conn = _StubConnection()

    manager.configure(conn)

    commands = conn.cursor().commands
    assert commands[0] == "INSTALL ducklake;"
    assert commands[1] == "LOAD ducklake;"
    assert commands[-2].startswith("ATTACH 'ducklake:secret_lake'")
    assert commands[-1] == "USE lake;"

    attach_clause = manager._build_attach_options_clause()
    assert attach_clause == " (API_VERSION '0.2', OVERRIDE_DATA_PATH true)"


def test_ducklake_calculator_requires_extensions() -> None:
    """Calculator should always request ducklake + hashfuncs extensions."""

    def _dummy_hash_sql(*args, **kwargs) -> str:
        return "SELECT 1"

    hash_generators: dict[HashAlgorithm, HashSQLGenerator] = {
        HashAlgorithm.XXHASH64: _dummy_hash_sql
    }

    calc = DuckLakeDataVersionCalculator(
        hash_sql_generators=hash_generators, alias="demo"
    )
    assert calc.default_algorithm == HashAlgorithm.XXHASH64

    # extensions attribute defined on DuckDBDataVersionCalculator
    assert "ducklake" in calc.extensions
    assert "hashfuncs" in calc.extensions
