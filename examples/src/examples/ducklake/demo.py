"""Interactive walkthrough for the DuckLake example.

The goal is to showcase how the DuckLake integration works after the Narwhals
migration without requiring a real DuckLake deployment. We focus on:

1. Attachment flow - the SQL emitted while configuring DuckLake.
2. Narwhals-powered join & diff components tailored for DuckLake catalogs.
3. Calculator configuration - how DuckLake ensures the right extensions.
"""

import narwhals as nw
import polars as pl

from metaxy.data_versioning.calculators.ducklake import DuckLakeDataVersionCalculator
from metaxy.data_versioning.diff.ducklake import DuckLakeDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.ducklake import DuckLakeJoiner
from metaxy.metadata_store.ducklake import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    DuckLakePostgresMetadataBackend,
    DuckLakeS3StorageBackend,
)
from metaxy.models.plan import FeaturePlan

from .features import DuckLakeChildFeature, DuckLakeParentFeature


def _to_polars(df) -> pl.DataFrame:
    """Convert a Narwhals DataFrame to Polars for display."""
    if hasattr(df, "to_polars"):
        return df.to_polars()
    if hasattr(df, "to_dicts"):
        return pl.DataFrame(df.to_dicts())
    raise AttributeError("Unable to convert Narwhals DataFrame to Polars")


class _StubCursor:
    """Records SQL commands for illustration purposes."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, command: str) -> None:
        self.commands.append(command.strip())

    def close(self) -> None:
        pass


class _StubConnection:
    """Matches the minimal DuckDB cursor API needed by DuckLakeAttachmentManager."""

    def __init__(self) -> None:
        self._cursor = _StubCursor()

    def cursor(self) -> _StubCursor:
        return self._cursor


class _LoggingBackend:
    """Lightweight stand-in for an Ibis backend that logs USE statements."""

    def __init__(self, alias: str) -> None:
        self.alias = alias
        self.commands: list[str] = []

    def raw_sql(self, command: str) -> None:
        self.commands.append(command.strip())
        print(f"[duckdb] {command.strip()}")


def _build_feature_plan() -> FeaturePlan:
    """Construct a FeaturePlan linking the child feature to its parent."""
    return FeaturePlan(
        feature=DuckLakeChildFeature.spec,
        deps=[DuckLakeParentFeature.spec],
    )


def show_attachment_sequence() -> None:
    """Display the DuckLake attachment SQL emitted during configuration."""
    print("🔧 DuckLake attachment manager")
    config = DuckLakeAttachmentConfig(
        metadata_backend=DuckLakePostgresMetadataBackend(
            database="catalog_db",
            user="ducklake",
            password="secret",
        ),
        storage_backend=DuckLakeS3StorageBackend(
            endpoint_url="https://object-store",
            bucket="demo-bucket",
            aws_access_key_id="env:DUCKLAKE_AWS_KEY",
            aws_secret_access_key="env:DUCKLAKE_AWS_SECRET",
            region="us-east-1",
        ),
        alias="demo_lake",
        plugins=["ducklake"],
        attach_options={"api_version": "0.2"},
    )

    manager = DuckLakeAttachmentManager(config)
    conn = _StubConnection()
    manager.configure(conn)

    print("Recorded SQL statements:")
    for statement in conn.cursor().commands:
        print(f"  {statement}")
    print()


def show_narwhals_join_and_diff() -> None:
    """Show how the Narwhals-based components behave for DuckLake."""
    print("🔗 Narwhals joiner and diff resolver")
    backend = _LoggingBackend(alias="demo_lake")
    joiner = DuckLakeJoiner(backend=backend, alias="demo_lake")

    # Build upstream references: parent feature data_versions in DuckLake.
    parent_df = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [
                {"ingested": "v1"},
                {"ingested": "v1"},
                {"ingested": "v2"},
            ],
        }
    ).lazy()

    upstream_refs = {
        DuckLakeParentFeature.spec.key.to_string(): nw.from_native(parent_df),
    }

    plan = _build_feature_plan()
    joined, mapping = joiner.join_upstream(
        upstream_refs,
        DuckLakeChildFeature.spec,
        plan,
    )

    print("Column mapping from joiner:")
    for upstream_key, column_name in mapping.items():
        print(f"  {upstream_key} -> {column_name}")

    joined_polars = _to_polars(joined.collect())
    print("\nJoined upstream view:")
    print(joined_polars)

    # Demonstrate the diff resolver with simple Narwhals frames.
    target_versions = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [1, 2, 3, 4],
                "data_version": [
                    {"metrics": "hash_a"},
                    {"metrics": "hash_b"},
                    {"metrics": "hash_c"},
                    {"metrics": "hash_new"},
                ],
            }
        ).lazy()
    )
    current_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"metrics": "hash_a"},
                    {"metrics": "hash_old"},
                    {"metrics": "hash_c"},
                ],
            }
        ).lazy()
    )

    diff_resolver = DuckLakeDiffResolver(backend=backend, alias="demo_lake")
    diff = diff_resolver.find_changes(target_versions, current_metadata)

    added = _to_polars(diff.added.collect())
    changed = _to_polars(diff.changed.collect())
    removed = _to_polars(diff.removed.collect())

    print("\nDiff summary:")
    print(f"  Added rows:\n{added}")
    print(f"  Changed rows:\n{changed}")
    print(f"  Removed rows:\n{removed}")
    print()


def show_calculator_configuration() -> None:
    """Highlight DuckLake calculator defaults and extension requirements."""
    print("🧮 DuckLake data version calculator")

    def _md5_sql(table, concat_columns: dict[str, str]) -> str:
        """Minimal hash SQL generator using MD5."""
        hash_selects: list[str] = []
        for field_key, concat_col in concat_columns.items():
            hash_selects.append(
                f"CAST(md5({concat_col}) AS VARCHAR) AS __hash_{field_key}"
            )
        select_clause = ", ".join(hash_selects)
        table_sql = table.compile()
        return f"SELECT *, {select_clause} FROM ({table_sql}) AS __metaxy_temp"

    calculator = DuckLakeDataVersionCalculator(
        hash_sql_generators={HashAlgorithm.MD5: _md5_sql},
        alias="demo_lake",
    )

    print(f"  Default algorithm: {calculator.default_algorithm}")
    print(f"  Supported algorithms: {calculator.supported_algorithms}")
    print(f"  Required extensions: {calculator.extensions}")
    print(
        "  (Call set_connection() with an Ibis backend before calculating data versions.)"
    )
    print()


def run_demo() -> None:
    """Run all DuckLake demo steps."""
    show_attachment_sequence()
    show_narwhals_join_and_diff()
    show_calculator_configuration()


if __name__ == "__main__":
    run_demo()
