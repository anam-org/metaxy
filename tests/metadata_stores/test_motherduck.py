"""Live MotherDuck integration tests.

Requires MOTHERDUCK_TOKEN environment variable. Skipped automatically when not set.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from metaxy._utils import collect_to_polars
from metaxy.ext.metadata_stores._ducklake_support import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
)
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore

pytestmark = pytest.mark.motherduck


def test_motherduck_direct_connection(motherduck_token: str, motherduck_database: str) -> None:
    """DuckDBMetadataStore should connect to MotherDuck directly and list tables."""
    store = DuckDBMetadataStore(database=f"md:{motherduck_database}?motherduck_token={motherduck_token}")

    with store:
        raw_conn = store._duckdb_raw_connection()
        tables = raw_conn.execute("SHOW TABLES").fetchall()
        assert isinstance(tables, list)


def test_motherduck_ducklake_attachment(
    motherduck_ducklake_database: str,
    motherduck_region: str | None,
) -> None:
    """Fully managed MotherDuck DuckLake should USE the database directly."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {
                "type": "motherduck",
                "database": motherduck_ducklake_database,
                "region": motherduck_region,
            },
            "alias": "md_lake",
        }
    )

    commands = DuckLakeAttachmentManager(config).preview_sql()
    expected = []
    if motherduck_region is not None:
        expected.append(f"SET s3_region='{motherduck_region}';")
    expected.append(f"USE {motherduck_ducklake_database};")
    assert commands == expected


def test_motherduck_ducklake_write_read_roundtrip(
    test_features,
    motherduck_token: str,
    motherduck_ducklake_database: str,
    motherduck_region: str | None,
) -> None:
    """Write metadata to MotherDuck-backed DuckLake and read it back."""
    store = DuckDBMetadataStore(
        database=f"md:?motherduck_token={motherduck_token}",
        ducklake=DuckLakeAttachmentConfig.model_validate(
            {
                "metadata_backend": {
                    "type": "motherduck",
                    "database": motherduck_ducklake_database,
                    "region": motherduck_region,
                },
            }
        ),
    )

    feature = test_features["UpstreamFeatureA"]
    payload = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"frames": "hash_f1", "audio": "hash_a1"},
                {"frames": "hash_f2", "audio": "hash_a2"},
                {"frames": "hash_f3", "audio": "hash_a3"},
            ],
        }
    )

    with store.open("w"):
        store.write(feature, payload)
        result = collect_to_polars(store.read(feature))
        actual = result.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"])
        assert_frame_equal(actual, payload.sort("sample_uid"))
