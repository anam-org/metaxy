"""Tests for DuckLake integration via DuckDB metadata store."""

import tempfile
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from polars.testing.parametric import column, dataframes

from metaxy._utils import collect_to_polars
from metaxy.ext.metadata_stores._ducklake_support import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    GCSStorageBackendConfig,
    MotherDuckMetadataBackendConfig,
    PostgresMetadataBackendConfig,
    R2StorageBackendConfig,
    S3StorageBackendConfig,
    format_attach_options,
)
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore


# Hypothesis strategies for generating metadata dataframes using Polars parametric testing
@st.composite
def metadata_dataframe_strategy(draw, fields=None, size=None):
    """Generate a metadata DataFrame using Polars parametric testing.

    Args:
        draw: Hypothesis draw function
        fields: List of field names for the provenance_by_field struct (default: ["default"])
        size: Number of rows (default: between 1-10)
    """
    if fields is None:
        fields = ["default"]
    if size is None:
        size = draw(st.integers(min_value=1, max_value=10))

    # Define struct type for provenance_by_field
    provenance_dtype = pl.Struct({field: pl.String for field in fields})

    # Generate dataframe with sample_uid and provenance_by_field columns
    df_strategy = dataframes(
        [
            column("sample_uid", dtype=pl.Int64),
            column("metaxy_provenance_by_field", dtype=provenance_dtype),
        ],
        size=size,
    )

    df = draw(df_strategy)

    # Ensure sample_uid is unique and sorted (required for test logic)
    df = df.with_columns(pl.Series("sample_uid", range(1, len(df) + 1)))

    return df


def test_ducklake_attachment_sequence() -> None:
    """DuckLakeAttachmentManager should issue expected setup statements."""
    attachment = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {
                "type": "postgres",
                "secret_name": "pg_meta",
                "database": "ducklake_meta",
                "user": "ducklake",
                "password": "secret",
                "host": "localhost",
            },
            "storage_backend": {
                "type": "s3",
                "secret_name": "s3_store",
                "key_id": "key",
                "secret": "secret",
                "endpoint": "https://object-store",
                "bucket": "ducklake",
            },
            "alias": "lake",
            "attach_options": {"api_version": "0.2", "override_data_path": True},
        }
    )

    manager = DuckLakeAttachmentManager(attachment)
    commands = manager.preview_sql()

    assert commands[-2].startswith("ATTACH IF NOT EXISTS 'ducklake:metaxy_generated_lake'")
    assert commands[-1] == "USE lake;"

    options_clause = format_attach_options(attachment.attach_options)
    assert options_clause == " (API_VERSION '0.2', OVERRIDE_DATA_PATH true)"


def test_ducklake_data_inlining_option() -> None:
    """ATTACH SQL should include DATA_INLINING_ROW_LIMIT when set."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "duckdb", "uri": "/tmp/cat.duckdb"},
            "storage_backend": {"type": "local", "path": "/tmp/data"},
            "data_inlining_row_limit": 100,
        }
    )
    commands = DuckLakeAttachmentManager(config).preview_sql()
    attach_stmt = [s for s in commands if s.startswith("ATTACH")][0]
    assert "DATA_INLINING_ROW_LIMIT 100" in attach_stmt


def test_ducklake_data_inlining_absent_when_not_set() -> None:
    """ATTACH SQL should not include DATA_INLINING_ROW_LIMIT when not set."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "duckdb", "uri": "/tmp/cat.duckdb"},
            "storage_backend": {"type": "local", "path": "/tmp/data"},
        }
    )
    commands = DuckLakeAttachmentManager(config).preview_sql()
    attach_stmt = [s for s in commands if s.startswith("ATTACH")][0]
    assert "DATA_INLINING_ROW_LIMIT" not in attach_stmt


def test_format_attach_options_handles_types() -> None:
    """_format_attach_options should stringify values similarly to DuckLake resource."""
    options = {
        "api_version": "0.2",
        "override_data_path": True,
        "max_retries": 3,
        "skip": None,
    }

    clause = format_attach_options(options)
    assert clause == " (API_VERSION '0.2', MAX_RETRIES 3, OVERRIDE_DATA_PATH true)"


# ---------------------------------------------------------------------------
# MotherDuck DuckLake tests
# ---------------------------------------------------------------------------


def test_motherduck_attachment_sequence() -> None:
    """Fully managed MotherDuck DuckLake should USE the database directly."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "motherduck", "database": "my_lake"},
            "alias": "lake",
        }
    )

    commands = DuckLakeAttachmentManager(config).preview_sql()

    assert commands == ["USE my_lake;"]


def test_motherduck_attachment_with_region() -> None:
    """MotherDuck DuckLake with explicit region should SET s3_region before USE."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "motherduck", "database": "my_lake", "region": "eu-central-1"},
            "alias": "lake",
        }
    )

    commands = DuckLakeAttachmentManager(config).preview_sql()

    assert commands == ["SET s3_region='eu-central-1';", "USE my_lake;"]


def test_motherduck_does_not_require_storage_backend() -> None:
    """MotherDuck metadata backend should validate without a storage_backend."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "motherduck", "database": "my_lake"},
        }
    )

    assert isinstance(config.metadata_backend, MotherDuckMetadataBackendConfig)
    assert config.storage_backend is None


def test_non_motherduck_requires_storage_backend() -> None:
    """Non-MotherDuck metadata backends should raise ValueError when storage_backend is missing."""
    for backend in [
        {"type": "duckdb", "uri": "/tmp/meta.duckdb"},
        {"type": "sqlite", "uri": "/tmp/meta.sqlite"},
        {"type": "postgres", "secret_name": "pg", "database": "db", "user": "u", "password": "p", "host": "localhost"},
    ]:
        with pytest.raises(ValueError, match="storage_backend is required"):
            DuckLakeAttachmentConfig.model_validate({"metadata_backend": backend})


def test_duckdb_store_accepts_motherduck_ducklake_config() -> None:
    """DuckDBMetadataStore should accept MotherDuck DuckLake config and include ducklake extension."""
    store = DuckDBMetadataStore(
        database=":memory:",
        ducklake=DuckLakeAttachmentConfig.model_validate(
            {"metadata_backend": {"type": "motherduck", "database": "my_lake"}, "alias": "lake"}
        ),
    )

    assert "ducklake" in [ext.name for ext in store.extensions]
    assert "motherduck" in [ext.name for ext in store.extensions]
    commands = store.preview_ducklake_sql()
    assert commands == ["USE my_lake;"]


def _ducklake_config_payload() -> dict[str, object]:
    return {
        "metadata_backend": {
            "type": "postgres",
            "secret_name": "pg_meta",
            "database": "ducklake_meta",
            "user": "ducklake",
            "password": "secret",
            "host": "localhost",
        },
        "storage_backend": {
            "type": "s3",
            "secret_name": "s3_store",
            "key_id": "key",
            "secret": "secret",
            "endpoint": "https://object-store",
            "bucket": "ducklake",
        },
        "attach_options": {"override_data_path": True},
    }


def test_duckdb_store_accepts_ducklake_config() -> None:
    """DuckDBMetadataStore should accept ducklake configuration inline."""
    store = DuckDBMetadataStore(
        database=":memory:",
        extensions=["json"],
        ducklake=DuckLakeAttachmentConfig.model_validate(_ducklake_config_payload()),
    )

    assert "ducklake" in [ext.name for ext in store.extensions]
    commands = store.preview_ducklake_sql()
    assert commands[-1] == "USE ducklake;"


def test_duckdb_store_preview_via_config_manager() -> None:
    """DuckDBMetadataStore exposes attachment manager helpers when configured."""
    store = DuckDBMetadataStore(
        database=":memory:",
        ducklake=DuckLakeAttachmentConfig.model_validate(_ducklake_config_payload()),
    )

    manager = store.ducklake_attachment
    preview = manager.preview_sql()
    assert preview[-1] == "USE ducklake;"


# Module-level state for tracking recorded commands in tests
_test_recorded_commands: list[str] = []


@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(size=st.integers(min_value=1, max_value=5))
def test_ducklake_store_read_write_roundtrip(test_features, monkeypatch, size) -> None:
    """DuckLake-configured store should still support read/write API.

    Uses hypothesis to generate test payloads with varying sizes and data.
    Uses tempfile to avoid data persistence across hypothesis examples.
    """
    # Use module-level list to capture commands
    global _test_recorded_commands

    # Set up monkeypatch once per test function
    if not _test_recorded_commands:
        original_preview = DuckLakeAttachmentManager.preview_sql

        def fake_configure(self, conn):
            _test_recorded_commands.extend(original_preview(self))

        monkeypatch.setattr(DuckLakeAttachmentManager, "configure", fake_configure)

    # Clear commands from previous hypothesis example
    _test_recorded_commands.clear()

    # Use tempfile to get fresh directory for each hypothesis example
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "ducklake_roundtrip.duckdb"
        metadata_path = tmp_path / "ducklake_catalog.duckdb"
        storage_dir = tmp_path / "ducklake_storage"

        ducklake_config = DuckLakeAttachmentConfig.model_validate(
            {
                "alias": "lake",
                "metadata_backend": {"type": "duckdb", "uri": str(metadata_path)},
                "storage_backend": {"type": "local", "path": str(storage_dir)},
            }
        )

        feature = test_features["UpstreamFeatureA"]

        # Generate payload using hypothesis size parameter
        sample_uids = list(range(1, size + 1))
        payload = pl.DataFrame(
            {
                "sample_uid": sample_uids,
                "metaxy_provenance_by_field": [
                    {"frames": f"hash_frames_{i}", "audio": f"hash_audio_{i}"} for i in sample_uids
                ],
            }
        )

        store = DuckDBMetadataStore(
            database=db_path,
            extensions=["json"],
            ducklake=ducklake_config,
        )

        # Explicit write mode for clarity (bare `with store:` defaults to auto_create_tables env var)
        with store.open("w"):
            store.write(feature, payload)
            result = collect_to_polars(store.read(feature))
            actual = result.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"])
            expected = payload.sort("sample_uid")
            assert_frame_equal(actual, expected)

        assert any(cmd.startswith("ATTACH IF NOT EXISTS 'ducklake:") for cmd in _test_recorded_commands)
        assert _test_recorded_commands[-1] == "USE lake;"


@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    num_samples=st.integers(min_value=1, max_value=5),
)
def test_ducklake_e2e_with_dependencies(test_graph, test_features, num_samples) -> None:
    """Real end-to-end integration test for DuckLake with DuckDB catalog and local filesystem storage.

    This is a real integration test that actually uses the DuckLake extension (no mocking).
    Uses hypothesis to generate test data with varying sample counts.

    This test exercises the full workflow:
    1. Write metadata for upstream features
    2. Write metadata for downstream features with dependencies
    3. Read metadata back and verify field provenances
    4. Test metadata updates and versioning
    5. Verify persistence across store reopening
    """
    # Extract graph from test_graph fixture
    graph = test_graph

    # Activate the test graph for list_features() calls
    with graph.use():
        # Create temp directory manually (hypothesis doesn't work with function-scoped fixtures)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Setup paths for DuckLake catalog and storage
            db_path = tmp_path / "ducklake_e2e.duckdb"
            metadata_path = tmp_path / "ducklake_catalog.duckdb"
            storage_dir = tmp_path / "ducklake_storage"

            ducklake_config = DuckLakeAttachmentConfig.model_validate(
                {
                    "alias": "e2e_lake",
                    "metadata_backend": {"type": "duckdb", "uri": str(metadata_path)},
                    "storage_backend": {"type": "local", "path": str(storage_dir)},
                    "attach_options": {"override_data_path": True},
                }
            )

            # Create store - this will actually attach DuckLake (no mocking)
            store = DuckDBMetadataStore(
                database=db_path,
                extensions=["json"],
                ducklake=ducklake_config,
            )

            # Test 1: Write upstream feature metadata
            upstream_a = test_features["UpstreamFeatureA"]
            upstream_b = test_features["UpstreamFeatureB"]
            downstream = test_features["DownstreamFeature"]

            # Generate test data using hypothesis-driven sample count
            sample_uids = list(range(1, num_samples + 1))

            upstream_a_data = pl.DataFrame(
                {
                    "sample_uid": sample_uids,
                    "metaxy_provenance_by_field": [
                        {"frames": f"hash_frames_{i}", "audio": f"hash_audio_{i}"} for i in sample_uids
                    ],
                }
            )

            upstream_b_data = pl.DataFrame(
                {
                    "sample_uid": sample_uids,
                    "metaxy_provenance_by_field": [{"default": f"hash_b_{i}"} for i in sample_uids],
                }
            )

            with store.open("w"):
                # Write upstream features
                store.write(upstream_a, upstream_a_data)
                store.write(upstream_b, upstream_b_data)

                # Verify upstream features can be read back
                result_a = collect_to_polars(store.read(upstream_a))
                result_b = collect_to_polars(store.read(upstream_b))

                assert_frame_equal(
                    result_a.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"]),
                    upstream_a_data.sort("sample_uid"),
                )
                assert_frame_equal(
                    result_b.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"]),
                    upstream_b_data.sort("sample_uid"),
                )

                # Test 2: Write downstream feature with dependencies
                downstream_data = pl.DataFrame(
                    {
                        "sample_uid": sample_uids,
                        "metaxy_provenance_by_field": [{"default": f"hash_d_{i}"} for i in sample_uids],
                    }
                )

                store.write(downstream, downstream_data)

                # Verify downstream feature can be read back
                result_d = collect_to_polars(store.read(downstream))
                assert_frame_equal(
                    result_d.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"]),
                    downstream_data.sort("sample_uid"),
                )

                # Verify DuckLake tracks data files (not plain DuckDB)
                raw_conn = store._duckdb_raw_connection()
                for feat in (upstream_a, upstream_b, downstream):
                    tbl = store.get_table_name(feat.spec.key)
                    files = raw_conn.execute(f"FROM ducklake_list_files('e2e_lake', '{tbl}')").fetchall()
                    assert files, f"ducklake_list_files should return data files for {tbl}"

                # Test 3: List features from active graph
                features_list = graph.list_features()
                assert len(features_list) == 3
                feature_keys = {fk.to_string() for fk in features_list}
                assert upstream_a.spec.key.to_string() in feature_keys
                assert upstream_b.spec.key.to_string() in feature_keys
                assert downstream.spec.key.to_string() in feature_keys

                # Test 4: Update metadata (append-only write)
                # Metaxy uses immutable, append-only metadata storage
                new_sample_uid = num_samples + 1
                updated_upstream_a = pl.DataFrame(
                    {
                        "sample_uid": [new_sample_uid],  # Add just a new sample
                        "metaxy_provenance_by_field": [
                            {
                                "frames": f"hash_frames_{new_sample_uid}",
                                "audio": f"hash_audio_{new_sample_uid}",
                            },
                        ],
                    }
                )

                store.write(upstream_a, updated_upstream_a)

                # Verify updated metadata - should have num_samples + 1 total
                result_updated = collect_to_polars(store.read(upstream_a))
                assert len(result_updated) == num_samples + 1
                assert set(result_updated["sample_uid"].to_list()) == set(range(1, num_samples + 2))

            # Test 5: Verify persistence by reopening the store
            # Reopen store and verify data persisted through DuckLake
            store2 = DuckDBMetadataStore(
                database=db_path,
                extensions=["json"],
                ducklake=ducklake_config,
            )

            with store2.open("w"):
                # Verify we can still read all features after reopening
                result_a2 = collect_to_polars(store2.read(upstream_a))
                assert len(result_a2) == num_samples + 1  # Original samples + 1 appended
                assert set(result_a2["sample_uid"].to_list()) == set(range(1, num_samples + 2))

                result_b2 = collect_to_polars(store2.read(upstream_b))
                assert len(result_b2) == num_samples

                result_d2 = collect_to_polars(store2.read(downstream))
                assert len(result_d2) == num_samples

                # Verify feature list persists (from active graph)
                features_list2 = graph.list_features()
                assert len(features_list2) == 3

            # Verify physical artifacts on disk
            assert metadata_path.exists(), "DuckLake metadata catalog should exist"
            parquet_files = list(storage_dir.rglob("*.parquet"))
            assert parquet_files, f"DuckLake should write parquet files to {storage_dir}"


# ---------------------------------------------------------------------------
# S3 secret_parameters tests
# ---------------------------------------------------------------------------


def test_s3_storage_with_secret_parameters() -> None:
    """S3 secret_parameters should be merged into the secret SQL."""
    config = S3StorageBackendConfig(
        type="s3",
        secret_name="my_s3",
        key_id="key",
        secret="secret",
        bucket="my-bucket",
        secret_parameters={"chain": "credential_chain", "kms_key_id": "my-kms-key"},
    )
    secret_sql, _ = config.sql_parts("test")
    assert "CHAIN 'credential_chain'" in secret_sql
    assert "KMS_KEY_ID 'my-kms-key'" in secret_sql
    assert "TYPE S3" in secret_sql


def test_s3_storage_credential_chain() -> None:
    """S3 with credential_chain via secret_parameters should include PROVIDER and omit KEY_ID."""
    config = S3StorageBackendConfig(
        type="s3",
        secret_name="my_s3",
        bucket="my-bucket",
        secret_parameters={"provider": "credential_chain"},
    )
    secret_sql, _ = config.sql_parts("test")
    assert "PROVIDER 'credential_chain'" in secret_sql
    assert "KEY_ID" not in secret_sql


# ---------------------------------------------------------------------------
# S3 integration test (moto)
# ---------------------------------------------------------------------------


@pytest.mark.ducklake
@pytest.mark.duckdb
def test_ducklake_s3_storage_roundtrip(
    test_features,
    s3_bucket_and_storage_options: tuple[str, dict],
) -> None:
    """DuckLake with S3 storage backend should accept S3 secret config and support read/write roundtrip."""
    bucket_name, storage_options = s3_bucket_and_storage_options

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "ducklake_s3.duckdb"
        metadata_path = tmp_path / "ducklake_catalog.duckdb"

        ducklake_config = DuckLakeAttachmentConfig.model_validate(
            {
                "alias": "s3_lake",
                "metadata_backend": {"type": "duckdb", "uri": str(metadata_path)},
                "storage_backend": {
                    "type": "s3",
                    "secret_name": "s3_test",
                    "key_id": storage_options["AWS_ACCESS_KEY_ID"],
                    "secret": storage_options["AWS_SECRET_ACCESS_KEY"],
                    "endpoint": storage_options["AWS_ENDPOINT_URL"],
                    "region": storage_options["AWS_REGION"],
                    "bucket": bucket_name,
                    "url_style": "path",
                    "use_ssl": False,
                },
                "attach_options": {"override_data_path": True},
            }
        )

        feature = test_features["UpstreamFeatureA"]
        payload = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "hash_a", "audio": "hash_b"},
                    {"frames": "hash_c", "audio": "hash_d"},
                    {"frames": "hash_e", "audio": "hash_f"},
                ],
            }
        )

        store = DuckDBMetadataStore(
            database=db_path,
            extensions=["json"],
            ducklake=ducklake_config,
        )

        with store.open("w"):
            store.write(feature, payload)
            result = collect_to_polars(store.read(feature))
            actual = result.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"])
            assert_frame_equal(actual, payload.sort("sample_uid"))

            # Verify DuckLake tracks data files for the written table
            table_name = store.get_table_name(feature.spec.key)
            raw_conn = store._duckdb_raw_connection()
            files = raw_conn.execute(f"FROM ducklake_list_files('s3_lake', '{table_name}')").fetchall()
            assert files, f"ducklake_list_files should return data files for {table_name}"
            assert all(f[0].endswith(".parquet") for f in files), "DuckLake data files should be parquet"

        assert metadata_path.exists(), f"DuckLake metadata catalog should exist at {metadata_path}"


# ---------------------------------------------------------------------------
# R2 storage backend tests
# ---------------------------------------------------------------------------


def test_r2_storage_with_explicit_auth() -> None:
    """R2 with explicit auth should include TYPE R2, ACCOUNT_ID, KEY_ID, and SECRET."""
    config = R2StorageBackendConfig(
        type="r2",
        secret_name="my_r2",
        key_id="r2key",
        secret="r2secret",
        account_id="my-account-id",
        data_path="r2://my-bucket/data/",
    )
    secret_sql, data_part = config.sql_parts("test")
    assert "TYPE R2" in secret_sql
    assert "ACCOUNT_ID 'my-account-id'" in secret_sql
    assert "KEY_ID 'r2key'" in secret_sql
    assert "SECRET 'r2secret'" in secret_sql
    assert data_part == "DATA_PATH 'r2://my-bucket/data/'"


def test_r2_storage_credential_chain() -> None:
    """R2 with credential_chain via secret_parameters should include PROVIDER and ACCOUNT_ID."""
    config = R2StorageBackendConfig(
        type="r2",
        secret_name="my_r2",
        account_id="my-account-id",
        data_path="r2://my-bucket/data/",
        secret_parameters={"provider": "credential_chain"},
    )
    secret_sql, _ = config.sql_parts("test")
    assert "TYPE R2" in secret_sql
    assert "PROVIDER 'credential_chain'" in secret_sql
    assert "ACCOUNT_ID 'my-account-id'" in secret_sql
    assert "KEY_ID" not in secret_sql


# ---------------------------------------------------------------------------
# GCS storage backend tests
# ---------------------------------------------------------------------------


def test_gcs_storage_with_explicit_auth() -> None:
    """GCS with explicit auth should include TYPE GCS, KEY_ID, and SECRET."""
    config = GCSStorageBackendConfig(
        type="gcs",
        secret_name="my_gcs",
        key_id="gcskey",
        secret="gcssecret",
        data_path="gs://my-bucket/data/",
    )
    secret_sql, data_part = config.sql_parts("test")
    assert "TYPE GCS" in secret_sql
    assert "KEY_ID 'gcskey'" in secret_sql
    assert "SECRET 'gcssecret'" in secret_sql
    assert data_part == "DATA_PATH 'gs://my-bucket/data/'"


def test_gcs_storage_credential_chain() -> None:
    """GCS with credential_chain via secret_parameters should include PROVIDER and omit KEY_ID."""
    config = GCSStorageBackendConfig(
        type="gcs",
        secret_name="my_gcs",
        data_path="gs://my-bucket/data/",
        secret_parameters={"provider": "credential_chain"},
    )
    secret_sql, _ = config.sql_parts("test")
    assert "TYPE GCS" in secret_sql
    assert "PROVIDER 'credential_chain'" in secret_sql
    assert "KEY_ID" not in secret_sql


def test_non_motherduck_accepts_r2_and_gcs_storage() -> None:
    """R2 and GCS storage backends should produce valid DUCKLAKE secret SQL."""
    for storage_config in [
        {
            "type": "r2",
            "secret_name": "r2_store",
            "key_id": "key",
            "secret": "secret",
            "account_id": "acct-123",
            "data_path": "r2://bucket/prefix/",
        },
        {
            "type": "gcs",
            "secret_name": "gcs_store",
            "key_id": "key",
            "secret": "secret",
            "data_path": "gs://bucket/prefix/",
        },
    ]:
        config = DuckLakeAttachmentConfig.model_validate(
            {
                "metadata_backend": {"type": "duckdb", "uri": "/tmp/meta.duckdb"},
                "storage_backend": storage_config,
                "alias": "lake",
            }
        )
        commands = DuckLakeAttachmentManager(config).preview_sql()
        ducklake_secret_sql = [c for c in commands if "TYPE DUCKLAKE" in c]
        assert len(ducklake_secret_sql) == 1
        assert "DATA_PATH" in ducklake_secret_sql[0]


# ---------------------------------------------------------------------------
# secret_name tests
# ---------------------------------------------------------------------------


def test_postgres_with_secret_name_skips_create_secret() -> None:
    """PostgreSQL with secret_name should return empty secret SQL and reference the user-provided name."""
    config = PostgresMetadataBackendConfig(type="postgres", secret_name="my_pg_secret")
    secret_sql, metadata_params = config.sql_parts("lake")
    assert secret_sql == ""
    assert "'my_pg_secret'" in metadata_params
    assert "TYPE" in metadata_params and "postgres" in metadata_params


def test_postgres_secret_name_with_inline_creates_named_secret() -> None:
    """PostgreSQL with secret_name + inline credentials should CREATE SECRET with the user-provided name."""
    config = PostgresMetadataBackendConfig(
        type="postgres",
        secret_name="my_pg",
        host="localhost",
        database="db",
        user="u",
        password="p",
    )
    secret_sql, metadata_params = config.sql_parts("lake")
    assert secret_sql.startswith("CREATE OR REPLACE SECRET my_pg")
    assert "TYPE postgres" in secret_sql
    assert "'my_pg'" in metadata_params


def test_postgres_inline_requires_all_credentials() -> None:
    """PostgreSQL with partial inline credentials should require all of them."""
    with pytest.raises(ValueError, match="Missing required inline credentials"):
        PostgresMetadataBackendConfig(type="postgres", secret_name="pg", host="localhost", database="db")


def test_s3_with_secret_name_skips_create_secret() -> None:
    """S3 with secret_name only should return empty secret SQL but still include DATA_PATH."""
    config = S3StorageBackendConfig(type="s3", secret_name="my_s3_secret", bucket="my-bucket")
    secret_sql, data_part = config.sql_parts("lake")
    assert secret_sql == ""
    assert "DATA_PATH" in data_part
    assert "my-bucket" in data_part


def test_s3_secret_name_with_inline_creates_named_secret() -> None:
    """S3 with secret_name + inline credentials should CREATE SECRET with the user-provided name."""
    config = S3StorageBackendConfig(type="s3", secret_name="my_s3", key_id="key", secret="secret", bucket="b")
    secret_sql, _ = config.sql_parts("lake")
    assert secret_sql.startswith("CREATE OR REPLACE SECRET my_s3")
    assert "TYPE S3" in secret_sql
    assert "KEY_ID 'key'" in secret_sql


def test_r2_with_secret_name_skips_create_secret() -> None:
    """R2 with secret_name only should return empty secret SQL."""
    config = R2StorageBackendConfig(type="r2", secret_name="my_r2_secret", data_path="r2://bucket/data/")
    secret_sql, data_part = config.sql_parts("lake")
    assert secret_sql == ""
    assert "DATA_PATH" in data_part


def test_r2_inline_requires_account_id() -> None:
    """R2 with inline key_id/secret should require account_id."""
    with pytest.raises(ValueError, match="account_id"):
        R2StorageBackendConfig(
            type="r2", secret_name="my_r2", key_id="key", secret="secret", data_path="r2://bucket/data/"
        )


def test_gcs_with_secret_name_skips_create_secret() -> None:
    """GCS with secret_name only should return empty secret SQL."""
    config = GCSStorageBackendConfig(type="gcs", secret_name="my_gcs_secret", data_path="gs://bucket/data/")
    secret_sql, data_part = config.sql_parts("lake")
    assert secret_sql == ""
    assert "DATA_PATH" in data_part


def test_full_attachment_with_secret_names() -> None:
    """End-to-end: preview_sql() should have no CREATE SECRET for catalog/storage when using secret_name."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "postgres", "secret_name": "pg_catalog"},
            "storage_backend": {"type": "s3", "secret_name": "s3_storage", "bucket": "data-bucket"},
            "alias": "lake",
        }
    )
    commands = DuckLakeAttachmentManager(config).preview_sql()

    create_secret_cmds = [c for c in commands if c.startswith("CREATE OR REPLACE SECRET")]
    assert len(create_secret_cmds) == 1
    assert "TYPE DUCKLAKE" in create_secret_cmds[0]

    assert any("pg_catalog" in c for c in commands)
    assert any("ATTACH IF NOT EXISTS" in c for c in commands)
    assert commands[-1] == "USE lake;"


# ---------------------------------------------------------------------------
# secret_name required tests
# ---------------------------------------------------------------------------


def test_s3_without_secret_name_raises_validation_error() -> None:
    """S3 without secret_name should raise a Pydantic validation error."""
    with pytest.raises(ValueError, match="secret_name"):
        S3StorageBackendConfig.model_validate({"type": "s3", "bucket": "my-bucket"})


def test_r2_without_secret_name_raises_validation_error() -> None:
    """R2 without secret_name should raise a Pydantic validation error."""
    with pytest.raises(ValueError, match="secret_name"):
        R2StorageBackendConfig.model_validate({"type": "r2", "data_path": "r2://bucket/data/"})


def test_gcs_without_secret_name_raises_validation_error() -> None:
    """GCS without secret_name should raise a Pydantic validation error."""
    with pytest.raises(ValueError, match="secret_name"):
        GCSStorageBackendConfig.model_validate({"type": "gcs", "data_path": "gs://bucket/data/"})


def test_postgres_without_secret_name_raises_validation_error() -> None:
    """PostgreSQL without secret_name should raise a Pydantic validation error."""
    with pytest.raises(ValueError, match="secret_name"):
        PostgresMetadataBackendConfig.model_validate(
            {"type": "postgres", "host": "localhost", "database": "db", "user": "u", "password": "p"}
        )


# ---------------------------------------------------------------------------
# MotherDuck BYOB tests
# ---------------------------------------------------------------------------


def test_motherduck_byob_with_inline_credentials() -> None:
    """MotherDuck BYOB with inline S3 credentials should create secret IN MOTHERDUCK."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "motherduck", "database": "my_ducklake", "region": "eu-central-1"},
            "storage_backend": {
                "type": "s3",
                "secret_name": "my_s3_secret",
                "key_id": "AKIA...",
                "secret": "secret",
                "region": "eu-central-1",
                "scope": "s3://mybucket/",
                "bucket": "mybucket",
            },
        }
    )
    commands = DuckLakeAttachmentManager(config).preview_sql()

    assert commands[0] == "SET s3_region='eu-central-1';"
    assert commands[1].startswith("CREATE DATABASE IF NOT EXISTS my_ducklake")
    assert "TYPE DUCKLAKE" in commands[1]
    assert "DATA_PATH 's3://mybucket/'" in commands[1]
    assert "IN MOTHERDUCK" in commands[2]
    assert commands[2].startswith("CREATE OR REPLACE SECRET my_s3_secret IN MOTHERDUCK")
    assert "TYPE S3" in commands[2]
    assert commands[-1] == "USE my_ducklake;"


def test_motherduck_byob_use_existing_secret() -> None:
    """MotherDuck BYOB with secret_name only should not create a secret."""
    config = DuckLakeAttachmentConfig.model_validate(
        {
            "metadata_backend": {"type": "motherduck", "database": "my_ducklake"},
            "storage_backend": {
                "type": "s3",
                "secret_name": "my_s3_secret",
                "bucket": "mybucket",
            },
        }
    )
    commands = DuckLakeAttachmentManager(config).preview_sql()

    assert any("CREATE DATABASE IF NOT EXISTS my_ducklake" in c for c in commands)
    assert not any("CREATE OR REPLACE SECRET" in c for c in commands)
    assert commands[-1] == "USE my_ducklake;"
