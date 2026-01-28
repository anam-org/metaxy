"""Tests for DuckLake integration via DuckDB metadata store."""

import tempfile
from pathlib import Path

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from polars.testing.parametric import column, dataframes

from metaxy._utils import collect_to_polars
from metaxy.metadata_store._ducklake_support import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    _PreviewConnection,
    _PreviewCursor,
    format_attach_options,
)
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


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


class _StubCursor(_PreviewCursor):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


class _StubConnection(_PreviewConnection):
    def __init__(self) -> None:
        super().__init__()
        self._cursor = _StubCursor()

    def cursor(self) -> _StubCursor:
        return self._cursor  # ty: ignore[invalid-return-type]


def test_ducklake_attachment_sequence() -> None:
    """DuckLakeAttachmentManager should issue expected setup statements."""
    attachment = DuckLakeAttachmentConfig(
        metadata_backend={
            "type": "postgres",
            "database": "ducklake_meta",
            "user": "ducklake",
            "password": "secret",
        },
        storage_backend={
            "type": "s3",
            "endpoint_url": "https://object-store",
            "bucket": "ducklake",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
        },
        alias="lake",
        plugins=("ducklake",),
        attach_options={"api_version": "0.2", "override_data_path": True},
    )

    manager = DuckLakeAttachmentManager(attachment)
    conn = _StubConnection()

    manager.configure(conn)

    commands = conn.cursor().commands
    assert commands[0] == "INSTALL ducklake;"
    assert commands[1] == "LOAD ducklake;"
    assert commands[-2].startswith("ATTACH 'ducklake:secret_lake'")
    assert commands[-1] == "USE lake;"

    options_clause = format_attach_options(attachment.attach_options)
    assert options_clause == " (API_VERSION '0.2', OVERRIDE_DATA_PATH true)"


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


def _ducklake_config_payload() -> dict[str, object]:
    return {
        "metadata_backend": {
            "type": "postgres",
            "database": "ducklake_meta",
            "user": "ducklake",
            "password": "secret",
        },
        "storage_backend": {
            "type": "s3",
            "endpoint_url": "https://object-store",
            "bucket": "ducklake",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
        },
        "attach_options": {"override_data_path": True},
    }


def test_duckdb_store_accepts_ducklake_config() -> None:
    """DuckDBMetadataStore should accept ducklake configuration inline."""
    store = DuckDBMetadataStore(
        database=":memory:",
        extensions=["json"],
        ducklake=_ducklake_config_payload(),
    )

    assert "ducklake" in store.extensions
    commands = store.preview_ducklake_sql()
    assert commands[0] == "INSTALL ducklake;"
    assert commands[1] == "LOAD ducklake;"
    assert commands[-1] == "USE ducklake;"


def test_duckdb_store_preview_via_config_manager() -> None:
    """DuckDBMetadataStore exposes attachment manager helpers when configured."""
    store = DuckDBMetadataStore(
        database=":memory:",
        ducklake=_ducklake_config_payload(),
    )

    manager = store.ducklake_attachment
    preview = manager.preview_sql()
    assert preview[0] == "INSTALL ducklake;"
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
        original_configure = DuckLakeAttachmentManager.configure

        def fake_configure(self, conn):
            preview_conn = _PreviewConnection()
            original_configure(self, preview_conn)
            _test_recorded_commands.extend(preview_conn.cursor().commands)

        monkeypatch.setattr(DuckLakeAttachmentManager, "configure", fake_configure)

    # Clear commands from previous hypothesis example
    _test_recorded_commands.clear()

    # Use tempfile to get fresh directory for each hypothesis example
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "ducklake_roundtrip.duckdb"
        metadata_path = tmp_path / "ducklake_catalog.duckdb"
        storage_dir = tmp_path / "ducklake_storage"

        ducklake_config = {
            "alias": "lake",
            "metadata_backend": {"type": "duckdb", "path": str(metadata_path)},
            "storage_backend": {"type": "local", "path": str(storage_dir)},
        }

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

        with store:
            store.write_metadata(feature, payload)
            result = collect_to_polars(store.read_metadata(feature))
            actual = result.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"])
            expected = payload.sort("sample_uid")
            assert_frame_equal(actual, expected)

        assert _test_recorded_commands[:2] == ["INSTALL ducklake;", "LOAD ducklake;"]
        assert any(cmd.startswith("ATTACH 'ducklake:") for cmd in _test_recorded_commands)
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

            ducklake_config = {
                "alias": "e2e_lake",
                "metadata_backend": {"type": "duckdb", "path": str(metadata_path)},
                "storage_backend": {"type": "local", "path": str(storage_dir)},
                "attach_options": {"override_data_path": True},
            }

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

            with store:
                # Write upstream features
                store.write_metadata(upstream_a, upstream_a_data)
                store.write_metadata(upstream_b, upstream_b_data)

                # Verify upstream features can be read back
                result_a = collect_to_polars(store.read_metadata(upstream_a))
                result_b = collect_to_polars(store.read_metadata(upstream_b))

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

                store.write_metadata(downstream, downstream_data)

                # Verify downstream feature can be read back
                result_d = collect_to_polars(store.read_metadata(downstream))
                assert_frame_equal(
                    result_d.sort("sample_uid").select(["sample_uid", "metaxy_provenance_by_field"]),
                    downstream_data.sort("sample_uid"),
                )

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

                store.write_metadata(upstream_a, updated_upstream_a)

                # Verify updated metadata - should have num_samples + 1 total
                result_updated = collect_to_polars(store.read_metadata(upstream_a))
                assert len(result_updated) == num_samples + 1
                assert set(result_updated["sample_uid"].to_list()) == set(range(1, num_samples + 2))

            # Test 5: Verify persistence by reopening the store
            # Reopen store and verify data persisted through DuckLake
            store2 = DuckDBMetadataStore(
                database=db_path,
                extensions=["json"],
                ducklake=ducklake_config,
            )

            with store2:
                # Verify we can still read all features after reopening
                result_a2 = collect_to_polars(store2.read_metadata(upstream_a))
                assert len(result_a2) == num_samples + 1  # Original samples + 1 appended
                assert set(result_a2["sample_uid"].to_list()) == set(range(1, num_samples + 2))

                result_b2 = collect_to_polars(store2.read_metadata(upstream_b))
                assert len(result_b2) == num_samples

                result_d2 = collect_to_polars(store2.read_metadata(downstream))
                assert len(result_d2) == num_samples

                # Verify feature list persists (from active graph)
                features_list2 = graph.list_features()
                assert len(features_list2) == 3

            # Verify DuckLake catalog database exists (storage dir may not exist if no tables were created)
            assert metadata_path.exists(), "DuckLake catalog database should exist"
