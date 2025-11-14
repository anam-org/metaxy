"""Tests for metadata store."""

from collections.abc import Iterator

import narwhals as nw
import polars as pl
import pytest

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    SampleFeatureSpec,
)
from metaxy._utils import collect_to_polars
from metaxy.metadata_store import (
    DependencyError,
    FeatureNotFoundError,
    InMemoryMetadataStore,
    MetadataSchemaError,
    StoreNotOpenError,
)


@pytest.fixture
def graph() -> Iterator[FeatureGraph]:
    """Create a clean FeatureGraph for testing."""
    reg = FeatureGraph()
    with reg.use():
        yield reg


# Test fixtures - Define features for testing - they will use the graph from context


class UpstreamFeatureA(
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["upstream", "a"]),
        fields=[
            FieldSpec(key=FieldKey(["frames"]), code_version="1"),
            FieldSpec(key=FieldKey(["audio"]), code_version="1"),
        ],
    ),
):
    pass


class UpstreamFeatureB(
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["upstream", "b"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="1"),
        ],
    ),
):
    pass


class DownstreamFeature(
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["downstream"]),
        deps=[
            FeatureDep(feature=FeatureKey(["upstream", "a"])),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["upstream", "a"]),
                        fields=[FieldKey(["frames"]), FieldKey(["audio"])],
                    )
                ],
            ),
        ],
    ),
):
    pass


@pytest.fixture
def empty_store(graph: FeatureGraph) -> Iterator[InMemoryMetadataStore]:
    """Empty in-memory store."""
    yield InMemoryMetadataStore()


@pytest.fixture
def populated_store(graph: FeatureGraph) -> Iterator[InMemoryMetadataStore]:
    """Store with sample upstream data."""
    store = InMemoryMetadataStore()

    with store:
        # Add upstream feature A
        upstream_a_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "path": ["/data/1.mp4", "/data/2.mp4", "/data/3.mp4"],
                "metaxy_provenance_by_field": [
                    {"frames": "hash_a1_frames", "audio": "hash_a1_audio"},
                    {"frames": "hash_a2_frames", "audio": "hash_a2_audio"},
                    {"frames": "hash_a3_frames", "audio": "hash_a3_audio"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(UpstreamFeatureA, upstream_a_data)

    yield store


@pytest.fixture
def multi_env_stores(
    graph: FeatureGraph,
) -> Iterator[dict[str, InMemoryMetadataStore]]:
    """Multi-environment store setup (prod, staging, dev)."""
    prod = InMemoryMetadataStore()
    staging = InMemoryMetadataStore(fallback_stores=[prod])
    dev = InMemoryMetadataStore(fallback_stores=[staging])

    with prod:
        # Populate prod with upstream data
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "prod_hash1", "audio": "prod_hash1"},
                    {"frames": "prod_hash2", "audio": "prod_hash2"},
                    {"frames": "prod_hash3", "audio": "prod_hash3"},
                ],
            }
        )
        with prod.allow_cross_project_writes():
            prod.write_metadata(UpstreamFeatureA, upstream_data)

    yield {"prod": prod, "staging": staging, "dev": dev}


# Basic CRUD Tests


def test_write_and_read_metadata(empty_store: InMemoryMetadataStore) -> None:
    """Test basic write and read operations."""
    with empty_store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "hash1", "audio": "hash1"},
                    {"frames": "hash2", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash3"},
                ],
            }
        )

        with empty_store.allow_cross_project_writes():
            empty_store.write_metadata(UpstreamFeatureA, metadata)
        result = collect_to_polars(empty_store.read_metadata(UpstreamFeatureA))

        assert len(result) == 3
        assert "sample_uid" in result.columns
        assert "metaxy_provenance_by_field" in result.columns


def test_write_invalid_schema(empty_store: InMemoryMetadataStore) -> None:
    """Test that writing without provenance_by_field column raises error."""
    with empty_store:
        invalid_df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "path": ["/a", "/b", "/c"],
            }
        )

        with pytest.raises(MetadataSchemaError, match="metaxy_provenance_by_field"):
            with empty_store.allow_cross_project_writes():
                empty_store.write_metadata(UpstreamFeatureA, invalid_df)


def test_write_append(empty_store: InMemoryMetadataStore) -> None:
    """Test that writes are append-only."""
    with empty_store:
        df1 = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )

        df2 = pl.DataFrame(
            {
                "sample_uid": [3, 4],
                "metaxy_provenance_by_field": [
                    {"frames": "h3", "audio": "h3"},
                    {"frames": "h4", "audio": "h4"},
                ],
            }
        )

        with empty_store.allow_cross_project_writes():
            empty_store.write_metadata(UpstreamFeatureA, df1)
            empty_store.write_metadata(UpstreamFeatureA, df2)

        result = collect_to_polars(empty_store.read_metadata(UpstreamFeatureA))
        assert len(result) == 4
        assert set(result["sample_uid"].to_list()) == {1, 2, 3, 4}


def test_read_with_filters(populated_store: InMemoryMetadataStore) -> None:
    """Test reading with Polars filter expressions."""
    with populated_store:
        result = collect_to_polars(
            populated_store.read_metadata(
                UpstreamFeatureA, filters=[nw.col("sample_uid") > 1]
            )
        )

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {2, 3}


def test_read_with_column_selection(populated_store: InMemoryMetadataStore) -> None:
    """Test reading specific columns."""
    with populated_store:
        result = collect_to_polars(
            populated_store.read_metadata(
                UpstreamFeatureA, columns=["sample_uid", "metaxy_provenance_by_field"]
            )
        )

        assert set(result.columns) == {"sample_uid", "metaxy_provenance_by_field"}
        assert "path" not in result.columns


def test_read_nonexistent_feature(empty_store: InMemoryMetadataStore) -> None:
    """Test that reading nonexistent feature raises error."""
    with empty_store:
        with pytest.raises(FeatureNotFoundError):
            empty_store.read_metadata(UpstreamFeatureA)


# Feature Existence Tests


def test_has_feature_local(populated_store: InMemoryMetadataStore) -> None:
    """Test has_feature for local store."""
    with populated_store:
        assert populated_store.has_feature(UpstreamFeatureA, check_fallback=False)
        assert not populated_store.has_feature(UpstreamFeatureB, check_fallback=False)


def test_has_feature_with_fallback(
    multi_env_stores: dict[str, InMemoryMetadataStore],
) -> None:
    """Test has_feature checking fallback stores."""
    dev = multi_env_stores["dev"]
    staging = multi_env_stores["staging"]
    prod = multi_env_stores["prod"]

    with dev, staging, prod:
        # UpstreamFeatureA is in prod, not in dev
        assert not dev.has_feature(UpstreamFeatureA, check_fallback=False)
        assert dev.has_feature(UpstreamFeatureA, check_fallback=True)


# Fallback Store Tests


def test_read_from_fallback(multi_env_stores: dict[str, InMemoryMetadataStore]) -> None:
    """Test reading from fallback store."""
    dev = multi_env_stores["dev"]
    staging = multi_env_stores["staging"]
    prod = multi_env_stores["prod"]

    with dev, staging, prod:
        # Read from prod via fallback chain
        result = collect_to_polars(
            dev.read_metadata(UpstreamFeatureA, allow_fallback=True)
        )
        assert len(result) == 3


def test_read_no_fallback(multi_env_stores: dict[str, InMemoryMetadataStore]) -> None:
    """Test that allow_fallback=False doesn't check fallback stores."""
    dev = multi_env_stores["dev"]

    with dev:
        with pytest.raises(FeatureNotFoundError):
            dev.read_metadata(UpstreamFeatureA, allow_fallback=False)


def test_write_to_dev_not_prod(
    multi_env_stores: dict[str, InMemoryMetadataStore],
) -> None:
    """Test that writes go to dev, not prod."""
    dev = multi_env_stores["dev"]
    prod = multi_env_stores["prod"]

    with dev, prod:
        new_data = pl.DataFrame(
            {
                "sample_uid": [4, 5],
                "metaxy_provenance_by_field": [
                    {"default": "hash4"},
                    {"default": "hash5"},
                ],
            }
        )

        with dev.allow_cross_project_writes():
            dev.write_metadata(UpstreamFeatureB, new_data)

        # Should be in dev
        assert dev.has_feature(UpstreamFeatureB, check_fallback=False)

        # Should NOT be in prod
        assert not prod.has_feature(UpstreamFeatureB, check_fallback=False)


# Dependency Resolution Tests


def test_read_upstream_metadata(populated_store: InMemoryMetadataStore) -> None:
    """Test reading upstream dependencies."""
    with populated_store:
        upstream = populated_store.read_upstream_metadata(DownstreamFeature)

        assert "upstream/a" in upstream
        upstream_df = collect_to_polars(upstream["upstream/a"])
        assert len(upstream_df) == 3
        assert "metaxy_provenance_by_field" in upstream_df.columns


def test_read_upstream_metadata_missing_dep(empty_store: InMemoryMetadataStore) -> None:
    """Test that missing dependencies raise error."""
    with empty_store:
        with pytest.raises(DependencyError, match="upstream/a"):
            empty_store.read_upstream_metadata(DownstreamFeature, allow_fallback=False)


def test_read_upstream_metadata_from_fallback(
    multi_env_stores: dict[str, InMemoryMetadataStore],
) -> None:
    """Test reading upstream from fallback stores."""
    dev = multi_env_stores["dev"]
    staging = multi_env_stores["staging"]
    prod = multi_env_stores["prod"]

    with dev, staging, prod:
        upstream = dev.read_upstream_metadata(DownstreamFeature, allow_fallback=True)

        assert "upstream/a" in upstream
        upstream_df = collect_to_polars(upstream["upstream/a"])
        assert len(upstream_df) == 3


# Clear/Reset Tests


def test_clear_store(populated_store: InMemoryMetadataStore) -> None:
    """Test clearing store."""
    with populated_store:
        assert populated_store.has_feature(UpstreamFeatureA)

        populated_store.clear()

        assert not populated_store.has_feature(UpstreamFeatureA)


# Store Open/Close Tests


def test_store_not_open_write_raises(empty_store: InMemoryMetadataStore) -> None:
    """Test that writing to a closed store raises StoreNotOpenError."""
    metadata = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"frames": "hash1", "audio": "hash1"},
                {"frames": "hash2", "audio": "hash2"},
                {"frames": "hash3", "audio": "hash3"},
            ],
        }
    )

    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        empty_store.write_metadata(UpstreamFeatureA, metadata)


def test_store_not_open_read_raises(populated_store: InMemoryMetadataStore) -> None:
    """Test that reading from a closed store raises StoreNotOpenError."""
    # Store is already closed after fixture setup
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        populated_store.read_metadata(UpstreamFeatureA)


def test_store_not_open_has_feature_raises(
    populated_store: InMemoryMetadataStore,
) -> None:
    """Test that has_feature on a closed store raises StoreNotOpenError."""
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        populated_store.has_feature(UpstreamFeatureA)


def test_store_context_manager_opens_and_closes(
    empty_store: InMemoryMetadataStore,
) -> None:
    """Test that context manager properly opens and closes store."""
    # Initially closed
    assert not empty_store._is_open

    with empty_store:
        # Should be open inside context
        assert empty_store._is_open

    # Should be closed after context
    assert not empty_store._is_open
