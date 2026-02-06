"""Tests for metadata store."""

from collections.abc import Iterator, Sequence

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing.models import SampleFeatureSpec

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
)
from metaxy._utils import collect_to_polars
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.metadata_store import (
    FeatureNotFoundError,
    MetadataSchemaError,
    StoreNotOpenError,
)


@pytest.fixture
def features(graph: FeatureGraph):
    class UpstreamFeatureA(
        BaseFeature,
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
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream", "b"]),
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DownstreamFeature(
        BaseFeature,
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

    return {
        "UpstreamFeatureA": UpstreamFeatureA,
        "UpstreamFeatureB": UpstreamFeatureB,
        "DownstreamFeature": DownstreamFeature,
    }


@pytest.fixture
def make_upstream_a_data():
    def _make(
        *,
        sample_uids: Sequence[int],
        prefix: str,
        include_path: bool,
    ) -> pl.DataFrame:
        data = {
            "sample_uid": list(sample_uids),
            "metaxy_provenance_by_field": [
                {
                    "frames": f"{prefix}{uid}_frames",
                    "audio": f"{prefix}{uid}_audio",
                }
                for uid in sample_uids
            ],
        }
        if include_path:
            data["path"] = [f"/data/{uid}.mp4" for uid in sample_uids]
        return pl.DataFrame(data)

    return _make


@pytest.fixture
def UpstreamFeatureA(features: dict[str, type[BaseFeature]]):
    return features["UpstreamFeatureA"]


@pytest.fixture
def UpstreamFeatureB(features: dict[str, type[BaseFeature]]):
    return features["UpstreamFeatureB"]


@pytest.fixture
def empty_store(graph: FeatureGraph, tmp_path) -> Iterator[DeltaMetadataStore]:
    """Empty delta store."""
    yield DeltaMetadataStore(root_path=tmp_path / "delta_store")


@pytest.fixture
def populated_store(
    graph: FeatureGraph, UpstreamFeatureA, make_upstream_a_data, tmp_path
) -> Iterator[DeltaMetadataStore]:
    """Store with sample upstream data."""
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")

    with store.open("w"):
        # Add upstream feature A
        upstream_a_data = make_upstream_a_data(
            sample_uids=[1, 2, 3],
            prefix="hash_a",
            include_path=True,
        )
        store.write(UpstreamFeatureA, upstream_a_data)

    yield store


@pytest.fixture
def multi_env_stores(
    graph: FeatureGraph, UpstreamFeatureA, make_upstream_a_data, tmp_path
) -> Iterator[dict[str, DeltaMetadataStore]]:
    """Multi-environment store setup (prod, staging, dev)."""
    prod = DeltaMetadataStore(root_path=tmp_path / "delta_prod")
    staging = DeltaMetadataStore(root_path=tmp_path / "delta_staging", fallback_stores=[prod])
    dev = DeltaMetadataStore(root_path=tmp_path / "delta_dev", fallback_stores=[staging])

    with prod:
        # Populate prod with upstream data
        upstream_data = make_upstream_a_data(
            sample_uids=[1, 2, 3],
            prefix="prod_hash",
            include_path=False,
        )
        prod.write(UpstreamFeatureA, upstream_data)

    yield {"prod": prod, "staging": staging, "dev": dev}


# Basic CRUD Tests


def test_write_and_read(empty_store: DeltaMetadataStore, UpstreamFeatureA, make_upstream_a_data) -> None:
    """Test basic write and read operations."""
    with empty_store.open("w"):
        metadata = make_upstream_a_data(
            sample_uids=[1, 2, 3],
            prefix="hash",
            include_path=False,
        )

        empty_store.write(UpstreamFeatureA, metadata)
        result = collect_to_polars(empty_store.read(UpstreamFeatureA))

        assert len(result) == 3
        assert "sample_uid" in result.columns
        assert "metaxy_provenance_by_field" in result.columns


def test_write_invalid_schema(empty_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
    """Test that writing without provenance_by_field column raises error."""
    with empty_store.open("w"):
        invalid_df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "path": ["/a", "/b", "/c"],
            }
        )

        with pytest.raises(MetadataSchemaError, match="metaxy_provenance_by_field"):
            empty_store.write(UpstreamFeatureA, invalid_df)


def test_write_append(
    empty_store: DeltaMetadataStore,
    UpstreamFeatureA,
    make_upstream_a_data,
) -> None:
    """Test that writes are append-only."""
    with empty_store.open("w"):
        df1 = make_upstream_a_data(
            sample_uids=[1, 2],
            prefix="h",
            include_path=False,
        )
        df2 = make_upstream_a_data(
            sample_uids=[3, 4],
            prefix="h",
            include_path=False,
        )

        empty_store.write(UpstreamFeatureA, df1)
        empty_store.write(UpstreamFeatureA, df2)

        result = collect_to_polars(empty_store.read(UpstreamFeatureA))
        assert len(result) == 4
        assert set(result["sample_uid"].to_list()) == {1, 2, 3, 4}


def test_read_with_filters(populated_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
    """Test reading with Polars filter expressions."""
    with populated_store.open("w"):
        result = collect_to_polars(populated_store.read(UpstreamFeatureA, filters=[nw.col("sample_uid") > 1]))

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {2, 3}


def test_read_with_column_selection(populated_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
    """Test reading specific columns."""
    with populated_store.open("w"):
        result = collect_to_polars(
            populated_store.read(UpstreamFeatureA, columns=["sample_uid", "metaxy_provenance_by_field"])
        )

        assert set(result.columns) == {"sample_uid", "metaxy_provenance_by_field"}
        assert "path" not in result.columns


def test_read_nonexistent_feature(empty_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
    """Test that reading nonexistent feature raises error."""
    with empty_store.open("w"):
        with pytest.raises(FeatureNotFoundError):
            empty_store.read(UpstreamFeatureA)


# Feature Existence Tests


def test_has_feature_local(populated_store: DeltaMetadataStore, UpstreamFeatureA, UpstreamFeatureB) -> None:
    """Test has_feature for local store."""
    with populated_store.open("w"):
        assert populated_store.has_feature(UpstreamFeatureA, check_fallback=False)
        assert not populated_store.has_feature(UpstreamFeatureB, check_fallback=False)


def test_has_feature_with_fallback(
    multi_env_stores: dict[str, DeltaMetadataStore],
    UpstreamFeatureA,
    UpstreamFeatureB,
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


def test_read_from_fallback(multi_env_stores: dict[str, DeltaMetadataStore], UpstreamFeatureA) -> None:
    """Test reading from fallback store."""
    dev = multi_env_stores["dev"]
    staging = multi_env_stores["staging"]
    prod = multi_env_stores["prod"]

    with dev, staging, prod:
        # Read from prod via fallback chain
        result = collect_to_polars(dev.read(UpstreamFeatureA, allow_fallback=True))
        assert len(result) == 3


def test_read_no_fallback(multi_env_stores: dict[str, DeltaMetadataStore], UpstreamFeatureA) -> None:
    """Test that allow_fallback=False doesn't check fallback stores."""
    dev = multi_env_stores["dev"]

    with dev:
        with pytest.raises(FeatureNotFoundError):
            dev.read(UpstreamFeatureA, allow_fallback=False)


def test_soft_delete_from_fallback_creates_soft_deletion_marker(
    multi_env_stores: dict[str, DeltaMetadataStore], UpstreamFeatureA
) -> None:
    """Soft delete should allow targeting fallback data and write markers locally."""
    dev = multi_env_stores["dev"]
    staging = multi_env_stores["staging"]
    prod = multi_env_stores["prod"]

    with dev, staging, prod:
        assert not dev.has_feature(UpstreamFeatureA, check_fallback=False)

        dev.delete(
            UpstreamFeatureA,
            filters=nw.col("sample_uid") == 1,
            soft=True,
            with_feature_history=False,
        )

        soft_deletion_markers = collect_to_polars(
            dev.read(
                UpstreamFeatureA,
                include_soft_deleted=True,
                allow_fallback=False,
            )
        )
        assert soft_deletion_markers.height == 1
        assert soft_deletion_markers["sample_uid"].to_list() == [1]
        assert soft_deletion_markers["metaxy_deleted_at"].is_not_null().all()

        active = collect_to_polars(
            dev.read(
                UpstreamFeatureA,
                filters=[nw.col("sample_uid") == 1],
                allow_fallback=True,
            )
        )
        assert active.is_empty()


def test_write_to_dev_not_prod(
    multi_env_stores: dict[str, DeltaMetadataStore],
    UpstreamFeatureA,
    UpstreamFeatureB,
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

        dev.write(UpstreamFeatureB, new_data)

        # Should be in dev
        assert dev.has_feature(UpstreamFeatureB, check_fallback=False)

        # Should NOT be in prod
        assert not prod.has_feature(UpstreamFeatureB, check_fallback=False)


# Store Open/Close Tests


def test_store_not_open_write_raises(empty_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
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
        empty_store.write(UpstreamFeatureA, metadata)


def test_store_not_open_read_raises(populated_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
    """Test that reading from a closed store raises StoreNotOpenError."""
    # Store is already closed after fixture setup
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        populated_store.read(UpstreamFeatureA)


def test_store_not_open_has_feature_raises(populated_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
    """Test that has_feature on a closed store raises StoreNotOpenError."""
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        populated_store.has_feature(UpstreamFeatureA)


def test_store_context_manager_opens_and_closes(
    empty_store: DeltaMetadataStore,
) -> None:
    """Test that context manager properly opens and closes store."""
    # Initially closed
    assert not empty_store._is_open

    with empty_store.open("w"):
        # Should be open inside context
        assert empty_store._is_open

    # Should be closed after context
    assert not empty_store._is_open


def test_write_casts_null_typed_system_columns(empty_store: DeltaMetadataStore, UpstreamFeatureA) -> None:
    """Test that system columns with Null dtype are cast to correct types."""
    # Create a DataFrame with Null-typed system columns
    # This can happen with empty frames or certain Polars operations
    df = pl.DataFrame(
        {
            "sample_uid": pl.Series([1, 2], dtype=pl.Int64),
            "metaxy_provenance_by_field": [
                {"frames": "hash1", "audio": "hash1"},
                {"frames": "hash2", "audio": "hash2"},
            ],
            # Explicitly create Null-typed columns for all castable system columns
            "metaxy_provenance": pl.Series([None, None], dtype=pl.Null),
            "metaxy_feature_version": pl.Series([None, None], dtype=pl.Null),
            "metaxy_snapshot_version": pl.Series([None, None], dtype=pl.Null),
            "metaxy_data_version": pl.Series([None, None], dtype=pl.Null),
            "metaxy_created_at": pl.Series([None, None], dtype=pl.Null),
            "metaxy_materialization_id": pl.Series([None, None], dtype=pl.Null),
        }
    )

    # Verify columns are Null typed before write
    assert df.schema["metaxy_provenance"] == pl.Null
    assert df.schema["metaxy_feature_version"] == pl.Null
    assert df.schema["metaxy_snapshot_version"] == pl.Null
    assert df.schema["metaxy_data_version"] == pl.Null
    assert df.schema["metaxy_created_at"] == pl.Null
    assert df.schema["metaxy_materialization_id"] == pl.Null

    with empty_store.open("w"):
        empty_store.write(UpstreamFeatureA, df)

        # Read back and verify columns are now correctly typed
        result = collect_to_polars(empty_store.read(UpstreamFeatureA))

        assert result.schema["metaxy_provenance"] == pl.String
        assert result.schema["metaxy_feature_version"] == pl.String
        assert result.schema["metaxy_snapshot_version"] == pl.String
        assert result.schema["metaxy_data_version"] == pl.String
        assert result.schema["metaxy_created_at"] == pl.Datetime("us", time_zone="UTC")
        assert result.schema["metaxy_materialization_id"] == pl.String


def test_resolve_update_accepts_feature_key(
    populated_store: DeltaMetadataStore, features: dict[str, type[BaseFeature]]
) -> None:
    """Test that resolve_update accepts FeatureKey in addition to feature class."""
    # UpstreamFeatureA = features["UpstreamFeatureA"]
    DownstreamFeature = features["DownstreamFeature"]

    with populated_store.open("w"):
        # Test with feature class (existing behavior)
        increment_from_class = populated_store.resolve_update(DownstreamFeature)
        assert increment_from_class.new is not None

        # Test with FeatureKey (new behavior)
        feature_key = FeatureKey(["downstream"])
        increment_from_key = populated_store.resolve_update(feature_key)
        assert increment_from_key.new is not None

        # Both should return equivalent results
        assert len(increment_from_class.new) == len(increment_from_key.new)

        # Test with string path (also a CoercibleToFeatureKey)
        increment_from_string = populated_store.resolve_update("downstream")
        assert increment_from_string.new is not None
        assert len(increment_from_string.new) == len(increment_from_class.new)
