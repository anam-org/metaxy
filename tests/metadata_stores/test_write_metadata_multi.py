"""Comprehensive tests for MetadataStore.write_metadata_multi() method.

Tests cover:
- Verification that writes occur in reverse topological order
- Different key types (FeatureKey, string paths, BaseFeature class references)
- Empty dict returns without error (no-op)
- Single feature
- Multiple features with dependencies
- materialization_id parameter
- Integration with actual metadata stores
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import polars as pl
import pytest

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
)
from metaxy._testing.models import SampleFeatureSpec
from metaxy._utils import collect_to_polars
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


@pytest.fixture
def features_with_deps(graph: FeatureGraph):
    """Create a set of features with dependencies for testing.

    Dependency graph:
        FeatureA (root)
        FeatureB (root)
        FeatureC -> FeatureA
        FeatureD -> FeatureB, FeatureC
    """

    class FeatureA(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "a"]),
            fields=[FieldSpec(key=FieldKey(["x"]))],
        ),
    ):
        pass

    class FeatureB(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "b"]),
            fields=[FieldSpec(key=FieldKey(["y"]))],
        ),
    ):
        pass

    class FeatureC(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "c"]),
            fields=[FieldSpec(key=FieldKey(["z"]))],
            deps=[FeatureDep(feature=FeatureA)],
        ),
    ):
        pass

    class FeatureD(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "d"]),
            fields=[FieldSpec(key=FieldKey(["w"]))],
            deps=[FeatureDep(feature=FeatureB), FeatureDep(feature=FeatureC)],
        ),
    ):
        pass

    return {
        "FeatureA": FeatureA,
        "FeatureB": FeatureB,
        "FeatureC": FeatureC,
        "FeatureD": FeatureD,
    }


@pytest.fixture
def store() -> Iterator[DuckDBMetadataStore]:
    """Create an empty in-memory metadata store."""
    store = DuckDBMetadataStore()
    with store.open("write"):
        yield store


class TestBasicFunctionality:
    """Test basic functionality of write_metadata_multi."""

    def test_empty_dict_is_noop(self, store: DuckDBMetadataStore):
        """Test that passing an empty dict does nothing (no-op)."""
        # Should not raise any errors
        store.write_metadata_multi({})

    def test_single_feature(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test writing metadata for a single feature."""
        FeatureA = features_with_deps["FeatureA"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "metaxy_provenance_by_field": [
                        {"x": "hash1"},
                        {"x": "hash2"},
                        {"x": "hash3"},
                    ],
                }
            )
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        # Verify data was written
        result = collect_to_polars(store.read_metadata(FeatureA))
        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}

    def test_multiple_features_no_dependencies(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test writing metadata for multiple features without dependencies."""
        FeatureA = features_with_deps["FeatureA"]
        FeatureB = features_with_deps["FeatureB"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [{"x": "hash1"}, {"x": "hash2"}],
                }
            ),
            FeatureB: pl.DataFrame(
                {
                    "sample_uid": [3, 4],
                    "metaxy_provenance_by_field": [{"y": "hash3"}, {"y": "hash4"}],
                }
            ),
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        # Verify both features were written
        result_a = collect_to_polars(store.read_metadata(FeatureA))
        result_b = collect_to_polars(store.read_metadata(FeatureB))

        assert len(result_a) == 2
        assert len(result_b) == 2


class TestReverseTopologicalOrder:
    """Test that writes occur in reverse topological order."""

    def test_writes_in_reverse_topological_order(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that features are written in reverse topological order (dependents first).

        For the dependency graph:
            A (root), B (root)
            C -> A
            D -> B, C

        Reverse topological order should be: [D, C, B, A] or [D, B, C, A] (since B and C are at same level)
        """
        FeatureA = features_with_deps["FeatureA"]
        FeatureB = features_with_deps["FeatureB"]
        FeatureC = features_with_deps["FeatureC"]
        FeatureD = features_with_deps["FeatureD"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            ),
            FeatureB: pl.DataFrame(
                {
                    "sample_uid": [2],
                    "metaxy_provenance_by_field": [{"y": "hash2"}],
                }
            ),
            FeatureC: pl.DataFrame(
                {
                    "sample_uid": [3],
                    "metaxy_provenance_by_field": [{"z": "hash3"}],
                }
            ),
            FeatureD: pl.DataFrame(
                {
                    "sample_uid": [4],
                    "metaxy_provenance_by_field": [{"w": "hash4"}],
                }
            ),
        }

        # Track write order by spying on write_metadata
        write_order = []

        original_write = store.write_metadata

        def tracked_write(feature, df, materialization_id=None):
            # Resolve the feature key
            feature_key = store._resolve_feature_key(feature)
            write_order.append(feature_key)
            return original_write(feature, df, materialization_id)

        with patch.object(store, "write_metadata", side_effect=tracked_write):
            with store.allow_cross_project_writes():
                store.write_metadata_multi(metadata)

        # Verify D was written before C, and C before A
        d_idx = write_order.index(FeatureD.spec().key)
        c_idx = write_order.index(FeatureC.spec().key)
        a_idx = write_order.index(FeatureA.spec().key)

        assert d_idx < c_idx, "D should be written before C"
        assert c_idx < a_idx, "C should be written before A"

        # All features should have been written
        assert len(write_order) == 4

    def test_writes_linear_chain_in_reverse_order(self, store, graph: FeatureGraph):
        """Test a simple linear chain is written in reverse order: C -> B -> A for A -> B -> C."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "chain_a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "chain_b"]),
                fields=[FieldSpec(key=FieldKey(["y"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "chain_c"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
                deps=[FeatureDep(feature=FeatureB)],
            ),
        ):
            pass

        metadata = {
            FeatureA: pl.DataFrame(
                {"sample_uid": [1], "metaxy_provenance_by_field": [{"x": "hash1"}]}
            ),
            FeatureB: pl.DataFrame(
                {"sample_uid": [2], "metaxy_provenance_by_field": [{"y": "hash2"}]}
            ),
            FeatureC: pl.DataFrame(
                {"sample_uid": [3], "metaxy_provenance_by_field": [{"z": "hash3"}]}
            ),
        }

        # Track write order
        write_order = []
        original_write = store.write_metadata

        def tracked_write(feature, df, materialization_id=None):
            feature_key = store._resolve_feature_key(feature)
            write_order.append(feature_key)
            return original_write(feature, df, materialization_id)

        with patch.object(store, "write_metadata", side_effect=tracked_write):
            with store.allow_cross_project_writes():
                store.write_metadata_multi(metadata)

        # Verify reverse order: C, B, A
        assert write_order[0] == FeatureC.spec().key
        assert write_order[1] == FeatureB.spec().key
        assert write_order[2] == FeatureA.spec().key


class TestInputVariations:
    """Test that the method accepts various input types for keys."""

    def test_accepts_feature_keys(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that feature keys can be provided as FeatureKey objects."""
        FeatureA = features_with_deps["FeatureA"]

        metadata = {
            FeatureA.spec().key: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            )
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        result = collect_to_polars(store.read_metadata(FeatureA))
        assert len(result) == 1

    def test_accepts_string_paths(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that feature keys can be provided as string paths."""
        FeatureA = features_with_deps["FeatureA"]

        metadata = {
            "test/a": pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            )
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        result = collect_to_polars(store.read_metadata(FeatureA))
        assert len(result) == 1

    def test_accepts_feature_classes(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that feature keys can be provided as Feature classes."""
        FeatureA = features_with_deps["FeatureA"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            )
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        result = collect_to_polars(store.read_metadata(FeatureA))
        assert len(result) == 1

    def test_accepts_mixed_key_types(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that feature keys can be provided as mixed types."""
        FeatureA = features_with_deps["FeatureA"]
        FeatureB = features_with_deps["FeatureB"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            ),
            "test/b": pl.DataFrame(
                {
                    "sample_uid": [2],
                    "metaxy_provenance_by_field": [{"y": "hash2"}],
                }
            ),
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        result_a = collect_to_polars(store.read_metadata(FeatureA))
        result_b = collect_to_polars(store.read_metadata(FeatureB))

        assert len(result_a) == 1
        assert len(result_b) == 1


class TestMaterializationId:
    """Test materialization_id parameter."""

    def test_materialization_id_parameter(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that materialization_id is passed to write_metadata calls."""
        FeatureA = features_with_deps["FeatureA"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            )
        }

        materialization_id = "test-run-123"

        # Spy on write_metadata to verify materialization_id is passed
        original_write = store.write_metadata
        write_calls = []

        def tracked_write(feature, df, materialization_id=None):
            write_calls.append(materialization_id)
            return original_write(feature, df, materialization_id)

        with patch.object(store, "write_metadata", side_effect=tracked_write):
            with store.allow_cross_project_writes():
                store.write_metadata_multi(
                    metadata, materialization_id=materialization_id
                )

        # Verify materialization_id was passed
        assert len(write_calls) == 1
        assert write_calls[0] == materialization_id

    def test_materialization_id_propagates_to_all_writes(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that materialization_id is passed to all write_metadata calls."""
        FeatureA = features_with_deps["FeatureA"]
        FeatureB = features_with_deps["FeatureB"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            ),
            FeatureB: pl.DataFrame(
                {
                    "sample_uid": [2],
                    "metaxy_provenance_by_field": [{"y": "hash2"}],
                }
            ),
        }

        materialization_id = "batch-write-456"

        # Spy on write_metadata
        original_write = store.write_metadata
        write_calls = []

        def tracked_write(feature, df, materialization_id=None):
            write_calls.append(materialization_id)
            return original_write(feature, df, materialization_id)

        with patch.object(store, "write_metadata", side_effect=tracked_write):
            with store.allow_cross_project_writes():
                store.write_metadata_multi(
                    metadata, materialization_id=materialization_id
                )

        # Verify materialization_id was passed to all writes
        assert len(write_calls) == 2
        assert all(mat_id == materialization_id for mat_id in write_calls)


class TestDataIntegrity:
    """Test that data is written correctly and can be read back."""

    def test_data_integrity_after_multi_write(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that data written via write_metadata_multi can be read back correctly."""
        FeatureA = features_with_deps["FeatureA"]
        FeatureB = features_with_deps["FeatureB"]
        FeatureC = features_with_deps["FeatureC"]

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [{"x": "hash1"}, {"x": "hash2"}],
                }
            ),
            FeatureB: pl.DataFrame(
                {
                    "sample_uid": [3, 4],
                    "metaxy_provenance_by_field": [{"y": "hash3"}, {"y": "hash4"}],
                }
            ),
            FeatureC: pl.DataFrame(
                {
                    "sample_uid": [5, 6],
                    "metaxy_provenance_by_field": [{"z": "hash5"}, {"z": "hash6"}],
                }
            ),
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        # Read back and verify
        result_a = collect_to_polars(store.read_metadata(FeatureA))
        result_b = collect_to_polars(store.read_metadata(FeatureB))
        result_c = collect_to_polars(store.read_metadata(FeatureC))

        assert len(result_a) == 2
        assert set(result_a["sample_uid"].to_list()) == {1, 2}

        assert len(result_b) == 2
        assert set(result_b["sample_uid"].to_list()) == {3, 4}

        assert len(result_c) == 2
        assert set(result_c["sample_uid"].to_list()) == {5, 6}


class TestErrorHandling:
    """Test error handling for write_metadata_multi."""

    def test_store_not_open_raises(
        self, features_with_deps: dict[str, type[BaseFeature]]
    ):
        """Test that calling write_metadata_multi on a closed store raises StoreNotOpenError."""
        from metaxy.metadata_store import StoreNotOpenError

        FeatureA = features_with_deps["FeatureA"]

        store = DuckDBMetadataStore()

        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            )
        }

        # Store is not opened, should raise
        with pytest.raises(StoreNotOpenError):
            store.write_metadata_multi(metadata)

    def test_invalid_schema_raises(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test that invalid schema raises MetadataSchemaError."""
        from metaxy.metadata_store import MetadataSchemaError

        FeatureA = features_with_deps["FeatureA"]

        # Missing metaxy_provenance_by_field column
        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    # Missing metaxy_provenance_by_field
                }
            )
        }

        with pytest.raises(MetadataSchemaError, match="metaxy_provenance_by_field"):
            with store.allow_cross_project_writes():
                store.write_metadata_multi(metadata)


class TestSubsetOfFeatures:
    """Test writing only a subset of features from a larger graph."""

    def test_write_subset_of_features(
        self,
        store,
        features_with_deps: dict[str, type[BaseFeature]],
    ):
        """Test writing metadata for only some features in a dependency graph."""
        FeatureA = features_with_deps["FeatureA"]
        FeatureC = features_with_deps["FeatureC"]

        # Write only A and C (not B or D)
        metadata = {
            FeatureA: pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"x": "hash1"}],
                }
            ),
            FeatureC: pl.DataFrame(
                {
                    "sample_uid": [2],
                    "metaxy_provenance_by_field": [{"z": "hash2"}],
                }
            ),
        }

        with store.allow_cross_project_writes():
            store.write_metadata_multi(metadata)

        # Verify A and C were written
        result_a = collect_to_polars(store.read_metadata(FeatureA))
        result_c = collect_to_polars(store.read_metadata(FeatureC))

        assert len(result_a) == 1
        assert len(result_c) == 1
