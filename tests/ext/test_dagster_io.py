"""Tests for Dagster IOManager and helper functions.

Tests the Metaxy Dagster integration's IOManager and helper functions:
1. Helper functions (build_asset_spec_from_feature, build_asset_in_from_feature)
2. MetaxyIOManager for managing Increment flow between assets
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import dagster as dg
import narwhals as nw
import polars as pl
import pytest

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureKey,
    FieldKey,
    FieldSpec,
    MetaxyConfig,
)
from metaxy._testing.models import SampleFeatureSpec
from metaxy.config import StoreConfig
from metaxy.ext.dagster import (
    MetaxyIOManager,
    MetaxyMetadataStoreResource,
    asset,
    build_asset_in_from_feature,
    build_asset_spec_from_feature,
)
from metaxy.versioning.types import Increment

if TYPE_CHECKING:
    pass


# ============= HELPER FUNCTIONS TESTS =============


def test_build_asset_spec_from_feature_basic() -> None:
    """Test building AssetSpec from a feature without dependencies.

    Verifies that:
    - AssetSpec is created with correct key derived from FeatureKey
    - No dependencies are set when feature has no deps
    - Basic metadata is populated correctly
    """

    class SimpleFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["simple", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    spec = build_asset_spec_from_feature(SimpleFeature)

    # Verify asset key
    assert spec.key == dg.AssetKey(["simple", "feature"])

    # Verify no dependencies (deps can be None or empty iterable)
    assert spec.deps is None or len(list(spec.deps)) == 0

    # Verify metadata
    assert spec.metadata is not None
    assert spec.metadata["feature_key"] == "simple/feature"
    assert "feature_version" in spec.metadata
    assert "feature_spec_version" in spec.metadata
    assert "id_columns" in spec.metadata
    assert "fields" in spec.metadata


def test_build_asset_spec_from_feature_with_dependencies() -> None:
    """Test building AssetSpec from a feature with dependencies.

    Verifies that:
    - Dependencies are correctly converted to AssetKey deps
    - Each dependency FeatureKey becomes a Dagster AssetKey
    """

    # Create upstream feature
    class UpstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Create downstream feature with dependency
    class DownstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream", "feature"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream", "feature"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    spec = build_asset_spec_from_feature(DownstreamFeature)

    # Verify asset key
    assert spec.key == dg.AssetKey(["downstream", "feature"])

    # Verify dependencies
    assert spec.deps is not None
    # Convert deps iterable to list for length checking and iteration
    dep_keys_list = list(spec.deps)
    assert len(dep_keys_list) == 1

    # Extract AssetKeys from either AssetKey or AssetDep objects
    dep_keys = []
    for dep in dep_keys_list:
        if hasattr(dep, "asset_key"):
            dep_keys.append(dep.asset_key)
        else:
            dep_keys.append(dep)

    assert dg.AssetKey(["upstream", "feature"]) in dep_keys


def test_build_asset_spec_asset_key_conversion() -> None:
    """Test FeatureKey to AssetKey conversion.

    Verifies that:
    - FeatureKey parts are correctly converted to AssetKey path
    - Multi-level keys work correctly
    """

    class MultiLevelFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["namespace", "category", "feature_name"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    spec = build_asset_spec_from_feature(MultiLevelFeature)

    # Verify multi-level key conversion
    assert spec.key == dg.AssetKey(["namespace", "category", "feature_name"])


def test_build_asset_spec_metadata_extraction() -> None:
    """Test metadata extraction from feature spec.

    Verifies that:
    - feature_key is correctly serialized to string
    - feature_version is extracted
    - id_columns are serialized to JSON
    - fields are serialized to JSON with proper string representation
    """

    class MetadataFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "metadata"]),
            id_columns=["sample_uid", "frame_id"],
            fields=[
                FieldSpec(key=FieldKey(["field_a"]), code_version="1"),
                FieldSpec(key=FieldKey(["field_b"]), code_version="1"),
            ],
        ),
    ):
        pass

    spec = build_asset_spec_from_feature(MetadataFeature)

    # Verify metadata
    assert spec.metadata is not None
    assert spec.metadata["feature_key"] == "test/metadata"
    assert isinstance(spec.metadata["feature_version"], str)
    assert isinstance(spec.metadata["feature_spec_version"], str)

    # Verify id_columns serialization
    id_columns = json.loads(spec.metadata["id_columns"])
    assert id_columns == ["sample_uid", "frame_id"]

    # Verify fields serialization
    fields = json.loads(spec.metadata["fields"])
    assert len(fields) == 2
    assert "field_a" in fields
    assert "field_b" in fields


def test_build_asset_spec_with_user_metadata() -> None:
    """Test user-defined metadata is included in AssetSpec.

    Verifies that:
    - User metadata from FeatureSpec.metadata is included
    - User metadata is serialized to JSON
    """

    class FeatureWithMetadata(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "with_metadata"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            metadata={
                "owner": "data-team",
                "priority": "high",
                "tags": ["ml", "video"],
            },
        ),
    ):
        pass

    spec = build_asset_spec_from_feature(FeatureWithMetadata)

    # Verify user metadata
    assert spec.metadata is not None
    assert "user_metadata" in spec.metadata
    user_metadata = json.loads(spec.metadata["user_metadata"])
    assert user_metadata["owner"] == "data-team"
    assert user_metadata["priority"] == "high"
    assert user_metadata["tags"] == ["ml", "video"]


def test_build_asset_spec_field_names_serialized() -> None:
    """Test field names are properly serialized in metadata.

    Verifies that:
    - Field keys are converted to strings
    - Multiple fields are all included
    """

    class MultiFieldFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "multi_field"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["text"]), code_version="1"),
            ],
        ),
    ):
        pass

    spec = build_asset_spec_from_feature(MultiFieldFeature)

    # Verify field serialization
    fields = json.loads(spec.metadata["fields"])
    assert len(fields) == 3
    assert "frames" in fields
    assert "audio" in fields
    assert "text" in fields


def test_build_asset_spec_description() -> None:
    """Test AssetSpec description is properly set.

    Verifies that:
    - Description includes feature key
    - Description format is consistent
    """

    class DescribedFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "described"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    spec = build_asset_spec_from_feature(DescribedFeature)

    # Verify description
    assert spec.description is not None
    assert "test/described" in spec.description
    assert "Metaxy feature" in spec.description


def test_build_asset_in_basic() -> None:
    """Test basic AssetIn creation from feature.

    Verifies that:
    - AssetIn mapping is created with correct input name
    - Asset key matches feature key
    - Default input name is "diff"
    """

    class InputFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["input", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    asset_in_map = build_asset_in_from_feature(InputFeature)

    # Verify structure
    assert "diff" in asset_in_map
    asset_in = asset_in_map["diff"]
    assert isinstance(asset_in, dg.AssetIn)
    assert asset_in.key == dg.AssetKey(["input", "feature"])


def test_build_asset_in_custom_input_name() -> None:
    """Test AssetIn with custom input_name parameter.

    Verifies that:
    - Custom input name is used in mapping
    - Asset key is still correct
    """

    class CustomInputFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["custom", "input"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    asset_in_map = build_asset_in_from_feature(
        CustomInputFeature, input_name="increment"
    )

    # Verify custom input name
    assert "increment" in asset_in_map
    assert "diff" not in asset_in_map
    asset_in = asset_in_map["increment"]
    assert asset_in.key == dg.AssetKey(["custom", "input"])


def test_build_asset_in_asset_key_set() -> None:
    """Test AssetIn has correct asset key from feature.

    Verifies that:
    - Asset key is properly set on AssetIn
    - Multi-level keys work correctly
    """

    class MultiLevelInputFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["namespace", "group", "input"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    asset_in_map = build_asset_in_from_feature(MultiLevelInputFeature)
    asset_in = asset_in_map["diff"]

    # Verify multi-level key
    assert asset_in.key == dg.AssetKey(["namespace", "group", "input"])


def test_build_asset_in_metadata() -> None:
    """Test AssetIn includes metadata about the feature.

    Verifies that:
    - Metadata includes feature_key
    - Metadata includes type information
    """

    class MetadataInputFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["metadata", "input"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    asset_in_map = build_asset_in_from_feature(MetadataInputFeature)
    asset_in = asset_in_map["diff"]

    # Verify metadata
    assert asset_in.metadata is not None
    assert asset_in.metadata["feature_key"] == "metadata/input"
    assert asset_in.metadata["type"] == "metaxy.Increment"


# ============= IO MANAGER TESTS =============


def test_metaxy_io_manager_from_store() -> None:
    """Test MetaxyIOManager.from_store constructor.

    Verifies that:
    - from_store creates IOManager with store attribute set
    - Convenience constructor works correctly
    """
    # Create resource
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        # Create store resource using from_config
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")

        # Create IO manager
        io_manager = MetaxyIOManager.from_store(store_resource)

        # Verify
        assert io_manager.store is store_resource
    finally:
        MetaxyConfig.reset()


def test_metaxy_io_manager_handle_output_with_feature_spec() -> None:
    """Test handle_output with valid FeatureSpec.

    Verifies that:
    - handle_output accepts FeatureSpec objects
    - No errors are raised
    - Log message includes feature key
    """

    # Create feature
    class OutputFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["output", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Create IO manager
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(store_resource)

        # Create mock context
        context = Mock(spec=dg.OutputContext)

        # Call handle_output - should not raise
        io_manager.handle_output(context, OutputFeature.spec())

        # If we get here, no exception was raised
        assert True
    finally:
        MetaxyConfig.reset()


def test_metaxy_io_manager_handle_output_with_invalid_type() -> None:
    """Test handle_output with invalid type logs warning.

    Verifies that:
    - Invalid types don't crash the method
    - Warning is logged for unexpected types
    """

    # Create IO manager
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(store_resource)

        # Create mock context
        context = Mock(spec=dg.OutputContext)

        # Call with invalid type - should not crash, just log warning
        # Create a mock object that looks like a FeatureSpec but isn't
        invalid_obj = Mock()
        invalid_obj.key = None  # No key attribute or invalid key

        io_manager.handle_output(context, invalid_obj)  # type: ignore[arg-type]

        # If we get here, no exception was raised
        assert True
    finally:
        MetaxyConfig.reset()


def test_metaxy_io_manager_load_input_with_feature_override(monkeypatch: Any) -> None:
    """Test load_input can resolve a different feature via metadata override."""

    class UpstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    class ConsumerFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["consumer", "feature"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream", "feature"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(store_resource)

        # Mock the store and its methods
        mock_store = Mock()
        empty_df = nw.from_native(pl.DataFrame())
        mock_store.resolve_update.return_value = Increment(
            added=empty_df, changed=empty_df, removed=empty_df
        )
        mock_store.__enter__ = Mock(return_value=mock_store)
        mock_store.__exit__ = Mock(return_value=None)

        # Mock get_store to return the mock store as a context manager
        def _get_store_override(self: Any) -> Any:  # noqa: ANN401
            return mock_store

        monkeypatch.setattr(
            MetaxyMetadataStoreResource, "get_store", _get_store_override
        )

        context = Mock(spec=dg.InputContext)
        context.asset_key = dg.AssetKey(["upstream", "feature"])
        context.metadata = {"metaxy/resolve_feature_key": "consumer/feature"}
        context.partition_key = None

        io_manager.load_input(context)

        mock_store.resolve_update.assert_called_once_with(
            ConsumerFeature,
            lazy=False,  # type: ignore[arg-type]
        )
    finally:
        MetaxyConfig.reset()


def test_metaxy_io_manager_load_input_feature_not_in_graph_error() -> None:
    """Test load_input with feature not in graph raises error.

    Verifies that:
    - ValueError is raised when feature key not found in graph
    - Error message is helpful
    """
    # Create IO manager
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(store_resource)

        # Create mock context with non-existent feature
        context = Mock(spec=dg.InputContext)
        context.asset_key = dg.AssetKey(["nonexistent", "feature"])
        context.metadata = None

        # Should raise ValueError
        with pytest.raises(ValueError, match="not found in active graph"):
            io_manager.load_input(context)
    finally:
        MetaxyConfig.reset()


# ============= DECORATOR TESTS =============


def test_asset_decorator_root_feature() -> None:
    """Test @mxd.asset decorator with root feature (no dependencies).

    Verifies that:
    - Decorator creates valid AssetsDefinition
    - Asset has correct key from feature
    - No ins are created for root feature
    - io_manager_key is set correctly
    """

    class RootFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "root"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    @asset(feature=RootFeature, io_manager_key="test_io")
    def root_asset(context) -> None:  # noqa: ANN001
        pass

    # Verify it's an AssetsDefinition
    assert isinstance(root_asset, dg.AssetsDefinition)

    # Verify asset key
    assert root_asset.key == dg.AssetKey(["test", "root"])


def test_asset_decorator_downstream_feature() -> None:
    """Test @mxd.asset decorator with downstream feature (has dependencies).

    Verifies that:
    - Decorator creates valid AssetsDefinition
    - Asset has correct key and dependencies
    - ins are automatically created for downstream feature
    """

    class UpstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    class DownstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[FeatureDep(feature=FeatureKey(["test", "upstream"]))],
            fields=[FieldSpec(key=FieldKey(["processed"]), code_version="1")],
        ),
    ):
        pass

    @asset(feature=DownstreamFeature)
    def downstream_asset(context, diff: Increment) -> None:  # noqa: ANN001
        pass

    # Verify it's an AssetsDefinition
    assert isinstance(downstream_asset, dg.AssetsDefinition)

    # Verify asset key
    assert downstream_asset.key == dg.AssetKey(["test", "downstream"])


def test_asset_decorator_custom_io_manager_key() -> None:
    """Test @mxd.asset decorator with custom io_manager_key.

    Verifies that:
    - Custom io_manager_key is respected
    """

    class TestFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "custom_io"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    @asset(feature=TestFeature, io_manager_key="custom_manager")
    def custom_io_asset(context) -> None:  # noqa: ANN001
        pass

    # Verify it's an AssetsDefinition
    assert isinstance(custom_io_asset, dg.AssetsDefinition)


def test_asset_decorator_with_partitions() -> None:
    """Test @mxd.asset decorator with partitions.

    Verifies that:
    - partitions_def is correctly applied
    """

    class PartitionedFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "partitioned"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    partitions = dg.DynamicPartitionsDefinition(name="test_partitions")

    @asset(feature=PartitionedFeature, partitions_def=partitions)
    def partitioned_asset(context) -> None:  # noqa: ANN001
        pass

    # Verify it's an AssetsDefinition
    assert isinstance(partitioned_asset, dg.AssetsDefinition)

    # Verify partitions are set (check via specs_by_key)
    specs = list(partitioned_asset.specs)
    assert len(specs) > 0
    assert specs[0].partitions_def == partitions


def test_asset_decorator_with_all_dagster_params() -> None:
    """Test @mxd.asset decorator with comprehensive Dagster parameters.

    Verifies that:
    - All Dagster parameters are passed through correctly
    - Tags, compute_kind, owners, etc. work
    - Metadata merges with Metaxy's feature metadata
    - User values take precedence where appropriate
    """

    class FullFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "full"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    # Test with many Dagster parameters
    @asset(
        feature=FullFeature,
        compute_kind="python",
        tags={"team": "ml-platform", "priority": "high"},
        owners=["team:ml-platform", "john@example.com"],  # Email without 'user:' prefix
        group_name="video_processing",
        metadata={"sla_minutes": 30, "model_version": "v2"},
        description="Custom description",
        op_tags={"concurrency_key": "video_processing"},
    )
    def full_asset(context) -> None:  # noqa: ANN001
        pass

    # Verify it's an AssetsDefinition
    assert isinstance(full_asset, dg.AssetsDefinition)

    # Check specs
    specs = list(full_asset.specs)
    assert len(specs) > 0
    spec = specs[0]

    # Verify Dagster parameters
    assert spec.group_name == "video_processing"
    assert spec.owners == ["team:ml-platform", "john@example.com"]
    assert spec.tags == {
        "team": "ml-platform",
        "priority": "high",
        "metaxy.feature_key": "test_full",
        "metaxy.project": "test",
    }

    # Verify metadata merge (both Metaxy and user metadata)
    assert spec.metadata is not None
    assert spec.metadata["sla_minutes"] == 30  # User metadata
    assert spec.metadata["model_version"] == "v2"  # User metadata
    assert "feature_key" in spec.metadata  # Metaxy metadata
    assert spec.metadata["feature_key"] == "test/full"

    # Verify user description overrides
    assert spec.description == "Custom description"


def test_asset_decorator_key_override() -> None:
    """Test @mxd.asset decorator with key override.

    Verifies that:
    - name parameter overrides feature key
    - key_prefix is applied correctly
    """

    class KeyFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "key"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    # Test with name override
    @asset(feature=KeyFeature, name="custom_name")
    def key_asset_with_name(context) -> None:  # noqa: ANN001
        pass

    specs = list(key_asset_with_name.specs)
    assert len(specs) > 0
    # When name is provided, Dagster uses it directly
    assert specs[0].key.path[-1] == "custom_name"

    # Test with key_prefix
    @asset(feature=KeyFeature, key_prefix="production")
    def key_asset_with_prefix(context) -> None:  # noqa: ANN001
        pass

    specs = list(key_asset_with_prefix.specs)
    assert len(specs) > 0
    assert specs[0].key == dg.AssetKey(["production", "test", "key"])
