"""Tests for the feature_to_dagster_type helper."""

from typing import Any, cast

import dagster as dg
import pandas as pd
import polars as pl
import pytest

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_COLUMN_LINEAGE_METADATA_KEY,
    DAGSTER_COLUMN_SCHEMA_METADATA_KEY,
)
from metaxy.ext.dagster.dagster_type import feature_to_dagster_type
from metaxy.ext.dagster.table_metadata import build_column_lineage, build_column_schema


@pytest.fixture
def simple_feature() -> type[mx.BaseFeature]:
    """Create a simple feature for testing."""
    from pydantic import Field

    spec = mx.FeatureSpec(
        key=["test", "simple"],
        id_columns=["id"],
        fields=["name", "value"],
    )

    class SimpleFeature(mx.BaseFeature, spec=spec):
        """A simple test feature."""

        id: int = Field(description="Primary key")
        name: str = Field(description="Name field")
        value: float | None = Field(default=None, description="Optional value")

    return SimpleFeature


@pytest.fixture
def feature_no_docstring() -> type[mx.BaseFeature]:
    """Create a feature without a docstring."""
    spec = mx.FeatureSpec(
        key=["test", "no_docs"],
        id_columns=["id"],
        fields=["data"],
    )

    class NoDocsFeature(mx.BaseFeature, spec=spec):
        id: str
        data: str

    return NoDocsFeature


@pytest.fixture
def upstream_feature() -> type[mx.BaseFeature]:
    """Create an upstream feature for testing lineage."""
    from pydantic import Field

    spec = mx.FeatureSpec(
        key=["test", "upstream"],
        id_columns=["id"],
        fields=["value"],
    )

    class UpstreamFeature(mx.BaseFeature, spec=spec):
        """Upstream feature."""

        id: int = Field(description="Primary key")
        value: str = Field(description="Value field")

    return UpstreamFeature


@pytest.fixture
def downstream_feature(
    upstream_feature: type[mx.BaseFeature],
) -> type[mx.BaseFeature]:
    """Create a downstream feature that depends on upstream."""
    from pydantic import Field

    spec = mx.FeatureSpec(
        key=["test", "downstream"],
        id_columns=["id"],
        fields=["result", "value"],
        deps=[mx.FeatureDep(feature=upstream_feature)],
    )

    class DownstreamFeature(mx.BaseFeature, spec=spec):
        """Downstream feature with dependency."""

        id: int = Field(description="Primary key")
        value: str = Field(description="Pass-through value")
        result: str = Field(description="Computed result")

    return DownstreamFeature


class TestFeatureToDagsterType:
    """Tests for feature_to_dagster_type function."""

    def test_creates_dagster_type(self, simple_feature: type[mx.BaseFeature]):
        """Test that a DagsterType is created."""
        dagster_type = feature_to_dagster_type(simple_feature)
        assert isinstance(dagster_type, dg.DagsterType)

    def test_type_name_from_feature_key(self, simple_feature: type[mx.BaseFeature]):
        """Test that the type name comes from the feature's table name."""
        dagster_type = feature_to_dagster_type(simple_feature)
        assert dagster_type.display_name == "test__simple"

    def test_custom_type_name(self, simple_feature: type[mx.BaseFeature]):
        """Test that a custom name can be provided."""
        dagster_type = feature_to_dagster_type(simple_feature, name="custom_name")
        assert dagster_type.display_name == "custom_name"

    def test_description_from_docstring(self, simple_feature: type[mx.BaseFeature]):
        """Test that description comes from the feature's docstring."""
        dagster_type = feature_to_dagster_type(simple_feature)
        assert dagster_type.description == "A simple test feature."

    def test_description_fallback_no_docstring(self, feature_no_docstring: type[mx.BaseFeature]):
        """Test fallback description when feature has no docstring."""
        dagster_type = feature_to_dagster_type(feature_no_docstring)
        assert dagster_type.description is not None
        assert "Metaxy feature" in dagster_type.description
        assert "test/no_docs" in dagster_type.description

    def test_custom_description(self, simple_feature: type[mx.BaseFeature]):
        """Test that a custom description can be provided."""
        dagster_type = feature_to_dagster_type(simple_feature, description="Custom description")
        assert dagster_type.description == "Custom description"

    def test_column_schema_metadata(self, simple_feature: type[mx.BaseFeature]):
        """Test that column schema is included in metadata."""
        dagster_type = feature_to_dagster_type(simple_feature)
        assert dagster_type.metadata is not None
        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY in dagster_type.metadata

        schema = dagster_type.metadata[DAGSTER_COLUMN_SCHEMA_METADATA_KEY]
        assert isinstance(schema, dg.TableSchemaMetadataValue)

        column_names = [col.name for col in schema.value.columns]
        assert "id" in column_names
        assert "name" in column_names
        assert "value" in column_names

    def test_column_schema_disabled(self, simple_feature: type[mx.BaseFeature]):
        """Test that column schema can be disabled."""
        dagster_type = feature_to_dagster_type(simple_feature, inject_column_schema=False)
        # metaxy/info is always present, but column schema should not be
        assert dagster_type.metadata is not None
        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY not in dagster_type.metadata

    def test_type_check_valid_polars_dataframe(self, simple_feature: type[mx.BaseFeature]):
        """Test type check passes for Polars DataFrame."""
        dagster_type = feature_to_dagster_type(simple_feature)
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "value": [1.0, 2.0]})

        result = dagster_type.type_check(cast(Any, None), df)
        assert result.success is True

    def test_type_check_valid_polars_lazyframe(self, simple_feature: type[mx.BaseFeature]):
        """Test type check passes for Polars LazyFrame."""
        dagster_type = feature_to_dagster_type(simple_feature)
        lf = pl.LazyFrame({"id": [1, 2], "name": ["a", "b"], "value": [1.0, 2.0]})

        result = dagster_type.type_check(cast(Any, None), lf)
        assert result.success is True

    def test_type_check_valid_pandas_dataframe(self, simple_feature: type[mx.BaseFeature]):
        """Test type check passes for Pandas DataFrame."""
        dagster_type = feature_to_dagster_type(simple_feature)
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "value": [1.0, 2.0]})

        result = dagster_type.type_check(cast(Any, None), df)
        assert result.success is True

    def test_type_check_valid_none(self, simple_feature: type[mx.BaseFeature]):
        """Test type check passes for None (valid MetaxyOutput)."""
        dagster_type = feature_to_dagster_type(simple_feature)

        result = dagster_type.type_check(cast(Any, None), None)
        assert result.success is True

    def test_type_check_invalid_string(self, simple_feature: type[mx.BaseFeature]):
        """Test type check fails for invalid string."""
        dagster_type = feature_to_dagster_type(simple_feature)

        result = dagster_type.type_check(cast(Any, None), "invalid")
        assert result.success is False
        assert result.description is not None
        assert "test/simple" in result.description
        assert "str" in result.description

    def test_type_check_invalid_dict(self, simple_feature: type[mx.BaseFeature]):
        """Test type check fails for invalid dict."""
        dagster_type = feature_to_dagster_type(simple_feature)

        result = dagster_type.type_check(cast(Any, None), {"key": "value"})
        assert result.success is False
        assert result.description is not None
        assert "dict" in result.description

    def test_type_check_invalid_list(self, simple_feature: type[mx.BaseFeature]):
        """Test type check fails for invalid list."""
        dagster_type = feature_to_dagster_type(simple_feature)

        result = dagster_type.type_check(cast(Any, None), [1, 2, 3])
        assert result.success is False
        assert result.description is not None
        assert "list" in result.description

    def test_accepts_feature_key(self, simple_feature: type[mx.BaseFeature]):
        """Test that a FeatureKey can be passed instead of a feature class."""
        feature_key = simple_feature.spec().key
        dagster_type = feature_to_dagster_type(feature_key)
        assert dagster_type.display_name == "test__simple"

    def test_accepts_string_key(self, simple_feature: type[mx.BaseFeature]):
        """Test that a string key can be passed."""
        dagster_type = feature_to_dagster_type("test/simple")
        assert dagster_type.display_name == "test__simple"


class TestBuildColumnSchema:
    """Tests for build_column_schema function."""

    def test_builds_schema_from_feature(self, simple_feature: type[mx.BaseFeature]):
        """Test that a TableSchema is built from a feature class."""
        schema = build_column_schema(simple_feature)
        assert isinstance(schema, dg.TableSchema)

    def test_includes_all_fields(self, simple_feature: type[mx.BaseFeature]):
        """Test that all Pydantic fields are included."""
        schema = build_column_schema(simple_feature)
        assert schema is not None
        column_names = {col.name for col in schema.columns}

        # User-defined fields
        assert "id" in column_names
        assert "name" in column_names
        assert "value" in column_names

        # System columns (inherited from BaseFeature)
        assert "metaxy_created_at" in column_names
        assert "metaxy_data_version" in column_names

    def test_columns_sorted_alphabetically(self, simple_feature: type[mx.BaseFeature]):
        """Test that columns are sorted alphabetically."""
        schema = build_column_schema(simple_feature)
        assert schema is not None
        column_names = [col.name for col in schema.columns]
        assert column_names == sorted(column_names)

    def test_field_descriptions_preserved(self, simple_feature: type[mx.BaseFeature]):
        """Test that field descriptions are preserved."""
        schema = build_column_schema(simple_feature)
        assert schema is not None
        columns_by_name = {col.name: col for col in schema.columns}

        assert columns_by_name["id"].description == "Primary key"
        assert columns_by_name["name"].description == "Name field"
        assert columns_by_name["value"].description == "Optional value"

    def test_field_types_converted(self, simple_feature: type[mx.BaseFeature]):
        """Test that field types are converted to strings."""
        schema = build_column_schema(simple_feature)
        assert schema is not None
        columns_by_name = {col.name: col for col in schema.columns}

        assert columns_by_name["id"].type == "int"
        assert columns_by_name["name"].type == "str"
        # Optional types strip None
        assert columns_by_name["value"].type == "float"


class TestBuildColumnLineage:
    """Tests for build_column_lineage function."""

    def test_no_lineage_for_root_feature(self, simple_feature: type[mx.BaseFeature]):
        """Test that root features have no lineage."""
        lineage = build_column_lineage(simple_feature)
        assert lineage is None

    def test_builds_lineage_for_downstream(self, downstream_feature: type[mx.BaseFeature]):
        """Test that lineage is built for downstream features."""
        lineage = build_column_lineage(downstream_feature)
        assert lineage is not None
        assert isinstance(lineage, dg.TableColumnLineage)

    def test_includes_pass_through_columns(self, downstream_feature: type[mx.BaseFeature]):
        """Test that pass-through columns are tracked."""
        lineage = build_column_lineage(downstream_feature)
        assert lineage is not None
        # 'value' exists in both upstream and downstream
        assert "value" in lineage.deps_by_column

    def test_includes_id_columns(self, downstream_feature: type[mx.BaseFeature]):
        """Test that ID columns are tracked."""
        lineage = build_column_lineage(downstream_feature)
        assert lineage is not None
        # 'id' is an ID column in both
        assert "id" in lineage.deps_by_column

    def test_includes_system_columns_with_lineage(self, downstream_feature: type[mx.BaseFeature]):
        """Test that system columns with lineage are tracked."""
        lineage = build_column_lineage(downstream_feature)
        assert lineage is not None
        # System columns with lineage should be present
        assert "metaxy_provenance" in lineage.deps_by_column
        assert "metaxy_provenance_by_field" in lineage.deps_by_column


class TestColumnLineageInDagsterType:
    """Tests for column lineage in feature_to_dagster_type."""

    def test_no_lineage_for_root_feature(self, simple_feature: type[mx.BaseFeature]):
        """Test that root features have no lineage metadata."""
        dagster_type = feature_to_dagster_type(simple_feature)
        assert dagster_type.metadata is not None
        # Only column schema should be present, not lineage
        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY in dagster_type.metadata
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY not in dagster_type.metadata

    def test_lineage_for_downstream_feature(self, downstream_feature: type[mx.BaseFeature]):
        """Test that downstream features have lineage metadata."""
        dagster_type = feature_to_dagster_type(downstream_feature)
        assert dagster_type.metadata is not None
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY in dagster_type.metadata

    def test_lineage_disabled(self, downstream_feature: type[mx.BaseFeature]):
        """Test that column lineage can be disabled."""
        dagster_type = feature_to_dagster_type(downstream_feature, inject_column_lineage=False)
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY not in (dagster_type.metadata or {})


class TestWithDagsterAsset:
    """Test integration with @dg.asset decorator."""

    def test_asset_with_dagster_type(self, simple_feature: type[mx.BaseFeature]):
        """Test that feature_to_dagster_type works with @dg.asset."""
        from metaxy.ext.dagster import metaxify

        @metaxify
        @dg.asset(
            metadata={"metaxy/feature": "test/simple"},
            dagster_type=feature_to_dagster_type(simple_feature),
        )
        def my_asset():
            return pl.DataFrame({"id": [1], "name": ["test"], "value": [1.0]})

        assert isinstance(my_asset, dg.AssetsDefinition)
        output_def = my_asset.node_def.output_defs[0]
        assert output_def.dagster_type.display_name == "test__simple"


@pytest.fixture
def external_feature_definition(graph: mx.FeatureGraph) -> mx.FeatureDefinition:
    """Create an external feature definition for testing."""
    spec = mx.FeatureSpec(
        key=["test", "external"],
        id_columns=["id"],
        fields=["data"],
    )
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "data": {"type": "number"},
        },
    }
    external_def = mx.FeatureDefinition.external(
        spec=spec,
        feature_schema=schema,
        project="external-project",
    )
    graph.add_feature_definition(external_def)
    return external_def


class TestExternalFeatureDefinitions:
    """Tests for external feature definitions in Dagster integration."""

    def test_build_column_schema_raises_for_external(self, external_feature_definition: mx.FeatureDefinition):
        """build_column_schema raises ImportError for external features."""
        with pytest.raises(ImportError, match="external feature"):
            build_column_schema(external_feature_definition)

    def test_build_column_lineage_returns_none_for_external(self, external_feature_definition: mx.FeatureDefinition):
        """build_column_lineage returns None for external features (no deps)."""
        # External feature has no deps, so lineage should be None
        lineage = build_column_lineage(external_feature_definition)
        assert lineage is None

    def test_feature_to_dagster_type_skips_schema_for_external(self, external_feature_definition: mx.FeatureDefinition):
        """feature_to_dagster_type works without errors for external features."""
        # Should not raise, should just skip column schema/lineage
        dagster_type = feature_to_dagster_type(external_feature_definition)

        assert dagster_type is not None
        assert dagster_type.metadata is not None
        # Column schema should NOT be present (skipped for external)
        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY not in dagster_type.metadata
        # Column lineage should NOT be present (no deps)
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY not in dagster_type.metadata
