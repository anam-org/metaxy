"""Tests for automatic error table generation and system feature handling.

Tests cover:
- Error table auto-creation when adding features
- System feature filtering in graph operations
- is_system property on FeatureSpec
- include_system parameter in graph methods
"""

from __future__ import annotations

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
)
from metaxy._testing.models import SampleFeatureSpec
from metaxy.graph.diff.models import GraphData
from metaxy.models.feature_spec import (
    ERROR_TABLE_METADATA_KEY,
    SYSTEM_METADATA_KEY,
    FeatureSpec,
)


class TestErrorTableCreation:
    """Test automatic error table creation when adding features."""

    def test_error_table_created_for_each_feature(self, graph: FeatureGraph):
        """Test that adding a feature automatically creates its error table."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "my_feature"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Feature should be in features_by_key
        assert FeatureKey(["test", "my_feature"]) in graph.features_by_key

        # Error table should be in feature_specs_by_key but NOT in features_by_key
        error_key = FeatureKey(["test", "my_feature", "errors"])
        assert error_key in graph.feature_specs_by_key
        assert error_key not in graph.features_by_key

    def test_error_table_has_correct_structure(self, graph: FeatureGraph):
        """Test that error tables have the correct spec structure."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "my_feature"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        error_key = FeatureKey(["test", "my_feature", "errors"])
        error_spec = graph.feature_specs_by_key[error_key]

        # Error table should have same ID columns as original
        assert tuple(error_spec.id_columns) == tuple(MyFeature.spec().id_columns)

        # Error table should have no dependencies
        assert error_spec.deps == []

        # Error table should have a default field
        assert len(error_spec.fields) == 1
        assert error_spec.fields[0].key == FieldKey(["default"])

    def test_error_table_has_system_metadata(self, graph: FeatureGraph):
        """Test that error tables are marked as system features."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "my_feature"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        error_key = FeatureKey(["test", "my_feature", "errors"])
        error_spec = graph.feature_specs_by_key[error_key]

        # Error table should have system metadata
        assert error_spec.metadata.get(ERROR_TABLE_METADATA_KEY) is True
        assert error_spec.metadata.get(SYSTEM_METADATA_KEY) is True

        # is_system property should return True
        assert error_spec.is_system is True


class TestIsSystemProperty:
    """Test the is_system property on FeatureSpec."""

    def test_regular_feature_is_not_system(self, graph: FeatureGraph):
        """Test that regular features are not marked as system."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "regular"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        assert MyFeature.spec().is_system is False

    def test_feature_with_explicit_system_metadata(self):
        """Test is_system property with explicit metadata."""
        spec = FeatureSpec(
            key=FeatureKey(["test", "system"]),
            id_columns=("sample_uid",),
            fields=[FieldSpec(key=FieldKey(["default"]))],
            metadata={SYSTEM_METADATA_KEY: True},
        )
        assert spec.is_system is True

    def test_feature_with_system_metadata_false(self):
        """Test is_system property when explicitly set to False."""
        spec = FeatureSpec(
            key=FeatureKey(["test", "not_system"]),
            id_columns=("sample_uid",),
            fields=[FieldSpec(key=FieldKey(["default"]))],
            metadata={SYSTEM_METADATA_KEY: False},
        )
        assert spec.is_system is False


class TestSystemFeatureFiltering:
    """Test filtering of system features in graph operations."""

    def test_topological_sort_excludes_system_by_default(self, graph: FeatureGraph):
        """Test that topological_sort_features excludes system features by default."""

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
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        # Default behavior: exclude system features
        sorted_keys = graph.topological_sort_features()
        assert len(sorted_keys) == 2
        assert FeatureKey(["test", "a"]) in sorted_keys
        assert FeatureKey(["test", "b"]) in sorted_keys
        assert FeatureKey(["test", "a", "errors"]) not in sorted_keys
        assert FeatureKey(["test", "b", "errors"]) not in sorted_keys

    def test_topological_sort_includes_system_when_requested(self, graph: FeatureGraph):
        """Test that topological_sort_features includes system features when requested."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "my_feature"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Include system features
        sorted_keys = graph.topological_sort_features(include_system=True)
        assert len(sorted_keys) == 2
        assert FeatureKey(["test", "my_feature"]) in sorted_keys
        assert FeatureKey(["test", "my_feature", "errors"]) in sorted_keys

    def test_get_downstream_features_excludes_system_by_default(
        self, graph: FeatureGraph
    ):
        """Test that get_downstream_features excludes system features by default."""

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
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        downstream = graph.get_downstream_features([FeatureKey(["test", "a"])])
        assert len(downstream) == 1
        assert FeatureKey(["test", "b"]) in downstream
        # Error tables should not be in downstream
        assert FeatureKey(["test", "a", "errors"]) not in downstream
        assert FeatureKey(["test", "b", "errors"]) not in downstream


class TestGraphDataSystemFeatures:
    """Test GraphData handling of system features."""

    def test_from_feature_graph_excludes_system_by_default(self, graph: FeatureGraph):
        """Test that GraphData.from_feature_graph excludes system features by default."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "my_feature"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        graph_data = GraphData.from_feature_graph(graph)

        # Only the main feature should be included
        assert len(graph_data.nodes) == 1
        assert "test/my_feature" in graph_data.nodes
        assert "test/my_feature/errors" not in graph_data.nodes

    def test_from_feature_graph_includes_system_when_requested(
        self, graph: FeatureGraph
    ):
        """Test that GraphData.from_feature_graph includes system features when requested."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "my_feature"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        graph_data = GraphData.from_feature_graph(graph, include_system=True)

        # Both main feature and error table should be included
        assert len(graph_data.nodes) == 2
        assert "test/my_feature" in graph_data.nodes
        assert "test/my_feature/errors" in graph_data.nodes


class TestMultipleFeatures:
    """Test error table creation with multiple features."""

    def test_each_feature_gets_own_error_table(self, graph: FeatureGraph):
        """Test that each feature gets its own error table."""

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
            ),
        ):
            pass

        # Each feature should have its own error table
        assert FeatureKey(["test", "a", "errors"]) in graph.feature_specs_by_key
        assert FeatureKey(["test", "b", "errors"]) in graph.feature_specs_by_key
        assert FeatureKey(["test", "c", "errors"]) in graph.feature_specs_by_key

        # Without system features, should only have 3 features
        sorted_keys = graph.topological_sort_features()
        assert len(sorted_keys) == 3

        # With system features, should have 6 (3 features + 3 error tables)
        sorted_keys_with_system = graph.topological_sort_features(include_system=True)
        assert len(sorted_keys_with_system) == 6
