"""Comprehensive tests for FeatureGraph methods.

Tests cover:
- add_feature: Feature registration and replacement behavior
- topological_sort_features:
  - Basic topological sorting (linear chains, diamonds, no dependencies)
  - Subset sorting (sorting specific features from larger graph)
  - All features sorting (feature_keys=None)
  - Deterministic ordering (alphabetical ordering for same-level features)
  - Edge cases (empty graph, single feature, disconnected components)
  - Complex graphs (multi-level, wide graphs, multiple roots)
"""

from __future__ import annotations

import pytest
from metaxy_testing.models import SampleFeatureSpec

from metaxy import (
    BaseFeature,
    FeatureDefinition,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FeatureSpec,
    FieldKey,
    FieldSpec,
)


class TestAddFeature:
    """Test FeatureGraph.add_feature() behavior."""

    def test_add_feature_same_import_path_replaces_quietly(self, graph: FeatureGraph):
        """Test that re-registering the same class (same import path) replaces quietly."""

        class MyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "my_feature"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Feature is already registered via metaclass
        assert MyFeature.spec().key in graph.feature_definitions_by_key

        # Re-adding the same class should not raise an error
        graph.add_feature(MyFeature)

        # Should still be registered
        assert graph.feature_definitions_by_key[MyFeature.spec().key].key == MyFeature.spec().key

    def test_add_feature_different_class_same_key_raises_during_definition(self, graph: FeatureGraph):
        """Test that defining a different class with the same key raises ValueError."""

        class FeatureV1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "shared_key"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Defining a different class with the same key should raise during class definition
        with pytest.raises(ValueError, match="already registered"):

            class FeatureV2(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "shared_key"]),
                    fields=[FieldSpec(key=FieldKey(["y"]))],
                ),
            ):
                pass

    def test_add_feature_different_module_same_key_raises_error(self, graph: FeatureGraph):
        """Test that different classes with same key from different modules raise ValueError."""

        class OriginalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "unique_key"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Remove the feature so we can test manual registration
        graph.remove_feature(OriginalFeature.spec().key)

        # Re-add original
        graph.add_feature(OriginalFeature)

        # Create a mock class that simulates a class from a different module
        # by modifying __module__ before calling add_feature
        class ConflictingFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "conflict_key"]),  # Use different key for definition
                fields=[FieldSpec(key=FieldKey(["y"]))],
            ),
        ):
            pass

        # Remove and modify to simulate different module
        graph.remove_feature(ConflictingFeature.spec().key)
        ConflictingFeature.__module__ = "some.other.module"
        # Override the spec key to conflict with OriginalFeature
        ConflictingFeature._spec = SampleFeatureSpec(
            key=FeatureKey(["test", "unique_key"]),
            fields=[FieldSpec(key=FieldKey(["y"]))],
        )

        with pytest.raises(ValueError, match="already registered"):
            graph.add_feature(ConflictingFeature)


class TestBasicTopologicalSorting:
    """Test basic topological sorting scenarios."""

    def test_simple_linear_chain(self, graph: FeatureGraph):
        """Test sorting a simple linear dependency chain A -> B -> C."""

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

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "c"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
                deps=[FeatureDep(feature=FeatureB)],
            ),
        ):
            pass

        # Sort all features
        sorted_keys = graph.topological_sort_features()

        # Verify order: A before B before C
        assert len(sorted_keys) == 3
        a_idx = sorted_keys.index(FeatureA.spec().key)
        b_idx = sorted_keys.index(FeatureB.spec().key)
        c_idx = sorted_keys.index(FeatureC.spec().key)

        assert a_idx < b_idx < c_idx

    def test_diamond_dependency(self, graph: FeatureGraph):
        """Test sorting a diamond dependency pattern: A -> B, A -> C, B -> D, C -> D."""

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

        sorted_keys = graph.topological_sort_features()

        # Verify dependencies appear before dependents
        assert len(sorted_keys) == 4
        a_idx = sorted_keys.index(FeatureA.spec().key)
        b_idx = sorted_keys.index(FeatureB.spec().key)
        c_idx = sorted_keys.index(FeatureC.spec().key)
        d_idx = sorted_keys.index(FeatureD.spec().key)

        # A must come before B and C
        assert a_idx < b_idx
        assert a_idx < c_idx
        # B and C must come before D
        assert b_idx < d_idx
        assert c_idx < d_idx

    def test_no_dependencies(self, graph: FeatureGraph):
        """Test sorting features with no dependencies (all independent)."""

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

        sorted_keys = graph.topological_sort_features()

        # All features should be included, and since they're independent,
        # they should be in alphabetical order
        assert len(sorted_keys) == 3
        assert sorted_keys[0] == FeatureA.spec().key
        assert sorted_keys[1] == FeatureB.spec().key
        assert sorted_keys[2] == FeatureC.spec().key

    def test_single_feature(self, graph: FeatureGraph):
        """Test sorting a graph with a single feature."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        assert len(sorted_keys) == 1
        assert sorted_keys[0] == FeatureA.spec().key


class TestSubsetSorting:
    """Test sorting a subset of features from a larger graph."""

    def test_subset_sorting_linear_chain(self, graph: FeatureGraph):
        """Test sorting a subset of features from a linear chain."""

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

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "c"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
                deps=[FeatureDep(feature=FeatureB)],
            ),
        ):
            pass

        class FeatureD(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "d"]),
                fields=[FieldSpec(key=FieldKey(["w"]))],
                deps=[FeatureDep(feature=FeatureC)],
            ),
        ):
            pass

        # Sort only B and C (dependencies: A -> B -> C -> D)
        subset = [FeatureB.spec().key, FeatureC.spec().key]
        sorted_keys = graph.topological_sort_features(subset)

        # Should only include B and C in correct order
        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureB.spec().key
        assert sorted_keys[1] == FeatureC.spec().key

    def test_subset_with_dependencies_outside_subset(self, graph: FeatureGraph):
        """Test that dependencies outside the subset are handled correctly."""

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

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "c"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
                deps=[FeatureDep(feature=FeatureB)],
            ),
        ):
            pass

        # Sort only C (which depends on B, which depends on A)
        # A and B should not be in result
        sorted_keys = graph.topological_sort_features([FeatureC.spec().key])

        assert len(sorted_keys) == 1
        assert sorted_keys[0] == FeatureC.spec().key

    def test_subset_with_shared_dependency(self, graph: FeatureGraph):
        """Test subset where multiple features depend on same parent (not in subset)."""

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

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "c"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        # Sort B and C (both depend on A, which is not in subset)
        subset = [FeatureB.spec().key, FeatureC.spec().key]
        sorted_keys = graph.topological_sort_features(subset)

        # Should include B and C (alphabetically since same level)
        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureB.spec().key
        assert sorted_keys[1] == FeatureC.spec().key

    def test_subset_includes_dependency_in_subset(self, graph: FeatureGraph):
        """Test subset where both parent and child are in the subset."""

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

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "c"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
            ),
        ):
            pass

        # Sort A and B (A -> B)
        subset = [FeatureA.spec().key, FeatureB.spec().key]
        sorted_keys = graph.topological_sort_features(subset)

        # Should include A before B
        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureA.spec().key
        assert sorted_keys[1] == FeatureB.spec().key


class TestAllFeaturesSorting:
    """Test sorting all features (feature_keys=None)."""

    def test_all_features_none_parameter(self, graph: FeatureGraph):
        """Test that feature_keys=None sorts all features in the graph."""

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

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "c"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
            ),
        ):
            pass

        # Sort all features (explicitly pass None)
        sorted_keys = graph.topological_sort_features(None)

        # Should include all 3 features
        assert len(sorted_keys) == 3
        assert FeatureA.spec().key in sorted_keys
        assert FeatureB.spec().key in sorted_keys
        assert FeatureC.spec().key in sorted_keys

        # Verify A comes before B
        a_idx = sorted_keys.index(FeatureA.spec().key)
        b_idx = sorted_keys.index(FeatureB.spec().key)
        assert a_idx < b_idx

    def test_all_features_default_parameter(self, graph: FeatureGraph):
        """Test that calling without parameter sorts all features."""

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

        # Sort all features (no parameter)
        sorted_keys = graph.topological_sort_features()

        # Should include all features
        assert len(sorted_keys) == 2
        assert FeatureA.spec().key in sorted_keys
        assert FeatureB.spec().key in sorted_keys


class TestDeterministicOrdering:
    """Test that features at the same level are sorted alphabetically."""

    def test_alphabetical_ordering_same_level(self, graph: FeatureGraph):
        """Test features at same level are sorted alphabetically."""

        class FeatureParent(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "parent"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Create children with names in non-alphabetical order
        class FeatureZ(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "z"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
                deps=[FeatureDep(feature=FeatureParent)],
            ),
        ):
            pass

        class FeatureM(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "m"]),
                fields=[FieldSpec(key=FieldKey(["m"]))],
                deps=[FeatureDep(feature=FeatureParent)],
            ),
        ):
            pass

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "a"]),
                fields=[FieldSpec(key=FieldKey(["a"]))],
                deps=[FeatureDep(feature=FeatureParent)],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # Parent should be first
        assert sorted_keys[0] == FeatureParent.spec().key

        # Children should be in alphabetical order
        assert sorted_keys[1] == FeatureA.spec().key
        assert sorted_keys[2] == FeatureM.spec().key
        assert sorted_keys[3] == FeatureZ.spec().key

    def test_alphabetical_ordering_multiple_levels(self, graph: FeatureGraph):
        """Test alphabetical ordering is maintained at multiple levels."""

        # Root level - multiple roots
        class FeatureZ(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "z"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "a"]),
                fields=[FieldSpec(key=FieldKey(["y"]))],
            ),
        ):
            pass

        # Second level - depend on roots
        class FeatureM2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "m2"]),
                fields=[FieldSpec(key=FieldKey(["m"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureB2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "b2"]),
                fields=[FieldSpec(key=FieldKey(["b"]))],
                deps=[FeatureDep(feature=FeatureZ)],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # Roots should be in alphabetical order
        a_idx = sorted_keys.index(FeatureA.spec().key)
        z_idx = sorted_keys.index(FeatureZ.spec().key)
        assert a_idx < z_idx

        # Children should come after parents
        m2_idx = sorted_keys.index(FeatureM2.spec().key)
        b2_idx = sorted_keys.index(FeatureB2.spec().key)
        assert a_idx < m2_idx
        assert z_idx < b2_idx

    def test_alphabetical_ordering(self, graph: FeatureGraph):
        """Test that same-level features are sorted alphabetically."""

        class FeatureParent(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "parent"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Create children with names that test alphabetical ordering
        class FeatureZulu(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "zulu"]),
                fields=[FieldSpec(key=FieldKey(["z"]))],
                deps=[FeatureDep(feature=FeatureParent)],
            ),
        ):
            pass

        class FeatureAlpha(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "alpha"]),
                fields=[FieldSpec(key=FieldKey(["a"]))],
                deps=[FeatureDep(feature=FeatureParent)],
            ),
        ):
            pass

        class FeatureMike(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "mike"]),
                fields=[FieldSpec(key=FieldKey(["m"]))],
                deps=[FeatureDep(feature=FeatureParent)],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # Parent should be first
        assert sorted_keys[0] == FeatureParent.spec().key

        # Children should be in alphabetical order
        # alpha < mike < zulu
        assert sorted_keys[1] == FeatureAlpha.spec().key
        assert sorted_keys[2] == FeatureMike.spec().key
        assert sorted_keys[3] == FeatureZulu.spec().key


class TestEdgeCases:
    """Test edge cases for topological sorting."""

    def test_empty_graph(self, graph: FeatureGraph):
        """Test sorting an empty graph returns empty list."""
        sorted_keys = graph.topological_sort_features()
        assert sorted_keys == []

    def test_empty_subset(self, graph: FeatureGraph):
        """Test sorting an empty subset returns empty list."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features([])
        assert sorted_keys == []

    def test_single_feature_in_larger_graph(self, graph: FeatureGraph):
        """Test sorting a single feature from a larger graph."""

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

        sorted_keys = graph.topological_sort_features([FeatureA.spec().key])
        assert len(sorted_keys) == 1
        assert sorted_keys[0] == FeatureA.spec().key

    def test_disconnected_components(self, graph: FeatureGraph):
        """Test graph with multiple disconnected components."""

        # Component 1: A -> B
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

        # Component 2: X -> Y (independent from A and B)
        class FeatureX(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "x"]),
                fields=[FieldSpec(key=FieldKey(["x2"]))],
            ),
        ):
            pass

        class FeatureY(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "y"]),
                fields=[FieldSpec(key=FieldKey(["y2"]))],
                deps=[FeatureDep(feature=FeatureX)],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # All features should be included
        assert len(sorted_keys) == 4

        # Verify dependencies within each component
        a_idx = sorted_keys.index(FeatureA.spec().key)
        b_idx = sorted_keys.index(FeatureB.spec().key)
        x_idx = sorted_keys.index(FeatureX.spec().key)
        y_idx = sorted_keys.index(FeatureY.spec().key)

        assert a_idx < b_idx
        assert x_idx < y_idx

    def test_feature_with_no_downstream_dependents(self, graph: FeatureGraph):
        """Test feature that has no downstream dependents (leaf node)."""

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

        sorted_keys = graph.topological_sort_features()

        # B is a leaf node (no downstream dependents)
        # Should still be included in correct order
        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureA.spec().key
        assert sorted_keys[1] == FeatureB.spec().key


class TestComplexGraphs:
    """Test complex graph patterns."""

    def test_multi_level_deep_dependencies(self, graph: FeatureGraph):
        """Test graph with 5+ levels of dependencies."""

        class FeatureL1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "l1"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureL2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "l2"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureL1)],
            ),
        ):
            pass

        class FeatureL3(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "l3"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureL2)],
            ),
        ):
            pass

        class FeatureL4(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "l4"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureL3)],
            ),
        ):
            pass

        class FeatureL5(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "l5"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureL4)],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # All levels should be in order
        assert len(sorted_keys) == 5
        l1_idx = sorted_keys.index(FeatureL1.spec().key)
        l2_idx = sorted_keys.index(FeatureL2.spec().key)
        l3_idx = sorted_keys.index(FeatureL3.spec().key)
        l4_idx = sorted_keys.index(FeatureL4.spec().key)
        l5_idx = sorted_keys.index(FeatureL5.spec().key)

        assert l1_idx < l2_idx < l3_idx < l4_idx < l5_idx

    def test_wide_graph_many_children(self, graph: FeatureGraph):
        """Test graph with many features at same level depending on same parent."""

        class FeatureParent(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "parent"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Create 10 children all depending on parent
        children = []
        for i in range(10):
            child_name = f"child_{i:02d}"

            class ChildFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", child_name]),
                    fields=[FieldSpec(key=FieldKey(["x"]))],
                    deps=[FeatureDep(feature=FeatureParent)],
                ),
            ):
                pass

            children.append(ChildFeature)

        sorted_keys = graph.topological_sort_features()

        # Parent should be first
        assert sorted_keys[0] == FeatureParent.spec().key

        # All children should come after parent and be in alphabetical order
        parent_idx = 0
        for i, child in enumerate(children):
            child_idx = sorted_keys.index(child.spec().key)
            assert child_idx > parent_idx
            # Verify alphabetical ordering of children
            if i > 0:
                prev_child_idx = sorted_keys.index(children[i - 1].spec().key)
                assert prev_child_idx < child_idx

    def test_multiple_root_features(self, graph: FeatureGraph):
        """Test graph with multiple root features (no dependencies)."""

        # Create 5 root features
        class FeatureR1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "r1"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureR2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "r2"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureR3(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "r3"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureR4(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "r4"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureR5(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "r5"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # All roots should be included in alphabetical order
        assert len(sorted_keys) == 5
        assert sorted_keys[0] == FeatureR1.spec().key
        assert sorted_keys[1] == FeatureR2.spec().key
        assert sorted_keys[2] == FeatureR3.spec().key
        assert sorted_keys[3] == FeatureR4.spec().key
        assert sorted_keys[4] == FeatureR5.spec().key

    def test_complex_multi_path_dependencies(self, graph: FeatureGraph):
        """Test complex graph where features can be reached via multiple paths."""

        # Create a graph where D can be reached from A via multiple paths:
        # Path 1: A -> B -> D
        # Path 2: A -> C -> D
        # Path 3: A -> E -> F -> D
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
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "c"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureE(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "e"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureF(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "f"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureE)],
            ),
        ):
            pass

        class FeatureD(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "d"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[
                    FeatureDep(feature=FeatureB),
                    FeatureDep(feature=FeatureC),
                    FeatureDep(feature=FeatureF),
                ],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # Verify all features are included
        assert len(sorted_keys) == 6

        # A must come first
        a_idx = sorted_keys.index(FeatureA.spec().key)
        assert a_idx == 0

        # B, C, E must come after A
        b_idx = sorted_keys.index(FeatureB.spec().key)
        c_idx = sorted_keys.index(FeatureC.spec().key)
        e_idx = sorted_keys.index(FeatureE.spec().key)
        assert a_idx < b_idx
        assert a_idx < c_idx
        assert a_idx < e_idx

        # F must come after E
        f_idx = sorted_keys.index(FeatureF.spec().key)
        assert e_idx < f_idx

        # D must come after B, C, and F
        d_idx = sorted_keys.index(FeatureD.spec().key)
        assert b_idx < d_idx
        assert c_idx < d_idx
        assert f_idx < d_idx

    def test_nested_namespace_keys(self, graph: FeatureGraph):
        """Test that topological sort works with nested namespace keys."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "raw", "frames"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processed", "embeddings"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["audio", "raw", "waveform"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features()

        # All features should be included
        assert len(sorted_keys) == 3

        # Verify alphabetical ordering by full path string
        # audio/raw/waveform < video/processed/embeddings < video/raw/frames
        # But also verify dependency: frames before embeddings
        frames_idx = sorted_keys.index(FeatureA.spec().key)
        embeddings_idx = sorted_keys.index(FeatureB.spec().key)
        assert frames_idx < embeddings_idx


class TestExternalDefinitions:
    """Test external feature definitions with FeatureGraph."""

    def test_add_external_definition_to_graph(self, graph: FeatureGraph):
        """External definitions can be added to a FeatureGraph."""

        spec = FeatureSpec(
            key=FeatureKey(["external", "test"]),
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

        external_def = FeatureDefinition.external(
            spec=spec,
            feature_schema=schema,
            project="other-project",
        )

        graph.add_feature_definition(external_def)

        # Verify it can be retrieved
        retrieved = graph.get_feature_definition(["external", "test"])
        assert retrieved == external_def
        assert retrieved.is_external is True
        assert retrieved.project == "other-project"

    def test_external_definition_excluded_from_graph_snapshot(self, graph: FeatureGraph):
        """External definitions are excluded from graph snapshots."""

        spec = FeatureSpec(
            key=FeatureKey(["snap", "external"]),
            id_columns=["uid"],
        )
        schema = {"type": "object", "properties": {"uid": {"type": "string"}}}

        external_def = FeatureDefinition.external(
            spec=spec,
            feature_schema=schema,
            project="snapshot-proj",
        )

        graph.add_feature_definition(external_def)
        # Pass project explicitly since graph only has external features
        snapshot = graph.to_snapshot(project="snapshot-proj")

        # External features should NOT be in the snapshot
        assert "snap/external" not in snapshot
        assert len(snapshot) == 0

    def test_external_definition_coexists_with_class_definitions(self, graph: FeatureGraph):
        """External definitions and class-based definitions can coexist in the same graph."""

        # Add a class-based feature
        class ClassFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "class_based"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Add an external feature
        external_spec = FeatureSpec(
            key=FeatureKey(["test", "external_based"]),
            id_columns=["id"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            project="external-proj",
        )
        graph.add_feature_definition(external_def)

        # Both should be retrievable
        class_def = graph.get_feature_definition(["test", "class_based"])
        ext_def = graph.get_feature_definition(["test", "external_based"])

        assert class_def.is_external is False
        assert ext_def.is_external is True

        # Both should appear in list_features
        all_keys = graph.list_features(only_current_project=False)
        assert FeatureKey(["test", "class_based"]) in all_keys
        assert FeatureKey(["test", "external_based"]) in all_keys

    def test_external_definition_in_topological_sort(self, graph: FeatureGraph):
        """External definitions participate in topological sorting."""

        # Add two external features (no dependencies between them)
        for name in ["alpha", "beta"]:
            spec = FeatureSpec(
                key=FeatureKey(["topo", name]),
                id_columns=["id"],
            )
            definition = FeatureDefinition.external(
                spec=spec,
                feature_schema={"type": "object", "properties": {"id": {"type": "string"}}},
                project="topo-proj",
            )
            graph.add_feature_definition(definition)

        sorted_keys = graph.topological_sort_features()

        # Both should be in sorted result in alphabetical order (alpha before beta)
        assert len(sorted_keys) == 2
        alpha_idx = sorted_keys.index(FeatureKey(["topo", "alpha"]))
        beta_idx = sorted_keys.index(FeatureKey(["topo", "beta"]))
        assert alpha_idx < beta_idx

    def test_normal_feature_replaces_external_feature(self, graph: FeatureGraph):
        """Non-external feature replaces external feature with same key."""

        # First add an external feature
        external_spec = FeatureSpec(
            key=FeatureKey(["test", "replaceable"]),
            id_columns=["id"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            project="external-proj",
        )
        graph.add_feature_definition(external_def)

        # Verify it's in the graph as external
        assert graph.get_feature_definition(["test", "replaceable"]).is_external is True

        # Now define a class-based feature with the same key
        class ReplaceableFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "replaceable"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
        ):
            pass

        # The class-based feature should replace the external one
        retrieved = graph.get_feature_definition(["test", "replaceable"])
        assert retrieved.is_external is False
        assert retrieved.feature_class_path is not None
        assert "ReplaceableFeature" in retrieved.feature_class_path

    def test_external_feature_does_not_overwrite_normal_feature(self, graph: FeatureGraph):
        """External feature does not overwrite existing non-external feature."""

        # First define a class-based feature
        class ExistingFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "existing"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
        ):
            pass

        original_class_path = graph.get_feature_definition(["test", "existing"]).feature_class_path

        # Try to add an external feature with the same key
        external_spec = FeatureSpec(
            key=FeatureKey(["test", "existing"]),
            id_columns=["id"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            project="external-proj",
        )
        graph.add_feature_definition(external_def)

        # The original class-based feature should still be there
        retrieved = graph.get_feature_definition(["test", "existing"])
        assert retrieved.is_external is False
        assert retrieved.feature_class_path == original_class_path

    def test_external_features_excluded_from_snapshot_with_normal_features(self, graph: FeatureGraph):
        """Snapshot only includes non-external features."""

        # Add a class-based feature
        class NormalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "normal"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
        ):
            pass

        # Add an external feature
        external_spec = FeatureSpec(
            key=FeatureKey(["test", "external_snap"]),
            id_columns=["id"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            project="external-proj",
        )
        graph.add_feature_definition(external_def)

        # Both should be in the graph
        assert len(graph.feature_definitions_by_key) == 2

        # Only the normal feature should be in the snapshot
        snapshot = graph.to_snapshot()
        assert "test/normal" in snapshot
        assert "test/external_snap" not in snapshot
        assert len(snapshot) == 1

    def test_external_feature_with_missing_external_dependency_raises_error(self, graph: FeatureGraph):
        """External feature depending on missing external parent should raise error on graph operations."""
        import pytest

        from metaxy.utils.exceptions import MetaxyMissingFeatureDependency

        # Add an external feature that depends on another external feature that doesn't exist
        child_spec = FeatureSpec(
            key=FeatureKey(["external", "child"]),
            id_columns=["id"],
            deps=[FeatureDep(feature=FeatureKey(["external", "parent"]))],
        )
        child_def = FeatureDefinition.external(
            spec=child_spec,
            feature_schema={"type": "object"},
            project="external-project",
        )
        graph.add_feature_definition(child_def)

        # The child is in the graph
        assert FeatureKey(["external", "child"]) in graph.feature_definitions_by_key

        # But the parent is NOT in the graph
        assert FeatureKey(["external", "parent"]) not in graph.feature_definitions_by_key

        # Trying to get the feature plan should raise an error
        with pytest.raises(MetaxyMissingFeatureDependency) as exc_info:
            graph.get_feature_plan(["external", "child"])

        assert "external/parent" in str(exc_info.value)
        assert "external/child" in str(exc_info.value)


class TestExternalProvenanceOverride:
    """Tests for external features with provenance overrides in FeatureGraph."""

    def test_get_field_version_returns_override_for_external(self, graph: FeatureGraph):
        """get_field_version returns provenance override instead of computing."""
        from metaxy.models.plan import FQFieldKey

        external_spec = FeatureSpec(
            key=FeatureKey(["ext", "upstream"]),
            id_columns=["id"],
            fields=["value", "other"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={
                "type": "object",
                "properties": {"id": {}, "value": {}, "other": {}},
            },
            project="external-proj",
            provenance_by_field={"value": "override_abc", "other": "override_def"},
        )
        graph.add_feature_definition(external_def)

        # get_field_version should return the override directly
        value_version = graph.get_field_version(
            FQFieldKey(feature=FeatureKey(["ext", "upstream"]), field=FieldKey(["value"]))
        )
        other_version = graph.get_field_version(
            FQFieldKey(feature=FeatureKey(["ext", "upstream"]), field=FieldKey(["other"]))
        )

        assert value_version == "override_abc"
        assert other_version == "override_def"

    def test_get_feature_version_uses_overridden_field_versions(self, graph: FeatureGraph):
        """get_feature_version computes correctly from overridden field versions."""
        external_spec = FeatureSpec(
            key=FeatureKey(["ext", "versioned"]),
            id_columns=["id"],
            fields=["field_a", "field_b"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={
                "type": "object",
                "properties": {"id": {}, "field_a": {}, "field_b": {}},
            },
            project="ext-proj",
            provenance_by_field={"field_a": "hash_a", "field_b": "hash_b"},
        )
        graph.add_feature_definition(external_def)

        # get_feature_version_by_field should return the overrides
        version_by_field = graph.get_feature_version_by_field(["ext", "versioned"])
        assert version_by_field == {"field_a": "hash_a", "field_b": "hash_b"}

        # get_feature_version should be deterministic based on overrides
        version1 = graph.get_feature_version(["ext", "versioned"])
        version2 = graph.get_feature_version(["ext", "versioned"])
        assert version1 == version2
        assert len(version1) > 0

    def test_downstream_version_uses_upstream_override(self, graph: FeatureGraph):
        """Downstream feature version is computed correctly from external override."""
        # Add external feature with provenance override
        external_spec = FeatureSpec(
            key=FeatureKey(["ext", "parent"]),
            id_columns=["id"],
            fields=["parent_field"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={
                "type": "object",
                "properties": {"id": {}, "parent_field": {}},
            },
            project="ext-proj",
            provenance_by_field={"parent_field": "stable_hash_123"},
        )
        graph.add_feature_definition(external_def)

        # Add downstream feature that depends on the external
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                fields=[FieldSpec(key=FieldKey(["child_field"]), code_version="1")],
                deps=[FeatureDep(feature=FeatureKey(["ext", "parent"]))],
            ),
        ):
            pass

        # Version should be computable
        version = graph.get_feature_version(["test", "downstream"])
        assert len(version) > 0

        # Version should be stable (based on override, not dynamically computed)
        version2 = graph.get_feature_version(["test", "downstream"])
        assert version == version2

    def test_snapshot_succeeds_with_provenance_override(self, graph: FeatureGraph):
        """to_snapshot succeeds when external deps have provenance overrides."""
        # Add external feature with provenance override
        external_spec = FeatureSpec(
            key=FeatureKey(["ext", "snap_parent"]),
            id_columns=["id"],
            fields=["data"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {}, "data": {}}},
            project="ext-proj",
            provenance_by_field={"data": "provenance_hash"},
        )
        graph.add_feature_definition(external_def)

        # Add downstream feature that depends on external
        class SnapshotableFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "snapshotable"]),
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                deps=[FeatureDep(feature=FeatureKey(["ext", "snap_parent"]))],
            ),
        ):
            pass

        # Snapshot should succeed (external has override)
        snapshot = graph.to_snapshot()

        # Downstream feature is in snapshot
        assert "test/snapshotable" in snapshot
        # External feature is NOT in snapshot (still external)
        assert "ext/snap_parent" not in snapshot

    def test_snapshot_succeeds_without_provenance_override(self, graph: FeatureGraph):
        """to_snapshot succeeds even when external deps lack provenance override.

        This enables entangled multi-project setups where projects can push
        to fresh stores without requiring external dependencies to be resolved first.
        External features are excluded from the snapshot.
        """
        # Add external feature WITHOUT provenance override
        external_spec = FeatureSpec(
            key=FeatureKey(["ext", "no_override"]),
            id_columns=["id"],
            fields=["data"],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {}, "data": {}}},
            project="ext-proj",
            # No provenance_by_field!
        )
        graph.add_feature_definition(external_def)

        # Add downstream feature that depends on external
        class SnapshotFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "snap_feature"]),
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                deps=[FeatureDep(feature=FeatureKey(["ext", "no_override"]))],
            ),
        ):
            pass

        # Snapshot should succeed - external features are excluded
        snapshot = graph.to_snapshot()

        # Only the non-external feature should be in snapshot
        assert "test/snap_feature" in snapshot
        assert "ext/no_override" not in snapshot

    def test_external_without_override_works_when_chain_loaded(self, graph: FeatureGraph):
        """External feature without override works when full chain is loaded in graph."""

        # Add the "real" parent feature (non-external) first
        class RealParentFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["real", "parent"]),
                fields=[FieldSpec(key=FieldKey(["parent_data"]), code_version="1")],
            ),
        ):
            pass

        # Add external feature that depends on the real parent
        external_spec = FeatureSpec(
            key=FeatureKey(["ext", "middle"]),
            id_columns=["id"],
            fields=["middle_data"],
            deps=[FeatureDep(feature=FeatureKey(["real", "parent"]))],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {}, "middle_data": {}}},
            project="ext-proj",
            # No provenance override - relies on full chain
        )
        graph.add_feature_definition(external_def)

        # Version computation should work because full chain is available
        version = graph.get_feature_version(["ext", "middle"])
        assert len(version) > 0

    def test_version_differs_with_different_overrides(self, graph: FeatureGraph):
        """Feature versions differ based on different provenance overrides."""
        from metaxy import FeatureGraph

        # Create two graphs with same structure but different overrides
        graph1 = FeatureGraph()
        graph2 = FeatureGraph()

        for g, override_value in [(graph1, "hash_v1"), (graph2, "hash_v2")]:
            ext_spec = FeatureSpec(
                key=FeatureKey(["ext", "ver_test"]),
                id_columns=["id"],
                fields=["field"],
            )
            ext_def = FeatureDefinition.external(
                spec=ext_spec,
                feature_schema={"type": "object", "properties": {"id": {}, "field": {}}},
                project="proj",
                provenance_by_field={"field": override_value},
            )
            g.add_feature_definition(ext_def)

        # Versions should differ
        version1 = graph1.get_feature_version(["ext", "ver_test"])
        version2 = graph2.get_feature_version(["ext", "ver_test"])
        assert version1 != version2

    def test_partial_override_falls_back_to_computation(self, graph: FeatureGraph):
        """Fields not in override fall back to normal computation."""
        from metaxy.models.plan import FQFieldKey

        # Add a root feature that the external depends on
        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "root"]),
                fields=[FieldSpec(key=FieldKey(["root_field"]), code_version="1")],
            ),
        ):
            pass

        # Add external with partial override (only some fields)
        external_spec = FeatureSpec(
            key=FeatureKey(["ext", "partial"]),
            id_columns=["id"],
            fields=[
                FieldSpec(key=FieldKey(["overridden"]), code_version="1"),
                FieldSpec(key=FieldKey(["computed"]), code_version="1"),
            ],
            deps=[FeatureDep(feature=FeatureKey(["test", "root"]))],
        )
        external_def = FeatureDefinition.external(
            spec=external_spec,
            feature_schema={"type": "object", "properties": {"id": {}, "overridden": {}, "computed": {}}},
            project="ext-proj",
            provenance_by_field={"overridden": "manual_hash"},  # Only override one field
        )
        graph.add_feature_definition(external_def)

        # Overridden field returns override
        overridden_version = graph.get_field_version(
            FQFieldKey(feature=FeatureKey(["ext", "partial"]), field=FieldKey(["overridden"]))
        )
        assert overridden_version == "manual_hash"

        # Non-overridden field is computed normally (has normal hash format)
        computed_version = graph.get_field_version(
            FQFieldKey(feature=FeatureKey(["ext", "partial"]), field=FieldKey(["computed"]))
        )
        assert computed_version != "manual_hash"
        assert len(computed_version) > 0  # Has a computed hash
