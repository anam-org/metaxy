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
    FeatureDep,
    FeatureGraph,
    FeatureKey,
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
        assert MyFeature.spec().key in graph.features_by_key

        # Re-adding the same class should not raise an error
        graph.add_feature(MyFeature)

        # Should still be registered
        assert graph.features_by_key[MyFeature.spec().key] is MyFeature

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
