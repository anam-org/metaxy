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

from metaxy import (
    BaseFeature,
    CascadeMode,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
)
from metaxy._testing.models import SampleFeatureSpec


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

    def test_simple_linear_chain(self, graph: FeatureGraph, linear_chain):
        """Test sorting a simple linear dependency chain A -> B -> C."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Sort all features
        sorted_keys = graph.topological_sort_features()

        # Verify order: A before B before C
        assert len(sorted_keys) == 3
        a_idx = sorted_keys.index(a.spec().key)
        b_idx = sorted_keys.index(b.spec().key)
        c_idx = sorted_keys.index(c.spec().key)

        assert a_idx < b_idx < c_idx

    def test_diamond_dependency(self, graph: FeatureGraph, diamond_graph):
        """Test sorting a diamond dependency pattern: A -> B, A -> C, B -> D, C -> D."""
        a, b, c, d = diamond_graph["a"], diamond_graph["b"], diamond_graph["c"], diamond_graph["d"]

        sorted_keys = graph.topological_sort_features()

        # Verify dependencies appear before dependents
        assert len(sorted_keys) == 4
        a_idx = sorted_keys.index(a.spec().key)
        b_idx = sorted_keys.index(b.spec().key)
        c_idx = sorted_keys.index(c.spec().key)
        d_idx = sorted_keys.index(d.spec().key)

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


class TestGetCascadeFeatures:
    """Tests for FeatureGraph.get_cascade_features method."""

    def test_cascade_downstream_linear_chain(self, graph: FeatureGraph, linear_chain):
        """Test downstream cascade on a linear dependency chain."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Cascade downstream from A should get [C, B, A] (leaf first)
        result = graph.get_cascade_features("a", "downstream")
        assert len(result) == 3
        assert result == [c.spec().key, b.spec().key, a.spec().key]

    def test_cascade_upstream_linear_chain(self, graph: FeatureGraph, linear_chain):
        """Test upstream cascade on a linear dependency chain."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Cascade upstream from C should get [A, B, C] (root first)
        result = graph.get_cascade_features("c", "upstream")
        assert len(result) == 3
        assert result == [a.spec().key, b.spec().key, c.spec().key]

    def test_cascade_both_linear_chain(self, graph: FeatureGraph, linear_chain):
        """Test both cascade directions on a linear chain."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Cascade both from B should get [A, B, C] (dependencies first, then self, then dependents)
        result = graph.get_cascade_features("b", "both")
        assert len(result) == 3
        assert result == [a.spec().key, b.spec().key, c.spec().key]

    def test_cascade_downstream_diamond_pattern(self, graph: FeatureGraph, diamond_graph):
        """Test downstream cascade on diamond dependency pattern."""
        a, b, c, d = diamond_graph["a"], diamond_graph["b"], diamond_graph["c"], diamond_graph["d"]

        # Cascade downstream from A should include all features, with D before B and C
        result = graph.get_cascade_features("a", "downstream")
        assert len(result) == 4
        assert a.spec().key in result
        assert b.spec().key in result
        assert c.spec().key in result
        assert d.spec().key in result

        # D should come before B and C (leaf first)
        d_idx = result.index(d.spec().key)
        b_idx = result.index(b.spec().key)
        c_idx = result.index(c.spec().key)
        a_idx = result.index(a.spec().key)
        assert d_idx < b_idx
        assert d_idx < c_idx
        # A should be last (root)
        assert a_idx == len(result) - 1

    def test_cascade_upstream_diamond_pattern(self, graph: FeatureGraph, diamond_graph):
        """Test upstream cascade on diamond dependency pattern."""
        a, b, c, d = diamond_graph["a"], diamond_graph["b"], diamond_graph["c"], diamond_graph["d"]

        # Cascade upstream from D should include A, B, C in dependency order
        result = graph.get_cascade_features("d", "upstream")
        assert len(result) == 4
        assert a.spec().key in result
        assert b.spec().key in result
        assert c.spec().key in result
        assert d.spec().key in result

        # A should come before B and C (root first)
        a_idx = result.index(a.spec().key)
        b_idx = result.index(b.spec().key)
        c_idx = result.index(c.spec().key)
        d_idx = result.index(d.spec().key)
        assert a_idx < b_idx
        assert a_idx < c_idx
        # D should be last
        assert d_idx == len(result) - 1

    def test_cascade_source_included(self, graph: FeatureGraph):
        """Test that source feature is always included in result."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Downstream from A (no dependents)
        result = graph.get_cascade_features("a", "downstream")
        assert len(result) == 1
        assert result == [FeatureA.spec().key]

        # Upstream from A (no dependencies)
        result = graph.get_cascade_features("a", "upstream")
        assert len(result) == 1
        assert result == [FeatureA.spec().key]

        # Both from A
        result = graph.get_cascade_features("a", "both")
        assert len(result) == 1
        assert result == [FeatureA.spec().key]

    def test_cascade_invalid_direction_raises_error(self, graph: FeatureGraph):
        """Test that invalid cascade direction raises ValueError."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        with pytest.raises(ValueError, match="Invalid cascade option: 'invalid'"):
            graph.get_cascade_features("a", "invalid")

        with pytest.raises(ValueError, match="Invalid cascade option: ''"):
            graph.get_cascade_features("a", "")

    def test_cascade_with_string_key(self, graph: FeatureGraph):
        """Test cascade with string key notation."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "raw"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processed"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        # Use string notation
        result = graph.get_cascade_features("video/raw", "downstream")
        assert len(result) == 2
        assert FeatureB.spec().key in result
        assert FeatureA.spec().key in result

    def test_cascade_no_dependents(self, graph: FeatureGraph):
        """Test downstream cascade on a leaf node (no dependents)."""

        # DAG: A -> B -> C (C is leaf)
        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["b"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["c"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureB)],
            ),
        ):
            pass

        # Cascade downstream from C (leaf node)
        result = graph.get_cascade_features("c", "downstream")
        assert len(result) == 1
        assert result == [FeatureC.spec().key]

    def test_cascade_no_dependencies(self, graph: FeatureGraph):
        """Test upstream cascade on a root node (no dependencies)."""

        # DAG: A -> B -> C (A is root)
        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["b"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["c"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureB)],
            ),
        ):
            pass

        # Cascade upstream from A (root node)
        result = graph.get_cascade_features("a", "upstream")
        assert len(result) == 1
        assert result == [FeatureA.spec().key]

    def test_cascade_none_mode(self, graph: FeatureGraph, linear_chain):
        """Test NONE cascade mode returns only the source feature."""
        b = linear_chain["b"]

        # Test with CascadeMode enum
        result = graph.get_cascade_features("b", CascadeMode.NONE)
        assert len(result) == 1
        assert result == [b.spec().key]

        # Test with string "none"
        result_str = graph.get_cascade_features("b", "none")
        assert result_str == result

    def test_cascade_with_enum(self, graph: FeatureGraph, linear_chain):
        """Test cascade with CascadeMode enum instead of strings."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Test DOWNSTREAM with enum
        result = graph.get_cascade_features("a", CascadeMode.DOWNSTREAM)
        assert result == [c.spec().key, b.spec().key, a.spec().key]

        # Test UPSTREAM with enum
        result = graph.get_cascade_features("c", CascadeMode.UPSTREAM)
        assert result == [a.spec().key, b.spec().key, c.spec().key]

        # Test BOTH with enum
        result = graph.get_cascade_features("b", CascadeMode.BOTH)
        assert result == [a.spec().key, b.spec().key, c.spec().key]

    def test_cascade_case_insensitive_strings(self, graph: FeatureGraph, linear_chain):
        """Test that cascade string arguments are case-insensitive."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Test various case combinations
        result_lower = graph.get_cascade_features("a", "downstream")
        result_upper = graph.get_cascade_features("a", "DOWNSTREAM")
        result_mixed = graph.get_cascade_features("a", "DownStream")

        assert result_lower == result_upper == result_mixed
        assert result_lower == [c.spec().key, b.spec().key, a.spec().key]

    def test_cascade_invalid_string_raises(self, graph: FeatureGraph, linear_chain):
        """Test that invalid cascade strings raise helpful error."""
        with pytest.raises(ValueError, match="Invalid cascade option"):
            graph.get_cascade_features("a", "invalid")

        # Error message should suggest valid options
        with pytest.raises(ValueError, match="none.*downstream.*upstream.*both"):
            graph.get_cascade_features("a", "bad_mode")


class TestGetCascadeDeletionOrder:
    """Tests for FeatureGraph.get_cascade_deletion_order method."""

    def test_deletion_order_none(self, graph: FeatureGraph, linear_chain):
        """Test NONE mode returns only the source feature."""
        b = linear_chain["b"]

        result = graph.get_cascade_deletion_order(b.spec().key, CascadeMode.NONE)
        assert result == [b.spec().key]

    def test_deletion_order_downstream(self, graph: FeatureGraph, linear_chain):
        """Test DOWNSTREAM mode includes dependents in deletion order (leaf-first)."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Delete A (which has B and C as downstream dependents)
        result = graph.get_cascade_deletion_order(a.spec().key, CascadeMode.DOWNSTREAM)

        # Should include A, B, C with dependents first (C -> B -> A)
        assert len(result) == 3
        assert result == [c.spec().key, b.spec().key, a.spec().key]

    def test_deletion_order_upstream(self, graph: FeatureGraph, linear_chain):
        """Test UPSTREAM mode includes dependencies in deletion order (dependent-first)."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Delete C (which has B and A as upstream dependencies)
        result = graph.get_cascade_deletion_order(c.spec().key, CascadeMode.UPSTREAM)

        # Should include A, B, C with dependents first (C -> B -> A)
        assert len(result) == 3
        assert result == [c.spec().key, b.spec().key, a.spec().key]

    def test_deletion_order_both(self, graph: FeatureGraph, linear_chain):
        """Test BOTH mode includes both dependencies and dependents."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Delete B (middle node)
        result = graph.get_cascade_deletion_order(b.spec().key, CascadeMode.BOTH)

        # Should include all features (C -> B -> A for safe deletion)
        assert len(result) == 3
        assert result == [c.spec().key, b.spec().key, a.spec().key]

    def test_deletion_order_diamond(self, graph: FeatureGraph, diamond_graph):
        """Test deletion order with diamond dependency pattern."""
        a, b, c, d = diamond_graph["a"], diamond_graph["b"], diamond_graph["c"], diamond_graph["d"]

        # Delete A (root) with downstream cascade
        result = graph.get_cascade_deletion_order(a.spec().key, CascadeMode.DOWNSTREAM)

        # All features should be included
        assert len(result) == 4
        assert set(result) == {a.spec().key, b.spec().key, c.spec().key, d.spec().key}

        # D must come before B and C (leaf first)
        d_idx = result.index(d.spec().key)
        b_idx = result.index(b.spec().key)
        c_idx = result.index(c.spec().key)
        a_idx = result.index(a.spec().key)

        assert d_idx < b_idx
        assert d_idx < c_idx
        assert b_idx < a_idx
        assert c_idx < a_idx

    def test_deletion_order_upstream_diamond(self, graph: FeatureGraph, diamond_graph):
        """Test upstream deletion order with diamond pattern."""
        a = diamond_graph["a"]
        d = diamond_graph["d"]

        # Delete D (leaf) with upstream cascade
        result = graph.get_cascade_deletion_order(d.spec().key, CascadeMode.UPSTREAM)

        # All features should be included
        assert len(result) == 4

        # D must come first (dependents before dependencies)
        assert result[0] == d.spec().key

        # A must come last (root)
        assert result[-1] == a.spec().key


class TestGetUpstreamFeatures:
    """Tests for FeatureGraph.get_upstream_features method."""

    def test_upstream_linear_chain(self, graph: FeatureGraph, linear_chain):
        """Test upstream collection on a simple linear dependency chain."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # Get upstream from C should return [A, B] (not C itself)
        result = graph.get_upstream_features([c])
        assert len(result) == 2
        assert result == [a.spec().key, b.spec().key]

        # Verify topological order: A before B
        a_idx = result.index(a.spec().key)
        b_idx = result.index(b.spec().key)
        assert a_idx < b_idx

    def test_upstream_diamond_pattern(self, graph: FeatureGraph, diamond_graph):
        """Test upstream collection handles diamond patterns correctly."""
        a, b, c, d = diamond_graph["a"], diamond_graph["b"], diamond_graph["c"], diamond_graph["d"]

        # Get upstream from D should return [A, B, C]
        result = graph.get_upstream_features([d])
        assert len(result) == 3
        assert a.spec().key in result
        assert b.spec().key in result
        assert c.spec().key in result

        # A should appear only once (no duplicates)
        assert result.count(a.spec().key) == 1

        # A should come before both B and C (topological order)
        a_idx = result.index(a.spec().key)
        b_idx = result.index(b.spec().key)
        c_idx = result.index(c.spec().key)
        assert a_idx < b_idx
        assert a_idx < c_idx

    def test_upstream_no_dependencies(self, graph: FeatureGraph):
        """Test upstream collection for root features with no dependencies."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        # Root feature has no upstream dependencies
        result = graph.get_upstream_features([FeatureA])
        assert len(result) == 0
        assert result == []

    def test_upstream_multiple_sources(self, graph: FeatureGraph):
        """Test upstream collection with multiple source features."""

        # DAG: a -> b -> c
        #      a -> d -> e
        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["b"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["c"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureB)],
            ),
        ):
            pass

        class FeatureD(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["d"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureE(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["e"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureD)],
            ),
        ):
            pass

        # Get upstream from both C and E
        result = graph.get_upstream_features([FeatureC, FeatureE])

        # Should include A, B, D (shared dependency A appears once)
        assert len(result) == 3
        assert FeatureA.spec().key in result
        assert FeatureB.spec().key in result
        assert FeatureD.spec().key in result

        # A should appear only once even though both chains depend on it
        assert result.count(FeatureA.spec().key) == 1

        # A should come before B and D
        a_idx = result.index(FeatureA.spec().key)
        b_idx = result.index(FeatureB.spec().key)
        d_idx = result.index(FeatureD.spec().key)
        assert a_idx < b_idx
        assert a_idx < d_idx

    def test_upstream_with_string_keys(self, graph: FeatureGraph):
        """Test upstream collection with string key notation."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "raw"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processed"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        # Use string notation
        result = graph.get_upstream_features(["video/processed"])
        assert len(result) == 1
        assert result == [FeatureA.spec().key]

    def test_upstream_complex_multi_level(self, graph: FeatureGraph, multi_level_graph):
        """Test upstream collection on a complex multi-level graph."""
        a = multi_level_graph["a"]
        b = multi_level_graph["b"]
        c = multi_level_graph["c"]
        d = multi_level_graph["d"]
        e = multi_level_graph["e"]
        f = multi_level_graph["f"]

        # Get upstream from F
        result = graph.get_upstream_features([f])

        # Should include A, B, C, D, E
        assert len(result) == 5
        assert a.spec().key in result
        assert b.spec().key in result
        assert c.spec().key in result
        assert d.spec().key in result
        assert e.spec().key in result

        # Verify topological ordering constraints
        a_idx = result.index(a.spec().key)
        b_idx = result.index(b.spec().key)
        c_idx = result.index(c.spec().key)
        d_idx = result.index(d.spec().key)
        e_idx = result.index(e.spec().key)

        # A must come before B and C
        assert a_idx < b_idx
        assert a_idx < c_idx
        # B must come before D
        assert b_idx < d_idx
        # C must come before E
        assert c_idx < e_idx

    def test_upstream_excludes_independent_sources(self, graph: FeatureGraph):
        """Test that source features are not added just for being sources.

        Note: If one source is a dependency of another source, it WILL be included
        in the result (as it's a true upstream dependency).
        """

        # DAG:  a -> b
        #       a -> c
        # (b and c are siblings, not in each other's dependency chain)
        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["b"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["c"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=[FeatureDep(feature=FeatureA)],
            ),
        ):
            pass

        # Source features B and C should not be in result (they're siblings)
        result = graph.get_upstream_features([FeatureB, FeatureC])
        assert FeatureB.spec().key not in result
        assert FeatureC.spec().key not in result
        # Only shared upstream dependency A
        assert result == [FeatureA.spec().key]

    def test_upstream_includes_source_if_dependency_of_another(self, graph: FeatureGraph, linear_chain):
        """Test that if one source is a dependency of another source, it's included."""
        a, b, c = linear_chain["a"], linear_chain["b"], linear_chain["c"]

        # When sources are [B, C], B is a dependency of C, so it should be included
        result = graph.get_upstream_features([b, c])
        assert len(result) == 2
        assert a.spec().key in result
        assert b.spec().key in result  # B is included because C depends on it
        assert c.spec().key not in result  # C is not a dependency of any source
