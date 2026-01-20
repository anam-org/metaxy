"""Comprehensive tests for FeatureGraph.topological_sort_features() with descending parameter.

Tests cover:
- Basic topological sorting with both orderings (descending=False, descending=True)
- Subset sorting (sorting only some feature keys)
- All features sorting (feature_keys=None)
- Deterministic ordering (alphabetically at each level)
- Edge cases (empty graph, single feature, disconnected components)
- Verification that descending=True is the exact reverse of descending=False
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


class TestDescendingOrdering:
    """Test topological sorting with descending=True."""

    def test_simple_linear_chain(self, graph: FeatureGraph):
        """Test sorting a simple linear dependency chain A -> B -> C returns [C, B, A] with dependents_first."""

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

        # Sort all features with dependents_first
        sorted_keys = graph.topological_sort_features(descending=True)

        # Verify order: C before B before A (dependents first)
        assert len(sorted_keys) == 3
        c_idx = sorted_keys.index(FeatureC.spec().key)
        b_idx = sorted_keys.index(FeatureB.spec().key)
        a_idx = sorted_keys.index(FeatureA.spec().key)

        assert c_idx < b_idx < a_idx

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

        sorted_keys = graph.topological_sort_features(descending=True)

        # Verify dependents appear before dependencies
        assert len(sorted_keys) == 4
        d_idx = sorted_keys.index(FeatureD.spec().key)
        b_idx = sorted_keys.index(FeatureB.spec().key)
        c_idx = sorted_keys.index(FeatureC.spec().key)
        a_idx = sorted_keys.index(FeatureA.spec().key)

        # D must come before B and C
        assert d_idx < b_idx
        assert d_idx < c_idx
        # B and C must come before A
        assert b_idx < a_idx
        assert c_idx < a_idx

    def test_no_dependencies(self, graph: FeatureGraph):
        """Test sorting features with no dependencies (all independent).

        For independent features, the dependents_first order should be reverse alphabetical.
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
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features(descending=True)

        # All features should be included, and since they're independent,
        # they should be in reverse alphabetical order
        assert len(sorted_keys) == 3
        assert sorted_keys[0] == FeatureC.spec().key
        assert sorted_keys[1] == FeatureB.spec().key
        assert sorted_keys[2] == FeatureA.spec().key

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

        sorted_keys = graph.topological_sort_features(descending=True)

        assert len(sorted_keys) == 1
        assert sorted_keys[0] == FeatureA.spec().key


class TestSubsetSortingWithOrder:
    """Test sorting a subset of features from a larger graph with order parameter."""

    def test_subset_sorting_linear_chain_dependents_first(self, graph: FeatureGraph):
        """Test sorting a subset of features from a linear chain with dependents_first."""

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
        sorted_keys = graph.topological_sort_features(subset, descending=True)

        # Should only include C and B in dependents_first order
        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureC.spec().key
        assert sorted_keys[1] == FeatureB.spec().key

    def test_subset_with_shared_dependency_dependents_first(self, graph: FeatureGraph):
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
        sorted_keys = graph.topological_sort_features(subset, descending=True)

        # Should include C and B (reverse alphabetically since same level)
        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureC.spec().key
        assert sorted_keys[1] == FeatureB.spec().key


class TestEdgeCases:
    """Test edge cases for topological sorting with order parameter."""

    def test_empty_graph_dependents_first(self, graph: FeatureGraph):
        """Test sorting an empty graph returns empty list."""
        sorted_keys = graph.topological_sort_features(descending=True)
        assert sorted_keys == []

    def test_empty_subset_dependents_first(self, graph: FeatureGraph):
        """Test sorting an empty subset returns empty list."""

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "a"]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
            ),
        ):
            pass

        sorted_keys = graph.topological_sort_features([], descending=True)
        assert sorted_keys == []

    def test_single_feature_in_larger_graph_dependents_first(self, graph: FeatureGraph):
        """Test sorting a single feature from a larger graph with dependents_first."""

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

        sorted_keys = graph.topological_sort_features([FeatureA.spec().key], descending=True)
        assert len(sorted_keys) == 1
        assert sorted_keys[0] == FeatureA.spec().key


class TestOrderParameterBehavior:
    """Test that dependents_first is the exact reverse of dependencies_first."""

    def test_orders_are_exact_reverse_linear_chain(self, graph: FeatureGraph):
        """Test that dependents_first is exactly reversed for linear chain."""

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

        forward = graph.topological_sort_features(descending=False)
        reverse = graph.topological_sort_features(descending=True)

        assert forward == list(reversed(reverse))
        assert reverse == list(reversed(forward))

    def test_orders_are_exact_reverse_diamond(self, graph: FeatureGraph):
        """Test that dependents_first is exactly reversed for diamond pattern."""

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

        forward = graph.topological_sort_features(descending=False)
        reverse = graph.topological_sort_features(descending=True)

        assert forward == list(reversed(reverse))
        assert reverse == list(reversed(forward))

    def test_orders_are_exact_reverse_with_subset(self, graph: FeatureGraph):
        """Test that dependents_first is exactly reversed even for subsets."""

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

        subset = [FeatureA.spec().key, FeatureB.spec().key]

        forward = graph.topological_sort_features(subset, descending=False)
        reverse = graph.topological_sort_features(subset, descending=True)

        assert forward == list(reversed(reverse))
        assert reverse == list(reversed(forward))

    def test_orders_are_exact_reverse_independent_features(self, graph: FeatureGraph):
        """Test that dependents_first is exactly reversed for independent features."""

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

        forward = graph.topological_sort_features(descending=False)
        reverse = graph.topological_sort_features(descending=True)

        assert forward == list(reversed(reverse))
        assert reverse == list(reversed(forward))


class TestDefaultOrderBehavior:
    """Test that the default order is dependencies_first."""

    def test_default_order_is_dependencies_first(self, graph: FeatureGraph):
        """Test that default order matches dependencies_first explicitly."""

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

        default_sorted = graph.topological_sort_features()
        explicit_deps_first = graph.topological_sort_features(descending=False)

        assert default_sorted == explicit_deps_first


class TestDeterministicOrdering:
    """Test that features at the same level are sorted deterministically."""

    def test_alphabetical_ordering_same_level_dependents_first(self, graph: FeatureGraph):
        """Test features at same level are sorted in reverse alphabetical order with dependents_first."""

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

        sorted_keys = graph.topological_sort_features(descending=True)

        # Children should be in reverse alphabetical order
        assert sorted_keys[0] == FeatureZ.spec().key
        assert sorted_keys[1] == FeatureM.spec().key
        assert sorted_keys[2] == FeatureA.spec().key

        # Parent should be last
        assert sorted_keys[3] == FeatureParent.spec().key


class TestInputVariations:
    """Test that the method accepts various input types for feature_keys parameter."""

    def test_accepts_string_paths_with_order(self, graph: FeatureGraph):
        """Test that feature keys can be provided as string paths with order parameter."""

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

        # Pass string paths instead of FeatureKey objects
        sorted_keys = graph.topological_sort_features(["test/a", "test/b"], descending=True)

        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureB.spec().key
        assert sorted_keys[1] == FeatureA.spec().key

    def test_accepts_feature_classes_with_order(self, graph: FeatureGraph):
        """Test that feature keys can be provided as Feature classes with order parameter."""

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

        # Pass Feature classes directly
        sorted_keys = graph.topological_sort_features([FeatureA, FeatureB], descending=True)

        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureB.spec().key
        assert sorted_keys[1] == FeatureA.spec().key

    def test_accepts_mixed_input_types_with_order(self, graph: FeatureGraph):
        """Test that feature keys can be provided as mixed types with order parameter."""

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

        # Mix of string, FeatureKey, and Feature class
        sorted_keys = graph.topological_sort_features(["test/a", FeatureB.spec().key], descending=True)

        assert len(sorted_keys) == 2
        assert sorted_keys[0] == FeatureB.spec().key
        assert sorted_keys[1] == FeatureA.spec().key
