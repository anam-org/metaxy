"""Fixtures for models tests, especially optional dependencies tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from metaxy_testing.models import SampleFeatureSpec

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy.models.feature import FeatureGraph


@pytest.fixture
def feature_factory(graph: FeatureGraph) -> Callable[[str, list[type[BaseFeature]] | None], type[BaseFeature]]:
    """Factory fixture for creating simple test features with minimal boilerplate.

    Args:
        name: Name for the feature (used in the feature key; field key is fixed as 'x')
        deps: Optional list of feature classes this feature depends on

    Returns:
        Feature class

    Example:
        def test_example(feature_factory):
            a = feature_factory("a")
            b = feature_factory("b", deps=[a])
            c = feature_factory("c", deps=[b])
    """

    def _create_feature(name: str, deps: list[type[BaseFeature]] | None = None) -> type[BaseFeature]:
        feature_deps = [FeatureDep(feature=dep) for dep in (deps or [])]

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey([name]),
                fields=[FieldSpec(key=FieldKey(["x"]))],
                deps=feature_deps,
            ),
        ):
            pass

        return TestFeature

    return _create_feature


@pytest.fixture
def linear_chain(feature_factory) -> dict[str, type[BaseFeature]]:
    """Create a linear dependency chain: a -> b -> c.

    Returns:
        Dictionary with keys 'a', 'b', 'c'
    """
    a = feature_factory("a")
    b = feature_factory("b", deps=[a])
    c = feature_factory("c", deps=[b])

    return {"a": a, "b": b, "c": c}


@pytest.fixture
def diamond_graph(feature_factory) -> dict[str, type[BaseFeature]]:
    """Create a diamond dependency pattern: a -> b, a -> c, b -> d, c -> d.

    Returns:
        Dictionary with keys 'a', 'b', 'c', 'd'
    """
    a = feature_factory("a")
    b = feature_factory("b", deps=[a])
    c = feature_factory("c", deps=[a])
    d = feature_factory("d", deps=[b, c])

    return {"a": a, "b": b, "c": c, "d": d}


@pytest.fixture
def multi_level_graph(feature_factory) -> dict[str, type[BaseFeature]]:
    """Create a multi-level graph with multiple branches.

    Structure:
        a -> b -> d
        a -> c -> e
        d -> f
        e -> f

    Returns:
        Dictionary with keys 'a', 'b', 'c', 'd', 'e', 'f'
    """
    a = feature_factory("a")
    b = feature_factory("b", deps=[a])
    c = feature_factory("c", deps=[a])
    d = feature_factory("d", deps=[b])
    e = feature_factory("e", deps=[c])
    f = feature_factory("f", deps=[d, e])

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}


@pytest.fixture
def upstream_features(graph: FeatureGraph) -> dict[str, type[BaseFeature]]:
    """Create commonly used upstream features for testing.

    Returns a dictionary with keys:
    - upstream1, upstream2, upstream3: Basic upstream features
    - upstream_a, upstream_b: Alternative naming for upstream features
    """

    class Upstream1(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream1"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    class Upstream2(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream2"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    class Upstream3(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream3"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Verify features registered in graph
    assert FeatureKey(["test", "upstream1"]) in graph.feature_definitions_by_key
    assert FeatureKey(["test", "upstream2"]) in graph.feature_definitions_by_key
    assert FeatureKey(["test", "upstream3"]) in graph.feature_definitions_by_key

    return {
        "upstream1": Upstream1,
        "upstream2": Upstream2,
        "upstream3": Upstream3,
    }
