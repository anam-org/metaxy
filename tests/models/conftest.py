"""Fixtures for models tests, especially optional dependencies tests."""

from __future__ import annotations

import pytest
from metaxy_testing.models import SampleFeatureSpec

from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy.models.feature import FeatureGraph


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
