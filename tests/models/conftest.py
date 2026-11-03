"""Fixtures for models tests, especially optional dependencies tests."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy.models.feature import FeatureGraph
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import CoercibleToFieldKey, ValidatedFieldKeyAdapter
from metaxy_testing.models import SampleFeatureSpec

from tests.models.diamond_graph import DEPTH, add_diamond


@pytest.fixture
def diamond_graph(graph: FeatureGraph) -> FeatureKey:
    """Diamond field graph in the active graph. Returns the key of the top layer."""
    return add_diamond(graph, ["1"] * (DEPTH + 1))


@pytest.fixture
def parent_field_lookups(monkeypatch: pytest.MonkeyPatch) -> list[FQFieldKey]:
    """Record the field of each `FeaturePlan.get_parent_fields_for_field` call.

    Field version computation looks up the parent fields once per computed field,
    so the recorded fields show which field versions were computed.
    """
    lookups: list[FQFieldKey] = []
    original = FeaturePlan.get_parent_fields_for_field

    def spy(plan: FeaturePlan, key: CoercibleToFieldKey) -> Mapping[FQFieldKey, FieldSpec]:
        lookups.append(FQFieldKey(feature=plan.feature.key, field=ValidatedFieldKeyAdapter.validate_python(key)))
        return original(plan, key)

    monkeypatch.setattr(FeaturePlan, "get_parent_fields_for_field", spy)
    return lookups


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
