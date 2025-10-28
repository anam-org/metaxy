import json

import pytest

from metaxy import Feature, FeatureSpec, FieldSpec
from metaxy.models.feature import FeatureGraph


def _compute_version(spec: FeatureSpec) -> str:
    """Helper to compute feature version for a standalone spec."""

    graph = FeatureGraph()
    with graph.use():

        class _TempFeature(Feature, spec=spec):  # type: ignore[name-defined]
            pass

    return graph.get_feature_version(spec.key)


def test_metadata_does_not_affect_version() -> None:
    base_fields = [FieldSpec(key="result")]

    spec_with_owner_a = FeatureSpec(
        key="metadata_test_feature",
        deps=None,
        fields=base_fields,
        metadata={"owner": "team-a"},
    )
    spec_with_owner_b = FeatureSpec(
        key="metadata_test_feature",
        deps=None,
        fields=base_fields,
        metadata={"owner": "team-b"},
    )

    assert _compute_version(spec_with_owner_a) == _compute_version(spec_with_owner_b)


def test_metadata_json_serializable() -> None:
    metadata = {
        "string": "value",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"key": "value"},
    }
    spec = FeatureSpec(key="metadata_serializable", metadata=metadata)
    assert json.dumps(spec.metadata) is not None

    with pytest.raises(ValueError):
        FeatureSpec(key="metadata_bad", metadata={"func": lambda x: x})


def test_metadata_is_immutable() -> None:
    spec = FeatureSpec(
        key="metadata_immutable", metadata={"key": "value", "tags": ["a"]}
    )

    with pytest.raises(TypeError):
        spec.metadata["key"] = "new_value"  # type: ignore[index]

    # Nested collections are frozen (lists converted to tuples)
    assert isinstance(spec.metadata["tags"], tuple)  # type: ignore[index]
