import json
from typing import Any, cast

import pytest
from pydantic import ValidationError
from pydantic.types import JsonValue

from metaxy import Feature, FeatureKey
from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature import FeatureGraph


def test_metadata_does_not_affect_version() -> None:
    """Metadata differences should not change feature version hashes."""
    graph_a = FeatureGraph()
    with graph_a.use():

        class MetadataFeatureA(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["tests", "metadata", "same"]),
                metadata={"owner": "team-a"},
            ),
        ):
            pass

        version_a = MetadataFeatureA.feature_version()

    graph_b = FeatureGraph()
    with graph_b.use():

        class MetadataFeatureB(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["tests", "metadata", "same"]),
                metadata={"owner": "team-b"},
            ),
        ):
            pass

        version_b = MetadataFeatureB.feature_version()

    assert version_a == version_b


def test_metadata_json_serializable() -> None:
    """Validate JSON serialization enforcement for metadata."""
    valid_metadata: dict[str, JsonValue] = {
        "string": "value",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"key": "value"},
    }

    spec = SampleFeatureSpec(
        key=FeatureKey(["tests", "metadata", "json"]),
        metadata=valid_metadata,
    )
    assert spec.metadata == valid_metadata
    assert json.dumps(spec.metadata) is not None

    with pytest.raises(ValidationError):
        _ = SampleFeatureSpec(
            key=FeatureKey(["tests", "metadata", "json"]),
            metadata=cast(Any, {"func": object()}),
        )


def test_metadata_immutable() -> None:
    """Metadata mapping should be immutable after initialization."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["tests", "metadata", "immutable"]),
        metadata={"key": "value"},
    )
    assert spec.metadata == {"key": "value"}

    with pytest.raises(Exception):
        spec.metadata = {"key": "new_value"}  # type: ignore[assignment]


def test_metadata_defaults_to_empty_dict() -> None:
    """Metadata defaults to an empty dict for easier usage."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["tests", "metadata", "default"]),
    )
    assert spec.metadata == {}
