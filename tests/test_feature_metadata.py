"""Tests for metadata parameter on FeatureSpec."""

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from metaxy import Feature, FeatureDep, FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.models.feature import FeatureGraph


def test_metadata_basic_usage() -> None:
    """Test basic metadata usage with FeatureSpec."""

    class TestFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "metadata"]),
            deps=None,
            metadata={
                "owner": "data-team",
                "sla": "24h",
                "description": "Test feature with metadata",
            },
        ),
    ):
        pass

    # Access metadata
    assert TestFeature.spec.metadata is not None
    assert TestFeature.spec.metadata["owner"] == "data-team"
    assert TestFeature.spec.metadata["sla"] == "24h"
    assert TestFeature.spec.metadata["description"] == "Test feature with metadata"


def test_metadata_does_not_affect_feature_version() -> None:
    """Metadata changes should NOT change feature_version()."""
    graph1 = FeatureGraph()
    graph2 = FeatureGraph()

    with graph1.use():

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["metadata_version_test"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                metadata={"owner": "team-a"},
            ),
        ):
            pass

    with graph2.use():

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["metadata_version_test"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                metadata={"owner": "team-b"},  # Different metadata!
            ),
        ):
            pass

    # feature_version should be the SAME (metadata doesn't affect it)
    assert Feature1.feature_version() == Feature2.feature_version()


def test_metadata_does_not_affect_code_version() -> None:
    """Metadata changes should NOT change code_version property."""
    graph1 = FeatureGraph()
    graph2 = FeatureGraph()

    with graph1.use():

        class FeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["metadata_code_test"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                metadata={"tag": "v1"},
            ),
        ):
            pass

        code_v1 = FeatureA.code_version()

    with graph2.use():

        class FeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["metadata_code_test"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                metadata={"tag": "v2"},  # Different metadata!
            ),
        ):
            pass

        code_v2 = FeatureB.code_version()

    # code_version should be the SAME (metadata doesn't affect it)
    assert code_v1 == code_v2


def test_metadata_affects_feature_spec_version() -> None:
    """Metadata changes SHOULD change feature_spec_version (for audit trail)."""
    spec1 = FeatureSpec(
        key=FeatureKey(["test", "spec_version"]),
        deps=None,
        metadata={"owner": "team-a"},
    )

    spec2 = FeatureSpec(
        key=FeatureKey(["test", "spec_version"]),
        deps=None,
        metadata={"owner": "team-b"},
    )

    # feature_spec_version should be DIFFERENT (includes metadata)
    assert spec1.feature_spec_version != spec2.feature_spec_version


def test_metadata_json_serializable_valid() -> None:
    """Test that valid JSON-serializable metadata is accepted."""
    valid_metadata = {
        "string": "value",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"key": "value"},
        "nested_list": [{"a": 1}, {"b": 2}],
    }

    spec = FeatureSpec(
        key=FeatureKey(["test", "json"]), deps=None, metadata=valid_metadata
    )

    # Should be able to serialize
    assert spec.metadata is not None
    serialized = json.dumps(dict(spec.metadata))
    deserialized: dict[str, Any] = json.loads(serialized)

    # Values should match
    assert deserialized["string"] == "value"
    assert deserialized["number"] == 42
    assert deserialized["float"] == 3.14
    assert deserialized["bool"] is True
    assert deserialized["null"] is None
    assert deserialized["list"] == [1, 2, 3]
    assert deserialized["nested"] == {"key": "value"}


def test_metadata_json_serializable_invalid() -> None:
    """Test that non-JSON-serializable metadata is rejected."""

    # Lambda function is not JSON-serializable
    with pytest.raises(ValueError, match="metadata must be JSON-serializable"):
        FeatureSpec(
            key=FeatureKey(["test", "invalid"]),
            deps=None,
            metadata={"func": lambda x: x},
        )

    # Set is not JSON-serializable
    with pytest.raises(ValueError, match="metadata must be JSON-serializable"):
        FeatureSpec(
            key=FeatureKey(["test", "invalid2"]),
            deps=None,
            metadata={"myset": {1, 2, 3}},
        )

    # Custom object is not JSON-serializable
    class CustomObject:
        pass

    with pytest.raises(ValueError, match="metadata must be JSON-serializable"):
        FeatureSpec(
            key=FeatureKey(["test", "invalid3"]),
            deps=None,
            metadata={"obj": CustomObject()},
        )


def test_metadata_not_none() -> None:
    """Test that metadata dict exists when set."""
    spec = FeatureSpec(
        key=FeatureKey(["test", "not_none"]),
        deps=None,
        metadata={"key": "value"},
    )

    # Metadata should exist
    assert spec.metadata is not None
    assert "key" in spec.metadata
    assert spec.metadata["key"] == "value"


def test_metadata_none_by_default() -> None:
    """Test that metadata is None by default."""
    spec = FeatureSpec(key=FeatureKey(["test", "default"]), deps=None)

    assert spec.metadata is None


def test_metadata_empty_dict() -> None:
    """Test that empty dict metadata is accepted."""
    spec = FeatureSpec(key=FeatureKey(["test", "empty"]), deps=None, metadata={})

    # Should be empty but not None
    assert spec.metadata is not None
    assert len(spec.metadata) == 0


def test_metadata_complex_structure() -> None:
    """Test metadata with complex nested structures."""
    complex_metadata = {
        "owner": "data-team",
        "sla": "24h",
        "tags": ["customer", "profile", "enrichment"],
        "cost_tier": "high",
        "pii": True,
        "custom_config": {
            "refresh_interval": "1h",
            "alert_threshold": 0.95,
            "retry_policy": {
                "max_retries": 3,
                "backoff_multiplier": 2,
            },
        },
        "contacts": [
            {"name": "Alice", "email": "alice@example.com"},
            {"name": "Bob", "email": "bob@example.com"},
        ],
    }

    spec = FeatureSpec(
        key=FeatureKey(["test", "complex"]), deps=None, metadata=complex_metadata
    )

    # Should be accessible
    assert spec.metadata is not None
    assert spec.metadata["owner"] == "data-team"
    assert spec.metadata["tags"] == ["customer", "profile", "enrichment"]
    assert spec.metadata["custom_config"]["refresh_interval"] == "1h"  # type: ignore[index]
    assert spec.metadata["custom_config"]["retry_policy"]["max_retries"] == 3  # type: ignore[index]
    assert spec.metadata["contacts"][0]["name"] == "Alice"  # type: ignore[index]


def test_metadata_with_feature_usage_example() -> None:
    """Test real-world usage example from documentation."""

    class ParentFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["example", "parent"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    class CustomerFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["example", "customer"]),
            deps=[FeatureDep(key=FeatureKey(["example", "parent"]))],
            fields=[
                FieldSpec(key=FieldKey(["age"]), code_version="1"),
                FieldSpec(key=FieldKey(["lifetime_value"]), code_version="1"),
            ],
            metadata={
                "owner": "data-team",
                "sla": "24h",
                "description": "Customer profile enrichment",
                "tags": ["customer", "profile", "enrichment"],
                "cost_tier": "high",
                "pii": True,
                "custom_config": {
                    "refresh_interval": "1h",
                    "alert_threshold": 0.95,
                },
            },
        ),
    ):
        pass

    # Access metadata
    assert CustomerFeature.spec.metadata is not None
    assert CustomerFeature.spec.metadata["owner"] == "data-team"
    assert CustomerFeature.spec.metadata["pii"] is True
    assert CustomerFeature.spec.metadata["custom_config"]["refresh_interval"] == "1h"  # type: ignore[index]


def test_metadata_serialization_in_model_dump() -> None:
    """Test that metadata is included in model_dump()."""
    spec = FeatureSpec(
        key=FeatureKey(["test", "dump"]),
        deps=None,
        metadata={"owner": "team-a"},
    )

    dumped = spec.model_dump(mode="json")

    # Metadata should be in the dump
    assert "metadata" in dumped
    assert dumped["metadata"]["owner"] == "team-a"  # type: ignore[index]


# Property-based tests using Hypothesis


@given(
    owner=st.text(min_size=1, max_size=50),
    sla=st.text(min_size=1, max_size=20),
)
def test_property_metadata_does_not_affect_feature_version(
    owner: str, sla: str
) -> None:
    """Property test: different metadata values don't change feature_version."""
    graph1 = FeatureGraph()
    graph2 = FeatureGraph()

    with graph1.use():

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property", "test"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                metadata={"owner": owner, "sla": sla},
            ),
        ):
            pass

        v1 = Feature1.feature_version()

    with graph2.use():

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property", "test"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                metadata={"owner": "different", "sla": "different"},
            ),
        ):
            pass

        v2 = Feature2.feature_version()

    # feature_version should be the same regardless of metadata
    assert v1 == v2


@given(
    metadata_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_property_metadata_json_serializable(metadata_dict: dict[str, Any]) -> None:
    """Property test: randomly generated metadata should be JSON-serializable."""
    spec = FeatureSpec(
        key=FeatureKey(["property", "json"]), deps=None, metadata=metadata_dict
    )

    # Should be able to serialize and deserialize
    assert spec.metadata is not None
    serialized = json.dumps(dict(spec.metadata))
    deserialized: dict[str, Any] = json.loads(serialized)

    # Check a sample key if present
    if metadata_dict:
        sample_key = list(metadata_dict.keys())[0]
        assert sample_key in deserialized


@given(
    num_keys=st.integers(min_value=1, max_value=20),
)
def test_property_metadata_access(num_keys: int) -> None:
    """Property test: metadata should be accessible with various numbers of keys."""
    metadata = {f"key_{i}": f"value_{i}" for i in range(num_keys)}

    spec = FeatureSpec(
        key=FeatureKey(["property", "access"]),
        deps=None,
        metadata=metadata,
    )

    # Should be able to access all keys
    assert spec.metadata is not None
    for i in range(num_keys):
        assert spec.metadata[f"key_{i}"] == f"value_{i}"
