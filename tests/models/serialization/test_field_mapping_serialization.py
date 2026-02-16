"""Tests for field mapping serialization and deserialization."""

import json

from syrupy.assertion import SnapshotAssertion

from metaxy import FieldKey
from metaxy.models.fields_mapping import (
    AllFieldsMapping,
    DefaultFieldsMapping,
    FieldsMapping,
    FieldsMappingAdapter,
    FieldsMappingType,
    SpecificFieldsMapping,
)


def test_default_fields_mapping_serialization_basic():
    """Test basic DefaultFieldsMapping serialization."""
    mapping = DefaultFieldsMapping()

    # Serialize to dict
    serialized = mapping.model_dump(mode="json")
    assert serialized == {
        "type": "default",
        "match_suffix": False,
        "exclude_fields": [],
    }

    # Deserialize back
    deserialized = FieldsMappingAdapter.validate_python(serialized)
    assert isinstance(deserialized, DefaultFieldsMapping)
    assert deserialized.match_suffix is False
    assert deserialized.exclude_fields == []


def test_default_fields_mapping_serialization_with_config():
    """Test DefaultFieldsMapping serialization with configuration."""
    field_key = FieldKey(["test_field"])

    mapping = DefaultFieldsMapping(
        match_suffix=True,
        exclude_fields=[field_key],
    )

    # Serialize to dict (FieldKey serializes as slashed string)
    serialized = mapping.model_dump(mode="json")
    assert serialized == {
        "type": "default",
        "match_suffix": True,
        "exclude_fields": ["test_field"],  # FieldKey serializes as string
    }

    # Deserialize back
    deserialized = FieldsMappingAdapter.validate_python(serialized)
    assert isinstance(deserialized, DefaultFieldsMapping)
    assert deserialized.match_suffix is True
    assert deserialized.exclude_fields == [field_key]


def test_default_fields_mapping_json_serialization():
    """Test DefaultFieldsMapping JSON serialization."""
    mapping = DefaultFieldsMapping(match_suffix=True)

    # Serialize to JSON string
    json_str = mapping.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed == {
        "type": "default",
        "match_suffix": True,
        "exclude_fields": [],
    }

    # Deserialize from JSON
    deserialized = FieldsMappingAdapter.validate_python(parsed)
    assert isinstance(deserialized, DefaultFieldsMapping)
    assert deserialized.match_suffix is True


def test_all_fields_mapping_serialization():
    """Test AllFieldsMapping serialization."""
    mapping = AllFieldsMapping()

    # Serialize to dict
    serialized = mapping.model_dump(mode="json")
    assert serialized == {"type": "all"}

    # Deserialize back
    deserialized = FieldsMappingAdapter.validate_python(serialized)
    assert isinstance(deserialized, AllFieldsMapping)
    assert deserialized.type == FieldsMappingType.ALL


def test_all_fields_mapping_json_serialization():
    """Test AllFieldsMapping JSON serialization."""
    mapping = AllFieldsMapping()

    # Serialize to JSON string
    json_str = mapping.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed == {"type": "all"}

    # Deserialize from JSON
    deserialized = FieldsMappingAdapter.validate_python(parsed)
    assert isinstance(deserialized, AllFieldsMapping)


def test_fields_mapping_classmethod_serialization():
    """Test serialization of mappings created via classmethods."""
    # Test FieldsMapping.default()
    default_mapping = FieldsMapping.default(match_suffix=True, exclude_fields=[FieldKey(["metadata"])])

    serialized = default_mapping.model_dump(mode="json")
    # The mapping field contains the actual DefaultFieldsMapping
    assert serialized == {
        "mapping": {
            "type": "default",
            "match_suffix": True,
            "exclude_fields": ["metadata"],  # FieldKey serializes as string
        }
    }

    # Test FieldsMapping.all()
    all_mapping = FieldsMapping.all()
    serialized = all_mapping.model_dump(mode="json")
    assert serialized == {"mapping": {"type": "all"}}


def test_backward_compatibility_direct_instantiation():
    """Test backward compatibility for direct DefaultFieldsMapping instantiation."""
    # Old-style direct dict without discriminated union structure
    old_style = {
        "match_suffix": True,
        "exclude_fields": [["metadata"]],  # FieldKey as list
    }

    # Should still validate properly
    mapping = DefaultFieldsMapping.model_validate(old_style)
    assert isinstance(mapping, DefaultFieldsMapping)
    assert mapping.match_suffix is True
    assert mapping.exclude_fields == [FieldKey(["metadata"])]


def test_discriminated_union_validation():
    """Test that discriminated union validation works correctly."""
    # DefaultFieldsMapping with discriminated union
    default_data = {
        "type": "default",
        "match_suffix": False,
        "exclude_fields": [],
    }
    mapping = FieldsMappingAdapter.validate_python(default_data)
    assert isinstance(mapping, DefaultFieldsMapping)

    # AllFieldsMapping with discriminated union
    all_data = {"type": "all"}
    mapping = FieldsMappingAdapter.validate_python(all_data)
    assert isinstance(mapping, AllFieldsMapping)


def test_round_trip_serialization():
    """Test round-trip serialization for all mapping types."""
    # Test individual mapping types (not FieldsMapping wrapper)
    mappings = [
        DefaultFieldsMapping(),
        DefaultFieldsMapping(match_suffix=True),
        DefaultFieldsMapping(
            exclude_fields=[FieldKey(["field1"]), FieldKey(["field2"])],
        ),
        AllFieldsMapping(),
    ]

    for original in mappings:
        # Serialize to dict
        serialized = original.model_dump()

        # Deserialize back
        deserialized = FieldsMappingAdapter.validate_python(serialized)

        # Check type matches
        assert type(deserialized) is type(original)

        # For DefaultFieldsMapping, check config
        if isinstance(original, DefaultFieldsMapping):
            assert isinstance(deserialized, DefaultFieldsMapping)
            assert deserialized.match_suffix == original.match_suffix
            assert deserialized.exclude_fields == original.exclude_fields

        # Test JSON round-trip too
        json_str = original.model_dump_json()
        json_data = json.loads(json_str)
        from_json = FieldsMappingAdapter.validate_python(json_data)
        assert type(from_json) is type(original)

    # Test FieldsMapping wrapper separately
    wrapped_mappings = [
        FieldsMapping.default(match_suffix=True),
        FieldsMapping.all(),
    ]

    for original in wrapped_mappings:
        # Serialize to dict
        serialized = original.model_dump()

        # Check that mapping field exists
        assert "mapping" in serialized

        # The inner mapping can be deserialized
        inner = FieldsMappingAdapter.validate_python(serialized["mapping"])
        assert isinstance(inner, (DefaultFieldsMapping, AllFieldsMapping))


def test_mixed_serialization_formats():
    """Test that both old and new serialization formats work."""
    # New discriminated union format
    new_format = {
        "type": "default",
        "match_suffix": True,
        "exclude_fields": [],
    }

    # Old direct format (backward compatibility)
    old_format = {"match_suffix": True, "exclude_fields": []}

    # Both should work
    from_new = DefaultFieldsMapping.model_validate(new_format)
    from_old = DefaultFieldsMapping.model_validate(old_format)

    assert from_new.match_suffix is True
    assert from_old.match_suffix is True
    assert from_new.exclude_fields == []
    assert from_old.exclude_fields == []


def test_specific_fields_mapping_serialization_deterministic(snapshot: SnapshotAssertion):
    """Test that SpecificFieldsMapping serialization is deterministic.

    The mapping values are sets, which have non-deterministic iteration order.
    Serialization must sort the field keys to ensure deterministic output.

    Uses snapshot testing to catch any future non-determinism regressions.
    """
    # Create a mapping with multiple field keys in each set
    # Using sets means the internal order is non-deterministic
    mapping = SpecificFieldsMapping(
        mapping={
            FieldKey(["scores"]): {
                FieldKey(["audio"]),
                FieldKey(["cropped", "frames"]),
                FieldKey(["video"]),
            },
            FieldKey(["summary"]): {
                FieldKey(["zebra"]),  # Intentionally out of alphabetical order
                FieldKey(["alpha"]),
                FieldKey(["middle"]),
            },
        }
    )

    # Serialize multiple times and verify the output is always the same
    serialized_outputs = []
    for _ in range(10):
        serialized = mapping.model_dump(mode="json")
        serialized_outputs.append(json.dumps(serialized, sort_keys=True))

    # All serializations should be identical
    assert len(set(serialized_outputs)) == 1, "SpecificFieldsMapping serialization is non-deterministic"

    # Snapshot test ensures determinism across test runs
    serialized = mapping.model_dump(mode="json")
    assert serialized == snapshot


def test_all_field_mappings_serialization_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for all field mapping types to catch serialization changes."""
    mappings = {
        "default_basic": DefaultFieldsMapping(),
        "default_with_config": DefaultFieldsMapping(
            match_suffix=True,
            exclude_fields=[FieldKey(["metadata"]), FieldKey(["internal", "field"])],
        ),
        "all": AllFieldsMapping(),
        "specific": SpecificFieldsMapping(
            mapping={
                FieldKey(["output"]): {
                    FieldKey(["input_a"]),
                    FieldKey(["input_b"]),
                },
            }
        ),
    }

    serialized = {name: mapping.model_dump(mode="json") for name, mapping in mappings.items()}
    assert serialized == snapshot


def test_fields_mapping_wrapper_serialization_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for FieldsMapping wrapper serialization."""
    mappings = {
        "default": FieldsMapping.default(),
        "default_with_suffix": FieldsMapping.default(match_suffix=True),
        "default_with_excludes": FieldsMapping.default(exclude_fields=[FieldKey(["private"])]),
        "all": FieldsMapping.all(),
        "none": FieldsMapping.none(),
        "specific": FieldsMapping.specific(
            mapping={
                FieldKey(["result"]): {FieldKey(["data"]), FieldKey(["config"])},
            }
        ),
    }

    serialized = {name: mapping.model_dump(mode="json") for name, mapping in mappings.items()}
    assert serialized == snapshot
