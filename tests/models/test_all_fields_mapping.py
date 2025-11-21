"""Tests for AllFieldsMapping functionality."""

from metaxy import Feature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.field import SpecialFieldDep
from metaxy.models.fields_mapping import AllFieldsMapping, FieldsMapping


def test_all_fields_mapping_returns_all():
    """Test that AllFieldsMapping always returns SpecialFieldDep.ALL."""

    # Define upstream feature
    class UpstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["video"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Define downstream feature with AllFieldsMapping
    class DownstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=UpstreamFeature, fields_mapping=FieldsMapping.all())
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["combined"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,  # Explicitly set to ALL
                ),
            ],
        ),
    ):
        pass

    # Check that field deps is SpecialFieldDep.ALL
    combined_field = DownstreamFeature.spec().fields_by_key[FieldKey(["combined"])]
    assert combined_field.deps == SpecialFieldDep.ALL


def test_fields_mapping_all_classmethod():
    """Test the FieldsMapping.all() classmethod."""

    # Define upstream feature
    class UpstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["video"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Use FieldsMapping.all() classmethod
    class DownstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=UpstreamFeature, fields_mapping=FieldsMapping.all())
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["combined"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,  # Explicitly ALL
                ),
            ],
        ),
    ):
        pass

    # Check that field deps is SpecialFieldDep.ALL
    combined_field = DownstreamFeature.spec().fields_by_key[FieldKey(["combined"])]
    assert combined_field.deps == SpecialFieldDep.ALL

    # Verify that FieldsMapping.all() returns FieldsMapping with inner AllFieldsMapping
    all_mapping = FieldsMapping.all()
    assert isinstance(all_mapping.mapping, AllFieldsMapping)
    from metaxy.models.fields_mapping import FieldsMappingType

    assert all_mapping.mapping.type == FieldsMappingType.ALL


def test_all_fields_mapping_with_multiple_upstreams():
    """Test AllFieldsMapping with multiple upstream features."""

    # Define first upstream feature
    class Upstream1(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream1"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Define second upstream feature
    class Upstream2(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream2"]),
            fields=[
                FieldSpec(key=FieldKey(["video"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Use AllFieldsMapping to depend on all fields from both upstreams
    class Downstream(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=Upstream1, fields_mapping=FieldsMapping.all()),
                FeatureDep(feature=Upstream2, fields_mapping=FieldsMapping.all()),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["aggregated"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,  # Explicitly ALL
                ),
            ],
        ),
    ):
        pass

    # Should be ALL (depends on everything)
    field = Downstream.spec().fields_by_key[FieldKey(["aggregated"])]
    assert field.deps == SpecialFieldDep.ALL


def test_all_fields_mapping_serialization():
    """Test that AllFieldsMapping serializes and deserializes correctly."""
    from metaxy.models.fields_mapping import FieldsMappingAdapter, FieldsMappingType

    # Create an AllFieldsMapping instance
    all_mapping = AllFieldsMapping()

    # Serialize to dict (for JSON serialization)
    serialized = all_mapping.model_dump(mode="json")
    assert serialized == {"type": "all"}  # Enum serializes to string value

    # Deserialize back
    deserialized = FieldsMappingAdapter.validate_python(serialized)
    assert isinstance(deserialized, AllFieldsMapping)
    assert deserialized.type == FieldsMappingType.ALL

    # Also test with FieldsMapping.all()
    all_mapping_2 = FieldsMapping.all()
    serialized_2 = all_mapping_2.model_dump(mode="json")
    assert serialized_2 == {"mapping": {"type": "all"}}  # Wrapped in mapping field


def test_all_fields_mapping_vs_explicit_all():
    """Test that AllFieldsMapping is equivalent to explicit SpecialFieldDep.ALL."""

    # Define upstream feature
    class UpstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["data"]), code_version="1"),
            ],
        ),
    ):
        pass

    # One field with AllFieldsMapping on FeatureDep
    class WithMapping(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "with_mapping"]),
            deps=[
                FeatureDep(feature=UpstreamFeature, fields_mapping=FieldsMapping.all())
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["field1"]),
                    code_version="1",
                    # Will be resolved to ALL via fields_mapping
                ),
            ],
        ),
    ):
        pass

    # Another field with explicit SpecialFieldDep.ALL
    class WithExplicit(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "with_explicit"]),
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=[
                FieldSpec(
                    key=FieldKey(["field1"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,  # Explicit
                ),
            ],
        ),
    ):
        pass

    # Both should result in ALL dependencies
    mapping_field = WithMapping.spec().fields_by_key[FieldKey(["field1"])]
    explicit_field = WithExplicit.spec().fields_by_key[FieldKey(["field1"])]

    # Note: The field with fields_mapping will have deps=[] at definition time
    # but will be resolved to ALL at runtime based on the AllFieldsMapping
    assert mapping_field.deps == []  # No explicit deps (empty list, not None)
    assert explicit_field.deps == SpecialFieldDep.ALL  # Explicit ALL


def test_all_fields_mapping_with_no_upstreams():
    """Test AllFieldsMapping on a root feature (no upstreams)."""

    # Root feature with no dependencies
    class RootFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "root"]),
            # No dependencies
            fields=[
                FieldSpec(
                    key=FieldKey(["data"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,  # Explicitly ALL for root feature
                ),
            ],
        ),
    ):
        pass

    # Should still be ALL (even with no upstreams)
    field = RootFeature.spec().fields_by_key[FieldKey(["data"])]
    assert field.deps == SpecialFieldDep.ALL
