"""Tests for FeatureSpec.description field and docstring extraction."""

from __future__ import annotations

import json

from metaxy_testing.models import SampleFeatureSpec

from metaxy import BaseFeature, FeatureDefinition, FeatureKey, FieldKey, FieldSpec


def test_feature_spec_accepts_description():
    """FeatureSpec accepts and stores a description field."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "with_description"]),
        description="This feature extracts video frames.",
    )
    assert spec.description == "This feature extracts video frames."


def test_feature_spec_description_defaults_to_none():
    """FeatureSpec.description defaults to None when not provided."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "no_description"]),
    )
    assert spec.description is None


def test_feature_spec_description_serializes_to_json():
    """Description is included in JSON serialization."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "serializable"]),
        description="A serializable description.",
    )
    spec_dict = spec.model_dump(mode="json")
    assert spec_dict["description"] == "A serializable description."

    # Verify round-trip
    spec_json = json.dumps(spec_dict)
    parsed = json.loads(spec_json)
    assert parsed["description"] == "A serializable description."


def test_feature_spec_description_affects_spec_version():
    """Changing description changes the feature_spec_version."""
    spec_a = SampleFeatureSpec(
        key=FeatureKey(["test", "version_check"]),
        description="Description A",
    )
    spec_b = SampleFeatureSpec(
        key=FeatureKey(["test", "version_check"]),
        description="Description B",
    )
    assert spec_a.feature_spec_version != spec_b.feature_spec_version


def test_from_feature_class_extracts_docstring(graph):
    """from_feature_class extracts class docstring as description."""

    class DocumentedFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "documented"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        """This is the feature docstring."""

    definition = FeatureDefinition.from_feature_class(DocumentedFeature)
    assert definition.spec.description == "This is the feature docstring."


def test_from_feature_class_strips_docstring(graph):
    """from_feature_class strips whitespace from docstrings."""

    class WhitespaceDocstringFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "whitespace"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        """
        This docstring has leading and trailing whitespace.
        """

    definition = FeatureDefinition.from_feature_class(WhitespaceDocstringFeature)
    assert definition.spec.description == "This docstring has leading and trailing whitespace."


def test_from_feature_class_preserves_explicit_description(graph):
    """Explicit FeatureSpec.description takes precedence over class docstring."""

    class ExplicitDescriptionFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "explicit"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            description="Explicit description from spec.",
        ),
    ):
        """This docstring should be ignored."""

    definition = FeatureDefinition.from_feature_class(ExplicitDescriptionFeature)
    assert definition.spec.description == "Explicit description from spec."


def test_from_feature_class_without_docstring(graph):
    """from_feature_class handles features without docstrings."""

    class NoDocstringFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "no_docstring"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    definition = FeatureDefinition.from_feature_class(NoDocstringFeature)
    assert definition.spec.description is None


def test_feature_spec_description_immutable():
    """FeatureSpec.description is immutable after initialization."""
    import pytest

    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "immutable"]),
        description="Original description",
    )

    with pytest.raises(Exception):
        spec.description = "New description"
