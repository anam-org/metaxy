# ty: ignore
"""Regression tests ensuring extra arguments are rejected by Pydantic models.

These tests verify that all key models have extra='forbid' configured,
preventing silent acceptance of typos or incorrect parameter names.

Note: These tests intentionally pass invalid arguments to verify runtime behavior.
The file-level type ignore is used since the type checker correctly flags these errors.
"""

import pytest
from pydantic import ValidationError


class TestFeatureSpecExtraArgsForbidden:
    """Test that FeatureSpec rejects extra arguments."""

    def test_feature_spec_rejects_extra_args(self):
        from metaxy.models.feature_spec import FeatureSpec

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FeatureSpec(
                key="test/feature",
                id_columns=["id"],
                extra_arg="should_fail",
            )

    def test_feature_spec_rejects_typo_in_field_name(self):
        from metaxy.models.feature_spec import FeatureSpec

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FeatureSpec(
                key="test/feature",
                id_column=["id"],  # typo: should be id_columns
            )


class TestFeatureDepExtraArgsForbidden:
    """Test that FeatureDep rejects extra arguments."""

    def test_feature_dep_rejects_extra_args(self):
        from metaxy.models.feature_spec import FeatureDep

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FeatureDep(
                feature="upstream/feature",
                extra_arg="should_fail",
            )

    def test_feature_dep_rejects_typo_in_field_name(self):
        from metaxy.models.feature_spec import FeatureDep

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FeatureDep(
                feature="upstream/feature",
                column=("a", "b"),  # typo: should be columns
            )


class TestFieldSpecExtraArgsForbidden:
    """Test that FieldSpec rejects extra arguments."""

    def test_field_spec_rejects_extra_args(self):
        from metaxy.models.field import FieldSpec

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FieldSpec(
                key="my_field",
                extra_arg="should_fail",
            )

    def test_field_spec_rejects_typo_in_field_name(self):
        from metaxy.models.field import FieldSpec

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FieldSpec(
                key="my_field",
                code_vers="v2",  # typo: should be code_version
            )


class TestFieldDepExtraArgsForbidden:
    """Test that FieldDep rejects extra arguments."""

    def test_field_dep_rejects_extra_args(self):
        from metaxy.models.field import FieldDep

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FieldDep(
                feature="upstream/feature",
                extra_arg="should_fail",
            )
