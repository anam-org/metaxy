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


class TestMigrationModelsExtraArgsForbidden:
    """Test that Migration models reject extra arguments."""

    def test_diff_migration_rejects_extra_args(self):
        from datetime import datetime, timezone

        from metaxy.migrations.models import DiffMigration

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DiffMigration(
                migration_id="test",
                parent="initial",
                created_at=datetime.now(timezone.utc),
                from_project_version="abc",
                to_project_version="def",
                ops=[],
                extra_arg="should_fail",
            )

    def test_migration_result_rejects_extra_args(self):
        from datetime import datetime, timezone

        from metaxy.migrations.models import MigrationResult

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MigrationResult(
                migration_id="test",
                status="completed",
                features_completed=1,
                features_failed=0,
                features_skipped=0,
                affected_features=[],
                errors={},
                rows_affected=0,
                duration_seconds=1.0,
                timestamp=datetime.now(timezone.utc),
                extra_arg="should_fail",
            )

    def test_migration_status_info_rejects_extra_args(self):
        from metaxy.migrations.models import MigrationStatusInfo

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MigrationStatusInfo(
                migration_id="test",
                status="pending",
                expected_features=[],
                completed_features=[],
                failed_features={},
                pending_features=[],
                extra_arg="should_fail",
            )
