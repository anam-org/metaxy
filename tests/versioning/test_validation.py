"""Tests for FeaturePlan column configuration validation."""

import pytest

from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldKey, FieldSpec
from metaxy.models.lineage import LineageRelationship
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey


@pytest.fixture
def upstream_spec() -> FeatureSpec:
    """A simple upstream feature spec."""
    return FeatureSpec(
        key=FeatureKey(["upstream"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
    )


@pytest.fixture
def another_upstream_spec() -> FeatureSpec:
    """Another upstream feature spec with different columns."""
    return FeatureSpec(
        key=FeatureKey(["another_upstream"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["other_value"]), code_version="1")],
    )


class TestRenameToSystemColumn:
    """Test that renaming to system column names is rejected."""

    def test_rename_to_metaxy_provenance_rejected(
        self, upstream_spec: FeatureSpec
    ) -> None:
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(
                    feature=upstream_spec.key,
                    rename={"value": "metaxy_provenance"},
                )
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        with pytest.raises(ValueError, match="system column name"):
            FeaturePlan(
                feature=downstream_spec,
                deps=[upstream_spec],
                feature_deps=downstream_spec.deps,
            )

    def test_rename_to_metaxy_data_version_rejected(
        self, upstream_spec: FeatureSpec
    ) -> None:
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(
                    feature=upstream_spec.key,
                    rename={"value": "metaxy_data_version"},
                )
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        with pytest.raises(ValueError, match="system column name"):
            FeaturePlan(
                feature=downstream_spec,
                deps=[upstream_spec],
                feature_deps=downstream_spec.deps,
            )


class TestRenameToUpstreamIdColumn:
    """Test that renaming to upstream ID column names is rejected."""

    def test_rename_to_upstream_id_column_rejected(
        self, upstream_spec: FeatureSpec
    ) -> None:
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(
                    feature=upstream_spec.key,
                    rename={"value": "id"},  # 'id' is upstream's ID column
                )
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        with pytest.raises(ValueError, match="ID column"):
            FeaturePlan(
                feature=downstream_spec,
                deps=[upstream_spec],
                feature_deps=downstream_spec.deps,
            )


class TestDuplicateRenameTargets:
    """Test that duplicate rename targets within a single dep are rejected."""

    def test_duplicate_rename_targets_rejected(self) -> None:
        upstream_spec = FeatureSpec(
            key=FeatureKey(["upstream"]),
            id_columns=("id",),
            fields=[
                FieldSpec(key=FieldKey(["value1"]), code_version="1"),
                FieldSpec(key=FieldKey(["value2"]), code_version="1"),
            ],
        )

        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(
                    feature=upstream_spec.key,
                    rename={"value1": "same_name", "value2": "same_name"},
                )
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        with pytest.raises(ValueError, match="Duplicate column names after renaming"):
            FeaturePlan(
                feature=downstream_spec,
                deps=[upstream_spec],
                feature_deps=downstream_spec.deps,
            )


class TestCrossDepColumnCollisions:
    """Test that column collisions across dependencies are rejected."""

    def test_same_column_name_across_deps_rejected(
        self,
        upstream_spec: FeatureSpec,
        another_upstream_spec: FeatureSpec,
    ) -> None:
        """Two deps with columns that would have the same name after rename."""
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(
                    feature=upstream_spec.key,
                    columns=("value",),
                    rename={"value": "colliding_name"},
                ),
                FeatureDep(
                    feature=another_upstream_spec.key,
                    columns=("other_value",),
                    rename={"other_value": "colliding_name"},
                ),
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        with pytest.raises(ValueError, match="duplicate column names"):
            FeaturePlan(
                feature=downstream_spec,
                deps=[upstream_spec, another_upstream_spec],
                feature_deps=downstream_spec.deps,
            )

    def test_id_columns_allowed_to_repeat(
        self,
        upstream_spec: FeatureSpec,
        another_upstream_spec: FeatureSpec,
    ) -> None:
        """ID columns are allowed to repeat across deps (they're joined on)."""
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(feature=upstream_spec.key),
                FeatureDep(feature=another_upstream_spec.key),
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        # Should not raise - 'id' column repeating is fine
        plan = FeaturePlan(
            feature=downstream_spec,
            deps=[upstream_spec, another_upstream_spec],
            feature_deps=downstream_spec.deps,
        )
        assert plan is not None


class TestAggregationLineageValidation:
    """Test validation with aggregation lineage."""

    def test_aggregation_on_columns_allowed_to_repeat(self) -> None:
        """Aggregation 'on' columns should be allowed to repeat."""
        upstream1 = FeatureSpec(
            key=FeatureKey(["upstream1"]),
            id_columns=("item_id",),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        )
        upstream2 = FeatureSpec(
            key=FeatureKey(["upstream2"]),
            id_columns=("item_id",),
            fields=[FieldSpec(key=FieldKey(["other_data"]), code_version="1")],
        )

        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("group_id",),
            deps=[
                FeatureDep(
                    feature=upstream1.key,
                    columns=("group_id",),
                    lineage=LineageRelationship.aggregation(on=["group_id"]),
                ),
                FeatureDep(
                    feature=upstream2.key,
                    columns=("group_id",),
                    lineage=LineageRelationship.aggregation(on=["group_id"]),
                ),
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        # Should not raise - aggregation 'on' columns are ID columns
        plan = FeaturePlan(
            feature=downstream_spec,
            deps=[upstream1, upstream2],
            feature_deps=downstream_spec.deps,
        )
        assert plan is not None


class TestValidConfigurationsPassing:
    """Test that valid configurations pass validation."""

    def test_simple_single_dep(self, upstream_spec: FeatureSpec) -> None:
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[FeatureDep(feature=upstream_spec.key)],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        plan = FeaturePlan(
            feature=downstream_spec,
            deps=[upstream_spec],
            feature_deps=downstream_spec.deps,
        )
        assert plan is not None

    def test_rename_to_non_conflicting_name(self, upstream_spec: FeatureSpec) -> None:
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(
                    feature=upstream_spec.key,
                    rename={"value": "renamed_value"},
                )
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        plan = FeaturePlan(
            feature=downstream_spec,
            deps=[upstream_spec],
            feature_deps=downstream_spec.deps,
        )
        assert plan is not None

    def test_columns_selection_avoids_collision(
        self,
        upstream_spec: FeatureSpec,
        another_upstream_spec: FeatureSpec,
    ) -> None:
        """Using columns= to select different columns avoids collision."""
        downstream_spec = FeatureSpec(
            key=FeatureKey(["downstream"]),
            id_columns=("id",),
            deps=[
                FeatureDep(
                    feature=upstream_spec.key,
                    columns=("value",),
                ),
                FeatureDep(
                    feature=another_upstream_spec.key,
                    columns=("other_value",),
                ),
            ],
            fields=[FieldSpec(key=FieldKey(["out"]), code_version="1")],
        )

        plan = FeaturePlan(
            feature=downstream_spec,
            deps=[upstream_spec, another_upstream_spec],
            feature_deps=downstream_spec.deps,
        )
        assert plan is not None
