"""Tests for column selection and renaming in feature dependencies."""

from typing import Any

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing.models import SampleFeatureSpec
from metaxy_testing.pytest_helpers import add_metaxy_system_columns

from metaxy import BaseFeature, FeatureDep, FeatureGraph, FeatureKey
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.plan import FeaturePlan
from metaxy.versioning.polars import PolarsVersioningEngine


# Simple test joiner that uses VersioningEngine
class TestJoiner:
    """Test utility that wraps PolarsVersioningEngine for column selection tests."""

    def join_upstream(
        self,
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
        feature_spec: Any,
        feature_plan: FeaturePlan,
        upstream_columns: dict[str, tuple[str, ...] | None],
        upstream_renames: dict[str, dict[str, str] | None],
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Join upstream feature metadata using PolarsVersioningEngine.

        This is a test utility that mimics the old NarwhalsJoiner interface
        but uses the new PolarsVersioningEngine internally.
        """
        from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD

        # Create a PolarsVersioningEngine for this feature
        engine = PolarsVersioningEngine(plan=feature_plan)

        # Convert string keys back to FeatureKey objects and ensure data is materialized
        upstream_by_key = {}
        for k, v in upstream_refs.items():
            # Materialize the lazy frame and ensure it has all metaxy system columns
            df = v.collect().to_polars()
            df = add_metaxy_system_columns(df)
            upstream_by_key[FeatureKey(k)] = nw.from_native(df.lazy(), eager_only=False)

        # Prepare upstream (handles filtering, selecting, renaming, and joining)
        joined = engine.prepare_upstream(upstream_by_key, filters=None)

        # Build the mapping of upstream_key -> provenance_by_field column name
        # The new naming convention is: {column_name}{feature_key.to_column_suffix()}
        mapping = {}
        for upstream_key_str in upstream_refs.keys():
            upstream_key = FeatureKey(upstream_key_str)
            # The provenance_by_field column is renamed using to_column_suffix()
            provenance_col_name = f"{METAXY_PROVENANCE_BY_FIELD}{upstream_key.to_column_suffix()}"
            mapping[upstream_key_str] = provenance_col_name

        return joined, mapping  # ty: ignore[invalid-return-type]


class TestColumnSelection:
    """Test column selection in feature dependencies."""

    def test_default_behavior_keeps_all_columns(self):
        """Test that all columns are kept by default (new behavior)."""

        # Create upstream feature with custom columns
        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Create downstream feature with default column handling
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["test", "upstream"]))],
            ),
        ):
            pass

        # Create test data with custom columns
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "custom_col1": ["a", "b", "c"],
                "custom_col2": [10, 20, 30],
            }
        )

        # Join upstream
        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        joined, mapping = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify all columns are present
        assert "sample_uid" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream" in joined_df.columns
        assert "custom_col1" in joined_df.columns
        assert "custom_col2" in joined_df.columns
        assert joined_df["custom_col1"].to_list() == ["a", "b", "c"]
        assert joined_df["custom_col2"].to_list() == [10, 20, 30]

    def test_column_selection_specific_columns(self):
        """Test selecting specific columns from upstream."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Select only custom_col1
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        select=("custom_col1",),
                    )
                ],
            ),
        ):
            pass

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "custom_col1": ["a", "b", "c"],
                "custom_col2": [10, 20, 30],
                "custom_col3": [100, 200, 300],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify only selected columns are present
        assert "sample_uid" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream" in joined_df.columns
        assert "custom_col1" in joined_df.columns
        assert "custom_col2" not in joined_df.columns
        assert "custom_col3" not in joined_df.columns

    def test_empty_columns_keeps_only_system_columns(self):
        """Test that empty tuple keeps only system columns."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Empty tuple - keep only system columns
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        select=(),  # Only system columns
                    )
                ],
            ),
        ):
            pass

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_feature_version": ["v1", "v1", "v1"],
                "custom_col1": ["a", "b", "c"],
                "custom_col2": [10, 20, 30],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify only essential system columns are present
        # Note: feature_version and project_version are NOT considered essential for joining
        # to avoid conflicts when joining multiple upstream features
        assert "sample_uid" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream" in joined_df.columns
        assert "metaxy_feature_version" not in joined_df.columns  # Not essential, dropped to avoid conflicts
        assert "custom_col1" not in joined_df.columns
        assert "custom_col2" not in joined_df.columns

    def test_column_renaming(self):
        """Test renaming columns to avoid conflicts."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Rename columns
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={
                            "custom_col1": "upstream_col1",
                            "custom_col2": "upstream_col2",
                        },
                    )
                ],
            ),
        ):
            pass

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "custom_col1": ["a", "b", "c"],
                "custom_col2": [10, 20, 30],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify columns are renamed
        assert "sample_uid" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream" in joined_df.columns
        assert "upstream_col1" in joined_df.columns
        assert "upstream_col2" in joined_df.columns
        assert "custom_col1" not in joined_df.columns
        assert "custom_col2" not in joined_df.columns
        assert joined_df["upstream_col1"].to_list() == ["a", "b", "c"]
        assert joined_df["upstream_col2"].to_list() == [10, 20, 30]

    def test_selection_and_renaming_combined(self):
        """Test combining column selection with renaming."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Select and rename
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={"custom_col1": "renamed_col1"},
                        select=("renamed_col1", "custom_col2"),
                    )
                ],
            ),
        ):
            pass

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "custom_col1": ["a", "b", "c"],
                "custom_col2": [10, 20, 30],
                "custom_col3": [100, 200, 300],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify selection and renaming
        assert "renamed_col1" in joined_df.columns
        assert "custom_col2" in joined_df.columns  # Not renamed
        assert "custom_col3" not in joined_df.columns  # Not selected
        assert joined_df["renamed_col1"].to_list() == ["a", "b", "c"]

    def test_column_conflict_detection(self):
        """Test that column conflicts are detected and raise errors."""

        class Upstream1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
            ),
        ):
            pass

        class Upstream2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
            ),
        ):
            pass

        # Both upstreams have 'conflict_col' without renaming
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["test", "upstream1"])),
                    FeatureDep(feature=FeatureKey(["test", "upstream2"])),
                ],
            ),
        ):
            pass

        upstream1_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "conflict_col": ["a", "b", "c"],
            }
        )

        upstream2_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h4"},
                    {"default": "h5"},
                    {"default": "h6"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h4"},
                    {"default": "h5"},
                    {"default": "h6"},
                ],
                "conflict_col": ["d", "e", "f"],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {
            "test/upstream1": nw.from_native(upstream1_data.lazy(), eager_only=False),
            "test/upstream2": nw.from_native(upstream2_data.lazy(), eager_only=False),
        }

        # Should raise error about column conflict
        with pytest.raises(ValueError, match="Found column collisions.*conflict_col"):
            DownstreamFeature.load_input(joiner, upstream_refs)

    def test_column_conflict_resolved_with_rename(self):
        """Test that column conflicts can be resolved with renaming."""

        class Upstream1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
            ),
        ):
            pass

        class Upstream2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
            ),
        ):
            pass

        # Rename conflicting columns
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]),
                        rename={"conflict_col": "upstream1_col"},
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]),
                        rename={"conflict_col": "upstream2_col"},
                    ),
                ],
            ),
        ):
            pass

        upstream1_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "conflict_col": ["a", "b", "c"],
            }
        )

        upstream2_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h4"},
                    {"default": "h5"},
                    {"default": "h6"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h4"},
                    {"default": "h5"},
                    {"default": "h6"},
                ],
                "conflict_col": ["d", "e", "f"],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {
            "test/upstream1": nw.from_native(upstream1_data.lazy(), eager_only=False),
            "test/upstream2": nw.from_native(upstream2_data.lazy(), eager_only=False),
        }

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify both renamed columns are present
        assert "upstream1_col" in joined_df.columns
        assert "upstream2_col" in joined_df.columns
        assert joined_df["upstream1_col"].to_list() == ["a", "b", "c"]
        assert joined_df["upstream2_col"].to_list() == ["d", "e", "f"]

    def test_essential_system_columns_preserved(self):
        """Test that essential system columns (sample_uid, provenance_by_field) are always preserved."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Select only a custom column, but system columns should be preserved
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        select=("custom_col",),  # Don't explicitly list system columns
                    )
                ],
            ),
        ):
            pass

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_feature_version": ["v1", "v1", "v1"],
                "metaxy_project_version": ["s1", "s1", "s1"],
                "custom_col": ["a", "b", "c"],
                "other_col": [10, 20, 30],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify essential system columns are preserved
        assert "sample_uid" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream" in joined_df.columns
        # Note: feature_version and project_version are NOT preserved to avoid conflicts
        assert "metaxy_feature_version" not in joined_df.columns
        assert "metaxy_project_version" not in joined_df.columns
        # Verify selected column
        assert "custom_col" in joined_df.columns
        # Verify non-selected column is not present
        assert "other_col" not in joined_df.columns

    def test_pydantic_validation_rename_to_system_column(self, graph: FeatureGraph):
        """Test that renaming to system column names is forbidden at plan creation time."""

        # Create upstream feature first
        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Renaming to provenance_by_field should raise an error when getting the plan
        class BadFeature1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad1"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={"old_col": "metaxy_provenance_by_field"},  # Not allowed
                    )
                ],
            ),
        ):
            pass

        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to system column name.*provenance_by_field",
        ):
            graph.get_feature_plan(FeatureKey(["test", "bad1"]))

        # Renaming to feature_version should raise an error
        class BadFeature2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad2"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={"old_col": "metaxy_feature_version"},  # Not allowed
                    )
                ],
            ),
        ):
            pass

        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to system column name.*feature_version",
        ):
            graph.get_feature_plan(FeatureKey(["test", "bad2"]))

        # Renaming to sample_uid should raise an error because upstream has sample_uid as its ID column
        class BadFeature3(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad3"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={"old_col": "sample_uid"},  # Not allowed - it's upstream's ID column
                    )
                ],
            ),
        ):
            pass

        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to ID column.*sample_uid",
        ):
            graph.get_feature_plan(FeatureKey(["test", "bad3"]))

    def test_rename_to_sample_uid_allowed_when_not_id_column(self):
        """Test that renaming to sample_uid is allowed when it's not an upstream ID column."""

        # Create upstream with custom ID columns (not sample_uid)
        class UpstreamWithCustomIDs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                id_columns=[
                    "user_id",
                    "session_id",
                ],  # sample_uid is NOT an ID column
            ),
        ):
            pass

        # Renaming to sample_uid should be allowed now since it's not an ID column
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={
                            "some_col": "sample_uid"  # Allowed - sample_uid is not upstream's ID column
                        },
                    )
                ],
                id_columns=[
                    "user_id",
                    "session_id",
                ],  # Must match upstream for joining
            ),
        ):
            pass

        # Verify the feature was created successfully
        deps = DownstreamFeature.spec().deps
        assert deps is not None
        assert deps[0].rename == {"some_col": "sample_uid"}

    def test_rename_column_to_different_name_than_id_columns(self):
        """Test that renaming columns to names other than ID columns is allowed."""

        # Create upstream with custom ID columns
        class UpstreamWithCustomIDs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                id_columns=["user_id", "session_id"],
            ),
        ):
            pass

        # Renaming to a name that is NOT an ID column or system column is allowed
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={"some_col": "user_id_renamed"},  # Allowed - not renaming to actual ID column name
                    )
                ],
                id_columns=[
                    "user_id",
                    "session_id",
                ],  # Must match upstream ID columns for joining
            ),
        ):
            pass

        upstream_data = pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                "session_id": ["a", "b", "c"],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "some_col": ["x", "y", "z"],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        # This should now work without raising an error
        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify the renamed column is present
        assert "user_id_renamed" in joined_df.columns
        assert joined_df["user_id_renamed"].to_list() == ["x", "y", "z"]

    def test_pydantic_serialization(self):
        """Test that FeatureDep with columns and rename serializes correctly."""
        dep = FeatureDep(
            feature=FeatureKey(["test", "upstream"]),
            rename={"col1": "new_col1"},
            select=("new_col1", "col2"),
            filters=["col1 = 'x'"],
        )

        # Serialize to dict - custom serializer always uses "filters" key
        # Test serialization with by_alias=True to use serialization_alias
        dep_dict = dep.model_dump(by_alias=True)
        # FeatureKey now serializes to string format for JSON dict key compatibility
        assert dep_dict["feature"] == "test/upstream"
        assert dep_dict["select"] == ("new_col1", "col2")
        assert dep_dict["rename"] == {"col1": "new_col1"}
        assert dep_dict["filters"] == ("col1 = 'x'",)
        assert "sql_filters" not in dep_dict

        # Verify filters property works (lazy evaluation)
        assert dep.filters
        assert len(dep.filters) == 1

        # Round-trip through JSON (using by_alias=True to preserve "filters" key)
        dep_json = dep.model_dump_json(by_alias=True)
        dep_restored = FeatureDep.model_validate_json(dep_json)
        assert dep_restored.feature == FeatureKey(["test", "upstream"])
        assert dep_restored.select == ("new_col1", "col2")
        assert dep_restored.rename == {"col1": "new_col1"}
        # Verify filters work after deserialization
        assert dep_restored.filters
        assert len(dep_restored.filters) == 1

    def test_multiple_upstream_features_complex_scenario(self):
        """Test complex scenario with multiple upstream features."""

        class Upstream1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
            ),
        ):
            pass

        class Upstream2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
            ),
        ):
            pass

        class Upstream3(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream3"]),
            ),
        ):
            pass

        # Complex downstream with different operations for each upstream
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    # Keep all columns from upstream1 (default)
                    FeatureDep(feature=FeatureKey(["test", "upstream1"])),
                    # Select specific columns from upstream2
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]),
                        select=("important_col",),
                    ),
                    # Rename columns from upstream3
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream3"]),
                        rename={"shared_col": "upstream3_shared"},
                        select=("upstream3_shared", "unique_col"),
                    ),
                ],
            ),
        ):
            pass

        upstream1_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
                "col_a": ["a1", "a2", "a3"],
                "col_b": [1, 2, 3],
            }
        )

        upstream2_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h4"},
                    {"default": "h5"},
                    {"default": "h6"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h4"},
                    {"default": "h5"},
                    {"default": "h6"},
                ],
                "important_col": ["i1", "i2", "i3"],
                "unimportant_col": ["u1", "u2", "u3"],
            }
        )

        upstream3_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h7"},
                    {"default": "h8"},
                    {"default": "h9"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h7"},
                    {"default": "h8"},
                    {"default": "h9"},
                ],
                "shared_col": ["s1", "s2", "s3"],
                "unique_col": ["q1", "q2", "q3"],
                "excluded_col": ["e1", "e2", "e3"],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {
            "test/upstream1": nw.from_native(upstream1_data.lazy(), eager_only=False),
            "test/upstream2": nw.from_native(upstream2_data.lazy(), eager_only=False),
            "test/upstream3": nw.from_native(upstream3_data.lazy(), eager_only=False),
        }

        joined, mapping = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify all expected columns
        # From upstream1 (all columns)
        assert "col_a" in joined_df.columns
        assert "col_b" in joined_df.columns
        # From upstream2 (only selected)
        assert "important_col" in joined_df.columns
        assert "unimportant_col" not in joined_df.columns
        # From upstream3 (selected and renamed)
        assert "upstream3_shared" in joined_df.columns  # Renamed
        assert "unique_col" in joined_df.columns  # Not renamed
        assert "excluded_col" not in joined_df.columns  # Not selected

        # Verify provenance_by_field columns
        assert "metaxy_provenance_by_field__test_upstream1" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream2" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream3" in joined_df.columns

        # Verify mapping
        assert mapping["test/upstream1"] == "metaxy_provenance_by_field__test_upstream1"
        assert mapping["test/upstream2"] == "metaxy_provenance_by_field__test_upstream2"
        assert mapping["test/upstream3"] == "metaxy_provenance_by_field__test_upstream3"

    def test_custom_load_input_with_filtering(self):
        """Test overriding load_input with custom filtering logic."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Override load_input to add custom filtering
        class CustomFilterFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "custom_filter"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={"value": "upstream_value"},
                        select=("upstream_value", "category"),
                    )
                ],
            ),
        ):
            @classmethod
            def load_input(cls, joiner, upstream_refs):
                """Custom load_input that filters data after joining."""
                # First, do the standard join with column selection/renaming
                joined, mapping = super().load_input(joiner, upstream_refs)

                # Then apply custom filtering
                # Keep only rows where category is "important"
                filtered = joined.filter(nw.col("category") == "important")

                return filtered, mapping

        # Create test data
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                    {"default": "h4"},
                    {"default": "h5"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                    {"default": "h4"},
                    {"default": "h5"},
                ],
                "value": [10, 20, 30, 40, 50],
                "category": [
                    "important",
                    "other",
                    "important",
                    "other",
                    "important",
                ],
                "extra_col": [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                ],  # Should be dropped by columns spec
            }
        )

        joiner = TestJoiner()
        upstream_refs = {"test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)}

        joined, mapping = CustomFilterFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify filtering worked
        assert len(joined_df) == 3  # Only 3 "important" rows
        assert joined_df["sample_uid"].to_list() == [1, 3, 5]

        # Verify column selection worked
        assert "upstream_value" in joined_df.columns  # Renamed
        assert "value" not in joined_df.columns  # Original name gone
        assert "category" in joined_df.columns  # Selected
        assert "extra_col" not in joined_df.columns  # Not selected

        # Verify renamed column has correct values
        assert joined_df["upstream_value"].to_list() == [10, 30, 50]

        # Verify mapping
        assert mapping["test/upstream"] == "metaxy_provenance_by_field__test_upstream"

    def test_columns_and_rename_serialized_to_snapshot(self, graph: FeatureGraph, tmp_path):
        """Test that columns and rename fields are properly serialized when pushing graph snapshot."""
        import json

        from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
        from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

        # Create features with columns and rename specified
        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={"col1": "renamed_col1"},
                        select=("renamed_col1", "col2"),
                    )
                ],
            ),
        ):
            pass

        # Create store and push snapshot
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            _ = SystemTableStorage(store).push_graph_snapshot()

            # Read the snapshot from feature_versions table
            versions = store.read(FEATURE_VERSIONS_KEY, with_feature_history=True).collect().to_polars()

            # Find the downstream feature record
            downstream_record = versions.filter(pl.col("feature_key") == "test/downstream")
            assert len(downstream_record) == 1

            # Parse the feature_spec JSON
            feature_spec_json = downstream_record["feature_spec"][0]
            feature_spec_dict = json.loads(feature_spec_json)

            # Verify the spec contains deps
            assert "deps" in feature_spec_dict
            assert len(feature_spec_dict["deps"]) == 1

            # Verify columns and rename are serialized
            dep = feature_spec_dict["deps"][0]
            # FeatureKey serializes to string format for JSON dict key compatibility
            assert dep["feature"] == "test/upstream"
            assert dep["select"] == [
                "renamed_col1",
                "col2",
            ]  # Pydantic serializes tuple as list
            assert dep["rename"] == {"col1": "renamed_col1"}

            # Verify round-trip through to_snapshot/from Pydantic
            snapshot_dict = graph.to_snapshot()
            downstream_snapshot = snapshot_dict["test/downstream"]

            # Verify feature_spec dict has the fields
            downstream_spec_dict = downstream_snapshot["feature_spec"]
            assert "deps" in downstream_spec_dict
            dep_dict = downstream_spec_dict["deps"][0]
            # FeatureKey serializes to string format for JSON dict key compatibility
            assert dep_dict["feature"] == "test/upstream"
            assert dep_dict["select"] == ["renamed_col1", "col2"]
            assert dep_dict["rename"] == {"col1": "renamed_col1"}

            # Verify Pydantic can deserialize it back
            reconstructed_spec = SampleFeatureSpec.model_validate(downstream_spec_dict)
            assert reconstructed_spec.deps and len(reconstructed_spec.deps) == 1
            reconstructed_dep = reconstructed_spec.deps[0]
            assert reconstructed_dep.select == ("renamed_col1", "col2")
            assert reconstructed_dep.rename == {"col1": "renamed_col1"}

    def test_duplicate_renamed_columns_within_single_dependency(self, graph: FeatureGraph):
        """Test that renaming multiple columns to the same name within a dependency is rejected at plan creation."""

        # Create upstream feature first
        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Feature class definition succeeds, but plan creation fails
        class BadFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={
                            "col1": "same_name",
                            "col2": "same_name",
                        },  # Both renamed to same name
                    )
                ],
            ),
        ):
            pass

        with pytest.raises(ValueError, match="Duplicate column names after renaming"):
            graph.get_feature_plan(FeatureKey(["test", "bad"]))

    def test_duplicate_columns_across_dependencies_validation(self, graph: FeatureGraph):
        """Test that duplicate columns across dependencies are detected at plan creation time."""

        # Create two upstream features
        class Upstream1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
            ),
        ):
            pass

        class Upstream2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
            ),
        ):
            pass

        # Feature class definition succeeds, but plan creation fails
        # because both dependencies rename columns to "duplicate_col"
        class BadDownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad_downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]),
                        rename={"col1": "duplicate_col"},
                        select=("duplicate_col",),
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]),
                        rename={"col2": "duplicate_col"},
                        select=("duplicate_col",),
                    ),
                ],
            ),
        ):
            pass

        with pytest.raises(ValueError, match="would have duplicate column names"):
            graph.get_feature_plan(FeatureKey(["test", "bad_downstream"]))

    def test_duplicate_columns_with_selected_columns(self, graph: FeatureGraph):
        """Test that duplicate detection works with column selection at plan creation time."""

        # Create two upstream features
        class Upstream1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
            ),
        ):
            pass

        class Upstream2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
            ),
        ):
            pass

        # Feature class definition succeeds, but plan creation fails
        # because both select "shared_col" without renaming
        class BadDownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad_downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]),
                        select=("shared_col", "col1"),  # Selects shared_col
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]),
                        select=(
                            "shared_col",
                            "col2",
                        ),  # Also selects shared_col
                    ),
                ],
            ),
        ):
            pass

        with pytest.raises(ValueError, match="would have duplicate column names"):
            graph.get_feature_plan(FeatureKey(["test", "bad_downstream"]))

    def test_renaming_to_upstream_id_columns_forbidden(self, graph: FeatureGraph):
        """Test that renaming to upstream's ID columns is forbidden at plan creation time."""

        # Create upstream with custom ID columns
        class UpstreamWithCustomIDs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                id_columns=["user_id", "session_id"],
            ),
        ):
            pass

        # Feature class definition succeeds, but plan creation fails
        class BadFeature1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad1"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={
                            "some_col": "user_id",  # Not allowed - upstream's ID column
                        },
                    )
                ],
                id_columns=["user_id", "session_id"],
            ),
        ):
            pass

        with pytest.raises(ValueError, match="Cannot rename column.*to ID column.*user_id"):
            graph.get_feature_plan(FeatureKey(["test", "bad1"]))

        # Renaming to another upstream ID column should also raise an error
        class BadFeature2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "bad2"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={
                            "some_col": "session_id",  # Not allowed - upstream's ID column
                        },
                    )
                ],
                id_columns=["user_id", "session_id"],
            ),
        ):
            pass

        with pytest.raises(ValueError, match="Cannot rename column.*to ID column.*session_id"):
            graph.get_feature_plan(FeatureKey(["test", "bad2"]))

    def test_renaming_to_system_columns_forbidden(self, graph: FeatureGraph):
        """Test that renaming to system columns and ID columns is forbidden at plan creation time."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                # Default ID columns: ["sample_uid"]
            ),
        ):
            pass

        # Feature class definition succeeds, but plan creation fails
        class DownstreamFeature1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream1"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={
                            "old_version": "metaxy_provenance_by_field",  # Not allowed - system column
                        },
                    )
                ],
            ),
        ):
            pass

        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to system column name.*provenance_by_field",
        ):
            graph.get_feature_plan(FeatureKey(["test", "downstream1"]))

        # Renaming to ID column sample_uid should raise an error
        class DownstreamFeature2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream2"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream"]),
                        rename={
                            "old_sample": "sample_uid",  # Not allowed - upstream's ID column
                        },
                    )
                ],
            ),
        ):
            pass

        with pytest.raises(ValueError, match="Cannot rename column.*to ID column.*sample_uid"):
            graph.get_feature_plan(FeatureKey(["test", "downstream2"]))

    def test_aggregation_lineage_column_selection_drops_non_join_id_columns(self):
        """Test that aggregation lineage drops upstream ID columns not in the aggregation key.

        When two upstreams share a column that is an ID column in one of them,
        column selection should drop it because it's not needed for the aggregation.
        The `on=` columns are auto-included, so users only specify the data columns.
        """
        from metaxy.models.lineage import LineageRelationship

        # Upstream1 has id_columns=["item_id", "group_id"]
        class Upstream1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
                id_columns=["item_id", "group_id"],
            ),
        ):
            pass

        # Upstream2 has id_columns=["item_id", "group_id"]
        class Upstream2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
                id_columns=["item_id", "group_id"],
            ),
        ):
            pass

        # Downstream aggregates both upstreams on group_id
        # Both upstreams have item_id as an ID column, which would collide
        # But with column selection, we only select the data columns we need
        # group_id is auto-included from the aggregation `on=`
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                id_columns=["group_id"],
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]),
                        select=("col1",),  # Only select col1, group_id comes from aggregation
                        lineage=LineageRelationship.aggregation(on=["group_id"]),
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]),
                        select=("col2",),  # Only select col2, group_id comes from aggregation
                        lineage=LineageRelationship.aggregation(on=["group_id"]),
                    ),
                ],
            ),
        ):
            pass

        # Prepare test data - both have item_id but we're not selecting it
        upstream1_data = pl.DataFrame(
            {
                "item_id": [1, 2, 3, 4],
                "group_id": ["g1", "g1", "g2", "g2"],
                "col1": ["a", "b", "c", "d"],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                    {"default": "h4"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "v1"},
                    {"default": "v2"},
                    {"default": "v3"},
                    {"default": "v4"},
                ],
            }
        )

        upstream2_data = pl.DataFrame(
            {
                "item_id": [
                    10,
                    20,
                    30,
                    40,
                ],  # Different item_ids, would collide if selected
                "group_id": ["g1", "g1", "g2", "g2"],
                "col2": [100, 200, 300, 400],
                "metaxy_provenance_by_field": [
                    {"default": "p1"},
                    {"default": "p2"},
                    {"default": "p3"},
                    {"default": "p4"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "d1"},
                    {"default": "d2"},
                    {"default": "d3"},
                    {"default": "d4"},
                ],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {
            "test/upstream1": nw.from_native(upstream1_data.lazy(), eager_only=False),
            "test/upstream2": nw.from_native(upstream2_data.lazy(), eager_only=False),
        }

        # This should NOT raise a column collision error because item_id is dropped
        joined, mapping = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify item_id is NOT in the result (was dropped by column selection)
        assert "item_id" not in joined_df.columns

        # Verify the selected columns are present
        assert "group_id" in joined_df.columns  # Auto-included from aggregation on=
        assert "col1" in joined_df.columns
        assert "col2" in joined_df.columns

        # Verify the join worked (we have data from both upstreams)
        assert len(joined_df) > 0
