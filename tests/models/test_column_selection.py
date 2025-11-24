"""Tests for column selection and renaming in feature dependencies."""

from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy import BaseFeature, FeatureDep, FeatureGraph, FeatureKey
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.plan import FeaturePlan
from metaxy.versioning.polars import PolarsVersioningEngine


# Helper function to add metaxy_provenance column to test data
def add_metaxy_provenance(df: pl.DataFrame) -> pl.DataFrame:
    """Add metaxy_provenance column (hash of provenance_by_field) to test data."""
    # For tests, we just use a simple hash-like string based on the provenance_by_field
    # In real usage, this would be calculated by the VersioningEngine
    df = df.with_columns(
        pl.col("metaxy_provenance_by_field")
        .map_elements(lambda x: f"hash_{x['default']}", return_dtype=pl.String)
        .alias("metaxy_provenance")
    )
    return df


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
            # Materialize the lazy frame and ensure it has metaxy_provenance
            df = v.collect().to_polars()
            if "metaxy_provenance" not in df.columns:
                df = add_metaxy_provenance(df)
            upstream_by_key[FeatureKey(k)] = nw.from_native(df.lazy(), eager_only=False)

        # Prepare upstream (handles filtering, selecting, renaming, and joining)
        joined = engine.prepare_upstream(upstream_by_key, filters=None)

        # Build the mapping of upstream_key -> provenance_by_field column name
        # The new naming convention is: {column_name}{feature_key.to_column_suffix()}
        mapping = {}
        for upstream_key_str in upstream_refs.keys():
            upstream_key = FeatureKey(upstream_key_str)
            # The provenance_by_field column is renamed using to_column_suffix()
            provenance_col_name = (
                f"{METAXY_PROVENANCE_BY_FIELD}{upstream_key.to_column_suffix()}"
            )
            mapping[upstream_key_str] = provenance_col_name

        return joined, mapping


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
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

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
                        columns=("custom_col1",),
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
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

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
                        columns=(),  # Only system columns
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
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify only essential system columns are present
        # Note: feature_version and snapshot_version are NOT considered essential for joining
        # to avoid conflicts when joining multiple upstream features
        assert "sample_uid" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream" in joined_df.columns
        assert (
            "metaxy_feature_version" not in joined_df.columns
        )  # Not essential, dropped to avoid conflicts
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
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

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
                        columns=("custom_col1", "custom_col2"),
                        rename={"custom_col1": "renamed_col1"},
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
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

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
        with pytest.raises(
            ValueError, match="Found additional shared columns.*conflict_col"
        ):
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
                        columns=("custom_col",),  # Don't explicitly list system columns
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
                "metaxy_snapshot_version": ["s1", "s1", "s1"],
                "custom_col": ["a", "b", "c"],
                "other_col": [10, 20, 30],
            }
        )

        joiner = TestJoiner()
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

        joined, _ = DownstreamFeature.load_input(joiner, upstream_refs)
        joined_df = joined.collect().to_polars()

        # Verify essential system columns are preserved
        assert "sample_uid" in joined_df.columns
        assert "metaxy_provenance_by_field__test_upstream" in joined_df.columns
        # Note: feature_version and snapshot_version are NOT preserved to avoid conflicts
        assert "metaxy_feature_version" not in joined_df.columns
        assert "metaxy_snapshot_version" not in joined_df.columns
        # Verify selected column
        assert "custom_col" in joined_df.columns
        # Verify non-selected column is not present
        assert "other_col" not in joined_df.columns

    def test_pydantic_validation_rename_to_system_column(self):
        """Test that renaming to system column names is forbidden."""

        # Create upstream feature first
        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Renaming to provenance_by_field should raise an error
        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to system column name.*provenance_by_field",
        ):

            class BadFeature1(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "bad1"]),
                    deps=[
                        FeatureDep(
                            feature=FeatureKey(["test", "upstream"]),
                            rename={
                                "old_col": "metaxy_provenance_by_field"
                            },  # Not allowed
                        )
                    ],
                ),
            ):
                pass

        # Renaming to feature_version should raise an error
        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to system column name.*feature_version",
        ):

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

        # Renaming to sample_uid should raise an error because upstream has sample_uid as its ID column
        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to ID column.*sample_uid",
        ):

            class BadFeature3(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "bad3"]),
                    deps=[
                        FeatureDep(
                            feature=FeatureKey(["test", "upstream"]),
                            rename={
                                "old_col": "sample_uid"
                            },  # Not allowed - it's upstream's ID column
                        )
                    ],
                ),
            ):
                pass

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
                        rename={
                            "some_col": "user_id_renamed"
                        },  # Allowed - not renaming to actual ID column name
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
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

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
            columns=("col1", "col2"),
            rename={"col1": "new_col1"},
        )

        # Serialize to dict
        dep_dict = dep.model_dump()
        assert dep_dict["feature"] == ["test", "upstream"]
        assert dep_dict["columns"] == ("col1", "col2")
        assert dep_dict["rename"] == {"col1": "new_col1"}

        # Round-trip through JSON
        dep_json = dep.model_dump_json()
        dep_restored = FeatureDep.model_validate_json(dep_json)
        assert dep_restored.feature == FeatureKey(["test", "upstream"])
        assert dep_restored.columns == ("col1", "col2")
        assert dep_restored.rename == {"col1": "new_col1"}

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
                        columns=("important_col",),
                    ),
                    # Rename columns from upstream3
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream3"]),
                        columns=("shared_col", "unique_col"),
                        rename={"shared_col": "upstream3_shared"},
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
                        columns=("value", "category"),
                        rename={"value": "upstream_value"},
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
        upstream_refs = {
            "test/upstream": nw.from_native(upstream_data.lazy(), eager_only=False)
        }

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

    def test_columns_and_rename_serialized_to_snapshot(self, graph: FeatureGraph):
        """Test that columns and rename fields are properly serialized when pushing graph snapshot."""
        import json

        from metaxy.metadata_store import InMemoryMetadataStore
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
                        columns=("col1", "col2"),
                        rename={"col1": "renamed_col1"},
                    )
                ],
            ),
        ):
            pass

        # Create store and push snapshot
        store = InMemoryMetadataStore()
        with store:
            _ = SystemTableStorage(store).push_graph_snapshot()

            # Read the snapshot from feature_versions table
            versions = (
                store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
                .collect()
                .to_polars()
            )

            # Find the downstream feature record
            downstream_record = versions.filter(
                pl.col("feature_key") == "test/downstream"
            )
            assert len(downstream_record) == 1

            # Parse the feature_spec JSON
            feature_spec_json = downstream_record["feature_spec"][0]
            feature_spec_dict = json.loads(feature_spec_json)

            # Verify the spec contains deps
            assert "deps" in feature_spec_dict
            assert len(feature_spec_dict["deps"]) == 1

            # Verify columns and rename are serialized
            dep = feature_spec_dict["deps"][0]
            assert dep["feature"] == ["test", "upstream"]
            assert dep["columns"] == [
                "col1",
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
            assert dep_dict["feature"] == ["test", "upstream"]
            assert dep_dict["columns"] == ["col1", "col2"]
            assert dep_dict["rename"] == {"col1": "renamed_col1"}

            # Verify Pydantic can deserialize it back
            reconstructed_spec = SampleFeatureSpec.model_validate(downstream_spec_dict)
            assert reconstructed_spec.deps and len(reconstructed_spec.deps) == 1
            reconstructed_dep = reconstructed_spec.deps[0]
            assert reconstructed_dep.columns == ("col1", "col2")
            assert reconstructed_dep.rename == {"col1": "renamed_col1"}

    def test_duplicate_renamed_columns_within_single_dependency(self):
        """Test that renaming multiple columns to the same name within a dependency is rejected."""

        # Create upstream feature first
        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
            ),
        ):
            pass

        # Should raise error when trying to rename multiple columns to the same name
        with pytest.raises(ValueError, match="Duplicate column names after renaming"):

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

    def test_duplicate_columns_across_dependencies_validation(self):
        """Test that duplicate columns across dependencies are detected at graph assembly time."""

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

        # This should raise an error at graph assembly time
        # because both dependencies rename columns to "duplicate_col"
        with pytest.raises(ValueError, match="would have duplicate column names"):

            class BadDownstreamFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "bad_downstream"]),
                    deps=[
                        FeatureDep(
                            feature=FeatureKey(["test", "upstream1"]),
                            columns=("col1",),
                            rename={"col1": "duplicate_col"},
                        ),
                        FeatureDep(
                            feature=FeatureKey(["test", "upstream2"]),
                            columns=("col2",),
                            rename={"col2": "duplicate_col"},
                        ),
                    ],
                ),
            ):
                pass

    def test_duplicate_columns_with_selected_columns(self):
        """Test that duplicate detection works with column selection."""

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

        # This should raise because both select "shared_col" without renaming
        with pytest.raises(ValueError, match="would have duplicate column names"):

            class BadDownstreamFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "bad_downstream"]),
                    deps=[
                        FeatureDep(
                            feature=FeatureKey(["test", "upstream1"]),
                            columns=("shared_col", "col1"),  # Selects shared_col
                        ),
                        FeatureDep(
                            feature=FeatureKey(["test", "upstream2"]),
                            columns=(
                                "shared_col",
                                "col2",
                            ),  # Also selects shared_col
                        ),
                    ],
                ),
            ):
                pass

    def test_renaming_to_upstream_id_columns_forbidden(self):
        """Test that renaming to upstream's ID columns is forbidden."""

        # Create upstream with custom ID columns
        class UpstreamWithCustomIDs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                id_columns=["user_id", "session_id"],
            ),
        ):
            pass

        # Renaming to upstream's ID column should raise an error
        with pytest.raises(
            ValueError, match="Cannot rename column.*to ID column.*user_id"
        ):

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

        # Renaming to another upstream ID column should also raise an error
        with pytest.raises(
            ValueError, match="Cannot rename column.*to ID column.*session_id"
        ):

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

    def test_renaming_to_system_columns_forbidden(self):
        """Test that renaming to system columns and ID columns is forbidden."""

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                # Default ID columns: ["sample_uid"]
            ),
        ):
            pass

        # Renaming to system column provenance_by_field should raise an error
        with pytest.raises(
            ValueError,
            match="Cannot rename column.*to system column name.*provenance_by_field",
        ):

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

        # Renaming to ID column sample_uid should raise an error
        with pytest.raises(
            ValueError, match="Cannot rename column.*to ID column.*sample_uid"
        ):

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
