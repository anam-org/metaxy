"""Test for filter + column selection bug.

Reproduces the issue where filtering on a column fails when reading a feature
that has dependencies with restricted columns.
"""

import time

import narwhals as nw
import polars as pl
from pytest_cases import parametrize_with_cases

from metaxy import BaseFeature, FeatureDep, FeatureSpec
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey

from .conftest import AllStoresCases


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_with_filter_on_own_column(store: MetadataStore):
    """Test filtering on a feature's OWN column when the feature has dependencies.

    This reproduces the bug where:
    1. ChildFeature has its own 'height' column (not inherited from parent)
    2. ChildFeature depends on ParentFeature with select=("dataset", "path", "size")
    3. User calls read on ChildFeature with a filter on 'height'
    4. The filter fails because column inference goes wrong

    The error was: IbisTypeError: Column 'height' is not found in table.
    """
    graph = FeatureGraph()

    with graph.use():
        # Parent feature (like SceneChunk)
        parent_key = FeatureKey(["test_own_col_filter", "parent"])

        class ParentFeature(
            BaseFeature,
            spec=FeatureSpec(key=parent_key, id_columns=["chunk_id"]),
        ):
            chunk_id: str
            dataset: str
            path: str
            size: int
            parent_height: int  # Parent has its own height

        # Child feature (like CroppedSceneChunk) - has its OWN height column
        child_key = FeatureKey(["test_own_col_filter", "child"])

        class ChildFeature(
            BaseFeature,
            spec=FeatureSpec(
                key=child_key,
                id_columns=["chunk_id"],
                deps=[
                    # Only select specific columns from parent - NOT including parent_height
                    FeatureDep(feature=ParentFeature, select=("dataset", "path", "size")),
                ],
            ),
        ):
            chunk_id: str
            height: int  # Child's OWN height column (different from parent's)
            width: int

        with store.open("w"):
            # Write parent data first
            parent_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3"],
                    "dataset": ["ds1", "ds1", "ds1"],
                    "path": ["/path/1", "/path/2", "/path/3"],
                    "size": [100, 200, 300],
                    "parent_height": [1080, 1080, 1080],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                    ],
                }
            )
            store.write(ParentFeature, parent_data)

            # Write child data with its own height values
            child_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3"],
                    "height": [480, 720, 1080],  # Child's own height
                    "width": [640, 1280, 1920],
                    "metaxy_provenance_by_field": [
                        {"default": "ch1"},
                        {"default": "ch2"},
                        {"default": "ch3"},
                    ],
                }
            )
            store.write(ChildFeature, child_data)

            # Now try to READ ChildFeature with a filter on its OWN 'height' column
            result = store.read(
                ChildFeature,
                filters=[nw.col("height") > 500],
            )

            df = result.collect().to_polars()
            assert set(df["chunk_id"].to_list()) == {"c2", "c3"}, (
                f"Expected chunks with height > 500, got {df['chunk_id'].to_list()}"
            )


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_with_filter_after_dedup(store: MetadataStore):
    """Test filtering on a feature's column after deduplication.

    This tests the scenario where:
    1. Feature has multiple versions of rows (needs dedup)
    2. Filter is applied on the feature's own column
    3. Filter should see the LATEST version of each row
    """
    graph = FeatureGraph()

    with graph.use():
        parent_key = FeatureKey(["test_own_col_filter_dedup", "parent"])

        class ParentFeature(
            BaseFeature,
            spec=FeatureSpec(key=parent_key, id_columns=["chunk_id"]),
        ):
            chunk_id: str
            dataset: str
            path: str
            size: int

        child_key = FeatureKey(["test_own_col_filter_dedup", "child"])

        class ChildFeature(
            BaseFeature,
            spec=FeatureSpec(
                key=child_key,
                id_columns=["chunk_id"],
                deps=[
                    FeatureDep(feature=ParentFeature, select=("dataset", "path", "size")),
                ],
            ),
        ):
            chunk_id: str
            height: int
            width: int

        with store.open("w"):
            # Write parent data
            parent_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3"],
                    "dataset": ["ds1", "ds1", "ds1"],
                    "path": ["/path/1", "/path/2", "/path/3"],
                    "size": [100, 200, 300],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                    ],
                }
            )
            store.write(ParentFeature, parent_data)

            # Write first version of child data
            child_data_v1 = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3"],
                    "height": [480, 720, 1080],
                    "width": [640, 1280, 1920],
                    "metaxy_provenance_by_field": [
                        {"default": "ch1"},
                        {"default": "ch2"},
                        {"default": "ch3"},
                    ],
                }
            )
            store.write(ChildFeature, child_data_v1)

            # Wait and write updated version for c1 and c2
            time.sleep(0.01)
            child_data_v2 = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2"],
                    "height": [500, 800],  # Updated heights: c1: 480->500, c2: 720->800
                    "width": [640, 1280],
                    "metaxy_provenance_by_field": [
                        {"default": "ch1_v2"},
                        {"default": "ch2_v2"},
                    ],
                }
            )
            store.write(ChildFeature, child_data_v2)

            # Filter on height > 600 - should see LATEST versions
            # c1 has height=500 (updated), c2 has height=800 (updated), c3 has height=1080
            # Only c2 and c3 should match
            result = store.read(
                ChildFeature,
                filters=[nw.col("height") > 600],
            )

            df = result.collect().to_polars()
            assert set(df["chunk_id"].to_list()) == {"c2", "c3"}, (
                f"Expected chunks with height > 600 after dedup, got {df['chunk_id'].to_list()}"
            )


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_with_columns_and_filter_on_different_column(store: MetadataStore):
    """Test reading with specific columns but filtering on a column NOT in that list.

    This reproduces the exact bug from the CLI:
        mx metadata status --feature chunk/crop --filter "height IS NULL"

    The status command passes:
    - columns=["chunk_id"]  (only ID columns for counting)
    - filters=[nw.col("height").is_null()]  (filter on a different column)

    The filter references 'height' which is NOT in the columns list.
    The code should read all columns to apply the filter, then select only the requested columns.
    """
    graph = FeatureGraph()

    with graph.use():
        feature_key = FeatureKey(["test_cols_filter_bug", "feature"])

        class MyFeature(
            BaseFeature,
            spec=FeatureSpec(key=feature_key, id_columns=["chunk_id"]),
        ):
            chunk_id: str
            height: int | None
            width: int

        with store.open("w"):
            # Write data with some NULL heights
            data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3", "c4"],
                    "height": [480, None, 1080, None],  # c2 and c4 have NULL height
                    "width": [640, 1280, 1920, 800],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                        {"default": "h4"},
                    ],
                }
            )
            store.write(MyFeature, data)

            # Read with columns=["chunk_id"] but filter on "height IS NULL"
            # This is what `mx metadata status --feature X --filter "height IS NULL"` does
            result = store.read(
                MyFeature,
                columns=["chunk_id"],  # Only request ID column
                filters=[nw.col("height").is_null()],  # But filter on height!
            )

            df = result.collect().to_polars()
            # Should only get c2 and c4 (where height IS NULL)
            assert set(df["chunk_id"].to_list()) == {"c2", "c4"}, (
                f"Expected chunks with NULL height, got {df['chunk_id'].to_list()}"
            )
            # And the result should only have the chunk_id column (what we requested)
            assert df.columns == ["chunk_id"], f"Expected only chunk_id column, got {df.columns}"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_with_deps_columns_and_filter_on_own_column(store: MetadataStore):
    """Test the EXACT scenario: feature with deps, columns param, filter on own column.

    This combines all the conditions from the bug report:
    1. Feature has dependencies with restricted columns (FeatureDep with select=(...))
    2. User calls read with columns=["chunk_id"] (ID columns only)
    3. User passes filter on "height" which is the feature's OWN column (not from parent)
    4. Filter column is NOT in the requested columns list

    This is the exact scenario:
        mx metadata status --feature chunk/crop --filter "height IS NULL"

    Where chunk/crop:
    - Has FeatureDep(feature=SceneChunk, select=("dataset", "path", "size"))
    - Has its own "height" column from _Video base class
    - Status command passes columns=["chunk_id"] but filters on "height"
    """
    graph = FeatureGraph()

    with graph.use():
        # Parent feature (like SceneChunk)
        parent_key = FeatureKey(["test_exact_bug", "parent"])

        class ParentFeature(
            BaseFeature,
            spec=FeatureSpec(key=parent_key, id_columns=["chunk_id"]),
        ):
            chunk_id: str
            dataset: str
            path: str
            size: int

        # Child feature (like CroppedSceneChunk) with deps and its own height
        child_key = FeatureKey(["test_exact_bug", "child"])

        class ChildFeature(
            BaseFeature,
            spec=FeatureSpec(
                key=child_key,
                id_columns=["chunk_id"],
                deps=[
                    # Restricted columns - height is NOT included
                    FeatureDep(feature=ParentFeature, select=("dataset", "path", "size")),
                ],
            ),
        ):
            chunk_id: str
            height: int | None  # Child's OWN height column (nullable)
            width: int

        with store.open("w"):
            # Write parent data first
            parent_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3", "c4"],
                    "dataset": ["ds1", "ds1", "ds1", "ds1"],
                    "path": ["/path/1", "/path/2", "/path/3", "/path/4"],
                    "size": [100, 200, 300, 400],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                        {"default": "h4"},
                    ],
                }
            )
            store.write(ParentFeature, parent_data)

            # Write child data with some NULL heights
            child_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3", "c4"],
                    "height": [480, None, 1080, None],  # c2 and c4 have NULL height
                    "width": [640, 1280, 1920, 800],
                    "metaxy_provenance_by_field": [
                        {"default": "ch1"},
                        {"default": "ch2"},
                        {"default": "ch3"},
                        {"default": "ch4"},
                    ],
                }
            )
            store.write(ChildFeature, child_data)

            # The EXACT bug scenario:
            # - columns=["chunk_id"] (only ID column)
            # - filter on "height IS NULL" (filter column NOT in columns list)
            # - feature has dependencies with restricted columns
            result = store.read(
                ChildFeature,
                columns=["chunk_id"],  # Only request ID column
                filters=[nw.col("height").is_null()],  # Filter on height (not in columns)
            )

            df = result.collect().to_polars()
            # Should only get c2 and c4 (where height IS NULL)
            assert set(df["chunk_id"].to_list()) == {"c2", "c4"}, (
                f"Expected chunks with NULL height, got {df['chunk_id'].to_list()}"
            )
            # And the result should only have the chunk_id column (what we requested)
            assert df.columns == ["chunk_id"], f"Expected only chunk_id column, got {df.columns}"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_resolve_update_with_target_filter_on_child_only_column(store: MetadataStore):
    """Test resolve_update with target_filters on a column that only exists in child.

    This tests the fix for:
        mx metadata status --feature chunk/crop --filter "height IS NULL"

    The solution is to use target_filters (--filter) for columns that only exist
    in the target feature, and global_filters (--global-filter) for columns that
    exist in ALL features.

    Expected behavior: target_filters are only applied to the target feature,
    so filtering on 'height' works even though upstream doesn't have it.
    """
    graph = FeatureGraph()

    with graph.use():
        # Parent feature (like SceneChunk) - does NOT have 'height'
        parent_key = FeatureKey(["test_target_filter", "parent"])

        class ParentFeature(
            BaseFeature,
            spec=FeatureSpec(key=parent_key, id_columns=["chunk_id"]),
        ):
            chunk_id: str
            dataset: str
            path: str
            size: int
            # Note: NO height column here!

        # Child feature (like CroppedSceneChunk) - HAS 'height'
        child_key = FeatureKey(["test_target_filter", "child"])

        class ChildFeature(
            BaseFeature,
            spec=FeatureSpec(
                key=child_key,
                id_columns=["chunk_id"],
                deps=[
                    FeatureDep(feature=ParentFeature, select=("dataset", "path", "size")),
                ],
            ),
        ):
            chunk_id: str
            height: int | None  # Child's OWN height column
            width: int

        with store.open("w"):
            # Write parent data
            parent_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3", "c4"],
                    "dataset": ["ds1", "ds1", "ds1", "ds1"],
                    "path": ["/path/1", "/path/2", "/path/3", "/path/4"],
                    "size": [100, 200, 300, 400],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                        {"default": "h4"},
                    ],
                }
            )
            store.write(ParentFeature, parent_data)

            # Write child data with some NULL heights
            child_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3", "c4"],
                    "height": [480, None, 1080, None],  # c2 and c4 have NULL height
                    "width": [640, 1280, 1920, 800],
                    "metaxy_provenance_by_field": [
                        {"default": "ch1"},
                        {"default": "ch2"},
                        {"default": "ch3"},
                        {"default": "ch4"},
                    ],
                }
            )
            store.write(ChildFeature, child_data)

            # Using target_filters should work - it only applies to the target feature
            result = store.resolve_update(
                ChildFeature,
                target_filters=[nw.col("height").is_null()],
            )
            # The key test is that it doesn't crash and returns filtered results
            # Since all child rows are materialized and filter is on height IS NULL,
            # we expect to see rows that need updates based on the filtered view
            assert result is not None


@parametrize_with_cases("store", cases=AllStoresCases)
def test_resolve_update_with_global_filter_on_child_only_column_fails(store: MetadataStore):
    """Test that resolve_update with global_filters FAILS on a child-only column.

    This verifies the expected behavior: global_filters are applied to ALL features,
    so if you filter on a column that doesn't exist in an upstream feature, it
    should fail with a column-not-found error.

    Users should use --filter (target_filters) for target-only columns, and
    --global-filter (global_filters) for columns that exist in all features.
    """
    import pytest

    graph = FeatureGraph()

    with graph.use():
        # Parent feature (like SceneChunk) - does NOT have 'height'
        parent_key = FeatureKey(["test_global_filter_fails", "parent"])

        class ParentFeature(
            BaseFeature,
            spec=FeatureSpec(key=parent_key, id_columns=["chunk_id"]),
        ):
            chunk_id: str
            dataset: str
            path: str
            size: int
            # Note: NO height column here!

        # Child feature (like CroppedSceneChunk) - HAS 'height'
        child_key = FeatureKey(["test_global_filter_fails", "child"])

        class ChildFeature(
            BaseFeature,
            spec=FeatureSpec(
                key=child_key,
                id_columns=["chunk_id"],
                deps=[
                    FeatureDep(feature=ParentFeature, select=("dataset", "path", "size")),
                ],
            ),
        ):
            chunk_id: str
            height: int | None  # Child's OWN height column
            width: int

        with store.open("w"):
            # Write parent data
            parent_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3", "c4"],
                    "dataset": ["ds1", "ds1", "ds1", "ds1"],
                    "path": ["/path/1", "/path/2", "/path/3", "/path/4"],
                    "size": [100, 200, 300, 400],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                        {"default": "h4"},
                    ],
                }
            )
            store.write(ParentFeature, parent_data)

            # Write child data with some NULL heights
            child_data = pl.DataFrame(
                {
                    "chunk_id": ["c1", "c2", "c3", "c4"],
                    "height": [480, None, 1080, None],  # c2 and c4 have NULL height
                    "width": [640, 1280, 1920, 800],
                    "metaxy_provenance_by_field": [
                        {"default": "ch1"},
                        {"default": "ch2"},
                        {"default": "ch3"},
                        {"default": "ch4"},
                    ],
                }
            )
            store.write(ChildFeature, child_data)

            # Using global_filters with a column that doesn't exist in parent SHOULD FAIL
            # This is the expected behavior - global_filters apply to ALL features
            with pytest.raises(Exception):  # Could be ColumnNotFoundError or IbisTypeError
                store.resolve_update(
                    ChildFeature,
                    global_filters=[nw.col("height").is_null()],
                )
