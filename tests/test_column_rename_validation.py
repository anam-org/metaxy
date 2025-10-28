"""Test validation for column renaming to prevent conflicts."""

import pytest

from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.types import FeatureKey


def test_cannot_rename_to_system_column():
    """Test that renaming to system columns is prevented."""
    with pytest.raises(ValueError, match="Cannot rename column to system column"):
        FeatureDep(
            key=FeatureKey(["upstream"]),
            rename={"old_col": "data_version"},  # System column
        )


def test_can_rename_to_match_feature_pydantic_field():
    """Test that renaming to a pydantic field on the Feature IS allowed.

    The feature's own fields are its own business and don't conflict with
    renamed columns from upstream features.
    """
    graph = FeatureGraph()

    with graph.use():
        # Create upstream feature
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
            ),
        ):
            pass

        # This should work - pydantic fields don't conflict with renamed columns
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        key=FeatureKey(["upstream"]),
                        rename={"some_col": "user_id"},  # Same name as pydantic field
                    )
                ],
            ),
        ):
            # This field exists on the Feature
            user_id: str
            score: float

        # Feature should be created successfully
        assert DownstreamFeature.spec.key == FeatureKey(["downstream"])


def test_can_rename_to_non_conflicting_name():
    """Test that renaming to non-conflicting names is allowed."""
    graph = FeatureGraph()

    with graph.use():
        # Create upstream feature
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
            ),
        ):
            pass

        # This should work fine - no conflicts
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        key=FeatureKey(["upstream"]),
                        rename={
                            "some_col": "upstream_some_col"
                        },  # Non-conflicting name
                    )
                ],
            ),
        ):
            user_id: str
            score: float


def test_can_rename_to_match_id_columns():
    """Test that renaming to match id_columns IS allowed.

    ID columns are the feature's own business and don't conflict with
    renamed columns from upstream features.
    """
    graph = FeatureGraph()

    with graph.use():
        # Create upstream feature
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
            ),
        ):
            pass

        # This should work - ID columns don't conflict with renamed columns
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        key=FeatureKey(["upstream"]),
                        rename={"some_col": "user_id"},  # Same as id_columns
                    )
                ],
                id_columns=["user_id", "session_id"],  # Custom ID columns
            ),
        ):
            pass

        # Feature should be created successfully
        assert DownstreamFeature.spec.key == FeatureKey(["downstream"])


def test_cannot_rename_to_duplicate_columns_across_deps():
    """Test that multiple dependencies cannot rename to the same column name."""
    graph = FeatureGraph()

    with graph.use():
        # Create two upstream features
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream1"]),
                deps=None,
            ),
        ):
            col1: str  # Define the column we're renaming

        class Upstream2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream2"]),
                deps=None,
            ),
        ):
            col2: int  # Define the column we're renaming

        # Should fail - both deps trying to rename to same column
        with pytest.raises(ValueError, match="Column name conflict after renaming"):

            class DownstreamFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["downstream"]),
                    deps=[
                        FeatureDep(
                            key=FeatureKey(["upstream1"]),
                            rename={"col1": "shared_name"},  # Rename to 'shared_name'
                        ),
                        FeatureDep(
                            key=FeatureKey(["upstream2"]),
                            rename={
                                "col2": "shared_name"
                            },  # Also rename to 'shared_name'
                        ),
                    ],
                ),
            ):
                pass
