"""Shared fixtures for CLI tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from metaxy._testing import TempMetaxyProject


# ============================================================================
# Reusable Functions (not fixtures, to allow use within with_features context)
# ============================================================================


def define_video_files_feature():
    """Create VideoFiles feature definition callable."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    return features


def define_multi_features():
    """Create VideoFiles, AudioFiles, and TextFiles feature definitions."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class AudioFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class TextFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["text", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    return features


def write_sample_data(
    metaxy_project: TempMetaxyProject,
    feature_key_str: str,
    store_name: str = "dev",
    sample_uids: list[int] | None = None,
    created_at_days_ago: list[int] | None = None,
    user_ids: list[str] | None = None,
    custom_columns: dict[str, list[Any]] | None = None,
) -> None:
    """Write sample metadata with timestamps and optional custom columns.

    Args:
        metaxy_project: Test project
        feature_key_str: Feature key as string (e.g., "video/files")
        store_name: Name of store to write to (default: "dev")
        sample_uids: List of sample IDs to use (default: [1, 2, 3])
        created_at_days_ago: Days ago for each sample's created_at (default: [0, 0, 0])
        user_ids: Optional user_id values for each sample
        custom_columns: Optional dict of column_name -> values to add
    """
    from metaxy.models.types import FeatureKey

    if sample_uids is None:
        sample_uids = [1, 2, 3]

    if created_at_days_ago is None:
        created_at_days_ago = [0] * len(sample_uids)

    # Parse feature key
    feature_key = FeatureKey(feature_key_str.split("/"))

    # Get feature class from project's graph
    graph = metaxy_project.graph
    feature_cls = graph.get_feature_by_key(feature_key)

    # Calculate timestamps
    now = datetime.now()
    timestamps = [now - timedelta(days=days) for days in created_at_days_ago]

    # Create sample data
    data = {
        "sample_uid": sample_uids,
        "value": [f"val_{i}" for i in sample_uids],
        "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in sample_uids],
        "metaxy_created_at": timestamps,
    }

    # Add user_ids if provided
    if user_ids:
        data["user_id"] = user_ids

    # Add custom columns if provided
    if custom_columns:
        for col_name, col_values in custom_columns.items():
            data[col_name] = col_values

    sample_data = pl.DataFrame(data)

    # Write metadata directly to store
    store = metaxy_project.stores[store_name]
    with graph.use():
        with store:
            store.write_metadata(feature_cls, sample_data)


# ============================================================================
# CLI Helper Functions
# ============================================================================


def run_cleanup_json(
    metaxy_project: TempMetaxyProject, *args: str, check: bool = True
) -> dict[str, Any]:
    """Run cleanup command and parse JSON result.

    Args:
        metaxy_project: Test project
        *args: CLI arguments (e.g., "--feature", "video/files", "--id", "1")
        check: If False, don't raise on non-zero exit code

    Returns:
        Parsed JSON dict
    """
    # Add --format json if not already present
    args_list = list(args)
    if "--format" not in args_list:
        args_list.extend(["--format", "json"])

    result = metaxy_project.run_cli(["metadata", "cleanup", *args_list], check=check)
    return json.loads(result.stdout)


def assert_cleanup_result(
    result: dict[str, Any],
    expected_deleted: int | None = None,
    deletion_mode: str | None = None,
    num_features: int | None = None,
) -> None:
    """Assert cleanup result has expected values.

    Args:
        result: Cleanup result dict from JSON output
        expected_deleted: Expected total_rows_deleted
        deletion_mode: Expected deletion_mode ("soft" or "hard")
        num_features: Expected number of features_affected
    """
    if expected_deleted is not None:
        assert result["total_rows_deleted"] == expected_deleted
    if deletion_mode is not None:
        assert result["deletion_mode"] == deletion_mode
    if num_features is not None:
        assert len(result["features_affected"]) == num_features
