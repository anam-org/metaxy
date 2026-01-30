"""Tests for metadata CLI commands."""

import json
from pathlib import Path

import polars as pl
import pytest
from metaxy_testing import TempMetaxyProject


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_up_to_date(metaxy_project: TempMetaxyProject, output_format: str):
    """Status command when metadata is up-to-date for both formats."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        # Create a root feature
        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Create a non-root feature with upstream dependency
        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write upstream metadata (this opens/closes store internally)
        metaxy_project.write_sample_metadata("video/files_root")

        # Now compute and write downstream metadata with correct provenance
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            # feature_cls removed - using feature_key directly
            increment = store.resolve_update(feature_key, lazy=False)

            # Write the computed metadata to the store
            store.write_metadata(feature_key, increment.added.to_polars())

        # Check status for the non-root feature
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            assert feature["feature_key"] == "video/files"
            assert feature["status"] == "up_to_date"
            assert feature["store_rows"] == 3
            assert feature["missing"] == 0
            assert feature["stale"] == 0
            assert feature["orphaned"] == 0
        else:
            # Check for table output with ✓ icon for up-to-date status
            assert "video/files" in result.stdout
            assert "✓" in result.stdout
            assert "3" in result.stdout  # Materialized count


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_missing_metadata(metaxy_project: TempMetaxyProject, output_format: str):
    """Status command when metadata is missing."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write upstream metadata but not for VideoFiles
        metaxy_project.write_sample_metadata("video/files_root")

        # Check status - should show missing metadata
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            assert feature["feature_key"] == "video/files"
            assert feature["status"] == "missing"
            assert feature["store_rows"] == 0
            assert feature["missing"] == 3
            assert feature["stale"] == 0
            assert feature["orphaned"] == 0
        else:
            # Check for table output with ✗ icon for missing metadata
            assert "video/files" in result.stdout
            assert "✗" in result.stdout
            assert "3" in result.stdout  # Missing count


def test_metadata_status_assert_in_sync_fails(metaxy_project: TempMetaxyProject):
    """Test that --assert-in-sync fails when metadata needs updates."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write upstream but not downstream - needs update
        metaxy_project.write_sample_metadata("video/files_root")

        # Check status with --assert-in-sync
        result = metaxy_project.run_cli(
            ["metadata", "status", "video/files", "--assert-in-sync"],
            check=False,
        )

        assert result.returncode == 1


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_multiple_features(metaxy_project: TempMetaxyProject, output_format: str):
    """Status command with multiple features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class FilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[FilesRoot],
            ),
        ):
            pass

        class AudioFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[FilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata for root and video but not audio
        metaxy_project.write_sample_metadata("files_root")
        metaxy_project.write_sample_metadata("video/files")

        # Check status for both
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "audio/files",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            assert set(data["features"]) == {"video/files", "audio/files"}
            assert data["features"]["audio/files"]["status"] == "missing"
        else:
            assert "video/files" in result.stdout
            assert "audio/files" in result.stdout


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_invalid_feature_key(metaxy_project: TempMetaxyProject, output_format: str):
    """Status command with a feature missing from the graph."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Check status for non-existent feature
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "nonexistent/feature",
                "--format",
                output_format,
            ],
            check=False,
        )

        # Should show warning and continue
        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            assert data["warning"] == "No valid features to check"
            assert data["features"] == {}
        else:
            assert "No valid features to check" in result.stdout


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_with_verbose(metaxy_project: TempMetaxyProject, output_format: str):
    """Status command with --verbose flag."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write upstream but not downstream - there will be samples to add
        metaxy_project.write_sample_metadata("video/files_root")
        # Check status with verbose
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                output_format,
                "--verbose",
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            assert feature["status"] == "missing"
            assert feature["sample_details"]
            assert any("sample_uid" in detail for detail in feature["sample_details"])
        else:
            assert "video/files" in result.stdout
            assert "Missing samples" in result.stdout


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_with_explicit_store(metaxy_project: TempMetaxyProject, output_format: str):
    """Status command with explicit --store flag."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write upstream metadata (this opens/closes store internally)
        metaxy_project.write_sample_metadata("video/files_root", store_name="dev")

        # Now compute and write downstream metadata with correct provenance
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            # feature_cls removed - using feature_key directly
            increment = store.resolve_update(feature_key, lazy=False)

            # Write the computed metadata to the store
            store.write_metadata(feature_key, increment.added.to_polars())

        # Check status with explicit store
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--store",
                "dev",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            assert feature["feature_key"] == "video/files"
            assert feature["status"] == "up_to_date"
            assert feature["needs_update"] is False
        else:
            # Check for table output with ✓ icon for up-to-date status
            assert "video/files" in result.stdout
            assert "✓" in result.stdout


def test_metadata_status_requires_feature_or_all(metaxy_project: TempMetaxyProject):
    """Test that status requires either feature arguments or --all-features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Try to check status without specifying features
        result = metaxy_project.run_cli(["metadata", "status", "--format", "json"], check=False)

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "MISSING_REQUIRED_FLAG"
        assert "--all-features" in str(error["required_flags"])
        assert "<features>" in str(error["required_flags"])


def test_metadata_status_cannot_specify_both_flags(metaxy_project: TempMetaxyProject):
    """Test that cannot specify both feature arguments and --all-features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Try to specify both flags
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--all-features",
                "--format",
                "json",
            ],
            check=False,
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "CONFLICTING_FLAGS"


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_all_features(metaxy_project: TempMetaxyProject, output_format: str):
    """Test status command with --all-features flag."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class FilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[FilesRoot],
            ),
        ):
            pass

        class AudioFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[FilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata for root only
        metaxy_project.write_sample_metadata("files_root")

        # Check status for all features
        result = metaxy_project.run_cli(["metadata", "status", "--all-features", "--format", output_format])

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            # Should include all 3 features
            assert len(data["features"]) == 3
            assert "files_root" in data["features"]
            assert "video/files" in data["features"]
            assert "audio/files" in data["features"]
            # Root feature should have root_feature status
            assert data["features"]["files_root"]["is_root_feature"] is True
            assert data["features"]["files_root"]["status"] == "root_feature"
        else:
            # Check for table output with feature names and ○ icon for root feature
            assert "files_root" in result.stdout
            assert "video/files" in result.stdout
            assert "audio/files" in result.stdout
            assert "○" in result.stdout  # Root feature icon


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_root_feature(metaxy_project: TempMetaxyProject, output_format: str):
    """Test status command for a root feature (no upstream dependencies)."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["root_feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata for the root feature
        metaxy_project.write_sample_metadata("root_feature")

        # Check status for the root feature
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "root_feature",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["root_feature"]
            assert feature["feature_key"] == "root_feature"
            assert feature["is_root_feature"] is True
            assert feature["status"] == "root_feature"
            assert feature["store_rows"] == 3
            # Root features don't have meaningful missing/stale/orphaned counts (excluded from JSON)
            assert "missing" not in feature
            assert "stale" not in feature
            assert "orphaned" not in feature
        else:
            # Check for table output with ○ icon for root feature
            assert "root_feature" in result.stdout
            assert "○" in result.stdout  # Root feature icon
            assert "3" in result.stdout  # Materialized count


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_root_feature_missing_metadata(metaxy_project: TempMetaxyProject, output_format: str):
    """Test status command for a root feature with no metadata."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["root_feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Don't write any metadata

        # Check status for the root feature
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "root_feature",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["root_feature"]
            assert feature["feature_key"] == "root_feature"
            # Missing metadata takes precedence over root feature status
            assert feature["status"] == "missing"
            assert feature["metadata_exists"] is False
            assert feature["store_rows"] == 0
        else:
            # Check for table output with ✗ icon for missing metadata
            assert "root_feature" in result.stdout
            assert "✗" in result.stdout  # Missing metadata icon
            assert "0" in result.stdout  # Materialized count


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_with_global_filter(metaxy_project: TempMetaxyProject, output_format: str):
    """Test status command with --global-filter flag to filter metadata by column value.

    --global-filter applies to all features (both upstream and target), which is needed
    when you want to filter the entire pipeline by a common column like 'category'.
    """

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write upstream metadata with a 'category' column for filtering
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "value": ["val_1", "val_2", "val_3", "val_4", "val_5"],
                "category": ["A", "A", "B", "B", "A"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 6)],
            }
        )

        feature_key_root = FeatureKey(["video", "files_root"])

        with graph.use(), store:
            store.write_metadata(feature_key_root, upstream_data)

        # Write downstream metadata for only category A samples (3 samples)
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            # feature_cls removed - using feature_key directly
            # Resolve the full increment first
            increment = store.resolve_update(feature_key, lazy=False)
            # Filter to only category A samples and write
            added_df = increment.added.to_polars().filter(pl.col("category") == "A")
            store.write_metadata(feature_key, added_df)

        # Check status with global-filter for category A - should be up-to-date
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--global-filter",
                "category = 'A'",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            assert feature["feature_key"] == "video/files"
            # With filter for category A, should show 3 rows and be up-to-date
            assert feature["store_rows"] == 3
            assert feature["status"] == "up_to_date"
            assert feature["needs_update"] is False
        else:
            # Check for table output with ✓ icon for up-to-date
            assert "video/files" in result.stdout
            assert "✓" in result.stdout  # Up-to-date icon
            assert "3" in result.stdout  # Materialized count

        # Check status with global-filter for category B - should need updates
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--global-filter",
                "category = 'B'",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            # With filter for category B, should show 0 rows (no B samples written)
            # and 2 samples need to be added
            assert feature["store_rows"] == 0
            assert feature["missing"] == 2
            assert feature["needs_update"] is True
        else:
            # Check for table output with ⚠ icon (needs update since missing=2)
            # Note: metadata exists, just 0 rows match the filter
            assert "video/files" in result.stdout
            assert "⚠" in result.stdout  # Needs update icon
            assert "2" in result.stdout  # Missing count


def test_metadata_status_with_invalid_filter(metaxy_project: TempMetaxyProject):
    """Test status command with invalid --filter syntax."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Try to check status with invalid filter syntax
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--filter",
                "invalid syntax !!!",
                "--format",
                "json",
            ],
            check=False,
        )

        assert result.returncode == 1
        # Error is handled by cyclopts and output to stderr
        assert "Invalid filter syntax" in result.stderr


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_with_multiple_global_filters(metaxy_project: TempMetaxyProject, output_format: str):
    """Test status command with multiple --global-filter flags (combined with AND)."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write upstream metadata with 'category' and 'status' columns
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "value": ["val_1", "val_2", "val_3", "val_4", "val_5"],
                "category": ["A", "A", "B", "B", "A"],
                "status": ["active", "inactive", "active", "inactive", "active"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 6)],
            }
        )

        feature_key_root = FeatureKey(["video", "files_root"])

        with graph.use(), store:
            store.write_metadata(feature_key_root, upstream_data)

        # Write downstream metadata for all samples
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            # feature_cls removed - using feature_key directly
            increment = store.resolve_update(feature_key, lazy=False)
            store.write_metadata(feature_key, increment.added.to_polars())

        # Check status with multiple global-filters: category A AND status active
        # Should match sample_uids 1 and 5
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--global-filter",
                "category = 'A'",
                "--global-filter",
                "status = 'active'",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            # Multiple filters are AND-ed: category A AND status active = 2 rows
            assert feature["store_rows"] == 2
            assert feature["status"] == "up_to_date"
        else:
            # Check for table output with ✓ icon for up-to-date
            assert "video/files" in result.stdout
            assert "✓" in result.stdout  # Up-to-date icon
            assert "2" in result.stdout  # Materialized count


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_root_feature_with_filter_after_overwrites(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Test status command with --filter on a root feature after writing updated rows.

    This tests the scenario where:
    1. Initial write has some rows with a column as NULL
    2. Later writes update those rows with non-NULL values
    3. Filter by IS NULL / IS NOT NULL should show the correct counts
       based on the LATEST version of each row (timestamp-based deduplication)
    """

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class RawVideoRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["raw_video", "root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        feature_key = FeatureKey(["raw_video", "root"])
        # feature_cls removed - using feature_key directly

        # Step 1: Write initial data where all rows have height=NULL
        # Cast height to Int64 to avoid DuckDB's NULL-typed column issue
        initial_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "value": ["val_1", "val_2", "val_3", "val_4", "val_5"],
                "height": pl.Series([None, None, None, None, None], dtype=pl.Int64),
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 6)],
            }
        )

        with graph.use(), store:
            store.write_metadata(feature_key, initial_data)

        # Step 2: Write updated data for samples 1, 2, 3 with height filled
        # (This simulates overwriting the same samples with updated values)
        import time

        time.sleep(0.01)  # Ensure different metaxy_created_at timestamps
        updated_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1_updated", "val_2_updated", "val_3_updated"],
                "height": [100, 200, 300],
                "metaxy_provenance_by_field": [{"default": f"hash{i}_v2"} for i in range(1, 4)],
            }
        )

        with graph.use(), store:
            store.write_metadata(feature_key, updated_data)

        # After deduplication:
        # - Samples 1, 2, 3 should have the latest version (height filled)
        # - Samples 4, 5 should still have height=NULL
        # Total: 5 unique samples

        # Check status WITHOUT filter - should show 5 rows
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "raw_video/root",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["raw_video/root"]
            assert feature["store_rows"] == 5  # Total unique samples after dedup
        else:
            assert "raw_video/root" in result.stdout
            assert "5" in result.stdout  # Materialized count

        # Check status with filter for height IS NULL - should show 2 rows (samples 4, 5)
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "raw_video/root",
                "--filter",
                "height IS NULL",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["raw_video/root"]
            # Only samples 4 and 5 should have NULL height after deduplication
            assert feature["store_rows"] == 2, (
                f"Expected 2 rows with height IS NULL, got {feature['store_rows']}. "
                "Filter should apply AFTER deduplication."
            )
        else:
            assert "raw_video/root" in result.stdout
            # Should show 2 in the Materialized column
            assert "2" in result.stdout

        # Check status with filter for height IS NOT NULL - should show 3 rows (samples 1, 2, 3)
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "raw_video/root",
                "--filter",
                "height IS NOT NULL",
                "--format",
                output_format,
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["raw_video/root"]
            # Samples 1, 2, 3 should have non-NULL height after deduplication
            assert feature["store_rows"] == 3, (
                f"Expected 3 rows with height IS NOT NULL, got {feature['store_rows']}. "
                "Filter should apply AFTER deduplication."
            )
        else:
            assert "raw_video/root" in result.stdout
            # Should show 3 in the Materialized column
            assert "3" in result.stdout


def test_metadata_status_error_representation():
    """Test that FullFeatureMetadataRepresentation handles error status correctly."""
    from metaxy.graph.status import FullFeatureMetadataRepresentation

    # Create an error representation
    error_rep = FullFeatureMetadataRepresentation(
        feature_key="audio/files",
        status="error",
        needs_update=False,
        metadata_exists=False,
        store_rows=0,
        missing=None,
        stale=None,
        orphaned=None,
        target_version="",
        is_root_feature=False,
        error_message="Simulated error: KeyError 'some_field'",
    )

    # Verify the model serializes correctly
    data = error_rep.model_dump()
    assert data["status"] == "error"
    assert data["error_message"] == "Simulated error: KeyError 'some_field'"
    assert data["feature_key"] == "audio/files"


def test_metadata_status_error_icon_exists():
    """Test that the error status icon is defined."""
    from metaxy.graph.status import _STATUS_ICONS, _STATUS_TEXTS

    # Verify error icon and text exist
    assert "error" in _STATUS_ICONS
    assert "error" in _STATUS_TEXTS
    assert _STATUS_ICONS["error"] == "[red]![/red]"
    assert _STATUS_TEXTS["error"] == "error"


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_with_progress_flag(
    metaxy_project: TempMetaxyProject,
    output_format: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test status command with --progress flag."""
    # Clear METAXY_STORE env var to ensure we use the project's config
    monkeypatch.delenv("METAXY_STORE", raising=False)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write upstream metadata (5 samples)
        metaxy_project.write_sample_metadata("video/files_root", id_values={"sample_uid": [1, 2, 3, 4, 5]})

        # Write downstream metadata for only 2 out of 5 samples (40% processed)
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            # feature_cls removed - using feature_key directly
            increment = store.resolve_update(feature_key, lazy=False)
            partial_data = increment.added.to_polars().head(2)
            store.write_metadata(feature_key, partial_data)

        # Check status with --progress flag
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                output_format,
                "--progress",
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            assert feature["feature_key"] == "video/files"
            assert feature["status"] == "needs_update"
            assert feature["progress_percentage"] is not None
            # 2 processed out of 5 = 40%
            assert abs(feature["progress_percentage"] - 40.0) < 0.1
        else:
            # Check for progress percentage in plain output
            assert "video/files" in result.stdout
            # Progress should be shown as (40%) next to status icon
            assert "40%" in result.stdout or "40.0%" in result.stdout


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_verbose_includes_progress(
    metaxy_project: TempMetaxyProject,
    output_format: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that --verbose flag also enables progress calculation."""
    # Clear METAXY_STORE env var to ensure we use the project's config
    monkeypatch.delenv("METAXY_STORE", raising=False)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write upstream metadata
        metaxy_project.write_sample_metadata("video/files_root")

        # Write downstream for only 1 out of 3 samples
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            # feature_cls removed - using feature_key directly
            increment = store.resolve_update(feature_key, lazy=False)
            partial_data = increment.added.to_polars().head(1)
            store.write_metadata(feature_key, partial_data)

        # Check status with --verbose (should also include progress)
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                output_format,
                "--verbose",
            ]
        )

        assert result.returncode == 0
        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["video/files"]
            # Verbose should include progress
            assert feature["progress_percentage"] is not None
            # 1 processed out of 3 = 33.33%
            assert abs(feature["progress_percentage"] - 33.33) < 0.1
            # Verbose should also include sample details
            assert feature["sample_details"] is not None
        else:
            # Check for progress in plain output
            assert "video/files" in result.stdout
            assert "33%" in result.stdout or "33.3%" in result.stdout


def test_metadata_status_progress_for_root_feature(metaxy_project: TempMetaxyProject, monkeypatch: pytest.MonkeyPatch):
    """Test that root features show no progress (None) since they have no upstream input."""
    # Clear METAXY_STORE env var to ensure we use the project's config
    monkeypatch.delenv("METAXY_STORE", raising=False)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["root_feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata for the root feature
        metaxy_project.write_sample_metadata("root_feature")

        # Check status with --progress flag
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "root_feature",
                "--format",
                "json",
                "--progress",
            ]
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        feature = data["features"]["root_feature"]
        assert feature["is_root_feature"] is True
        # Root features should have no progress (None excluded from JSON or null)
        assert "progress_percentage" not in feature or feature["progress_percentage"] is None


def test_metadata_status_progress_100_percent(metaxy_project: TempMetaxyProject, monkeypatch: pytest.MonkeyPatch):
    """Test that fully processed features show 100% progress."""
    # Clear METAXY_STORE env var to ensure we use the project's config
    monkeypatch.delenv("METAXY_STORE", raising=False)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write upstream metadata
        metaxy_project.write_sample_metadata("video/files_root")

        # Write downstream metadata for all samples
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            # feature_cls removed - using feature_key directly
            increment = store.resolve_update(feature_key, lazy=False)
            store.write_metadata(feature_key, increment.added.to_polars())

        # Check status with --progress flag
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                "json",
                "--progress",
            ]
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        feature = data["features"]["video/files"]
        assert feature["status"] == "up_to_date"
        assert feature["progress_percentage"] == 100.0


# ============================================================================
# Tests for metadata delete command
# ============================================================================


def test_metadata_delete_requires_yes_for_hard_delete_without_filters(
    metaxy_project: TempMetaxyProject,
):
    """Test that hard delete without filters requires --yes flag."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata
        metaxy_project.write_sample_metadata("logs")

        # Try hard delete without --yes and without filters
        result = metaxy_project.run_cli(
            ["metadata", "delete", "logs", "--soft=false"],
            check=False,
        )

        assert result.returncode == 1
        output = result.stdout + result.stderr
        assert "MISSING_CONFIRMATION" in output or "requires --yes" in output


def test_metadata_delete_soft_delete_with_filter(metaxy_project: TempMetaxyProject):
    """Test soft delete with a filter."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
            ),
        ):
            level: str | None = None

    with metaxy_project.with_features(features):
        from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write metadata with different log levels
        test_data = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "level": ["debug", "info", "warn"],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"level": "p1"},
                    {"level": "p2"},
                    {"level": "p3"},
                ],
            }
        )

        feature_key = FeatureKey(["logs"])
        # feature_cls removed - using feature_key directly

        with graph.use(), store:
            store.write_metadata(feature_key, test_data)

        # Soft delete debug logs
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs",
                "--filter",
                "level = 'debug'",
                "--soft",
            ]
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Deletion complete" in output
        assert "soft delete" in output

        # Verify data was soft deleted
        with graph.use(), store:
            remaining = store.read_metadata(feature_key).collect().to_polars()
            assert remaining.height == 2
            assert set(remaining["level"]) == {"info", "warn"}

            # Check soft deleted is still in store when including deleted
            all_data = store.read_metadata(feature_key, include_soft_deleted=True).collect().to_polars()
            assert all_data.height == 3


def test_metadata_delete_hard_delete_with_filter(metaxy_project: TempMetaxyProject):
    """Test hard delete with a filter."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
            ),
        ):
            level: str | None = None

    with metaxy_project.with_features(features):
        from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write metadata with different log levels
        test_data = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "level": ["debug", "info", "warn"],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"level": "p1"},
                    {"level": "p2"},
                    {"level": "p3"},
                ],
            }
        )

        feature_key = FeatureKey(["logs"])
        # feature_cls removed - using feature_key directly

        with graph.use(), store:
            store.write_metadata(feature_key, test_data)

        # Hard delete debug logs
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs",
                "--filter",
                "level = 'debug'",
                "--soft=false",
            ]
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Deletion complete" in output
        assert "hard delete" in output

        # Verify data was physically removed
        with graph.use(), store:
            remaining = store.read_metadata(feature_key).collect().to_polars()
            assert remaining.height == 2
            assert set(remaining["level"]) == {"info", "warn"}

            # Check that hard-deleted row is NOT in store even when including deleted
            all_data = store.read_metadata(feature_key, include_soft_deleted=True).collect().to_polars()
            assert all_data.height == 2


def test_metadata_delete_multiple_features(metaxy_project: TempMetaxyProject):
    """Test deleting metadata from multiple features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class LogsA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs_a"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class LogsB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs_b"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata to both features
        metaxy_project.write_sample_metadata("logs_a")
        metaxy_project.write_sample_metadata("logs_b")

        # Delete from both with filter
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs_a",
                "logs_b",
                "--filter",
                "sample_uid = 1",
                "--soft",
            ]
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Deletion complete" in output


def test_metadata_delete_all_features(metaxy_project: TempMetaxyProject):
    """Test deleting from all features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class LogsA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs_a"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class LogsB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs_b"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata to both features
        metaxy_project.write_sample_metadata("logs_a")
        metaxy_project.write_sample_metadata("logs_b")

        # Delete from all features with filter
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "--all-features",
                "--filter",
                "sample_uid = 1",
                "--soft",
            ]
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Deletion complete" in output


def test_metadata_delete_invalid_feature(metaxy_project: TempMetaxyProject):
    """Test that delete warns about missing features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
            ),
        ):
            level: str | None = None

    with metaxy_project.with_features(features):
        # Try to delete from non-existent feature
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "nonexistent/feature",
                "--filter",
                "sample_uid = 1",
                "--soft",
            ],
            check=False,
        )

        assert result.returncode == 1
        assert "NO_FEATURES" in result.stdout or "No valid features" in result.stdout


def test_metadata_delete_with_multiple_filters(metaxy_project: TempMetaxyProject):
    """Test delete with multiple filters combined with AND."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
            ),
        ):
            level: str | None = None

    with metaxy_project.with_features(features):
        from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write metadata with log levels and status
        test_data = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3", "s4"],
                "level": ["debug", "debug", "info", "warn"],
                "status": ["old", "new", "old", "new"],
                METAXY_PROVENANCE_BY_FIELD: [{"level": f"p{i}", "status": f"p{i}"} for i in range(1, 5)],
            }
        )

        feature_key = FeatureKey(["logs"])
        # feature_cls removed - using feature_key directly

        with graph.use(), store:
            store.write_metadata(feature_key, test_data)

        # Delete debug AND old (should delete only s1)
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs",
                "--filter",
                "level = 'debug'",
                "--filter",
                "status = 'old'",
                "--soft",
            ]
        )

        assert result.returncode == 0

        # Verify only s1 was deleted
        with graph.use(), store:
            remaining = store.read_metadata(feature_key).collect().to_polars()
            assert remaining.height == 3
            assert "s1" not in remaining["sample_uid"].to_list()


def test_metadata_delete_with_invalid_filter(metaxy_project: TempMetaxyProject):
    """Test delete with invalid filter syntax."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
            ),
        ):
            level: str | None = None

    with metaxy_project.with_features(features):
        # Try delete with invalid filter
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs",
                "--filter",
                "invalid syntax !!!",
                "--soft",
            ],
            check=False,
        )

        assert result.returncode == 1
        assert "Invalid filter syntax" in result.stderr


def test_metadata_delete_all_rows_with_yes(metaxy_project: TempMetaxyProject):
    """Test deleting all rows with --yes flag."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write metadata
        metaxy_project.write_sample_metadata("logs")

        # Hard delete all rows with --yes
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs",
                "--soft=false",
                "--yes",
            ]
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Deletion complete" in output

        # Verify all rows were deleted
        feature_key = FeatureKey(["logs"])
        # feature_cls removed - using feature_key directly

        with graph.use(), store:
            remaining = store.read_metadata(feature_key).collect().to_polars()
            assert remaining.height == 0


def test_metadata_delete_soft_delete_without_filter_no_yes_required(
    metaxy_project: TempMetaxyProject,
):
    """Test that soft delete without filters does not require --yes flag."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write metadata
        metaxy_project.write_sample_metadata("logs")

        # Soft delete all rows without --yes (should succeed)
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs",
                "--soft",
            ]
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Deletion complete" in output

        # Verify rows were soft deleted
        feature_key = FeatureKey(["logs"])
        # feature_cls removed - using feature_key directly

        with graph.use(), store:
            remaining = store.read_metadata(feature_key).collect().to_polars()
            assert remaining.height == 0

            # Check soft deleted rows still exist
            all_data = store.read_metadata(feature_key, include_soft_deleted=True).collect().to_polars()
            assert all_data.height == 3


def test_metadata_delete_requires_feature_or_all(metaxy_project: TempMetaxyProject):
    """Test that delete requires either feature arguments or --all-features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
            ),
        ):
            level: str | None = None

    with metaxy_project.with_features(features):
        # Try to delete without specifying features
        result = metaxy_project.run_cli(
            ["metadata", "delete", "--filter", "sample_uid = 1", "--soft"],
            check=False,
        )

        assert result.returncode == 1


def test_metadata_delete_dry_run(metaxy_project: TempMetaxyProject):
    """Test that --dry-run prints features and filters without deleting."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Logs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["logs"]),
                fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
            ),
        ):
            level: str | None = None

    with metaxy_project.with_features(features):
        from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        # Write some metadata
        test_data = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "level": ["debug", "info"],
                METAXY_PROVENANCE_BY_FIELD: [{"level": "p1"}, {"level": "p2"}],
            }
        )

        feature_key = FeatureKey(["logs"])
        # feature_cls removed - using feature_key directly

        with graph.use(), store:
            store.write_metadata(feature_key, test_data)

        # Run dry-run
        result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "logs",
                "--filter",
                "level = 'debug'",
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Dry run" in output
        assert "logs" in output
        assert "level" in output  # Filter should be printed
        assert "1 rows" in output  # Should show count of rows matching filter
        assert "Total rows to delete" in output  # Should show total count
        assert "Deletion complete" not in output  # Should not actually delete

        # Verify data was NOT deleted
        with graph.use(), store:
            remaining = store.read_metadata(feature_key).collect().to_polars()
            assert remaining.height == 2  # All rows still present


@pytest.mark.parametrize(
    "soft,current_only",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_metadata_delete_dry_run_count_matches_actual_deletion(
    metaxy_project: TempMetaxyProject,
    soft: bool,
    current_only: bool,
):
    """Test that --dry-run row counts match the actual number of rows deleted."""

    def features_v1():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Events(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["events"]),
                fields=[FieldSpec(key=FieldKey(["status"]), code_version="1")],
            ),
        ):
            status: str | None = None

    def features_v2():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class Events(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["events"]),
                fields=[FieldSpec(key=FieldKey(["status"]), code_version="2")],  # bumped version
            ),
        ):
            status: str | None = None

    # Write v1 metadata
    with metaxy_project.with_features(features_v1):
        from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
        from metaxy.models.types import FeatureKey

        graph_v1 = metaxy_project.graph
        store = metaxy_project.stores["dev"]
        feature_key = FeatureKey(["events"])

        # Write 3 rows with v1 feature version
        v1_data = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "status": ["active", "inactive", "active"],
                METAXY_PROVENANCE_BY_FIELD: [{"status": "p1"}, {"status": "p2"}, {"status": "p3"}],
            }
        )
        with graph_v1.use(), store:
            store.write_metadata(feature_key, v1_data)

    # Write v2 metadata (different feature version)
    with metaxy_project.with_features(features_v2):
        from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
        from metaxy.models.types import FeatureKey

        graph_v2 = metaxy_project.graph
        store = metaxy_project.stores["dev"]
        feature_key = FeatureKey(["events"])

        # Write 2 rows with v2 feature version
        v2_data = pl.DataFrame(
            {
                "sample_uid": ["s4", "s5"],
                "status": ["inactive", "inactive"],
                METAXY_PROVENANCE_BY_FIELD: [{"status": "p4"}, {"status": "p5"}],
            }
        )
        with graph_v2.use(), store:
            store.write_metadata(feature_key, v2_data)

        # Now we have:
        # - 3 rows with v1 feature version (2 active, 1 inactive)
        # - 2 rows with v2 feature version (0 active, 2 inactive)
        # Filter: status = 'inactive' should match:
        #   - current_only=True: 2 rows (v2 only)
        #   - current_only=False: 3 rows (1 from v1 + 2 from v2)

        # Run dry-run to get predicted count
        delete_mode = "--soft" if soft else "--hard"
        current_only_flag = "--current-only" if current_only else "--no-current-only"

        dry_run_result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "events",
                "--filter",
                "status = 'inactive'",
                delete_mode,
                current_only_flag,
                "--dry-run",
            ]
        )
        assert dry_run_result.returncode == 0

        # Extract the row count from dry-run output
        output = dry_run_result.stdout + dry_run_result.stderr
        # Output format: "events (N rows)"
        import re

        match = re.search(r"events\s+\((\d+)\s+rows\)", output)
        assert match, f"Could not find row count in output: {output}"
        dry_run_count = int(match.group(1))

        # Count rows before deletion
        with graph_v2.use(), store:
            before_count = (
                store.read_metadata(
                    feature_key,
                    include_soft_deleted=False,
                    current_only=False,  # count all versions
                )
                .collect()
                .to_polars()
                .height
            )

        # Actually delete
        delete_result = metaxy_project.run_cli(
            [
                "metadata",
                "delete",
                "events",
                "--filter",
                "status = 'inactive'",
                delete_mode,
                current_only_flag,
                "--yes",  # needed for hard delete
            ]
        )
        assert delete_result.returncode == 0

        # Count rows after deletion
        with graph_v2.use(), store:
            after_count = (
                store.read_metadata(
                    feature_key,
                    include_soft_deleted=False,
                    current_only=False,  # count all versions
                )
                .collect()
                .to_polars()
                .height
            )

        # Verify the dry-run count matches actual deletion
        actual_deleted = before_count - after_count
        assert dry_run_count == actual_deleted, (
            f"Dry-run predicted {dry_run_count} rows but actually deleted {actual_deleted}. "
            f"soft={soft}, current_only={current_only}, before={before_count}, after={after_count}"
        )

        # Verify expected counts based on parameters
        if current_only:
            # Only v2 rows affected (2 inactive)
            assert dry_run_count == 2, f"Expected 2 rows for current_only=True, got {dry_run_count}"
        else:
            # Both v1 and v2 rows affected (1 inactive from v1 + 2 inactive from v2)
            assert dry_run_count == 3, f"Expected 3 rows for current_only=False, got {dry_run_count}"


# ============================================================================


def test_metadata_status_progress_no_input_display(metaxy_project: TempMetaxyProject, monkeypatch: pytest.MonkeyPatch):
    """Test that non-root features with no upstream input show '(no input)' in plain format."""
    # Clear METAXY_STORE env var to ensure we use the project's config
    monkeypatch.delenv("METAXY_STORE", raising=False)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFilesRoot(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files_root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[VideoFilesRoot],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write upstream metadata with sample_uids
        metaxy_project.write_sample_metadata("video/files_root", id_values={"sample_uid": [1, 2, 3]})

        # Check status with --progress flag and a global-filter that excludes all rows
        # This simulates the "no input" scenario when all upstream data is filtered out
        result = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                "plain",
                "--progress",
                "--global-filter",
                "sample_uid > 999",  # No samples match this filter
            ]
        )

        assert result.returncode == 0
        # For non-root features with no input after filtering, should show "(no input)"
        assert "no input" in result.stdout

        # Check JSON format - should have null progress_percentage
        result_json = metaxy_project.run_cli(
            [
                "metadata",
                "status",
                "video/files",
                "--format",
                "json",
                "--progress",
                "--global-filter",
                "sample_uid > 999",  # No samples match this filter
            ]
        )

        assert result_json.returncode == 0
        data = json.loads(result_json.stdout)
        feature = data["features"]["video/files"]
        # Non-root feature with no upstream input should have null progress
        # (key may be absent due to exclude_none or present with None value)
        assert feature.get("progress_percentage") is None


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_with_fallback_stores(
    tmp_path: Path,
    output_format: str,
):
    """Test that metadata status correctly uses fallback stores for upstream metadata.

    This test reproduces a scenario where:
    1. Upstream feature metadata is only in the fallback store (e.g., production)
    2. Downstream feature was materialized by reading upstream from fallback
    3. Status should show downstream is up-to-date, NOT show inflated orphaned count

    This was a bug where `--allow-fallback-stores` defaulted to False in CLI,
    causing status to report orphaned samples when upstream was only in fallback.

    NOTE: We use DeltaMetadataStore for both stores since they both default to XXHASH64
    and work well together without Ibis backend conflicts.
    """
    # Create config with dev store that has prod as fallback (both Delta)
    dev_path = (tmp_path / "dev").as_posix()
    prod_path = (tmp_path / "prod").as_posix()

    config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{dev_path}"
fallback_stores = ["prod"]

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.prod.config]
root_path = "{prod_path}"
'''

    project = TempMetaxyProject(tmp_path, config_content=config_content)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                deps=[UpstreamFeature],
            ),
        ):
            pass

    with project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = project.graph
        dev_store = project.stores["dev"]
        prod_store = project.stores["prod"]

        # Write upstream metadata ONLY to prod (fallback) store
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in [1, 2, 3]],
            }
        )

        upstream_key = FeatureKey(["upstream"])

        with graph.use(), prod_store:
            prod_store.write_metadata(upstream_key, upstream_data)

        # Now compute downstream metadata using dev store (reads upstream from fallback)
        downstream_key = FeatureKey(["downstream"])

        with graph.use(), dev_store:
            # resolve_update should read upstream from fallback (prod) store
            increment = dev_store.resolve_update(downstream_key, lazy=False)
            # Write downstream to dev store
            dev_store.write_metadata(downstream_key, increment.added.to_polars())

        # Run CLI status command - should use fallback stores by default
        result = project.run_cli(["metadata", "status", "downstream", "--format", output_format])

        assert result.returncode == 0

        if output_format == "json":
            data = json.loads(result.stdout)
            feature = data["features"]["downstream"]
            assert feature["feature_key"] == "downstream"
            # Should be up-to-date with 0 orphaned (upstream is in fallback)
            assert feature["status"] == "up_to_date", (
                f"Expected status 'up_to_date' but got '{feature['status']}'. "
                f"Orphaned: {feature.get('orphaned')}, Missing: {feature.get('missing')}. "
                "This suggests fallback stores are not being used correctly."
            )
            assert feature["orphaned"] == 0, (
                f"Expected 0 orphaned but got {feature['orphaned']}. "
                "Orphaned count is inflated because upstream metadata in fallback store "
                "is not being read."
            )
            assert feature["store_rows"] == 3
        else:
            # Check for table output with ✓ icon for up-to-date status
            assert "downstream" in result.stdout
            assert "✓" in result.stdout, (
                f"Expected ✓ (up-to-date) but output was:\n{result.stdout}\n"
                "This suggests fallback stores are not being used correctly."
            )


def test_metadata_status_fallback_disabled_shows_missing_row_count(
    tmp_path: Path,
):
    """Test that disabling fallback stores affects how current feature's metadata is counted.

    When --no-allow-fallback-stores is used and the feature's metadata exists ONLY
    in the fallback store (not the primary), the status should show:
    - store_rows=0 (not reading from fallback)
    - metadata_exists=False

    This contrasts with --allow-fallback-stores where metadata would be found in fallback.

    NOTE: This test verifies the flag works for reading the CURRENT feature's metadata count.
    """
    # Create config with dev store that has prod as fallback (both Delta)
    dev_path = (tmp_path / "dev").as_posix()
    prod_path = (tmp_path / "prod").as_posix()

    config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{dev_path}"
fallback_stores = ["prod"]

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.prod.config]
root_path = "{prod_path}"
'''

    project = TempMetaxyProject(tmp_path, config_content=config_content)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["root"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            """Root feature with no dependencies."""

            pass

    with project.with_features(features):
        from metaxy.models.types import FeatureKey

        graph = project.graph
        prod_store = project.stores["prod"]

        # Write root feature metadata ONLY to prod (fallback) store
        root_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in [1, 2, 3]],
            }
        )

        root_key = FeatureKey(["root"])

        with graph.use(), prod_store:
            prod_store.write_metadata(root_key, root_data)

        # Run CLI status command WITH fallback enabled
        # Should find 3 rows in fallback store
        result_with_fallback = project.run_cli(
            [
                "metadata",
                "status",
                "root",
                "--format",
                "json",
                # --allow-fallback-stores is True by default
            ]
        )

        assert result_with_fallback.returncode == 0
        data = json.loads(result_with_fallback.stdout)
        feature = data["features"]["root"]
        assert feature["store_rows"] == 3, (
            f"With fallback enabled, expected store_rows=3 but got {feature['store_rows']}"
        )
        assert feature["metadata_exists"] is True
        # Root features show "root_feature" status
        assert feature["status"] == "root_feature"

        # Run CLI status command WITH fallback DISABLED
        # Should NOT find any rows (not looking in fallback)
        result_no_fallback = project.run_cli(
            [
                "metadata",
                "status",
                "root",
                "--format",
                "json",
                "--no-allow-fallback-stores",
            ]
        )

        assert result_no_fallback.returncode == 0
        data = json.loads(result_no_fallback.stdout)
        feature = data["features"]["root"]

        # Without fallback, metadata should not be found in primary store
        assert feature["store_rows"] == 0, (
            f"With fallback disabled, expected store_rows=0 but got {feature['store_rows']}"
        )
        assert feature["metadata_exists"] is False
        # Missing metadata takes precedence over root feature status
        assert feature["status"] == "missing"


# ============================================================================
# Tests for metadata copy command
# ============================================================================


def test_metadata_copy_requires_from_and_to(metaxy_project: TempMetaxyProject):
    """Test that copy requires both --from and --to flags."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Missing --from
        result = metaxy_project.run_cli(
            ["metadata", "copy", "video/files", "--to", "dev"],
            check=False,
        )
        assert result.returncode != 0

        # Missing --to
        result = metaxy_project.run_cli(
            ["metadata", "copy", "video/files", "--from", "dev"],
            check=False,
        )
        assert result.returncode != 0


def test_metadata_copy_requires_feature(metaxy_project: TempMetaxyProject):
    """Test that copy requires either feature arguments or --all-features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(
            ["metadata", "copy", "--from", "dev", "--to", "dev"],
            check=False,
        )

        # FeatureSelector validation requires features or --all-features
        assert result.returncode == 1
        assert "Must specify either --all-features or feature arguments" in result.stdout


def test_metadata_copy_single_feature(tmp_path: Path):
    """Test copying metadata for a single feature between stores."""
    # Create config with two stores
    dev_path = (tmp_path / "dev").as_posix()
    prod_path = (tmp_path / "prod").as_posix()

    config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{dev_path}"

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.prod.config]
root_path = "{prod_path}"
'''

    project = TempMetaxyProject(tmp_path, config_content=config_content)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with project.with_features(features):
        # Write metadata to dev store
        project.write_sample_metadata("video/files", store_name="dev")

        # Copy from dev to prod (positional args before keyword args)
        result = project.run_cli(["metadata", "copy", "video/files", "--from", "dev", "--to", "prod"])

        assert result.returncode == 0
        assert "Copy complete" in result.stdout
        assert "1 feature(s)" in result.stdout
        assert "3 row(s)" in result.stdout


def test_metadata_copy_multiple_features(tmp_path: Path):
    """Test copying metadata for multiple features."""
    dev_path = (tmp_path / "dev").as_posix()
    prod_path = (tmp_path / "prod").as_posix()

    config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{dev_path}"

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.prod.config]
root_path = "{prod_path}"
'''

    project = TempMetaxyProject(tmp_path, config_content=config_content)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

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

    with project.with_features(features):
        # Write metadata to dev store
        project.write_sample_metadata("video/files", store_name="dev")
        project.write_sample_metadata("audio/files", store_name="dev")

        # Copy multiple features from dev to prod using positional args
        result = project.run_cli(
            [
                "metadata",
                "copy",
                "video/files",
                "audio/files",
                "--from",
                "dev",
                "--to",
                "prod",
            ]
        )

        assert result.returncode == 0
        assert "Copy complete" in result.stdout
        assert "2 feature(s)" in result.stdout
        assert "6 row(s)" in result.stdout  # 3 rows per feature


def test_metadata_copy_with_filter(tmp_path: Path):
    """Test copying metadata with filter applied."""
    dev_path = (tmp_path / "dev").as_posix()
    prod_path = (tmp_path / "prod").as_posix()

    config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{dev_path}"

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.prod.config]
root_path = "{prod_path}"
'''

    project = TempMetaxyProject(tmp_path, config_content=config_content)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with project.with_features(features):
        # Write metadata with more samples
        project.write_sample_metadata("video/files", store_name="dev", id_values={"sample_uid": [1, 2, 3, 4, 5]})

        # Copy with filter - only sample_uid <= 2
        result = project.run_cli(
            [
                "metadata",
                "copy",
                "video/files",
                "--from",
                "dev",
                "--to",
                "prod",
                "--filter",
                "sample_uid <= 2",
            ]
        )

        assert result.returncode == 0
        assert "Copy complete" in result.stdout
        assert "1 feature(s)" in result.stdout
        assert "2 row(s)" in result.stdout  # Only 2 rows pass the filter


def test_metadata_copy_missing_feature_warning(tmp_path: Path):
    """Test that copy warns about missing features but continues."""
    dev_path = (tmp_path / "dev").as_posix()
    prod_path = (tmp_path / "prod").as_posix()

    config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{dev_path}"

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.prod.config]
root_path = "{prod_path}"
'''

    project = TempMetaxyProject(tmp_path, config_content=config_content)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with project.with_features(features):
        # Write metadata only for video/files
        project.write_sample_metadata("video/files", store_name="dev")

        # Try to copy including a non-existent feature
        result = project.run_cli(
            [
                "metadata",
                "copy",
                "video/files",
                "nonexistent/feature",
                "--from",
                "dev",
                "--to",
                "prod",
            ]
        )

        # Should succeed (copy what exists)
        assert result.returncode == 0
        assert "Copy complete" in result.stdout
        assert "1 feature(s)" in result.stdout
        # Should have warning about missing feature
        assert "Warning" in result.stdout
        assert "nonexistent/feature" in result.stdout


def test_metadata_copy_no_features_to_copy(tmp_path: Path):
    """Test that copy handles case when all specified features are missing."""
    dev_path = (tmp_path / "dev").as_posix()
    prod_path = (tmp_path / "prod").as_posix()

    config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{dev_path}"

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.prod.config]
root_path = "{prod_path}"
'''

    project = TempMetaxyProject(tmp_path, config_content=config_content)

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with project.with_features(features):
        # Try to copy a non-existent feature only
        result = project.run_cli(
            [
                "metadata",
                "copy",
                "nonexistent/feature",
                "--from",
                "dev",
                "--to",
                "prod",
            ]
        )

        assert result.returncode == 0
        assert "Warning" in result.stdout
        assert "No valid features to copy" in result.stdout
