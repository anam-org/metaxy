"""Tests for metadata CLI commands."""

import json

import polars as pl
import pytest

from metaxy._testing import TempMetaxyProject
from metaxy.metadata_store.system import SystemTableStorage


def _write_sample_metadata(
    metaxy_project: TempMetaxyProject,
    feature_key_str: str,
    store_name: str = "dev",
    sample_uids: list[int] | None = None,
):
    """Helper to write sample metadata for a feature.

    NOTE: This function must be called within a graph.use() context,
    and the same graph context must be active when reading the metadata.

    Args:
        metaxy_project: Test project
        feature_key_str: Feature key as string (e.g., "video/files")
        store_name: Name of store to write to (default: "dev")
        sample_uids: List of sample IDs to use (default: [1, 2, 3])
    """
    from metaxy.models.types import FeatureKey

    if sample_uids is None:
        sample_uids = [1, 2, 3]

    # Parse feature key
    feature_key = FeatureKey(feature_key_str.split("/"))

    # Get feature class from project's graph (imported from the features module)
    graph = metaxy_project.graph
    feature_cls = graph.get_feature_by_key(feature_key)

    # Create sample data with provenance_by_field column
    sample_data = pl.DataFrame(
        {
            "sample_uid": sample_uids,
            "value": [f"val_{i}" for i in sample_uids],
            "metaxy_provenance_by_field": [
                {"default": f"hash{i}"} for i in sample_uids
            ],
        }
    )

    # Write metadata directly to store
    store = metaxy_project.stores[store_name]
    # Use the project's graph context so the store can resolve feature plans
    with graph.use():
        with store:
            store.write_metadata(feature_cls, sample_data)
            # Record the feature graph snapshot so copy_metadata can determine snapshot_version
            SystemTableStorage(store).push_graph_snapshot()


def test_metadata_drop_requires_feature_or_all(metaxy_project: TempMetaxyProject):
    """Test that drop requires either --feature or --all-features."""

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

    with metaxy_project.with_features(features):
        # Write actual metadata for the feature
        _write_sample_metadata(metaxy_project, "video/files")

        # Try to drop without specifying features
        result = metaxy_project.run_cli(
            "metadata", "drop", "--confirm", "--format", "json", check=False
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "MISSING_REQUIRED_FLAG"
        assert "--all-features" in str(error["required_flags"])
        assert "--feature" in str(error["required_flags"])


def test_metadata_drop_requires_confirm(metaxy_project: TempMetaxyProject):
    """Test that drop requires --confirm flag."""

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

    with metaxy_project.with_features(features):
        # Write actual metadata for the feature
        _write_sample_metadata(metaxy_project, "video/files")

        # Try to drop without --confirm
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--feature",
            "video/files",
            "--format",
            "json",
            check=False,
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "MISSING_CONFIRMATION"
        assert "--confirm" in error["required_flag"]


def test_metadata_drop_single_feature(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for a single feature."""

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

    with metaxy_project.with_features(features):
        # Write actual metadata for both features
        _write_sample_metadata(metaxy_project, "video/files")
        _write_sample_metadata(metaxy_project, "audio/files")

        # Drop one feature
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--feature",
            "video/files",
            "--confirm",
            "--format",
            "json",
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["features_dropped"] == 1
        assert "video/files" in data["dropped"]


def test_metadata_drop_multiple_features(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for multiple features."""

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

    with metaxy_project.with_features(features):
        # Write actual metadata for all features
        _write_sample_metadata(metaxy_project, "video/files")
        _write_sample_metadata(metaxy_project, "audio/files")
        _write_sample_metadata(metaxy_project, "text/files")

        # Drop multiple features
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--feature",
            "video/files",
            "--feature",
            "audio/files",
            "--confirm",
            "--format",
            "json",
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["features_dropped"] == 2
        assert "video/files" in data["dropped"]
        assert "audio/files" in data["dropped"]


def test_metadata_drop_all_features(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for all features."""

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

    with metaxy_project.with_features(features):
        # Write actual metadata for both features
        _write_sample_metadata(metaxy_project, "video/files")
        _write_sample_metadata(metaxy_project, "audio/files")

        # Drop all features
        result = metaxy_project.run_cli(
            "metadata", "drop", "--all-features", "--confirm"
        )

    with metaxy_project.with_features(features):
        # Write actual metadata for both features
        _write_sample_metadata(metaxy_project, "video/files")
        _write_sample_metadata(metaxy_project, "audio/files")

        # Drop all features
        result = metaxy_project.run_cli(
            "metadata", "drop", "--all-features", "--confirm", "--format", "json"
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["features_dropped"] == 2
        assert len(data["dropped"]) == 2


def test_metadata_drop_empty_store(metaxy_project: TempMetaxyProject):
    """Test dropping from an empty store succeeds (idempotent operation)."""

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

    with metaxy_project.with_features(features):
        # Don't write any metadata - store is empty

        # Drop all features from empty store - should succeed (idempotent)
        result = metaxy_project.run_cli(
            "metadata", "drop", "--all-features", "--confirm", "--format", "json"
        )

        # Should succeed - drop is idempotent even if no metadata exists
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["features_dropped"] == 1


def test_metadata_drop_cannot_specify_both_flags(metaxy_project: TempMetaxyProject):
    """Test that cannot specify both --feature and --all-features."""

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

    with metaxy_project.with_features(features):
        # Try to specify both flags
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--feature",
            "video/files",
            "--all-features",
            "--confirm",
            "--format",
            "json",
            check=False,
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "CONFLICTING_FLAGS"


def test_metadata_drop_with_store_flag(metaxy_project: TempMetaxyProject):
    """Test dropping metadata with explicit --store flag."""

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

    with metaxy_project.with_features(features):
        # Write actual metadata for the feature
        _write_sample_metadata(metaxy_project, "video/files")

        # Drop with explicit store
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--store",
            "dev",
            "--feature",
            "video/files",
            "--confirm",
            "--format",
            "json",
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert "video/files" in data["dropped"]


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_up_to_date(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Status command when metadata is up-to-date for both formats."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "video/files_root")

        # Now compute and write downstream metadata with correct provenance
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            feature_cls = graph.get_feature_by_key(feature_key)
            increment = store.resolve_update(feature_cls, lazy=False)

            # Write the computed metadata to the store
            store.write_metadata(feature_cls, increment.added.to_polars())

        # Check status for the non-root feature
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--format",
            output_format,
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
def test_metadata_status_missing_metadata(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Status command when metadata is missing."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "video/files_root")

        # Check status - should show missing metadata
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--format",
            output_format,
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
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "video/files_root")

        # Check status with --assert-in-sync
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--assert-in-sync",
            check=False,
        )

        assert result.returncode == 1


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_multiple_features(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Status command with multiple features."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "files_root")
        _write_sample_metadata(metaxy_project, "video/files")

        # Check status for both
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--feature",
            "audio/files",
            "--format",
            output_format,
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
def test_metadata_status_invalid_feature_key(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Status command with a feature missing from the graph."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
            "metadata",
            "status",
            "--feature",
            "nonexistent/feature",
            "--format",
            output_format,
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
def test_metadata_status_with_verbose(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Status command with --verbose flag."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "video/files_root")
        # Check status with verbose
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--format",
            output_format,
            "--verbose",
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
def test_metadata_status_with_explicit_store(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Status command with explicit --store flag."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "video/files_root", store_name="dev")

        # Now compute and write downstream metadata with correct provenance
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            feature_cls = graph.get_feature_by_key(feature_key)
            increment = store.resolve_update(feature_cls, lazy=False)

            # Write the computed metadata to the store
            store.write_metadata(feature_cls, increment.added.to_polars())

        # Check status with explicit store
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--store",
            "dev",
            "--format",
            output_format,
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
    """Test that status requires either --feature or --all-features."""

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

    with metaxy_project.with_features(features):
        # Try to check status without specifying features
        result = metaxy_project.run_cli(
            "metadata", "status", "--format", "json", check=False
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "MISSING_REQUIRED_FLAG"
        assert "--all-features" in str(error["required_flags"])
        assert "--feature" in str(error["required_flags"])


def test_metadata_status_cannot_specify_both_flags(metaxy_project: TempMetaxyProject):
    """Test that cannot specify both --feature and --all-features."""

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

    with metaxy_project.with_features(features):
        # Try to specify both flags
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--all-features",
            "--format",
            "json",
            check=False,
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "CONFLICTING_FLAGS"


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_all_features(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Test status command with --all-features flag."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "files_root")

        # Check status for all features
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--all-features",
            "--format",
            output_format,
        )

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
def test_metadata_status_root_feature(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Test status command for a root feature (no upstream dependencies)."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        _write_sample_metadata(metaxy_project, "root_feature")

        # Check status for the root feature
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "root_feature",
            "--format",
            output_format,
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
def test_metadata_status_root_feature_missing_metadata(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Test status command for a root feature with no metadata."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
            "metadata",
            "status",
            "--feature",
            "root_feature",
            "--format",
            output_format,
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
def test_metadata_status_with_filter(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Test status command with --filter flag to filter metadata by column value."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
                "metaxy_provenance_by_field": [
                    {"default": f"hash{i}"} for i in range(1, 6)
                ],
            }
        )

        feature_key_root = FeatureKey(["video", "files_root"])
        feature_cls_root = graph.get_feature_by_key(feature_key_root)

        with graph.use(), store:
            store.write_metadata(feature_cls_root, upstream_data)

        # Write downstream metadata for only category A samples (3 samples)
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            feature_cls = graph.get_feature_by_key(feature_key)
            # Resolve the full increment first
            increment = store.resolve_update(feature_cls, lazy=False)
            # Filter to only category A samples and write
            added_df = increment.added.to_polars().filter(pl.col("category") == "A")
            store.write_metadata(feature_cls, added_df)

        # Check status with filter for category A - should be up-to-date
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--filter",
            "category = 'A'",
            "--format",
            output_format,
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

        # Check status with filter for category B - should need updates
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--filter",
            "category = 'B'",
            "--format",
            output_format,
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

    with metaxy_project.with_features(features):
        # Try to check status with invalid filter syntax
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--filter",
            "invalid syntax !!!",
            "--format",
            "json",
            check=False,
        )

        assert result.returncode == 1
        # Error is handled by cyclopts and output to stderr
        assert "Invalid filter syntax" in result.stderr


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_metadata_status_with_multiple_filters(
    metaxy_project: TempMetaxyProject, output_format: str
):
    """Test status command with multiple --filter flags (combined with AND)."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
                "metaxy_provenance_by_field": [
                    {"default": f"hash{i}"} for i in range(1, 6)
                ],
            }
        )

        feature_key_root = FeatureKey(["video", "files_root"])
        feature_cls_root = graph.get_feature_by_key(feature_key_root)

        with graph.use(), store:
            store.write_metadata(feature_cls_root, upstream_data)

        # Write downstream metadata for all samples
        with graph.use(), store:
            feature_key = FeatureKey(["video", "files"])
            feature_cls = graph.get_feature_by_key(feature_key)
            increment = store.resolve_update(feature_cls, lazy=False)
            store.write_metadata(feature_cls, increment.added.to_polars())

        # Check status with multiple filters: category A AND status active
        # Should match sample_uids 1 and 5
        result = metaxy_project.run_cli(
            "metadata",
            "status",
            "--feature",
            "video/files",
            "--filter",
            "category = 'A'",
            "--filter",
            "status = 'active'",
            "--format",
            output_format,
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
