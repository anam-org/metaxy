"""Tests for list CLI commands."""

import json

import pytest
from metaxy_testing import TempMetaxyProject


def test_list_features_basic(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test basic list features command with plain output."""

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
        result = metaxy_project.run_cli(["list", "features"], capsys=capsys)

        assert result.returncode == 0
        # Check for project grouping and table output
        assert "test" in result.stdout  # Project name as table title
        assert "video/files" in result.stdout
        assert "Feature" in result.stdout  # Table header
        assert "Version" in result.stdout  # Table header
        assert "Source" in result.stdout  # Table header
        assert "Total:" in result.stdout  # Summary


def test_list_features_json_format(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test list features with JSON output format."""

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
        result = metaxy_project.run_cli(["list", "features", "--format", "json"], capsys=capsys)

        assert result.returncode == 0
        data = json.loads(result.stdout)

        assert data["feature_count"] == 1
        assert len(data["features"]) == 1

        feature = data["features"][0]
        assert feature["key"] == "video/files"
        assert feature["is_root"] is True
        assert feature["field_count"] == 1
        assert len(feature["fields"]) == 1
        assert feature["fields"][0]["key"] == "default"
        assert feature["fields"][0]["code_version"] == "1"
        assert "version" in feature  # Has version hash


def test_list_features_multiple_features(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test listing multiple features."""

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

        class TextFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["text", "files"]),
                fields=[
                    FieldSpec(key=FieldKey(["content"]), code_version="1"),
                    FieldSpec(key=FieldKey(["metadata"]), code_version="1"),
                ],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "features", "--format", "json"], capsys=capsys)

        assert result.returncode == 0
        data = json.loads(result.stdout)

        assert data["feature_count"] == 3
        feature_keys = {f["key"] for f in data["features"]}
        assert feature_keys == {"video/files", "audio/files", "text/files"}

        # Check text/files has 2 fields
        text_feature = next(f for f in data["features"] if f["key"] == "text/files")
        assert text_feature["field_count"] == 2


def test_list_features_with_dependencies(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test listing features with dependencies shows dependent features correctly."""

    def root_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import (
            BaseFeature,
            FeatureDep,
            FeatureKey,
            FieldDep,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["video", "files"]),
                                fields=[FieldKey(["path"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            result = metaxy_project.run_cli(["list", "features", "--format", "json"], capsys=capsys)

            assert result.returncode == 0
            data = json.loads(result.stdout)

            assert data["feature_count"] == 2

            # Find features by key
            video_files = next(f for f in data["features"] if f["key"] == "video/files")
            video_processing = next(f for f in data["features"] if f["key"] == "video/processing")

            # video/files is root
            assert video_files["is_root"] is True

            # video/processing is dependent
            assert video_processing["is_root"] is False


def test_list_features_verbose_mode(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test verbose mode shows additional details."""

    def root_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import (
            BaseFeature,
            FeatureDep,
            FeatureKey,
            FieldDep,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["video", "files"]),
                                fields=[FieldKey(["path"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            # Test verbose with JSON format
            result = metaxy_project.run_cli(["list", "features", "--verbose", "--format", "json"], capsys=capsys)

            assert result.returncode == 0
            data = json.loads(result.stdout)

            # Find dependent feature
            video_processing = next(f for f in data["features"] if f["key"] == "video/processing")

            # Verbose mode includes deps list
            assert "deps" in video_processing, f"No deps in {video_processing}"
            assert "video/files" in video_processing["deps"]

            # Verbose mode includes field deps
            frames_field = video_processing["fields"][0]
            assert "deps" in frames_field, f"No deps in field {frames_field}"
            assert frames_field["deps"][0]["feature"] == "video/files"
            assert "path" in frames_field["deps"][0]["fields"]


def test_list_features_verbose_plain_output(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test verbose mode with plain output shows field tables."""

    def root_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["path"]), code_version="1")],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import (
            BaseFeature,
            FeatureDep,
            FeatureKey,
            FieldDep,
            FieldKey,
            FieldSpec,
        )

        class VideoProcessing(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["video", "files"]),
                                fields=[FieldKey(["path"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            result = metaxy_project.run_cli(["list", "features", "--verbose"], capsys=capsys)

            assert result.returncode == 0
            # Check for main table
            assert "Feature" in result.stdout
            assert "Dependencies" in result.stdout  # Verbose adds deps column

            # Check for field details section with feature title (key, project, import path)
            assert "video/files" in result.stdout
            assert "(test)" in result.stdout  # Project in title
            assert "VideoFiles" in result.stdout  # Import path in title
            assert "Code Version" in result.stdout
            assert "path" in result.stdout
            assert "frames" in result.stdout

            # Check for dependency display in field table
            assert "video/files.path" in result.stdout


def test_list_features_empty_project(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test listing features in empty project."""

    def features():
        # Empty features module - no features defined
        pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "features"], capsys=capsys)

        assert result.returncode == 0
        assert "No features found" in result.stdout


def test_list_features_empty_project_json(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test listing features in empty project with JSON format."""

    def features():
        # Empty features module - no features defined
        pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "features", "--format", "json"], capsys=capsys)

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["feature_count"] == 0
        assert data["features"] == []


def test_list_features_shows_root_and_dependent_icons(
    metaxy_project: TempMetaxyProject,
    capsys: pytest.CaptureFixture[str],
):
    """Test that plain output shows different icons for root vs dependent features."""

    def root_features():
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

    def dependent_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

        class VideoProcessing(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[FieldSpec(key=FieldKey(["frames"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            result = metaxy_project.run_cli(["list", "features"], capsys=capsys)

            assert result.returncode == 0
            # Check both features are listed
            assert "video/files" in result.stdout
            assert "video/processing" in result.stdout
            # Check summary shows counts
            assert "1 root" in result.stdout
            assert "1 dependent" in result.stdout


def test_list_features_short_flag(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test short flags work (-v for verbose, -f for format)."""

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
        # Test -f json
        result = metaxy_project.run_cli(["list", "features", "-f", "json"], capsys=capsys)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["feature_count"] == 1

        # Test -v
        result = metaxy_project.run_cli(["list", "features", "-v"], capsys=capsys)
        assert result.returncode == 0
        assert "Dependencies" in result.stdout  # Verbose column


def test_list_features_json_includes_version(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that JSON output includes full version hash."""

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
        # Get JSON output to see full version
        result_json = metaxy_project.run_cli(["list", "features", "--format", "json"], capsys=capsys)
        data = json.loads(result_json.stdout)
        full_version = data["features"][0]["version"]

        assert len(full_version) == 8

        # JSON should also include project and source
        assert "project" in data["features"][0]
        assert "source" in data["features"][0]


def test_list_features_verbose_auto_field_mapping(
    metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]
):
    """Test verbose mode shows auto-mapped field dependencies (no explicit deps)."""

    def root_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[
                    FieldSpec(key=FieldKey(["path"]), code_version="1"),
                    FieldSpec(key=FieldKey(["size"]), code_version="1"),
                ],
            ),
        ):
            pass

    def dependent_features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import (
            BaseFeature,
            FeatureDep,
            FeatureKey,
            FieldKey,
            FieldSpec,
        )

        # No explicit field deps - uses automatic mapping
        class VideoProcessing(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "processing"]),
                deps=[FeatureDep(feature=FeatureKey(["video", "files"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version="1",
                        # No explicit deps - automatic mapping applies
                    )
                ],
            ),
        ):
            pass

    with metaxy_project.with_features(root_features):
        with metaxy_project.with_features(dependent_features):
            result = metaxy_project.run_cli(["list", "features", "--verbose", "--format", "json"], capsys=capsys)

            assert result.returncode == 0
            data = json.loads(result.stdout)

            # Find dependent feature
            video_processing = next(f for f in data["features"] if f["key"] == "video/processing")

            # Auto-mapped field deps should be present
            frames_field = video_processing["fields"][0]
            assert "deps" in frames_field, f"No deps in field {frames_field}"
            # Auto-mapping should include all upstream fields
            assert frames_field["deps"][0]["feature"] == "video/files"
            # Should have both path and size from auto-mapping
            assert set(frames_field["deps"][0]["fields"]) == {"path", "size"}


def test_list_features_multiple_fields(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test listing a feature with multiple fields."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class MediaFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["media", "files"]),
                fields=[
                    FieldSpec(key=FieldKey(["path"]), code_version="1"),
                    FieldSpec(key=FieldKey(["size"]), code_version="1"),
                    FieldSpec(key=FieldKey(["hash"]), code_version="2"),
                ],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "features", "--format", "json"], capsys=capsys)

        assert result.returncode == 0
        data = json.loads(result.stdout)

        feature = data["features"][0]
        assert feature["field_count"] == 3

        field_keys = {f["key"] for f in feature["fields"]}
        assert field_keys == {"path", "size", "hash"}


def test_list_features_long_names_not_truncated(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that long feature keys and import paths are not truncated."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class CroppedSceneChunk720x480FaceLandmarksEyeFeatures(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["chunk", "crop", "face_landmarks", "eye_features", "extended"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "features"], capsys=capsys)

        assert result.returncode == 0
        # Full feature key should be present (not truncated with ...)
        assert "chunk/crop/face_landmarks/eye_features/extended" in result.stdout
        # Full class name should be present
        assert "CroppedSceneChunk720x480FaceLandmarksEyeFeatures" in result.stdout
        # Should not contain ellipsis truncation
        assert "…" not in result.stdout


def test_list_stores_basic(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test basic list stores command with plain output."""

    def features():
        pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "stores"], capsys=capsys)

        assert result.returncode == 0
        # Check for table output
        assert "Name" in result.stdout
        assert "Info" in result.stdout
        assert "Fallbacks" in result.stdout
        # Check for the default dev store
        assert "dev" in result.stdout
        assert "(default)" in result.stdout
        assert "Total:" in result.stdout


def test_list_stores_json_format(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test list stores with JSON output format."""

    def features():
        pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "stores", "--format", "json"], capsys=capsys)

        assert result.returncode == 0
        data = json.loads(result.stdout)

        assert "default_store" in data
        assert "store_count" in data
        assert data["store_count"] >= 1
        assert "stores" in data

        # Check that each store has required fields
        for store in data["stores"]:
            assert "name" in store
            assert "info" in store
            assert "is_default" in store
            assert "fallbacks" in store

        # Check the default store is marked correctly
        default_stores = [s for s in data["stores"] if s["is_default"]]
        assert len(default_stores) == 1
        assert default_stores[0]["name"] == data["default_store"]


def test_list_stores_short_flag(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test short flag works (-f for format)."""

    def features():
        pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "stores", "-f", "json"], capsys=capsys)

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "stores" in data


def test_list_features_shows_external_features(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that external features are marked as such in the output."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy.models.feature import FeatureDefinition, FeatureGraph

        # Define a regular feature
        class LocalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        # Add an external feature (use "test" project to match the project filter)
        graph = FeatureGraph.get_active()
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["external", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]))],
            ),
            feature_schema={"type": "object"},
            project="test",  # Must match the project to be listed
        )
        graph.add_feature_definition(external_def)

    with metaxy_project.with_features(features):
        # Test plain output
        result = metaxy_project.run_cli(["list", "features"], capsys=capsys)
        assert result.returncode == 0
        assert "local/feature" in result.stdout
        assert "external/feature" in result.stdout
        # Summary should mention external count
        assert "1 external" in result.stdout


def test_list_features_external_json_format(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that JSON output includes is_external field."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy.models.feature import FeatureDefinition, FeatureGraph

        # Define a regular feature
        class LocalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        # Add an external feature (use "test" project to match the project filter)
        graph = FeatureGraph.get_active()
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["external", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]))],
            ),
            feature_schema={"type": "object"},
            project="test",  # Must match the project to be listed
        )
        graph.add_feature_definition(external_def)

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "features", "--format", "json"], capsys=capsys)
        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Find both features
        local_feature = next(f for f in data["features"] if f["key"] == "local/feature")
        external_feature = next(f for f in data["features"] if f["key"] == "external/feature")

        # Check is_external field
        assert local_feature["is_external"] is False
        assert external_feature["is_external"] is True

        # Both should have source
        assert local_feature["source"] is not None
        assert external_feature["source"] is not None


def test_list_features_external_verbose(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test verbose mode with external features."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy.models.feature import FeatureDefinition, FeatureGraph

        # Define a regular feature
        class LocalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        # Add an external feature (use "test" project to match the project filter)
        graph = FeatureGraph.get_active()
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["external", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]))],
            ),
            feature_schema={"type": "object"},
            project="test",  # Must match the project to be listed
        )
        graph.add_feature_definition(external_def)

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["list", "features", "--verbose"], capsys=capsys)
        assert result.returncode == 0
        assert "external/feature" in result.stdout
