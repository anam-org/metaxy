"""Tests for metadata CLI commands."""

import polars as pl

from metaxy._testing import TempMetaxyProject


def _write_sample_metadata(metaxy_project: TempMetaxyProject, feature_key_str: str):
    """Helper to write sample metadata for a feature."""
    from metaxy.models.types import FeatureKey

    # Parse feature key
    feature_key = FeatureKey(feature_key_str.split("__"))

    # Get feature class from graph
    feature_cls = metaxy_project.graph.features_by_key[feature_key]

    # Create sample data with data_version column
    sample_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
            "data_version": [
                {"default": "hash1"},
                {"default": "hash2"},
                {"default": "hash3"},
            ],
        }
    )

    # Write metadata directly to store
    store = metaxy_project.stores["dev"]
    with store:
        store.write_metadata(feature_cls, sample_data)


def test_metadata_drop_requires_feature_or_all(metaxy_project: TempMetaxyProject):
    """Test that drop requires either --feature or --all-features."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write actual metadata for the feature
        _write_sample_metadata(metaxy_project, "video__files")

        # Try to drop without specifying features
        result = metaxy_project.run_cli("metadata", "drop", "--confirm", check=False)

        assert result.returncode == 1
        assert "Must specify either --all-features or --feature" in result.stdout


def test_metadata_drop_requires_confirm(metaxy_project: TempMetaxyProject):
    """Test that drop requires --confirm flag."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write actual metadata for the feature
        _write_sample_metadata(metaxy_project, "video__files")

        # Try to drop without --confirm
        result = metaxy_project.run_cli(
            "metadata", "drop", "--feature", "video__files", check=False
        )

        assert result.returncode == 1
        assert "Must specify --confirm flag" in result.stdout


def test_metadata_drop_single_feature(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for a single feature."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write actual metadata for both features
        _write_sample_metadata(metaxy_project, "video__files")
        _write_sample_metadata(metaxy_project, "audio__files")

        # Drop one feature
        result = metaxy_project.run_cli(
            "metadata", "drop", "--feature", "video__files", "--confirm"
        )

        assert result.returncode == 0
        assert "Dropped: video__files" in result.stdout
        assert "Drop complete: 1 feature(s) dropped" in result.stdout


def test_metadata_drop_multiple_features(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for multiple features."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class TextFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["text", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write actual metadata for all features
        _write_sample_metadata(metaxy_project, "video__files")
        _write_sample_metadata(metaxy_project, "audio__files")
        _write_sample_metadata(metaxy_project, "text__files")

        # Drop multiple features
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--feature",
            "video__files",
            "--feature",
            "audio__files",
            "--confirm",
        )

        assert result.returncode == 0
        assert "Dropped: video__files" in result.stdout
        assert "Dropped: audio__files" in result.stdout
        assert "Drop complete: 2 feature(s) dropped" in result.stdout


def test_metadata_drop_all_features(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for all features."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["audio", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write actual metadata for both features
        _write_sample_metadata(metaxy_project, "video__files")
        _write_sample_metadata(metaxy_project, "audio__files")

        # Drop all features
        result = metaxy_project.run_cli(
            "metadata", "drop", "--all-features", "--confirm"
        )

        assert result.returncode == 0
        assert "Dropping metadata for 2 feature(s)" in result.stdout
        assert (
            "Dropped: video__files" in result.stdout
            or "Dropped: audio__files" in result.stdout
        )
        assert "Drop complete: 2 feature(s) dropped" in result.stdout


def test_metadata_drop_empty_store(metaxy_project: TempMetaxyProject):
    """Test dropping from an empty store is a no-op."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Don't write any metadata - store is empty

        # Drop all features from empty store - should be a no-op
        result = metaxy_project.run_cli(
            "metadata", "drop", "--all-features", "--confirm"
        )

        # Should succeed with 0 features dropped
        assert result.returncode == 0
        assert "Dropping metadata for 0 feature(s)" in result.stdout
        assert "Drop complete: 0 feature(s) dropped" in result.stdout


def test_metadata_drop_cannot_specify_both_flags(metaxy_project: TempMetaxyProject):
    """Test that cannot specify both --feature and --all-features."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Try to specify both flags
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--feature",
            "video__files",
            "--all-features",
            "--confirm",
            check=False,
        )

        assert result.returncode == 1
        assert "Cannot specify both --all-features and --feature" in result.stdout


def test_metadata_drop_with_store_flag(metaxy_project: TempMetaxyProject):
    """Test dropping metadata with explicit --store flag."""

    def features():
        from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec

        class VideoFiles(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["video", "files"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write actual metadata for the feature
        _write_sample_metadata(metaxy_project, "video__files")

        # Drop with explicit store
        result = metaxy_project.run_cli(
            "metadata",
            "drop",
            "--store",
            "dev",
            "--feature",
            "video__files",
            "--confirm",
        )

        assert result.returncode == 0
        assert "Dropped: video__files" in result.stdout
