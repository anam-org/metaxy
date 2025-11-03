"""Tests for metadata CLI commands."""

import polars as pl

from metaxy._testing import TempMetaxyProject


def _write_sample_metadata(
    metaxy_project: TempMetaxyProject,
    feature_key_str: str,
    store_name: str = "dev",
    sample_uids: list[int] | None = None,
):
    """Helper to write sample metadata for a feature.

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

    # Get feature class from graph
    feature_cls = metaxy_project.graph.features_by_key[feature_key]

    # Create sample data with provenance_by_field column
    sample_data = pl.DataFrame(
        {
            "sample_uid": sample_uids,
            "value": [f"val_{i}" for i in sample_uids],
            "provenance_by_field": [{"default": f"hash{i}"} for i in sample_uids],
        }
    )

    # Write metadata directly to store
    # Must activate the graph before recording snapshot since record_feature_graph_snapshot uses get_active()
    graph = metaxy_project.graph
    with graph.use():
        store = metaxy_project.stores[store_name]
        with store:
            store.write_metadata(feature_cls, sample_data)
            # Record the feature graph snapshot so copy_metadata can determine snapshot_version
            store.record_feature_graph_snapshot()


def test_metadata_drop_requires_feature_or_all(metaxy_project: TempMetaxyProject):
    """Test that drop requires either --feature or --all-features."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write actual metadata for the feature
        _write_sample_metadata(metaxy_project, "video/files")

        # Try to drop without specifying features
        result = metaxy_project.run_cli("metadata", "drop", "--confirm", check=False)

        assert result.returncode == 1
        assert "Must specify either --all-features or --feature" in result.stdout


def test_metadata_drop_requires_confirm(metaxy_project: TempMetaxyProject):
    """Test that drop requires --confirm flag."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
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
            "metadata", "drop", "--feature", "video/files", check=False
        )

        assert result.returncode == 1
        assert "Must specify --confirm flag" in result.stdout


def test_metadata_drop_single_feature(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for a single feature."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
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
            "metadata", "drop", "--feature", "video/files", "--confirm"
        )

        assert result.returncode == 0
        assert "Dropped: video/files" in result.stdout


def test_metadata_copy_incremental_skips_duplicates(metaxy_project: TempMetaxyProject):
    """Test that incremental copy skips existing sample_uids."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata to dev store with sample_uids [1, 2, 3]
        _write_sample_metadata(
            metaxy_project, "video/files", store_name="dev", sample_uids=[1, 2, 3]
        )

        # Write metadata to staging store with sample_uids [2, 3, 4]
        # sample_uids 2 and 3 overlap with dev
        _write_sample_metadata(
            metaxy_project, "video/files", store_name="staging", sample_uids=[2, 3, 4]
        )

        # Copy from dev to staging with incremental=True (default)
        result = metaxy_project.run_cli(
            "metadata",
            "copy",
            "--from",
            "dev",
            "--to",
            "staging",
            "--feature",
            "video/files",
        )

        assert result.returncode == 0
        assert "Copy complete" in result.stdout

        # Verify staging now has [1, 2, 3, 4] (no duplicates)
        # Only sample_uid 1 should have been copied (2 and 3 were skipped)
        store = metaxy_project.stores["staging"]
        with store:
            from metaxy.models.types import FeatureKey

            feature_key = FeatureKey(["video", "files"])
            metadata = store.read_metadata(
                feature_key, allow_fallback=False, current_only=False
            )
            df = metadata.collect().to_polars()

            # Should have 4 total rows (original 3 + 1 new)
            assert df.height == 4

            # Check sample_uids
            sample_uids = sorted(df["sample_uid"].to_list())
            assert sample_uids == [1, 2, 3, 4]

            # Verify no duplicate sample_uids
            assert len(sample_uids) == len(set(sample_uids))


def test_metadata_copy_non_incremental_creates_duplicates(
    metaxy_project: TempMetaxyProject,
):
    """Test that non-incremental copy allows duplicate sample_uids."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata to dev store with sample_uids [1, 2, 3]
        _write_sample_metadata(
            metaxy_project, "video/files", store_name="dev", sample_uids=[1, 2, 3]
        )

        # Write metadata to staging store with sample_uids [2, 3, 4]
        _write_sample_metadata(
            metaxy_project, "video/files", store_name="staging", sample_uids=[2, 3, 4]
        )

        # Copy from dev to staging with incremental=False (--no-incremental)
        result = metaxy_project.run_cli(
            "metadata",
            "copy",
            "--from",
            "dev",
            "--to",
            "staging",
            "--feature",
            "video/files",
            "--no-incremental",
        )

        assert result.returncode == 0
        assert "Copy complete" in result.stdout

        # Verify staging now has duplicates for sample_uids 2 and 3
        store = metaxy_project.stores["staging"]
        with store:
            from metaxy.models.types import FeatureKey

            feature_key = FeatureKey(["video", "files"])
            metadata = store.read_metadata(
                feature_key, allow_fallback=False, current_only=False
            )
            df = metadata.collect().to_polars()

            # Should have 6 total rows (original 3 + all 3 from dev)
            assert df.height == 6

            # Check that we have duplicates
            sample_uids = df["sample_uid"].to_list()
            assert sample_uids.count(2) == 2  # sample_uid 2 appears twice
            assert sample_uids.count(3) == 2  # sample_uid 3 appears twice
            assert sample_uids.count(1) == 1  # sample_uid 1 appears once
            assert sample_uids.count(4) == 1  # sample_uid 4 appears once


def test_metadata_copy_incremental_empty_destination(metaxy_project: TempMetaxyProject):
    """Test that incremental copy works correctly with empty destination."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Write metadata to dev store only
        _write_sample_metadata(
            metaxy_project, "video/files", store_name="dev", sample_uids=[1, 2, 3]
        )

        # Copy from dev to empty staging with incremental=True
        result = metaxy_project.run_cli(
            "metadata",
            "copy",
            "--from",
            "dev",
            "--to",
            "staging",
            "--feature",
            "video/files",
        )

        assert result.returncode == 0
        assert "Copy complete" in result.stdout

        # Verify staging has all 3 rows
        store = metaxy_project.stores["staging"]
        with store:
            from metaxy.models.types import FeatureKey

            feature_key = FeatureKey(["video", "files"])
            metadata = store.read_metadata(
                feature_key, allow_fallback=False, current_only=False
            )
            df = metadata.collect().to_polars()

            assert df.height == 3
            sample_uids = sorted(df["sample_uid"].to_list())
            assert sample_uids == [1, 2, 3]


def test_metadata_drop_multiple_features(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for multiple features."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["audio", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class TextFiles(
            Feature,
            spec=TestingFeatureSpec(
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
        )

        assert result.returncode == 0
        assert "Dropped: video/files" in result.stdout
        assert "Dropped: audio/files" in result.stdout
        assert "Drop complete: 2 feature(s) dropped" in result.stdout


def test_metadata_drop_all_features(metaxy_project: TempMetaxyProject):
    """Test dropping metadata for all features."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class AudioFiles(
            Feature,
            spec=TestingFeatureSpec(
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

        assert result.returncode == 0
        assert "Dropping metadata for 2 feature(s)" in result.stdout
        assert (
            "Dropped: video/files" in result.stdout
            or "Dropped: audio/files" in result.stdout
        )
        assert "Drop complete: 2 feature(s) dropped" in result.stdout


def test_metadata_drop_empty_store(metaxy_project: TempMetaxyProject):
    """Test dropping from an empty store is a no-op."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
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
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
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
            check=False,
        )

        assert result.returncode == 1
        assert "Cannot specify both --all-features and --feature" in result.stdout


def test_metadata_drop_with_store_flag(metaxy_project: TempMetaxyProject):
    """Test dropping metadata with explicit --store flag."""

    def features():
        from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec

        class VideoFiles(
            Feature,
            spec=TestingFeatureSpec(
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
        )

        assert result.returncode == 0
        assert "Dropped: video/files" in result.stdout
