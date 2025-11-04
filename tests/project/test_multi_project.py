"""Tests for multi-project feature architecture."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec, FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


class TestProjectDetection:
    """Test automatic project detection from various sources."""

    def test_detect_project_from_entrypoints(self):
        """Test project detection from installed package with entry points."""
        # Install the test-project fixture in a temporary venv and test it
        test_project_path = Path(__file__).parent.parent / "fixtures" / "test-project"

        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "venv"

            # Create venv using uv
            subprocess.run(["uv", "venv", str(venv_path)], check=True)

            # Get python path
            if sys.platform == "win32":
                python = venv_path / "Scripts" / "python"
            else:
                python = venv_path / "bin" / "python"

            # Install metaxy and test-project using uv pip
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "-e",
                    str(Path.cwd()),
                    "--python",
                    str(python),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "-e",
                    str(test_project_path),
                    "--python",
                    str(python),
                ],
                check=True,
            )

            # Use the test script that's already in the project
            test_script = test_project_path / "entrypoint_detection.py"

            result = subprocess.run(
                [str(python), str(test_script)],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                pytest.fail(f"Test failed: {result.stderr}")

            assert "SUCCESS" in result.stdout

    def test_fallback_to_global_config(self):
        """Test fallback to global config when no entry points are found."""
        from unittest.mock import patch

        # Set a specific project in config
        config = MetaxyConfig(project="fallback_project")

        # Create a test graph with the custom config
        test_graph = FeatureGraph()
        with test_graph.use():
            # Patch MetaxyConfig.get to return our custom config
            with patch("metaxy.config.MetaxyConfig.get", return_value=config):
                # Create a feature with generic module name (no entry points for this module)
                class TestFeature(
                    Feature,
                    spec=FeatureSpec(
                        key="test_feature", fields=[FieldSpec(key="field1")]
                    ),
                ):
                    pass

                # Should fall back to config.project
                assert TestFeature.project == "fallback_project"

    def test_load_features_from_entrypoints(self):
        """Test that load_features() actually loads features from entry points."""
        from metaxy._testing import ExternalMetaxyProject

        test_project_path = Path(__file__).parent.parent / "fixtures" / "test-project"
        project = ExternalMetaxyProject(test_project_path, require_config=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup venv and install project
            project.setup_venv(Path(tmpdir) / "venv")

            # Run the test script in the venv
            test_script = test_project_path / "entrypoints_group_discovery.py"

            result = project.run_in_venv(
                "python",
                str(test_script),
                check=False,
            )

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                pytest.fail(f"Test script failed with return code {result.returncode}")

    def test_multiple_entrypoints_same_package_raises_error(self):
        """Test that multiple entry points from the same package raises an error."""
        from unittest.mock import MagicMock, patch

        from metaxy._packaging import get_all_project_entrypoints

        # Clear the cache first
        get_all_project_entrypoints.cache_clear()

        # Create mock entry points - two from the same package
        mock_ep1 = MagicMock()
        mock_ep1.name = "project-one"
        mock_ep1.value = "my_package.features"

        mock_ep2 = MagicMock()
        mock_ep2.name = "project-two"
        mock_ep2.value = "my_package.other"

        with patch("metaxy._packaging.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep1, mock_ep2]

            with pytest.raises(
                ValueError,
                match=r"Found multiple entries in `metaxy\.project` entrypoints group: 'project-one', 'project-two'\. "
                r"The key should be the Metaxy project name, thus only one entry is allowed\.",
            ):
                get_all_project_entrypoints()

        # Clear cache after test
        get_all_project_entrypoints.cache_clear()


class TestFeatureTrackingVersion:
    """Test feature tracking version that combines spec version and project."""

    def test_feature_tracking_version_calculation(self):
        """Test that feature_tracking_version correctly combines spec version and project."""
        # Create a test graph
        test_graph = FeatureGraph()

        with test_graph.use():
            # Create a feature with a specific project
            class TestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            # Override project for testing
            TestFeature.project = "project_a"  # type: ignore[attr-defined]

            # Get the tracking version
            tracking_version = TestFeature.feature_tracking_version()

            # Verify it's different from spec version alone
            spec_version = TestFeature.feature_spec_version()
            assert tracking_version != spec_version

            # Verify it changes when project changes
            TestFeature.project = "project_b"  # type: ignore[attr-defined]
            new_tracking_version = TestFeature.feature_tracking_version()
            assert new_tracking_version != tracking_version

            # Verify it's deterministic
            TestFeature.project = "project_a"  # type: ignore[attr-defined]
            assert TestFeature.feature_tracking_version() == tracking_version

    def test_tracking_version_in_snapshot(self):
        """Test that tracking version is included in graph snapshot."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class TestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            # Override project
            TestFeature.project = "test_project"  # type: ignore[attr-defined]

            # Get snapshot
            snapshot = test_graph.to_snapshot()

            # Verify tracking version is included
            feature_data = snapshot["test/feature"]
            assert "feature_tracking_version" in feature_data
            assert "project" in feature_data
            assert feature_data["project"] == "test_project"

            # Verify tracking version is computed correctly
            expected_tracking_version = TestFeature.feature_tracking_version()
            assert feature_data["feature_tracking_version"] == expected_tracking_version


class TestMultiProjectIsolation:
    """Test that features from different projects are properly isolated."""

    def test_features_with_different_projects(self):
        """Test that features can have different projects in the same graph."""
        test_graph = FeatureGraph()

        with test_graph.use():
            # Create first feature
            class FeatureA(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["feature", "a"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Override project for FeatureA
            FeatureA.project = "project_a"  # type: ignore[attr-defined]

            # Create second feature
            class FeatureB(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["feature", "b"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Override project for FeatureB
            FeatureB.project = "project_b"  # type: ignore[attr-defined]

            # Verify they have different projects
            assert FeatureA.project != FeatureB.project  # type: ignore[attr-defined]

            # Verify they have different tracking versions
            assert (
                FeatureA.feature_tracking_version()
                != FeatureB.feature_tracking_version()
            )

            # Verify snapshot includes both with correct projects
            snapshot = test_graph.to_snapshot()
            assert snapshot["feature/a"]["project"] == "project_a"
            assert snapshot["feature/b"]["project"] == "project_b"

    def test_migration_detection_across_projects(self):
        """Test that migrations are triggered when features move between projects."""
        # Create first graph with feature in project_a
        graph1 = FeatureGraph()

        with graph1.use():

            class FeatureV1(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            FeatureV1.project = "project_a"  # type: ignore[attr-defined]

        snapshot1 = graph1.to_snapshot()
        tracking_v1 = snapshot1["test/feature"]["feature_tracking_version"]

        # Create second graph with same feature in project_b
        graph2 = FeatureGraph()

        with graph2.use():

            class FeatureV2(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "feature"]),  # Same key
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1")
                    ],  # Same spec
                ),
            ):
                pass

            FeatureV2.project = "project_b"  # type: ignore[attr-defined]  # Different project

        snapshot2 = graph2.to_snapshot()
        tracking_v2 = snapshot2["test/feature"]["feature_tracking_version"]

        # Verify tracking versions are different (would trigger migration)
        assert tracking_v1 != tracking_v2

        # Verify feature versions are the same (no computational change)
        assert (
            snapshot1["test/feature"]["feature_version"]
            == snapshot2["test/feature"]["feature_version"]
        )


class TestSystemTableRecording:
    """Test that system tables correctly record feature tracking versions."""

    def test_record_snapshot_with_tracking_version(self):
        """Test that record_feature_graph_snapshot includes tracking version."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class TestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            TestFeature.project = "test_project"  # type: ignore[attr-defined]

            # Create a store and record snapshot (while the test graph is still active)
            with InMemoryMetadataStore() as store:
                result = store.record_feature_graph_snapshot()

                assert not result.already_recorded  # First time recording
                snapshot_version = result.snapshot_version

                # Read the recorded features
                features_df = store.read_features(
                    current=False, snapshot_version=snapshot_version
                )

                # Verify tracking version is recorded
                assert "feature_tracking_version" in features_df.columns

                # Verify the recorded tracking version matches the computed one
                rows = features_df.filter(
                    features_df["feature_key"] == "test/feature"
                ).to_dicts()
                assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {rows}"
                row = rows[0]
                assert (
                    row["feature_tracking_version"]
                    == TestFeature.feature_tracking_version()
                )
                assert row["project"] == "test_project"


class TestBackwardCompatibility:
    """Test backward compatibility with existing snapshots without tracking version."""

    def test_snapshot_without_tracking_version(self):
        """Test that snapshots without tracking version still work."""
        import json

        import polars as pl

        # Create a store with an old-style snapshot (no tracking version)
        with InMemoryMetadataStore() as store:
            # Manually insert old-style feature version record
            from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

            old_record = pl.DataFrame(
                {
                    "project": ["old_project"],
                    "feature_key": ["test/feature"],
                    "feature_version": ["abc123"],
                    "feature_spec_version": ["def456"],
                    "recorded_at": [pl.datetime(2024, 1, 1)],
                    "feature_spec": [
                        json.dumps(
                            {
                                "key": ["test", "feature"],
                                "fields": [{"key": ["value"], "code_version": 1}],
                            }
                        )
                    ],
                    "feature_class_path": ["test.TestFeature"],
                    "snapshot_version": ["snap123"],
                    # Note: no feature_tracking_version column in old data
                }
            )

            # Write with only the old columns
            store._write_metadata_impl(FEATURE_VERSIONS_KEY, old_record)

            # Try to read features - should handle missing tracking version
            features_df = store.read_features(current=False, snapshot_version="snap123")

            # Should still work even without tracking version
            assert features_df.height == 1
            assert features_df["feature_key"][0] == "test/feature"


class TestProjectValidation:
    """Test project validation in store operations."""

    def test_write_metadata_validates_project(self):
        """Test that write_metadata validates project matches config."""
        import polars as pl

        # Set up config with specific project
        config = MetaxyConfig(project="expected_project")

        test_graph = FeatureGraph()

        with test_graph.use():

            class TestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            # Override to different project
            TestFeature.project = "different_project"  # type: ignore[attr-defined]

            with InMemoryMetadataStore() as store:
                # Create some test data
                test_df = pl.DataFrame(
                    {
                        "sample_uid": [1, 2],
                        "provenance_by_field": [{"value": "hash1"}, {"value": "hash2"}],
                    }
                )

                # Should raise error when writing to feature with different project
                with patch("metaxy.config.MetaxyConfig.get", return_value=config):
                    with pytest.raises(
                        ValueError, match="Cannot write to feature .* from project"
                    ):
                        store.write_metadata(TestFeature, test_df)

                # Should work with allow_cross_project_writes context
                with patch("metaxy.config.MetaxyConfig.get", return_value=config):
                    with store.allow_cross_project_writes():
                        store.write_metadata(TestFeature, test_df)  # Should not raise
