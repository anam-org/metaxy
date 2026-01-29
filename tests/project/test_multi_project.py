"""Tests for multi-project feature architecture."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from metaxy_testing import SampleFeature
from metaxy_testing.models import SampleFeatureSpec

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_spec import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


class TestProjectDetection:
    """Test automatic project detection from Python package."""

    def test_detect_project_from_metaxy_project_variable(self):
        """Test project detection from __metaxy_project__ in installed package."""
        # Install the test-project fixture in a temporary venv and test it
        test_project_path = Path(__file__).parent.parent / "fixtures" / "test-project"

        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "venv"

            # Create venv using uv
            subprocess.run(["uv", "venv", str(venv_path)], check=True)

            # Get python path (Windows needs .exe extension for uv to find it)
            if sys.platform == "win32":
                python = venv_path / "Scripts" / "python.exe"
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

    def test_project_from_package_name_when_no_variable(self):
        """Test that features get project from package name when __metaxy_project__ not set."""
        # Create a test graph
        test_graph = FeatureGraph()
        with test_graph.use():
            # Create a feature - will get project from its module name
            class TestFeature(
                SampleFeature,
                spec=SampleFeatureSpec(key="test_feature", fields=[FieldSpec(key="field1")]),
            ):
                pass

            # Should get project from top-level package (tests or test_multi_project)
            expected_project = TestFeature.__module__.split(".")[0]
            assert TestFeature.metaxy_project() == expected_project

    def test_config_project_does_not_affect_feature_detection(self):
        """Test that MetaxyConfig.project doesn't affect feature project detection."""
        # Set a specific project in config
        config = MetaxyConfig(project="config_project")
        MetaxyConfig.set(config)

        try:
            # Create a test graph
            test_graph = FeatureGraph()
            with test_graph.use():
                # Create a feature - should get project from package, not config
                class TestFeature(
                    SampleFeature,
                    spec=SampleFeatureSpec(key="config_test_feature", fields=[FieldSpec(key="field1")]),
                ):
                    pass

                # Should get project from package name, not config
                expected_project = TestFeature.__module__.split(".")[0]
                assert TestFeature.metaxy_project() == expected_project
                assert TestFeature.metaxy_project() != "config_project"
        finally:
            MetaxyConfig.reset()

    def test_load_features_from_entrypoints(self):
        """Test that load_features() actually loads features from entry points."""
        from metaxy_testing import ExternalMetaxyProject

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


class TestFeatureTrackingVersion:
    """Test feature tracking version that combines spec version and project."""

    def test_full_definition_version_calculation(self):
        """Test that full_definition_version correctly combines spec version and project."""
        # Create a test graph
        test_graph = FeatureGraph()

        with test_graph.use():
            # Create a feature with a specific project
            class TestFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            # Override project for testing
            TestFeature.__metaxy_project__ = "project_a"

            # Get the tracking version
            tracking_version = TestFeature.full_definition_version()

            # Verify it's different from spec version alone
            spec_version = TestFeature.feature_spec_version()
            assert tracking_version != spec_version

            # Verify it changes when project changes
            TestFeature.__metaxy_project__ = "project_b"
            new_tracking_version = TestFeature.full_definition_version()
            assert new_tracking_version != tracking_version

            # Verify it's deterministic
            TestFeature.__metaxy_project__ = "project_a"
            assert TestFeature.full_definition_version() == tracking_version

    def test_tracking_version_in_snapshot(self):
        """Test that tracking version is included in graph snapshot."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class TestFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            # Override project
            TestFeature.__metaxy_project__ = "test_project"

            # Get snapshot
            snapshot = test_graph.to_snapshot()

            # Verify tracking version is included
            feature_data = snapshot["test/feature"]
            assert "metaxy_full_definition_version" in feature_data
            assert "project" in feature_data
            assert feature_data["project"] == "test_project"

            # Verify tracking version is computed correctly
            expected_tracking_version = TestFeature.full_definition_version()
            assert feature_data["metaxy_full_definition_version"] == expected_tracking_version


class TestMultiProjectIsolation:
    """Test that features from different projects are properly isolated."""

    def test_features_with_different_projects(self):
        """Test that features can have different projects in the same graph."""
        test_graph = FeatureGraph()

        with test_graph.use():
            # Create first feature
            class FeatureA(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature", "a"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Override project for FeatureA
            FeatureA.__metaxy_project__ = "project_a"

            # Create second feature
            class FeatureB(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature", "b"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Override project for FeatureB
            FeatureB.__metaxy_project__ = "project_b"

            # Verify they have different projects
            assert FeatureA.metaxy_project() != FeatureB.metaxy_project

            # Verify they have different tracking versions
            assert FeatureA.full_definition_version() != FeatureB.full_definition_version()

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
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            FeatureV1.__metaxy_project__ = "project_a"

        snapshot1 = graph1.to_snapshot()
        tracking_v1 = snapshot1["test/feature"]["metaxy_full_definition_version"]

        # Create second graph with same feature in project_b
        graph2 = FeatureGraph()

        with graph2.use():

            class FeatureV2(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),  # Same key
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],  # Same spec
                ),
            ):
                pass

            FeatureV2.__metaxy_project__ = "project_b"  # Different project

        snapshot2 = graph2.to_snapshot()
        tracking_v2 = snapshot2["test/feature"]["metaxy_full_definition_version"]

        # Verify tracking versions are different (would trigger migration)
        assert tracking_v1 != tracking_v2

        # Verify feature versions are the same (no computational change)
        assert (
            snapshot1["test/feature"]["metaxy_feature_version"] == snapshot2["test/feature"]["metaxy_feature_version"]
        )


class TestSystemTableRecording:
    """Test that system tables correctly record feature tracking versions."""

    def test_record_snapshot_with_tracking_version(self, tmp_path: Path):
        """Test that push_graph_snapshot includes tracking version."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class TestFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            TestFeature.__metaxy_project__ = "test_project"

            # Create a store and record snapshot (while the test graph is still active)
            with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
                storage = SystemTableStorage(store)
                result = storage.push_graph_snapshot()

                assert not result.already_pushed  # First time recording
                snapshot_version = result.snapshot_version

                # Read the recorded features
                features_df = storage.read_features(current=False, snapshot_version=snapshot_version)

                # Verify tracking version is recorded
                assert "metaxy_full_definition_version" in features_df.columns

                # Verify the recorded tracking version matches the computed one
                rows = features_df.filter(features_df["feature_key"] == "test/feature").to_dicts()
                assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {rows}"
                row = rows[0]
                assert row["metaxy_full_definition_version"] == TestFeature.full_definition_version()
                assert row["project"] == "test_project"
