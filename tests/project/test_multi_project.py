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

    def test_definition_version_calculation(self):
        """Test that feature_definition_version is based on spec + schema (not project)."""
        test_graph = FeatureGraph()

        with test_graph.use():
            # Create a feature
            class TestFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                    # Root feature
                ),
            ):
                pass

            # Get the definition from the graph
            definition = test_graph.get_feature_definition(FeatureKey(["test", "feature"]))
            definition_version = definition.feature_definition_version

            # Verify it's different from spec version alone (includes schema)
            spec_version = definition.spec.feature_spec_version
            assert definition_version != spec_version

            # Verify changing project does NOT affect feature_definition_version
            # (project is not part of the hash)
            original_project = TestFeature.__metaxy_project__
            TestFeature.__metaxy_project__ = "different_project"
            # Re-get the definition - the version should be the same
            definition_after = test_graph.get_feature_definition(FeatureKey(["test", "feature"]))
            assert definition_after.feature_definition_version == definition_version

            # Restore original project
            TestFeature.__metaxy_project__ = original_project

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

            # Get snapshot
            snapshot = test_graph.to_snapshot()

            # Verify tracking version is included
            feature_data = snapshot["test/feature"]
            assert "metaxy_definition_version" in feature_data
            assert "project" in feature_data
            # Project is detected from module name at class definition time
            assert feature_data["project"] == TestFeature.metaxy_project()

            # Verify tracking version is computed correctly
            # The snapshot's definition_version should match the definition's property
            definition = test_graph.get_feature_definition(FeatureKey(["test", "feature"]))
            expected_tracking_version = definition.feature_definition_version
            assert feature_data["metaxy_definition_version"] == expected_tracking_version


class TestMultiProjectIsolation:
    """Test that features from different projects are properly isolated."""

    def test_features_with_different_projects(self):
        """Test that features from same module share project, and project is stored separately."""
        test_graph = FeatureGraph()

        with test_graph.use():
            # Create two features in the same graph
            class FeatureA(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature", "a"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            class FeatureB(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature", "b"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Get definitions - both should have the same project (from module name)
            definition_a = test_graph.get_feature_definition(FeatureKey(["feature", "a"]))
            definition_b = test_graph.get_feature_definition(FeatureKey(["feature", "b"]))

            # Both features are defined in the same module, so they share the same project
            assert definition_a.project == definition_b.project

            # Project is stored in the definition and snapshot
            snapshot = test_graph.to_snapshot()
            assert "project" in snapshot["feature/a"]
            assert "project" in snapshot["feature/b"]
            assert snapshot["feature/a"]["project"] == definition_a.project
            assert snapshot["feature/b"]["project"] == definition_b.project

    def test_identical_features_have_same_definition_version(self):
        """Test that identical features in different graphs have the same definition version."""
        # Create first graph
        graph1 = FeatureGraph()

        with graph1.use():
            # Use same class name "TestFeature" in both graphs to ensure schema matches
            class TestFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

        snapshot1 = graph1.to_snapshot()
        definition_v1 = snapshot1["test/feature"]["metaxy_definition_version"]
        feature_v1 = snapshot1["test/feature"]["metaxy_feature_version"]

        # Create second graph with identical feature
        graph2 = FeatureGraph()

        with graph2.use():
            # Same class name to ensure identical schema
            class TestFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),  # Same key
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],  # Same spec
                ),
            ):
                pass

        snapshot2 = graph2.to_snapshot()
        definition_v2 = snapshot2["test/feature"]["metaxy_definition_version"]
        feature_v2 = snapshot2["test/feature"]["metaxy_feature_version"]

        # Verify definition versions are the SAME (identical specs and schemas)
        assert definition_v1 == definition_v2

        # Verify feature versions are also the same (no computational change)
        assert feature_v1 == feature_v2

        # Both have the same project (from module name)
        assert snapshot1["test/feature"]["project"] == snapshot2["test/feature"]["project"]


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

            # Note: Project is captured at class definition time from the module name.
            # Assigning __metaxy_project__ after class creation does NOT update the
            # already-stored FeatureDefinition.

            # Create a store and record snapshot (while the test graph is still active)
            with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
                storage = SystemTableStorage(store)
                result = storage.push_graph_snapshot()

                assert not result.already_pushed  # First time recording
                snapshot_version = result.snapshot_version

                # Read the recorded features
                features_df = storage.read_features(current=False, snapshot_version=snapshot_version)

                # Verify tracking version is recorded
                assert "metaxy_definition_version" in features_df.columns

                # Verify the recorded tracking version matches the computed one
                rows = features_df.filter(features_df["feature_key"] == "test/feature").to_dicts()
                assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {rows}"
                row = rows[0]
                definition = test_graph.get_feature_definition(FeatureKey(["test", "feature"]))
                assert row["metaxy_definition_version"] == definition.feature_definition_version
                # Project is captured from module name at class definition time
                assert row["project"] == definition.project
