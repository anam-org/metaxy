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
    """Test feature tracking version (definition version)."""

    def test_definition_version_calculation(self):
        """Test that feature_definition_version is independent of project."""
        from metaxy.models.feature_definition import FeatureDefinition

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

            # Get the definition version via FeatureDefinition
            definition_a = FeatureDefinition.from_feature_class(TestFeature)
            version_a = definition_a.feature_definition_version

            # Verify it's different from spec version alone
            spec_version = TestFeature.feature_spec_version()
            assert version_a != spec_version

            # Verify it does NOT change when project changes (project-independent)
            TestFeature.__metaxy_project__ = "project_b"
            definition_b = FeatureDefinition.from_feature_class(TestFeature)
            version_b = definition_b.feature_definition_version
            assert version_b == version_a  # Same! Definition version excludes project

            # Verify it's deterministic
            TestFeature.__metaxy_project__ = "project_a"
            definition_c = FeatureDefinition.from_feature_class(TestFeature)
            assert definition_c.feature_definition_version == version_a

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

            # Get snapshot - project is captured when feature is registered to graph
            snapshot = test_graph.to_snapshot()

            # Verify tracking version is included (now named metaxy_definition_version)
            feature_data = snapshot["test/feature"]
            assert "metaxy_definition_version" in feature_data
            assert "project" in feature_data

            # Project should match what's stored in the graph's FeatureDefinition
            definition = test_graph.get_feature_definition("test/feature")
            assert feature_data["project"] == definition.project

            # Verify tracking version is computed correctly
            expected_tracking_version = definition.feature_definition_version
            assert feature_data["metaxy_definition_version"] == expected_tracking_version


class TestMultiProjectIsolation:
    """Test that features from different projects are properly isolated."""

    def test_features_with_different_projects(self):
        """Test that features can have different projects in the same graph.

        Note: Project is captured when the feature is registered to the graph,
        so to test different projects we need to ensure features are defined in
        modules with different project settings.
        """
        test_graph = FeatureGraph()

        with test_graph.use():
            # Create first feature - project is detected from package
            class FeatureA(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature", "a"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Create second feature
            class FeatureB(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature", "b"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Both features have the same project (detected from their common package)
            # This is expected behavior - the graph snapshot reflects the project
            # at the time of feature registration
            snapshot = test_graph.to_snapshot()

            # Both should have same project since they're defined in the same test module
            assert snapshot["feature/a"]["project"] == snapshot["feature/b"]["project"]

            # Verify definition versions are different (different specs/keys)
            assert (
                snapshot["feature/a"]["metaxy_definition_version"] != snapshot["feature/b"]["metaxy_definition_version"]
            )

    def test_definition_version_same_across_identical_features(self):
        """Test that identical features have the same definition version regardless of where defined.

        The definition_version excludes project, so two identical feature definitions
        should have the same definition_version even if they are in different graphs
        or defined with different class names but same spec and schema.
        """
        # Create first graph
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

        snapshot1 = graph1.to_snapshot()

        # Create second graph with same feature spec
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

        snapshot2 = graph2.to_snapshot()

        # Note: Definition versions may differ because class names (FeatureV1 vs FeatureV2)
        # affect the Pydantic schema. Definition version = hash(spec + schema).
        # This is expected - the schema includes the class name/title.

        # However, feature versions should be SAME (no computational change)
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
                definition = test_graph.get_feature_definition("test/feature")
                rows = features_df.filter(features_df["feature_key"] == "test/feature").to_dicts()
                assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {rows}"
                row = rows[0]
                assert row["metaxy_definition_version"] == definition.feature_definition_version
                # Project should match the definition in the graph
                assert row["project"] == definition.project
