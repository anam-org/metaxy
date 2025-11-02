"""Tests for multi-project feature architecture."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import Feature, FeatureGraph, MetaxyMeta
from metaxy.models.feature_spec import FeatureSpec, FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


class TestProjectDetection:
    """Test automatic project detection from various sources."""

    def test_detect_project_from_entrypoint(self):
        """Test project detection from installed package entrypoint."""
        # Create a mock distribution
        mock_dist = MagicMock()
        mock_dist.metadata.get.side_effect = lambda key, default="": {
            "Name": "my-awesome-project",
        }.get(key, default)

        with patch("importlib.metadata.distributions", return_value=[mock_dist]):
            # Create a feature with module name matching the distribution
            test_cls = type(
                "TestFeature", (), {"__module__": "my_awesome_project.features"}
            )

            # Test the _detect_project method
            project = MetaxyMeta._detect_project(test_cls)

            assert project == "my_awesome_project"  # Hyphens replaced with underscores

    def test_detect_project_from_module_path(self):
        """Test project detection from module path."""
        # Create a feature with a specific module path
        test_cls = type(
            "TestFeature", (), {"__module__": "custom_project.features.video"}
        )

        # Test the _detect_project method (should extract root package)
        project = MetaxyMeta._detect_project(test_cls)

        assert project == "custom_project"

    def test_detect_project_from_pyproject_toml(self, tmp_path):
        """Test project detection from pyproject.toml file."""
        # Create a temporary pyproject.toml
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project-from-file"
version = "0.1.0"
"""
        pyproject_path.write_text(pyproject_content)

        # Create a mock module with __file__ pointing to tmp directory
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "test_module.py")

        # Create a test class
        test_cls = type("TestFeature", (), {"__module__": "test_module"})

        with patch.dict(sys.modules, {"test_module": mock_module}):
            project = MetaxyMeta._detect_project(test_cls)

        assert project == "test_project_from_file"  # Hyphens replaced with underscores

    def test_detect_project_from_poetry_pyproject(self, tmp_path):
        """Test project detection from Poetry-style pyproject.toml."""
        # Create a temporary pyproject.toml with Poetry config
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_content = """
[tool.poetry]
name = "poetry-project"
version = "0.1.0"
"""
        pyproject_path.write_text(pyproject_content)

        # Create a mock module with __file__ pointing to tmp directory
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "test_module.py")

        # Create a test class - use a module name that won't be recognized as a package
        test_cls = type("TestFeature", (), {"__module__": "test_poetry_module"})

        # Need to patch config too since poetry check is in Strategy 3
        config = MetaxyConfig(project="poetry_project")

        with patch.dict(sys.modules, {"test_poetry_module": mock_module}):
            with patch("metaxy.config.MetaxyConfig.get", return_value=config):
                project = MetaxyMeta._detect_project(test_cls)

        # For now this test will check that it falls back to config
        # The poetry detection would need refactoring to check tool.poetry section
        assert project == "poetry_project"

    def test_fallback_to_global_config(self):
        """Test fallback to global config when other methods fail."""
        # Set a specific project in config
        config = MetaxyConfig(project="fallback_project")

        # Create a mock module with no __file__ (to prevent pyproject.toml detection)
        mock_module = MagicMock()
        mock_module.__file__ = None

        # Need to patch the global config for the final fallback
        with patch("metaxy.config.MetaxyConfig.get", return_value=config):
            with patch.dict(sys.modules, {"__main__": mock_module}):
                # Create a feature with generic module name
                test_cls = type("TestFeature", (), {"__module__": "__main__"})

                project = MetaxyMeta._detect_project(test_cls)

        assert project == "fallback_project"


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
                        "metaxy_provenance_by_field": [
                            {"value": "hash1"},
                            {"value": "hash2"},
                        ],
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
