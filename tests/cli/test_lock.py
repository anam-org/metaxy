"""Tests for metaxy lock command."""

from __future__ import annotations

from pathlib import Path

import tomli


class TestLockCommand:
    """Tests for metaxy lock command."""

    def test_lock_creates_lock_file(self, tmp_path: Path):
        """Test that lock command creates metaxy.lock file with feature definitions."""
        from metaxy_testing import TempMetaxyProject

        # Create two projects - one to push features, one to lock them
        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # First project: push a feature to the store
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        source_project = TempMetaxyProject(tmp_path / "source", config_content=source_config)

        def source_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class SharedFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["shared", "feature_a"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

        with source_project.with_features(source_features):
            result = source_project.run_cli(["push"])
            assert result.returncode == 0

        # Second project: configure features list and run lock
        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true
features = ["shared/feature_a"]

[stores.shared]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        # Run lock command
        result = consumer_project.run_cli(["lock"])
        assert result.returncode == 0
        assert "Locked 1 feature(s)" in result.stderr

        # Verify lock file was created
        lock_file = consumer_project.project_dir / "metaxy.lock"
        assert lock_file.exists()

        # Parse and verify lock file content
        lock_content = tomli.loads(lock_file.read_text())

        # Verify features
        assert "features" in lock_content
        assert "shared/feature_a" in lock_content["features"]

        feature_data = lock_content["features"]["shared/feature_a"]
        assert feature_data["project"] == "source"
        assert "spec" in feature_data
        assert "feature_schema" in feature_data

    def test_lock_errors_on_missing_feature(self, tmp_path: Path):
        """Test that lock command errors when configured feature is not in store."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "store.duckdb").as_posix()

        # First, create a store by pushing a feature
        source_config = f'''project = "source"
store = "dev"
auto_create_tables = true

[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{store_path}"
'''
        source_project = TempMetaxyProject(tmp_path / "source", config_content=source_config)

        def source_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class ExistingFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["existing", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

        with source_project.with_features(source_features):
            source_project.run_cli(["push"])

        # Now try to lock a non-existent feature
        consumer_config = f'''project = "consumer"
store = "dev"
auto_create_tables = true
features = ["nonexistent/feature"]

[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{store_path}"
'''
        project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        result = project.run_cli(["lock"], check=False)
        assert result.returncode != 0
        assert "not found in metadata store" in result.stderr

    def test_lock_creates_empty_lock_file_when_no_features_configured(self, tmp_path: Path):
        """Test that lock command creates empty lock file when no features are configured."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "store.duckdb").as_posix()

        # Config without features list
        config = f'''project = "test"
store = "dev"
auto_create_tables = true

[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{store_path}"
'''
        project = TempMetaxyProject(tmp_path / "project", config_content=config)

        result = project.run_cli(["lock"], check=False)
        assert result.returncode == 0
        assert "Created empty lock file" in result.stderr

        # Verify lock file was created with empty features
        lock_file = project.project_dir / "metaxy.lock"
        assert lock_file.exists()
        lock_content = tomli.loads(lock_file.read_text())
        assert lock_content["features"] == {}

    def test_lock_multiple_features(self, tmp_path: Path):
        """Test that lock command handles multiple features."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # First project: push multiple features
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        source_project = TempMetaxyProject(tmp_path / "source", config_content=source_config)

        def source_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class FeatureA(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["multi", "feature_a"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            class FeatureB(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["multi", "feature_b"]),
                    fields=[FieldSpec(key=FieldKey(["data"]), code_version="2")],
                ),
            ):
                pass

        with source_project.with_features(source_features):
            result = source_project.run_cli(["push"])
            assert result.returncode == 0

        # Consumer project: lock both features
        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true
features = ["multi/feature_a", "multi/feature_b"]

[stores.shared]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        result = consumer_project.run_cli(["lock"])
        assert result.returncode == 0
        assert "Locked 2 feature(s)" in result.stderr

        # Verify lock file content
        lock_file = consumer_project.project_dir / "metaxy.lock"
        lock_content = tomli.loads(lock_file.read_text())

        assert "multi/feature_a" in lock_content["features"]
        assert "multi/feature_b" in lock_content["features"]

    def test_lock_custom_output_path(self, tmp_path: Path):
        """Test that lock command respects custom output path."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # Push a feature
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        source_project = TempMetaxyProject(tmp_path / "source", config_content=source_config)

        def source_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class TestFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

        with source_project.with_features(source_features):
            source_project.run_cli(["push"])

        # Consumer with custom output path
        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true
features = ["test/feature"]

[stores.shared]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)
        custom_output = tmp_path / "custom.lock"

        result = consumer_project.run_cli(["lock", "--output", str(custom_output)])
        assert result.returncode == 0
        assert custom_output.exists()
        assert not (consumer_project.project_dir / "metaxy.lock").exists()
