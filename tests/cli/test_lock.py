"""Tests for metaxy lock command."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli


class TestLockCommand:
    """Tests for metaxy lock command."""

    def test_lock_creates_lock_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that lock command creates metaxy.lock file with feature definitions."""
        from metaxy_testing import TempMetaxyProject

        # Create two projects - one to push features, one to lock them
        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # First project: push a feature to the store
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

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
            result = source_project.run_cli(["push"], capsys=capsys)
            assert result.returncode == 0

        # Second project: define a feature that depends on the shared feature
        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        def consumer_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class LocalFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["consumer", "local"]),
                    deps=[FeatureDep(feature="shared/feature_a")],
                    fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                ),
            ):
                pass

        # Run lock command
        with consumer_project.with_features(consumer_features):
            result = consumer_project.run_cli(["lock"], capsys=capsys)
        assert result.returncode == 0
        assert "Written 1 external feature(s)" in result.stderr

        # Verify lock file was created
        lock_file = consumer_project.project_dir / "metaxy.lock"
        assert lock_file.exists()

        # Parse and verify lock file content
        lock_content = tomli.loads(lock_file.read_text())

        # Verify features
        assert "features" in lock_content
        assert "shared/feature_a" in lock_content["features"]

        feature_data = lock_content["features"]["shared/feature_a"]
        # Verify info block
        assert "info" in feature_data
        assert "version" in feature_data["info"]
        assert "version_by_field" in feature_data["info"]
        assert "definition_version" in feature_data["info"]
        # Verify data block
        assert "data" in feature_data
        assert feature_data["data"]["project"] == "source"
        assert "spec" in feature_data["data"]
        assert "feature_schema" in feature_data["data"]

    def test_lock_errors_on_missing_feature(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that lock command errors when a dependency is not in store."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "store.duckdb").as_posix()

        # First, create a store by pushing a feature
        source_config = f'''project = "source"
store = "dev"
auto_create_tables = true

[stores.dev]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

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
            source_project.run_cli(["push"], capsys=capsys)

        # Consumer project depends on a non-existent feature
        consumer_config = f'''project = "consumer"
store = "dev"
auto_create_tables = true

[stores.dev]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{store_path}"
'''
        project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        def consumer_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class LocalFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["consumer", "local"]),
                    deps=[FeatureDep(feature="nonexistent/feature")],
                    fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                ),
            ):
                pass

        with project.with_features(consumer_features):
            result = project.run_cli(["lock"], check=False, capsys=capsys)
        assert result.returncode != 0
        assert "not found in metadata store" in result.stderr

    def test_lock_creates_empty_lock_file_when_no_external_deps(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Test that lock command creates empty lock file when no external dependencies."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "store.duckdb").as_posix()

        # Config without external dependencies
        config = f'''project = "test"
store = "dev"
auto_create_tables = true

[stores.dev]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{store_path}"
'''
        project = TempMetaxyProject(tmp_path / "project", config_content=config)

        def local_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class LocalFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["local", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

        with project.with_features(local_features):
            result = project.run_cli(["lock"], check=False, capsys=capsys)
        assert result.returncode == 0
        assert "Written 0 external feature(s)" in result.stderr

        # Verify lock file was created with empty features
        lock_file = project.project_dir / "metaxy.lock"
        assert lock_file.exists()
        lock_content = tomli.loads(lock_file.read_text())
        assert lock_content["features"] == {}

    def test_lock_multiple_features(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that lock command handles multiple external dependencies."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # First project: push multiple features
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

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
            result = source_project.run_cli(["push"], capsys=capsys)
            assert result.returncode == 0

        # Consumer project depends on both features
        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        def consumer_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class LocalFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["consumer", "local"]),
                    deps=[
                        FeatureDep(feature="multi/feature_a"),
                        FeatureDep(feature="multi/feature_b"),
                    ],
                    fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                ),
            ):
                pass

        with consumer_project.with_features(consumer_features):
            result = consumer_project.run_cli(["lock"], capsys=capsys)
        assert result.returncode == 0
        assert "Written 2 external feature(s)" in result.stderr

        # Verify lock file content
        lock_file = consumer_project.project_dir / "metaxy.lock"
        lock_content = tomli.loads(lock_file.read_text())

        assert "multi/feature_a" in lock_content["features"]
        assert "multi/feature_b" in lock_content["features"]

    def test_lock_transitive_dependencies(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that lock command resolves transitive dependencies."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # Source project: push features with a dependency chain (A -> B -> C)
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        source_project = TempMetaxyProject(tmp_path / "source", config_content=source_config)

        def source_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            # Base feature with no deps
            class FeatureC(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["chain", "c"]),
                    fields=[FieldSpec(key=FieldKey(["base"]), code_version="1")],
                ),
            ):
                pass

            # Feature B depends on C
            class FeatureB(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["chain", "b"]),
                    deps=[FeatureDep(feature=FeatureC)],
                    fields=[FieldSpec(key=FieldKey(["middle"]), code_version="1")],
                ),
            ):
                pass

            # Feature A depends on B
            class FeatureA(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["chain", "a"]),
                    deps=[FeatureDep(feature=FeatureB)],
                    fields=[FieldSpec(key=FieldKey(["top"]), code_version="1")],
                ),
            ):
                pass

        with source_project.with_features(source_features):
            result = source_project.run_cli(["push"], capsys=capsys)
            assert result.returncode == 0

        # Consumer depends only on A, but should get B and C transitively
        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        def consumer_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class LocalFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["consumer", "local"]),
                    deps=[FeatureDep(feature="chain/a")],
                    fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                ),
            ):
                pass

        with consumer_project.with_features(consumer_features):
            result = consumer_project.run_cli(["lock"], capsys=capsys)
        assert result.returncode == 0
        assert "Written 3 external feature(s)" in result.stderr

        # Verify all three features are in lock file
        lock_file = consumer_project.project_dir / "metaxy.lock"
        lock_content = tomli.loads(lock_file.read_text())

        assert "chain/a" in lock_content["features"]
        assert "chain/b" in lock_content["features"]
        assert "chain/c" in lock_content["features"]

    def test_lock_custom_output_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that lock command respects custom output path."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # Push a feature
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

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
            source_project.run_cli(["push"], capsys=capsys)

        # Consumer with custom output path
        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)
        custom_output = tmp_path / "custom.lock"

        def consumer_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class LocalFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["consumer", "local"]),
                    deps=[FeatureDep(feature="test/feature")],
                    fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
                ),
            ):
                pass

        with consumer_project.with_features(consumer_features):
            result = consumer_project.run_cli(["lock", "--output", str(custom_output)], capsys=capsys)
        assert result.returncode == 0
        assert custom_output.exists()
        assert not (consumer_project.project_dir / "metaxy.lock").exists()


def test_load_all_features_from_store_warns_on_invalid_feature(tmp_path: Path):
    """Test that _load_all_features_from_store warns and skips features that fail to validate."""
    import warnings

    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureGraph, FeatureKey, FieldKey, FieldSpec
    from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
    from metaxy.metadata_store.system.storage import SystemTableStorage
    from metaxy.utils.lock_file import _load_all_features_from_store

    store_path = tmp_path / "store.duckdb"

    # Push two valid features
    graph = FeatureGraph()
    with graph.use():

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "valid"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "will_corrupt"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DuckDBMetadataStore(database=store_path) as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Corrupt one feature's spec in the store
    with DuckDBMetadataStore(database=store_path).open("w") as store:
        store._duckdb_raw_connection().execute(
            "UPDATE metaxy_system__feature_versions "
            "SET feature_spec = 'not valid json' "
            "WHERE feature_key = 'test/will_corrupt'"
        )

    # Load features - should warn about the corrupted one and return the valid one
    from metaxy._warnings import InvalidStoredFeatureWarning

    with DuckDBMetadataStore(database=store_path) as store:
        storage = SystemTableStorage(store)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            definitions, db_versions = _load_all_features_from_store(storage, exclude_project=None)

    assert len(definitions) == 1
    assert FeatureKey(["test", "valid"]) in definitions
    assert FeatureKey(["test", "will_corrupt"]) not in definitions

    assert len(caught) == 1
    assert caught[0].category is InvalidStoredFeatureWarning
    assert "test/will_corrupt" in str(caught[0].message)


def test_generate_lock_file_errors_on_missing_transitive_dependency(tmp_path: Path):
    """Test that generate_lock_file errors when a transitive dependency is not in store."""
    import pytest
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureDep, FeatureGraph, FeatureKey, FieldKey, FieldSpec
    from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.metadata_store.system.storage import SystemTableStorage
    from metaxy.utils.lock_file import generate_lock_file

    store_path = tmp_path / "store.duckdb"

    # First graph: push A -> B -> C chain
    source_graph = FeatureGraph()
    with source_graph.use():

        class FeatureC(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["chain", "c"]),
                fields=[FieldSpec(key=FieldKey(["base"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["chain", "b"]),
                deps=[FeatureDep(feature=FeatureC)],
                fields=[FieldSpec(key=FieldKey(["middle"]), code_version="1")],
            ),
        ):
            pass

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["chain", "a"]),
                deps=[FeatureDep(feature=FeatureB)],
                fields=[FieldSpec(key=FieldKey(["top"]), code_version="1")],
            ),
        ):
            pass

        with DuckDBMetadataStore(database=store_path) as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Now delete chain/c from the store to simulate missing transitive dep
    with DuckDBMetadataStore(database=store_path).open("w") as store:
        store._duckdb_raw_connection().execute(
            "DELETE FROM metaxy_system__feature_versions WHERE feature_key = 'chain/c'"
        )

    # Consumer graph: local feature depends on A
    consumer_graph = FeatureGraph()
    with consumer_graph.use():

        class LocalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["consumer", "local"]),
                deps=[FeatureDep(feature="chain/a")],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        # Try to generate lock file - should fail because chain/c is missing
        store = DuckDBMetadataStore(database=store_path)
        with pytest.raises(FeatureNotFoundError, match="chain/c"):
            generate_lock_file(store, tmp_path / "metaxy.lock", exclude_project="consumer")
