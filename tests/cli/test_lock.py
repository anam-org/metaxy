"""Tests for metaxy lock command."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli

from metaxy import MetadataStore
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.utils.lock_file import LockedFeature, LockedFeatureInfo, LockFile


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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
        assert "+ shared/feature_a" in result.stderr
        assert "Updated metaxy.lock" in result.stderr

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
        assert "is up to date" in result.stderr

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
        assert "+ multi/feature_a" in result.stderr
        assert "+ multi/feature_b" in result.stderr
        assert "Updated metaxy.lock" in result.stderr

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
        assert "+ chain/a" in result.stderr
        assert "+ chain/b" in result.stderr
        assert "+ chain/c" in result.stderr
        assert "Updated metaxy.lock" in result.stderr

        # Verify all three features are in lock file
        lock_file = consumer_project.project_dir / "metaxy.lock"
        lock_content = tomli.loads(lock_file.read_text())

        assert "chain/a" in lock_content["features"]
        assert "chain/b" in lock_content["features"]
        assert "chain/c" in lock_content["features"]

    def test_lock_up_to_date_on_rerun(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that re-running lock without changes shows 'up to date'."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        source_project = TempMetaxyProject(tmp_path / "source", config_content=source_config)

        def source_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class Feature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["ext", "feat"]),
                    fields=[FieldSpec(key=FieldKey(["v"]), code_version="1")],
                ),
            ):
                pass

        with source_project.with_features(source_features):
            source_project.run_cli(["push"], capsys=capsys)

        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        def consumer_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class Local(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["consumer", "local"]),
                    deps=[FeatureDep(feature="ext/feat")],
                    fields=[FieldSpec(key=FieldKey(["r"]), code_version="1")],
                ),
            ):
                pass

        # First lock: should add
        with consumer_project.with_features(consumer_features):
            result = consumer_project.run_cli(["lock"], capsys=capsys)
        assert result.returncode == 0
        assert "+ ext/feat" in result.stderr
        assert "Updated metaxy.lock" in result.stderr

        # Second lock: should be up to date
        with consumer_project.with_features(consumer_features):
            result = consumer_project.run_cli(["lock"], capsys=capsys)
        assert result.returncode == 0
        assert "is up to date" in result.stderr
        assert "+" not in result.stderr
        assert "~" not in result.stderr

    def test_lock_shows_updated_features(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that re-running lock after a feature change shows updated features."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        source_project = TempMetaxyProject(tmp_path / "source", config_content=source_config)

        def source_features_v1():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class Feature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["ext", "feat"]),
                    fields=[FieldSpec(key=FieldKey(["v"]), code_version="1")],
                ),
            ):
                pass

        with source_project.with_features(source_features_v1):
            source_project.run_cli(["push"], capsys=capsys)

        consumer_config = f'''project = "consumer"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

[stores.shared.config]
database = "{store_path}"
'''
        consumer_project = TempMetaxyProject(tmp_path / "consumer", config_content=consumer_config)

        def consumer_features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class Local(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["consumer", "local"]),
                    deps=[FeatureDep(feature="ext/feat")],
                    fields=[FieldSpec(key=FieldKey(["r"]), code_version="1")],
                ),
            ):
                pass

        # First lock
        with consumer_project.with_features(consumer_features):
            consumer_project.run_cli(["lock"], capsys=capsys)

        # Update the source feature (bump code_version)
        def source_features_v2():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class Feature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["ext", "feat"]),
                    fields=[FieldSpec(key=FieldKey(["v"]), code_version="2")],
                ),
            ):
                pass

        with source_project.with_features(source_features_v2):
            source_project.run_cli(["push"], capsys=capsys)

        # Second lock: should show update
        with consumer_project.with_features(consumer_features):
            result = consumer_project.run_cli(["lock"], capsys=capsys)
        assert result.returncode == 0
        assert "~ ext/feat" in result.stderr
        assert "->" in result.stderr
        assert "Updated metaxy.lock" in result.stderr

    def test_lock_custom_output_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Test that lock command respects custom output path."""
        from metaxy_testing import TempMetaxyProject

        store_path = (tmp_path / "shared_store.duckdb").as_posix()

        # Push a feature
        source_config = f'''project = "source"
store = "shared"
auto_create_tables = true

[stores.shared]
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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
    from metaxy.ext.duckdb import DuckDBMetadataStore
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

        with DuckDBMetadataStore(database=store_path).open("w") as store:
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


class TestGenerateLockFileSelection:
    """Tests for the `selection` parameter on `generate_lock_file`."""

    @staticmethod
    def _push_project(store: MetadataStore, project: str, keys: list[str]) -> None:
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy.metadata_store.system.storage import SystemTableStorage
        from metaxy.models.feature import FeatureGraph

        g = FeatureGraph()
        with g.use():
            for key in keys:
                type(
                    f"_Feat_{key.replace('/', '_')}",
                    (BaseFeature,),
                    {"__metaxy_project__": project},
                    spec=SampleFeatureSpec(
                        key=FeatureKey(key),
                        fields=[FieldSpec(key=FieldKey(["v"]), code_version="1")],
                    ),
                )
            with store.open("w"):
                SystemTableStorage(store).push_graph_snapshot(project=project)

    def test_selection_by_projects(self, store: MetadataStore, tmp_path: Path):
        """Selection with projects= locks all features from those projects."""
        from metaxy.models.feature_selection import FeatureSelection
        from metaxy.utils.lock_file import generate_lock_file

        self._push_project(store, "upstream", ["up/a", "up/b"])

        lock_path = tmp_path / "metaxy.lock"
        result = generate_lock_file(store, lock_path, selection=FeatureSelection(projects=["upstream"]))

        assert result.count == 2
        content = tomli.loads(lock_path.read_text())
        assert set(content["features"].keys()) == {"up/a", "up/b"}

    def test_selection_by_keys(self, store: MetadataStore, tmp_path: Path):
        """Selection with keys= locks only the specified features."""
        from metaxy.models.feature_selection import FeatureSelection
        from metaxy.utils.lock_file import generate_lock_file

        self._push_project(store, "upstream", ["up/a", "up/b", "up/c"])

        lock_path = tmp_path / "metaxy.lock"
        result = generate_lock_file(store, lock_path, selection=FeatureSelection(keys=["up/a", "up/c"]))

        assert result.count == 2
        content = tomli.loads(lock_path.read_text())
        assert set(content["features"].keys()) == {"up/a", "up/c"}

    def test_selection_all(self, store: MetadataStore, tmp_path: Path):
        """Selection with all=True locks every feature in the store."""
        from metaxy.models.feature_selection import FeatureSelection
        from metaxy.utils.lock_file import generate_lock_file

        self._push_project(store, "proj-a", ["a/x"])
        self._push_project(store, "proj-b", ["b/y"])

        lock_path = tmp_path / "metaxy.lock"
        result = generate_lock_file(store, lock_path, selection=FeatureSelection(all=True))

        assert result.count == 2
        content = tomli.loads(lock_path.read_text())
        assert set(content["features"].keys()) == {"a/x", "b/y"}

    def test_selection_combined_with_graph_deps(self, store: MetadataStore, tmp_path: Path):
        """Selection features are unioned with graph-discovered dependencies."""
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.feature_selection import FeatureSelection
        from metaxy.utils.lock_file import generate_lock_file

        self._push_project(store, "upstream", ["up/dep", "up/extra"])

        # Local feature depends on up/dep (graph-discovered)
        g = FeatureGraph()
        with g.use():

            class Local(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey("local/feat"),
                    deps=[FeatureDep(feature="up/dep")],
                    fields=[FieldSpec(key=FieldKey(["v"]), code_version="1")],
                ),
            ):
                __metaxy_project__ = "local"

            lock_path = tmp_path / "metaxy.lock"
            # selection adds up/extra on top of graph-discovered up/dep
            result = generate_lock_file(
                store,
                lock_path,
                exclude_project="local",
                selection=FeatureSelection(keys=["up/extra"]),
            )

        assert result.count == 2
        content = tomli.loads(lock_path.read_text())
        assert set(content["features"].keys()) == {"up/dep", "up/extra"}

    def test_selection_missing_key_raises(self, store: MetadataStore, tmp_path: Path):
        """Selection with a key not in the store raises FeatureNotFoundError."""
        from metaxy.metadata_store.exceptions import FeatureNotFoundError
        from metaxy.models.feature_selection import FeatureSelection
        from metaxy.utils.lock_file import generate_lock_file

        self._push_project(store, "upstream", ["up/exists"])

        lock_path = tmp_path / "metaxy.lock"
        with pytest.raises(FeatureNotFoundError, match="up/ghost"):
            generate_lock_file(store, lock_path, selection=FeatureSelection(keys=["up/exists", "up/ghost"]))

    def test_selection_excludes_local_features(self, store: MetadataStore, tmp_path: Path):
        """Features defined locally in the graph are excluded even if matched by selection."""
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.feature_selection import FeatureSelection
        from metaxy.utils.lock_file import generate_lock_file

        self._push_project(store, "upstream", ["shared/feat", "up/other"])

        g = FeatureGraph()
        with g.use():
            # Local project also defines shared/feat
            class SharedFeat(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey("shared/feat"),
                    fields=[FieldSpec(key=FieldKey(["v"]), code_version="1")],
                ),
            ):
                __metaxy_project__ = "local"

            lock_path = tmp_path / "metaxy.lock"
            result = generate_lock_file(
                store,
                lock_path,
                exclude_project="local",
                selection=FeatureSelection(projects=["upstream"]),
            )

        # shared/feat is local, so only up/other should be locked
        assert result.count == 1
        content = tomli.loads(lock_path.read_text())
        assert set(content["features"].keys()) == {"up/other"}

    def test_selection_no_matches_produces_empty_lock(self, store: MetadataStore, tmp_path: Path):
        """Selection that matches nothing produces an empty lock file."""
        from metaxy.models.feature_selection import FeatureSelection
        from metaxy.utils.lock_file import generate_lock_file

        self._push_project(store, "upstream", ["up/a"])

        lock_path = tmp_path / "metaxy.lock"
        result = generate_lock_file(store, lock_path, selection=FeatureSelection(projects=["nonexistent"]))

        assert result.count == 0
        content = tomli.loads(lock_path.read_text())
        assert content["features"] == {}


class TestDiffLockFiles:
    """Unit tests for _diff_lock_files."""

    @staticmethod
    def _make_lock_file(features: dict[str, tuple[str, str, str]]) -> LockFile:
        """Build a LockFile from {key: (project, version, definition_version)}."""
        return LockFile(
            features={
                key: LockedFeature(
                    info=LockedFeatureInfo(version=version, version_by_field={}, definition_version=defn_version),
                    data=FeatureDefinition.from_stored_data(
                        feature_spec=(
                            f'{{"key": "{key}", "id_columns": ["uid"],'
                            f' "fields": [{{"key": ["v"], "code_version": "1"}}]}}'
                        ),
                        feature_schema="{}",
                        feature_class_path="test.Feature",
                        project=project,
                    ),
                )
                for key, (project, version, defn_version) in features.items()
            }
        )

    def test_no_existing_file_all_added(self, tmp_path: Path):
        """All features are reported as added when no lock file exists."""
        from metaxy.utils.lock_file import _diff_lock_files

        new_lock = self._make_lock_file({"a/x": ("a", "v1", "d1"), "b/y": ("b", "v2", "d2")})
        result = _diff_lock_files(tmp_path / "metaxy.lock", new_lock)

        assert len(result.added) == 2
        assert result.added[0].key == "a/x"
        assert result.added[1].key == "b/y"
        assert result.updated == []
        assert result.removed == []

    def test_corrupt_lock_file_warns_and_treats_as_fresh(self, tmp_path: Path):
        """Corrupt lock file emits a warning and treats all features as added."""
        from metaxy.utils.lock_file import _diff_lock_files

        lock_path = tmp_path / "metaxy.lock"
        lock_path.write_text("this is not valid toml [[[")

        new_lock = self._make_lock_file({"a/x": ("a", "v1", "d1")})

        with pytest.warns(match="Could not parse existing lock file"):
            result = _diff_lock_files(lock_path, new_lock)

        assert len(result.added) == 1
        assert result.added[0].key == "a/x"

    def test_identical_lock_files_unchanged(self, tmp_path: Path):
        """Identical lock files produce no changes."""
        from metaxy.utils.lock_file import _diff_lock_files

        lock = self._make_lock_file({"a/x": ("a", "v1", "d1"), "a/y": ("a", "v2", "d2")})
        lock_path = tmp_path / "metaxy.lock"
        lock_path.write_text("# Generated by `metaxy lock`.\n\n" + lock.to_toml())

        result = _diff_lock_files(lock_path, lock)

        assert result.added == []
        assert result.updated == []
        assert result.removed == []
        assert result.unchanged_count == 2
        assert not result.changed

    def test_removed_features(self, tmp_path: Path):
        """Features in old lock but not new are reported as removed."""
        from metaxy.utils.lock_file import _diff_lock_files

        old_lock = self._make_lock_file({"a/x": ("a", "v1", "d1"), "a/y": ("a", "v2", "d2")})
        lock_path = tmp_path / "metaxy.lock"
        lock_path.write_text("# Generated by `metaxy lock`.\n\n" + old_lock.to_toml())

        new_lock = self._make_lock_file({"a/x": ("a", "v1", "d1")})
        result = _diff_lock_files(lock_path, new_lock)

        assert result.added == []
        assert result.updated == []
        assert len(result.removed) == 1
        assert result.removed[0].key == "a/y"
        assert result.removed[0].project == "a"
        assert result.unchanged_count == 1

    def test_updated_features_version_change(self, tmp_path: Path):
        """Features with different definition_version and version are reported as updated."""
        from metaxy.utils.lock_file import _diff_lock_files

        old_lock = self._make_lock_file({"a/x": ("a", "v1", "d1")})
        lock_path = tmp_path / "metaxy.lock"
        lock_path.write_text("# Generated by `metaxy lock`.\n\n" + old_lock.to_toml())

        new_lock = self._make_lock_file({"a/x": ("a", "v2", "d2")})
        result = _diff_lock_files(lock_path, new_lock)

        assert result.added == []
        assert len(result.updated) == 1
        assert result.updated[0].key == "a/x"
        assert result.updated[0].old_version == "v1"
        assert result.updated[0].new_version == "v2"
        assert not result.updated[0].metadata_only

    def test_updated_features_metadata_only(self, tmp_path: Path):
        """Features with different definition_version but same version are metadata-only updates."""
        from metaxy.utils.lock_file import _diff_lock_files

        old_lock = self._make_lock_file({"a/x": ("a", "v1", "d1")})
        lock_path = tmp_path / "metaxy.lock"
        lock_path.write_text("# Generated by `metaxy lock`.\n\n" + old_lock.to_toml())

        new_lock = self._make_lock_file({"a/x": ("a", "v1", "d2")})
        result = _diff_lock_files(lock_path, new_lock)

        assert len(result.updated) == 1
        assert result.updated[0].metadata_only
        assert result.updated[0].old_version == "v1"
        assert result.updated[0].new_version == "v1"

    def test_mixed_changes(self, tmp_path: Path):
        """Added, updated, removed, and unchanged features in one diff."""
        from metaxy.utils.lock_file import _diff_lock_files

        old_lock = self._make_lock_file(
            {
                "a/keep": ("a", "v1", "d1"),
                "a/update": ("a", "v1", "d1"),
                "b/remove": ("b", "v1", "d1"),
            }
        )
        lock_path = tmp_path / "metaxy.lock"
        lock_path.write_text("# Generated by `metaxy lock`.\n\n" + old_lock.to_toml())

        new_lock = self._make_lock_file(
            {
                "a/keep": ("a", "v1", "d1"),
                "a/update": ("a", "v2", "d2"),
                "c/new": ("c", "v1", "d1"),
            }
        )
        result = _diff_lock_files(lock_path, new_lock)

        assert [c.key for c in result.added] == ["c/new"]
        assert [u.key for u in result.updated] == ["a/update"]
        assert [c.key for c in result.removed] == ["b/remove"]
        assert result.unchanged_count == 1
        assert result.count == 3  # added(1) + updated(1) + unchanged(1)
        assert result.changed


def test_generate_lock_file_errors_on_missing_transitive_dependency(tmp_path: Path):
    """Test that generate_lock_file errors when a transitive dependency is not in store."""
    import pytest
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureDep, FeatureGraph, FeatureKey, FieldKey, FieldSpec
    from metaxy.ext.duckdb import DuckDBMetadataStore
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

        with DuckDBMetadataStore(database=store_path).open("w") as store:
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
