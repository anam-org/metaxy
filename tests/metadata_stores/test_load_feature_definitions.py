"""Tests for loading feature definitions from metadata stores."""

from collections.abc import Sequence
from pathlib import Path

import narwhals as nw
import pytest
from metaxy_testing.models import SampleFeatureSpec

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureKey,
    FeatureSelection,
    FieldKey,
    FieldSpec,
    MetadataStore,
    sync_external_features,
)
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureDefinition, FeatureGraph


def test_load_feature_definitions_raw_into_graph(tmp_path: Path):
    """Test loading feature definitions into a graph."""
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_a"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_b"]),
                deps=[FeatureDep(feature=FeatureKey(["feature_a"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Load into a new graph
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            assert len(new_graph.feature_definitions_by_key) == 0
            definitions = storage._load_feature_definitions_raw(graph=new_graph)

            # Returns list of loaded definitions
            assert len(definitions) == 2
            assert {d.key for d in definitions} == {FeatureKey(["feature_a"]), FeatureKey(["feature_b"])}

            # Graph is populated
            assert len(new_graph.feature_definitions_by_key) == 2
            assert FeatureKey(["feature_a"]) in new_graph.feature_definitions_by_key
            assert FeatureKey(["feature_b"]) in new_graph.feature_definitions_by_key


def test_load_feature_definitions_raw_uses_active_graph_by_default(tmp_path: Path):
    """Test that load_feature_definitions uses the active graph when none is provided."""
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class FeatureX(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_x"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Load into active graph
    target_graph = FeatureGraph()
    with target_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw()

            assert len(definitions) == 1
            assert FeatureKey(["feature_x"]) in target_graph.feature_definitions_by_key


def test_load_feature_definitions_raw_by_project(tmp_path: Path):
    """Test loading feature definitions filtered by project."""
    graph = FeatureGraph()
    with graph.use():

        class FeatureOne(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_one"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureTwo(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_two"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            result = storage.push_graph_snapshot()

            # Get the project name from stored features
            features_df = storage.read_features(current=False, project_version=result.project_version)
            project = features_df["project"][0]

    # Load filtering by project
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw(projects=[project], graph=new_graph)

            assert len(definitions) == 2


def test_load_feature_definitions_raw_returns_empty_list_when_no_features(tmp_path: Path):
    """Test that load_feature_definitions returns empty list when no features found."""
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw(graph=new_graph)

            assert definitions == []
            assert len(new_graph.feature_definitions_by_key) == 0


def test_load_feature_definitions_raw_preserves_dependencies(tmp_path: Path):
    """Test that loaded features preserve their dependency information."""
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["upstream"]),
                        rename={"value": "upstream_value"},
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Load and verify dependencies
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage._load_feature_definitions_raw(graph=new_graph)

            downstream_def = new_graph.feature_definitions_by_key[FeatureKey(["downstream"])]
            assert downstream_def.spec.deps is not None
            assert len(downstream_def.spec.deps) == 1
            assert downstream_def.spec.deps[0].feature == FeatureKey(["upstream"])
            assert downstream_def.spec.deps[0].rename == {"value": "upstream_value"}


def test_load_feature_definitions_raw_loads_latest_snapshot(tmp_path: Path):
    """Test that load_feature_definitions loads from the latest snapshot."""
    # Create v1
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class FeatureV1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["evolving_feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create v2 with changed code_version
    graph_v2 = FeatureGraph()
    with graph_v2.use():

        class FeatureV2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["evolving_feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="2")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Load should get latest (v2)
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw(graph=new_graph)

            assert len(definitions) == 1
            assert definitions[0].spec.fields[0].code_version == "2"


def test_mx_load_feature_definitions_raw_public_api(tmp_path: Path):
    """Test the _load_feature_definitions_raw public API."""
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class PublicApiFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["public_api_test"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Use storage API to load into a new graph
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw(graph=new_graph)

        assert len(definitions) == 1
        assert definitions[0].key == FeatureKey(["public_api_test"])
        assert FeatureKey(["public_api_test"]) in new_graph.feature_definitions_by_key


def test_feature_depending_on_loaded_definition(tmp_path: Path):
    """Test that a feature can depend on a feature loaded from metadata store.

    The dependency should not produce an error at definition time - it should
    only error if the dependency is not eventually loaded before operations
    that require it (like get_feature_plan).
    """
    # First, save a feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Now create a new graph where we define a feature that depends on the stored feature
    new_graph = FeatureGraph()
    with new_graph.use():
        # Define a downstream feature that depends on "upstream" which doesn't exist yet
        # This should NOT error - dependencies are validated lazily

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        # At this point, "upstream" is not in the graph
        assert FeatureKey(["upstream"]) not in new_graph.feature_definitions_by_key
        assert FeatureKey(["downstream"]) in new_graph.feature_definitions_by_key

        # Now load the upstream feature from the store
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage._load_feature_definitions_raw()

        # Now both features are in the graph
        assert FeatureKey(["upstream"]) in new_graph.feature_definitions_by_key
        assert FeatureKey(["downstream"]) in new_graph.feature_definitions_by_key

        # And we can get the feature plan (which validates dependencies)
        plan = new_graph.get_feature_plan(FeatureKey(["downstream"]))
        assert plan.deps is not None
        assert len(plan.deps) == 1
        assert plan.deps[0].key == FeatureKey(["upstream"])


def test_load_feature_definitions_raw_with_filters(tmp_path: Path):
    """Test loading feature definitions with narwhals filter expressions."""
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class FeatureAlpha(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["alpha"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureBeta(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["beta"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureGamma(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["gamma"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Load only specific features using filter
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw(
                filters=[nw.col("feature_key").is_in(["alpha", "gamma"])],
                graph=new_graph,
            )

        assert len(definitions) == 2
        keys = {d.key for d in definitions}
        assert keys == {FeatureKey(["alpha"]), FeatureKey(["gamma"])}
        assert FeatureKey(["beta"]) not in new_graph.feature_definitions_by_key


def test_load_feature_definitions_raw_with_filters_via_storage(tmp_path: Path):
    """Test SystemTableStorage.load_feature_definitions with filters."""
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class FeatureOne(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["one"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class FeatureTwo(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["two"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Load using SystemTableStorage directly with filter
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw(
                filters=[nw.col("feature_key") == "one"],
                graph=new_graph,
            )

            assert len(definitions) == 1
            assert definitions[0].key == FeatureKey(["one"])
            assert FeatureKey(["two"]) not in new_graph.feature_definitions_by_key


def test_load_feature_definitions_raw_filters_applied_after_deduplication(tmp_path: Path):
    """Test that filters are applied after timestamp deduplication.

    This ensures we always load the latest version of a feature, regardless of filters.
    If filters were applied before deduplication, filtering by feature_key could
    incorrectly return an older version.
    """
    # Create v1 of the feature
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class EvolvingFeatureV1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["evolving"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create v2 of the feature with different code_version
    graph_v2 = FeatureGraph()
    with graph_v2.use():

        class EvolvingFeatureV2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["evolving"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="2")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Load with filter - should get v2 (latest), not v1
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage._load_feature_definitions_raw(
                filters=[nw.col("feature_key") == "evolving"],
                graph=new_graph,
            )

        # Should load exactly one feature
        assert len(definitions) == 1
        # And it should be the latest version (v2)
        assert definitions[0].spec.fields[0].code_version == "2"


def test_snapshot_with_unresolved_external_dependency_succeeds(tmp_path: Path):
    """Test that creating a snapshot with unresolved external dependencies succeeds.

    This enables entangled multi-project setups where projects can push independently
    without requiring all external dependencies to be resolved first.
    """
    from metaxy.models.feature import FeatureDefinition

    # Create a new graph with an external feature placeholder and a dependent feature
    new_graph = FeatureGraph()
    with new_graph.use():
        # Add external feature placeholder (simulating a feature we know exists but haven't loaded)
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["external_upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={"type": "object"},
            project="other-project",
        )
        new_graph.add_feature_definition(external_def)

        # Define a downstream feature that depends on the external placeholder
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local_downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["external_upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        # Creating a snapshot should succeed - external features are excluded
        snapshot = new_graph.to_snapshot()

        # Only the non-external feature should be in the snapshot
        assert "local_downstream" in snapshot
        assert "external_upstream" not in snapshot


def test_snapshot_succeeds_after_loading_external_dependencies(tmp_path: Path):
    """Test that snapshot works after loading external dependencies from the store."""
    from metaxy.models.feature import FeatureDefinition

    # First, save an upstream feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["shared_upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Now create a new graph with an external feature placeholder and a dependent feature
    new_graph = FeatureGraph()
    with new_graph.use():
        # Add external feature placeholder
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["shared_upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={"type": "object"},
            project="other-project",
        )
        new_graph.add_feature_definition(external_def)

        # Define a downstream feature that depends on the external placeholder
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["local_downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["shared_upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        # Verify external is in the graph
        assert new_graph.get_feature_definition(["shared_upstream"]).is_external is True

        # Load the real feature from the store - this should replace the external placeholder
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage._load_feature_definitions_raw()

        # Now the feature should no longer be external
        assert new_graph.get_feature_definition(["shared_upstream"]).is_external is False

        # And creating a snapshot should succeed
        snapshot = new_graph.to_snapshot()

        # Both features should be in the snapshot
        assert "shared_upstream" in snapshot
        assert "local_downstream" in snapshot


def test_external_features_never_pushed_to_metadata_store(tmp_path: Path):
    """Test that external features are never pushed to the metadata store."""
    from metaxy.models.feature import FeatureDefinition

    # First, save an upstream feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class OriginalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["original_feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create a new graph with the loaded feature plus an external feature
    new_graph = FeatureGraph()
    with new_graph.use():
        # Load the original feature
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage._load_feature_definitions_raw()

        # Add an external feature (from another project we don't control)
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["external_only"]),
                fields=[FieldSpec(key=FieldKey(["value"]))],
            ),
            feature_schema={"type": "object"},
            project="external-project",
        )
        new_graph.add_feature_definition(external_def)

        # Both features are in the graph
        assert FeatureKey(["original_feature"]) in new_graph.feature_definitions_by_key
        assert FeatureKey(["external_only"]) in new_graph.feature_definitions_by_key

        # Snapshot should only contain the non-external feature
        snapshot = new_graph.to_snapshot()
        assert "original_feature" in snapshot
        assert "external_only" not in snapshot

        # Push should succeed and only push the non-external feature
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            result = storage.push_graph_snapshot()

            # Verify only original_feature was pushed
            features_df = storage.read_features(current=False, project_version=result.project_version)
            feature_keys = features_df["feature_key"].to_list()
            assert "original_feature" in feature_keys
            assert "external_only" not in feature_keys


def test_resolve_update_loads_external_feature_definitions(tmp_path: Path):
    """Test that resolve_update automatically loads feature definitions from the store.

    This ensures that when a downstream feature depends on an external feature placeholder,
    the actual feature definition is loaded from the store before computing the update.
    Without this, version hashes would be computed incorrectly using stale external data.
    """
    import polars as pl

    # Step 1: Create and push an upstream feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["resolve_test", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

            # Also write some metadata for the upstream feature
            upstream_metadata = pl.DataFrame(
                {
                    "sample_uid": ["sample1", "sample2"],
                    "metaxy_provenance_by_field": [
                        {"data": "hash1"},
                        {"data": "hash2"},
                    ],
                    "metaxy_provenance": ["prov1", "prov2"],
                    "metaxy_data_version_by_field": [
                        {"data": "hash1"},
                        {"data": "hash2"},
                    ],
                    "metaxy_data_version": ["prov1", "prov2"],
                }
            )
            store.write(UpstreamFeature, upstream_metadata)

        # Get the expected version when upstream is fully defined
        expected_upstream_version = source_graph.get_feature_version(["resolve_test", "upstream"])

    # Step 2: Create a new graph with external placeholder for upstream
    # and define a downstream feature that depends on it
    new_graph = FeatureGraph()
    with new_graph.use():
        # Add external placeholder for upstream (simulating code that doesn't have the upstream class)
        external_upstream = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["resolve_test", "upstream"]),
                # Intentionally different field to verify it gets replaced
                fields=[FieldSpec(key=FieldKey(["different_field"]), code_version="999")],
            ),
            feature_schema={"wrong": "schema"},
            project="placeholder-project",
        )
        new_graph.add_feature_definition(external_upstream)

        # Verify it's external before resolve_update
        assert new_graph.get_feature_definition(["resolve_test", "upstream"]).is_external is True

        # Define downstream feature
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["resolve_test", "downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["resolve_test", "upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            pass

        # Step 3: Call resolve_update - this should automatically load the real upstream definition
        with DeltaMetadataStore(root_path=tmp_path / "delta_store").open("w") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

            # resolve_update should load feature definitions, replacing the external placeholder
            increment = store.resolve_update(DownstreamFeature)

            # Verify the external placeholder was replaced with the real definition
            upstream_def = new_graph.get_feature_definition(["resolve_test", "upstream"])
            assert upstream_def.is_external is False
            assert new_graph.get_feature_version(["resolve_test", "upstream"]) == expected_upstream_version

            # The increment should have computed correctly with the real upstream definition
            assert increment.new is not None
            assert len(increment.new) == 2  # Both upstream samples


def test_sync_external_features_warns_on_version_mismatch(tmp_path: Path):
    """Test that sync_external_features warns when external feature version mismatches."""
    import pytest

    # First, save a feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class SourceFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_mismatch_warn"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create a new graph with a mismatched external feature
    new_graph = FeatureGraph()
    with new_graph.use():
        # Add external feature with different code_version than what's in the store
        # Note: project must match the project of the saved feature (project doesn't need to match - sync uses feature keys)
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_mismatch_warn"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="warn",  # Should warn
        )
        new_graph.add_feature_definition(external_def)

        # Sync should warn about version mismatch
        with pytest.warns(UserWarning, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            sync_external_features(store)

        # But the feature should still be loaded (and no longer external)
        assert new_graph.get_feature_definition(["version_mismatch_warn"]).is_external is False


def test_sync_external_features_raises_on_version_mismatch_when_error(tmp_path: Path):
    """Test that sync_external_features raises when external feature version mismatches and on_version_mismatch='error'."""
    import pytest

    # First, save a feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class SourceFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_mismatch_error"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create a new graph with a mismatched external feature set to error
    new_graph = FeatureGraph()
    with new_graph.use():
        # Add external feature with different code_version than what's in the store
        # Note: project must match the project of the saved feature (project doesn't need to match - sync uses feature keys)
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_mismatch_error"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="error",  # Should raise
        )
        new_graph.add_feature_definition(external_def)

        # Sync should raise error about version mismatch
        from metaxy import ExternalFeatureVersionMismatchError

        with pytest.raises(ExternalFeatureVersionMismatchError, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            sync_external_features(store)


def test_sync_external_features_consolidates_multiple_mismatches(tmp_path: Path):
    """Test that load_feature_definitions issues a single consolidated warning for multiple mismatches."""
    import pytest

    # First, save two features to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi_mismatch", "a"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi_mismatch", "b"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create a new graph with TWO mismatched external features
    new_graph = FeatureGraph()
    with new_graph.use():
        # Note: project must match the project of the saved features (project doesn't need to match - sync uses feature keys)
        external_a = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi_mismatch", "a"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="warn",
        )
        new_graph.add_feature_definition(external_a)

        external_b = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi_mismatch", "b"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="999")],
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="warn",
        )
        new_graph.add_feature_definition(external_b)

        # Should get ONE warning mentioning BOTH features
        with pytest.warns(UserWarning, match="2 external feature") as record:
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            sync_external_features(store)

        # Verify both features are mentioned in the single warning
        warning_message = str(record[0].message)
        assert "multi_mismatch/a" in warning_message
        assert "multi_mismatch/b" in warning_message


def test_sync_external_features_no_warning_when_versions_match(tmp_path: Path):
    """Test that load_feature_definitions doesn't warn when versions match."""
    import warnings

    # First, save a feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class SourceFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_match"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        # Get the real feature version to use in our external definition
        expected_version_by_field = source_graph.get_feature_version_by_field(["version_match"])

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create a new graph with a MATCHING external feature (using provenance override)
    new_graph = FeatureGraph()
    with new_graph.use():
        # Add external feature with provenance override matching the real feature
        # Convert string keys to FieldKey for type safety
        # Note: project must match the project of the saved feature (project doesn't need to match - sync uses feature keys)
        provenance_override = {FieldKey([k]): v for k, v in expected_version_by_field.items()}
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_match"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={},
            project="any-project",
            provenance_by_field=provenance_override,  # Match the real version
            on_version_mismatch="error",  # Would raise if there's a mismatch
        )
        new_graph.add_feature_definition(external_def)

        # Sync should NOT warn or raise
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            sync_external_features(store)

        # Feature should be loaded
        assert new_graph.get_feature_definition(["version_match"]).is_external is False


def test_sync_external_features_on_version_mismatch_override_to_error(tmp_path: Path):
    """Test that on_version_mismatch='error' overrides a feature's 'warn' setting."""
    import pytest

    # First, save a feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class SourceFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["override_to_error"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create a new graph with a mismatched external feature set to "warn"
    new_graph = FeatureGraph()
    with new_graph.use():
        # Note: project must match the project of the saved feature (project doesn't need to match - sync uses feature keys)
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["override_to_error"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="warn",  # Feature says warn
        )
        new_graph.add_feature_definition(external_def)

        # Sync with on_version_mismatch="error" override should raise
        from metaxy import ExternalFeatureVersionMismatchError

        with pytest.raises(ExternalFeatureVersionMismatchError, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            sync_external_features(store, on_version_mismatch="error")


def test_sync_external_features_on_version_mismatch_override_to_warn(tmp_path: Path):
    """Test that on_version_mismatch='warn' overrides a feature's 'error' setting."""
    import pytest

    # First, save a feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class SourceFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["override_to_warn"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

    # Create a new graph with a mismatched external feature set to "error"
    new_graph = FeatureGraph()
    with new_graph.use():
        # Note: project must match the project of the saved feature (project doesn't need to match - sync uses feature keys)
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["override_to_warn"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="any-project",
            on_version_mismatch="error",  # Feature says error
        )
        new_graph.add_feature_definition(external_def)

        # Sync with on_version_mismatch="warn" override should only warn, not raise
        with pytest.warns(UserWarning, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            sync_external_features(store, on_version_mismatch="warn")

        # Feature should still be loaded
        assert new_graph.get_feature_definition(["override_to_warn"]).is_external is False


def test_sync_external_features_warns_on_invalid_stored_feature(tmp_path: Path):
    """Test that sync_external_features warns and skips individually corrupted features."""
    import warnings

    from metaxy._warnings import InvalidStoredFeatureWarning
    from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore

    store_path = tmp_path / "store.duckdb"

    # Push two features to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class ValidFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["sync_test", "valid"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        class CorruptFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["sync_test", "corrupt"]),
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
            "WHERE feature_key = 'sync_test/corrupt'"
        )

    # Create a graph with external placeholders for both features
    new_graph = FeatureGraph()
    with new_graph.use():
        external_valid = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["sync_test", "valid"]),
                fields=[FieldSpec(key=FieldKey(["value"]))],
            ),
            feature_schema={},
            project="placeholder",
        )
        new_graph.add_feature_definition(external_valid)

        external_corrupt = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["sync_test", "corrupt"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={},
            project="placeholder",
        )
        new_graph.add_feature_definition(external_corrupt)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sync_external_features(DuckDBMetadataStore(database=store_path))

    # Valid feature was loaded successfully
    assert len(result) == 1
    assert result[0].key == FeatureKey(["sync_test", "valid"])

    # One warning for the corrupted feature, one for it remaining unresolved
    invalid_warnings = [w for w in caught if w.category is InvalidStoredFeatureWarning]
    assert len(invalid_warnings) == 1
    assert "sync_test/corrupt" in str(invalid_warnings[0].message)


# ── resolve_selection tests ──────────────────────────────────────────


def _push_project(store: MetadataStore, project: str, keys: Sequence[str]) -> None:
    """Define features for a project, push them to the store, then clear the graph."""
    graph = FeatureGraph()
    with graph.use():
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
        with store:
            SystemTableStorage(store).push_graph_snapshot(project=project)


@pytest.fixture
def two_project_store(store: MetadataStore) -> MetadataStore:
    """Store pre-populated with features from two projects."""
    _push_project(store, "proj-a", ["sel/a1", "sel/a2"])
    _push_project(store, "proj-b", ["sel/b1"])
    return store


class TestResolveSelection:
    def test_by_projects(self, two_project_store: MetadataStore):
        with two_project_store:
            defs = SystemTableStorage(two_project_store).resolve_selection(
                FeatureSelection(projects=["proj-a"]),
            )
        assert {d.key for d in defs} == {FeatureKey("sel/a1"), FeatureKey("sel/a2")}

    def test_by_multiple_projects(self, two_project_store: MetadataStore):
        with two_project_store:
            defs = SystemTableStorage(two_project_store).resolve_selection(
                FeatureSelection(projects=["proj-a", "proj-b"]),
            )
        assert {d.key for d in defs} == {FeatureKey("sel/a1"), FeatureKey("sel/a2"), FeatureKey("sel/b1")}

    def test_by_keys(self, two_project_store: MetadataStore):
        with two_project_store:
            defs = SystemTableStorage(two_project_store).resolve_selection(
                FeatureSelection(keys=["sel/a1", "sel/b1"]),
            )
        assert {d.key for d in defs} == {FeatureKey("sel/a1"), FeatureKey("sel/b1")}

    def test_by_keys_missing_ignored(self, two_project_store: MetadataStore):
        with two_project_store:
            defs = SystemTableStorage(two_project_store).resolve_selection(
                FeatureSelection(keys=["sel/a1", "sel/nonexistent"]),
            )
        assert {d.key for d in defs} == {FeatureKey("sel/a1")}

    def test_all(self, two_project_store: MetadataStore):
        with two_project_store:
            defs = SystemTableStorage(two_project_store).resolve_selection(
                FeatureSelection(all=True),
            )
        assert {d.key for d in defs} == {FeatureKey("sel/a1"), FeatureKey("sel/a2"), FeatureKey("sel/b1")}

    def test_empty_store(self, store: MetadataStore):
        with store:
            defs = SystemTableStorage(store).resolve_selection(
                FeatureSelection(projects=["anything"]),
            )
        assert defs == []

    def test_or_merges_projects_and_keys(self, two_project_store: MetadataStore):
        sel = FeatureSelection(projects=["proj-a"]) | FeatureSelection(keys=["sel/b1"])
        with two_project_store:
            defs = SystemTableStorage(two_project_store).resolve_selection(sel)
        assert {d.key for d in defs} == {FeatureKey("sel/a1"), FeatureKey("sel/a2"), FeatureKey("sel/b1")}

    def test_projects_and_keys_combined(self, two_project_store: MetadataStore):
        sel = FeatureSelection(projects=["proj-a"], keys=["sel/b1"])
        with two_project_store:
            defs = SystemTableStorage(two_project_store).resolve_selection(sel)
        assert {d.key for d in defs} == {FeatureKey("sel/a1"), FeatureKey("sel/a2"), FeatureKey("sel/b1")}


# ── sync_external_features with selection tests ──────────────────────


def _push_project_with_deps(
    store: MetadataStore,
    project: str,
    features: dict[str, list[str]],
) -> None:
    """Push features with optional deps.

    Args:
        store: Target store.
        project: Project name for all features.
        features: Mapping of feature key to list of dep keys (empty list = no deps).
    """
    g = FeatureGraph()
    with g.use():
        for key, deps in features.items():
            type(
                f"_Feat_{key.replace('/', '_')}",
                (BaseFeature,),
                {"__metaxy_project__": project},
                spec=SampleFeatureSpec(
                    key=FeatureKey(key),
                    deps=[FeatureDep(feature=d) for d in deps],
                    fields=[FieldSpec(key=FieldKey(["v"]), code_version="1")],
                ),
            )
        with store:
            SystemTableStorage(store).push_graph_snapshot(project=project)


class TestSyncWithSelection:
    def test_by_projects(self, store: MetadataStore):
        """Selection loads all features from the specified projects."""
        _push_project(store, "upstream", ["load/a", "load/b"])

        result = sync_external_features(store, selection=FeatureSelection(projects=["upstream"]))
        assert {d.key for d in result} == {FeatureKey("load/a"), FeatureKey("load/b")}

    def test_by_keys(self, store: MetadataStore):
        """Selection loads only the specified keys."""
        _push_project(store, "upstream", ["load/x", "load/y", "load/z"])

        result = sync_external_features(store, selection=FeatureSelection(keys=["load/x", "load/z"]))
        assert {d.key for d in result} == {FeatureKey("load/x"), FeatureKey("load/z")}

    def test_all(self, store: MetadataStore):
        """Selection with all=True loads every feature in the store."""
        _push_project(store, "proj-a", ["load/a1"])
        _push_project(store, "proj-b", ["load/b1"])

        result = sync_external_features(store, selection=FeatureSelection(all=True))
        assert {d.key for d in result} == {FeatureKey("load/a1"), FeatureKey("load/b1")}

    def test_adds_to_graph(self, store: MetadataStore, graph: FeatureGraph):
        """Loaded features are added to the active graph."""
        _push_project(store, "upstream", ["load/added"])

        assert FeatureKey("load/added") not in graph.feature_definitions_by_key

        sync_external_features(store, selection=FeatureSelection(keys=["load/added"]))

        assert FeatureKey("load/added") in graph.feature_definitions_by_key
        assert graph.feature_definitions_by_key[FeatureKey("load/added")].is_external is False

    def test_transitive_deps_resolved(self, store: MetadataStore):
        """Transitive deps are loaded iteratively."""
        _push_project_with_deps(
            store,
            "upstream",
            {
                "chain/c": [],
                "chain/b": ["chain/c"],
                "chain/a": ["chain/b"],
            },
        )

        result = sync_external_features(store, selection=FeatureSelection(keys=["chain/a"]))
        assert {d.key for d in result} == {
            FeatureKey("chain/a"),
            FeatureKey("chain/b"),
            FeatureKey("chain/c"),
        }

    def test_transitive_deps_already_in_graph_not_reloaded(self, store: MetadataStore, graph: FeatureGraph):
        """Deps already present in the graph are not re-fetched."""
        _push_project_with_deps(
            store,
            "upstream",
            {
                "dep/base": [],
                "dep/top": ["dep/base"],
            },
        )

        sync_external_features(store, selection=FeatureSelection(keys=["dep/base"]))
        assert FeatureKey("dep/base") in graph.feature_definitions_by_key

        result = sync_external_features(store, selection=FeatureSelection(keys=["dep/top"]))
        result_keys = {d.key for d in result}
        assert FeatureKey("dep/top") in result_keys
        assert FeatureKey("dep/base") not in result_keys

    def test_sync_replaces_external_placeholders(self, store: MetadataStore, graph: FeatureGraph):
        """sync_external_features replaces external placeholders with real definitions."""
        _push_project(store, "upstream", ["sync/ext"])

        external = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey("sync/ext"),
                fields=[FieldSpec(key=FieldKey(["v"]))],
            ),
            feature_schema={},
            project="placeholder",
        )
        graph.add_feature_definition(external)

        result = sync_external_features(store)

        assert len(result) == 1
        assert result[0].key == FeatureKey("sync/ext")
        assert graph.feature_definitions_by_key[FeatureKey("sync/ext")].is_external is False

    def test_sync_no_externals_no_selection_returns_empty(self, store: MetadataStore):
        """sync_external_features with no external features and no selection returns empty list."""
        _push_project(store, "upstream", ["sync/noop"])

        result = sync_external_features(store)
        assert result == []

    def test_sync_warns_unresolved(self, store: MetadataStore, graph: FeatureGraph):
        """sync_external_features warns about external keys not found in the store."""
        import warnings

        from metaxy._warnings import UnresolvedExternalFeatureWarning

        external = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey("sync/ghost"),
                fields=[FieldSpec(key=FieldKey(["v"]))],
            ),
            feature_schema={},
            project="placeholder",
        )
        graph.add_feature_definition(external)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sync_external_features(store)

        unresolved = [w for w in caught if w.category is UnresolvedExternalFeatureWarning]
        assert len(unresolved) == 1
        assert "sync/ghost" in str(unresolved[0].message)
