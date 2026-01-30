"""Tests for loading feature definitions from metadata stores."""

from pathlib import Path

import narwhals as nw
from metaxy_testing.models import SampleFeatureSpec

import metaxy as mx
from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureDefinition, FeatureGraph


def test_load_feature_definitions_into_graph(tmp_path: Path):
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
            definitions = storage.load_feature_definitions(graph=new_graph)

            # Returns list of loaded definitions
            assert len(definitions) == 2
            assert {d.key for d in definitions} == {FeatureKey(["feature_a"]), FeatureKey(["feature_b"])}

            # Graph is populated
            assert len(new_graph.feature_definitions_by_key) == 2
            assert FeatureKey(["feature_a"]) in new_graph.feature_definitions_by_key
            assert FeatureKey(["feature_b"]) in new_graph.feature_definitions_by_key


def test_load_feature_definitions_uses_active_graph_by_default(tmp_path: Path):
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
            definitions = storage.load_feature_definitions()

            assert len(definitions) == 1
            assert FeatureKey(["feature_x"]) in target_graph.feature_definitions_by_key


def test_load_feature_definitions_by_project(tmp_path: Path):
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
            storage.push_graph_snapshot()

            # Get the project name from stored features
            features_df = storage.read_features(current=False, snapshot_version=graph.snapshot_version)
            project = features_df["project"][0]

    # Load filtering by project
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage.load_feature_definitions(projects=project, graph=new_graph)

            assert len(definitions) == 2


def test_load_feature_definitions_returns_empty_list_when_no_features(tmp_path: Path):
    """Test that load_feature_definitions returns empty list when no features found."""
    new_graph = FeatureGraph()
    with new_graph.use():
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            definitions = storage.load_feature_definitions(graph=new_graph)

            assert definitions == []
            assert len(new_graph.feature_definitions_by_key) == 0


def test_load_feature_definitions_preserves_dependencies(tmp_path: Path):
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
            storage.load_feature_definitions(graph=new_graph)

            downstream_def = new_graph.feature_definitions_by_key[FeatureKey(["downstream"])]
            assert downstream_def.spec.deps is not None
            assert len(downstream_def.spec.deps) == 1
            assert downstream_def.spec.deps[0].feature == FeatureKey(["upstream"])
            assert downstream_def.spec.deps[0].rename == {"value": "upstream_value"}


def test_load_feature_definitions_loads_latest_snapshot(tmp_path: Path):
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
            definitions = storage.load_feature_definitions(graph=new_graph)

            assert len(definitions) == 1
            assert definitions[0].spec.fields[0].code_version == "2"


def test_mx_load_feature_definitions_public_api(tmp_path: Path):
    """Test the mx.load_feature_definitions public API."""
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

    # Use public API to load into a new graph (store not pre-opened)
    new_graph = FeatureGraph()
    with new_graph.use():
        store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
        definitions = mx.load_feature_definitions(store)

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
        store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
        mx.load_feature_definitions(store)

        # Now both features are in the graph
        assert FeatureKey(["upstream"]) in new_graph.feature_definitions_by_key
        assert FeatureKey(["downstream"]) in new_graph.feature_definitions_by_key

        # And we can get the feature plan (which validates dependencies)
        plan = new_graph.get_feature_plan(FeatureKey(["downstream"]))
        assert plan.deps is not None
        assert len(plan.deps) == 1
        assert plan.deps[0].key == FeatureKey(["upstream"])


def test_load_feature_definitions_with_filters(tmp_path: Path):
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
        store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
        definitions = mx.load_feature_definitions(
            store,
            filters=[nw.col("feature_key").is_in(["alpha", "gamma"])],
        )

        assert len(definitions) == 2
        keys = {d.key for d in definitions}
        assert keys == {FeatureKey(["alpha"]), FeatureKey(["gamma"])}
        assert FeatureKey(["beta"]) not in new_graph.feature_definitions_by_key


def test_load_feature_definitions_with_filters_via_storage(tmp_path: Path):
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
            definitions = storage.load_feature_definitions(
                filters=[nw.col("feature_key") == "one"],
                graph=new_graph,
            )

            assert len(definitions) == 1
            assert definitions[0].key == FeatureKey(["one"])
            assert FeatureKey(["two"]) not in new_graph.feature_definitions_by_key


def test_load_feature_definitions_filters_applied_after_deduplication(tmp_path: Path):
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
        store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
        definitions = mx.load_feature_definitions(
            store,
            filters=[nw.col("feature_key") == "evolving"],
        )

        # Should load exactly one feature
        assert len(definitions) == 1
        # And it should be the latest version (v2)
        assert definitions[0].spec.fields[0].code_version == "2"


def test_snapshot_with_unresolved_external_dependency_raises_error(tmp_path: Path):
    """Test that creating a snapshot with unresolved external dependencies raises an error."""
    import pytest

    from metaxy.models.feature import FeatureDefinition
    from metaxy.utils.exceptions import MetaxyInvariantViolationError

    # First, save an upstream feature to the store
    source_graph = FeatureGraph()
    with source_graph.use():

        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["external_upstream"]),
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

        # Attempting to create a snapshot should fail because
        # local_downstream depends on an external feature
        with pytest.raises(MetaxyInvariantViolationError) as exc_info:
            new_graph.to_snapshot()

        assert "local_downstream" in str(exc_info.value)
        assert "external_upstream" in str(exc_info.value)
        assert "External dependencies must be replaced" in str(exc_info.value)


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
        store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
        mx.load_feature_definitions(store)

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
        store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
        mx.load_feature_definitions(store)

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
            storage.push_graph_snapshot()

            # Verify only original_feature was pushed
            features_df = storage.read_features(current=True)
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
            store.write_metadata(UpstreamFeature, upstream_metadata)

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
        with DeltaMetadataStore(root_path=tmp_path / "delta_store").open("write") as store:
            storage = SystemTableStorage(store)
            storage.push_graph_snapshot()

            # resolve_update should load feature definitions, replacing the external placeholder
            increment = store.resolve_update(DownstreamFeature)

            # Verify the external placeholder was replaced with the real definition
            upstream_def = new_graph.get_feature_definition(["resolve_test", "upstream"])
            assert upstream_def.is_external is False
            assert new_graph.get_feature_version(["resolve_test", "upstream"]) == expected_upstream_version

            # The increment should have computed correctly with the real upstream definition
            assert increment.added is not None
            assert len(increment.added) == 2  # Both upstream samples


def test_load_feature_definitions_warns_on_version_mismatch(tmp_path: Path):
    """Test that load_feature_definitions warns when external feature version mismatches."""
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
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_mismatch_warn"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="test-project",
            on_version_mismatch="warn",  # Should warn
        )
        new_graph.add_feature_definition(external_def)

        # Load should warn about version mismatch
        with pytest.warns(UserWarning, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            mx.load_feature_definitions(store)

        # But the feature should still be loaded (and no longer external)
        assert new_graph.get_feature_definition(["version_mismatch_warn"]).is_external is False


def test_load_feature_definitions_raises_on_version_mismatch_when_error(tmp_path: Path):
    """Test that load_feature_definitions raises when external feature version mismatches and on_version_mismatch='error'."""
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
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_mismatch_error"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="test-project",
            on_version_mismatch="error",  # Should raise
        )
        new_graph.add_feature_definition(external_def)

        # Load should raise ValueError about version mismatch
        with pytest.raises(ValueError, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            mx.load_feature_definitions(store)


def test_load_feature_definitions_consolidates_multiple_mismatches(tmp_path: Path):
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
        external_a = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi_mismatch", "a"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],
            ),
            feature_schema={},
            project="test-project",
            on_version_mismatch="warn",
        )
        new_graph.add_feature_definition(external_a)

        external_b = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi_mismatch", "b"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="999")],
            ),
            feature_schema={},
            project="test-project",
            on_version_mismatch="warn",
        )
        new_graph.add_feature_definition(external_b)

        # Should get ONE warning mentioning BOTH features
        with pytest.warns(UserWarning, match="2 external feature") as record:
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            mx.load_feature_definitions(store)

        # Verify both features are mentioned in the single warning
        warning_message = str(record[0].message)
        assert "multi_mismatch/a" in warning_message
        assert "multi_mismatch/b" in warning_message


def test_load_feature_definitions_no_warning_when_versions_match(tmp_path: Path):
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
        provenance_override = {FieldKey([k]): v for k, v in expected_version_by_field.items()}
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_match"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={},
            project="test-project",
            provenance_by_field=provenance_override,  # Match the real version
            on_version_mismatch="error",  # Would raise if there's a mismatch
        )
        new_graph.add_feature_definition(external_def)

        # Load should NOT warn or raise
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            mx.load_feature_definitions(store)

        # Feature should be loaded
        assert new_graph.get_feature_definition(["version_match"]).is_external is False


def test_load_feature_definitions_on_version_mismatch_override_to_error(tmp_path: Path):
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
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["override_to_error"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="test-project",
            on_version_mismatch="warn",  # Feature says warn
        )
        new_graph.add_feature_definition(external_def)

        # Load with on_version_mismatch="error" override should raise
        with pytest.raises(ValueError, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            mx.load_feature_definitions(store, on_version_mismatch="error")


def test_load_feature_definitions_on_version_mismatch_override_to_warn(tmp_path: Path):
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
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["override_to_warn"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="999")],  # Different!
            ),
            feature_schema={},
            project="test-project",
            on_version_mismatch="error",  # Feature says error
        )
        new_graph.add_feature_definition(external_def)

        # Load with on_version_mismatch="warn" override should only warn, not raise
        with pytest.warns(UserWarning, match="Version mismatch"):
            store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
            mx.load_feature_definitions(store, on_version_mismatch="warn")

        # Feature should still be loaded
        assert new_graph.get_feature_definition(["override_to_warn"]).is_external is False
