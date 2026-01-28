"""Tests for loading feature definitions from metadata stores."""

from pathlib import Path

import narwhals as nw
from metaxy_testing.models import SampleFeatureSpec

import metaxy as mx
from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph


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
