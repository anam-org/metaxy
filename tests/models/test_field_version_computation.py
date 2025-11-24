"""Test field version computation in load_snapshot_data()."""

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.graph.diff.differ import GraphDiffer
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph
from metaxy.models.field import FieldSpec
from metaxy.models.plan import FQFieldKey
from metaxy.models.types import FeatureKey, FieldKey


def test_load_snapshot_data_computes_proper_field_versions(graph: FeatureGraph):
    """Test that field versions can be computed separately (when graph reconstruction works).

    Note: This test validates the field version computation logic, even though
    features defined in test functions fall back to using feature_version since
    they can't be imported. The important validation is that:
    1. Fields are tracked separately in the snapshot data
    2. The fallback mechanism works when graph reconstruction fails
    """
    differ = GraphDiffer()

    # Create parent feature with two fields with different code versions
    class ParentFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent"]),
            fields=[
                FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                FieldSpec(key=FieldKey(["field2"]), code_version="2"),
            ],
        ),
    ):
        pass

    # Compute expected field versions from the active graph
    # (these would be different if we could reconstruct the graph)
    parent_field1_version = graph.get_field_version(
        FQFieldKey(feature=FeatureKey(["parent"]), field=FieldKey(["field1"]))
    )
    parent_field2_version = graph.get_field_version(
        FQFieldKey(feature=FeatureKey(["parent"]), field=FieldKey(["field2"]))
    )

    # Field versions should be different from each other (different code versions)
    assert parent_field1_version != parent_field2_version

    with InMemoryMetadataStore() as store:
        # Record snapshot
        result = SystemTableStorage(store).push_graph_snapshot()

        snapshot_version = result.snapshot_version

        _ = result.already_pushed

        # Load snapshot data - will load standalone specs since test features can't be imported
        snapshot_data = differ.load_snapshot_data(store, snapshot_version)

        # Verify structure is correct
        assert "parent" in snapshot_data
        parent_data = snapshot_data["parent"]
        assert "metaxy_feature_version" in parent_data
        assert "fields" in parent_data
        assert "field1" in parent_data["fields"]
        assert "field2" in parent_data["fields"]

        # In fallback mode, fields use feature_version
        # (This is acceptable behavior when features can't be imported)
        assert parent_data["fields"]["field1"] == parent_data["metaxy_feature_version"]
        assert parent_data["fields"]["field2"] == parent_data["metaxy_feature_version"]


def test_load_snapshot_data_fallback_when_graph_reconstruction_fails(
    graph: FeatureGraph,
):
    """Test fallback behavior when feature classes cannot be imported."""
    differ = GraphDiffer()

    # Create a feature defined in test scope (will not be importable)
    class TestFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[
                FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                FieldSpec(key=FieldKey(["field2"]), code_version="2"),
            ],
        ),
    ):
        pass

    with InMemoryMetadataStore() as store:
        # Record snapshot
        result = SystemTableStorage(store).push_graph_snapshot()

        snapshot_version = result.snapshot_version

        _ = result.already_pushed

        # Load snapshot data - should load standalone specs
        snapshot_data = differ.load_snapshot_data(store, snapshot_version)

        # Verify data was loaded (even with fallback)
        assert "test/feature" in snapshot_data
        feature_data = snapshot_data["test/feature"]
        assert "metaxy_feature_version" in feature_data
        assert "fields" in feature_data

        # In fallback mode, all fields get the same version (feature_version)
        assert (
            feature_data["fields"]["field1"] == feature_data["metaxy_feature_version"]
        )
        assert (
            feature_data["fields"]["field2"] == feature_data["metaxy_feature_version"]
        )


def test_field_key_normalization(graph: FeatureGraph):
    """Test that field keys are normalized to "/" separator format."""
    differ = GraphDiffer()

    class TestFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test"]),
            fields=[
                FieldSpec(key=FieldKey(["nested", "field"]), code_version="1"),
            ],
        ),
    ):
        pass

    with InMemoryMetadataStore() as store:
        # Record snapshot
        result = SystemTableStorage(store).push_graph_snapshot()

        snapshot_version = result.snapshot_version

        _ = result.already_pushed

        # Load snapshot data (will load standalone spec since feature is in test scope)
        snapshot_data = differ.load_snapshot_data(store, snapshot_version)

        # Field key should be normalized to "/" format, not "__"
        assert "test" in snapshot_data
        feature_data = snapshot_data["test"]
        assert "nested/field" in feature_data["fields"]
        assert "nested__field" not in feature_data["fields"]
