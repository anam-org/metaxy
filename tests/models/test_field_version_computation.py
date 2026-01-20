"""Test field version computation in load_snapshot_data()."""

from syrupy.assertion import SnapshotAssertion

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.graph.diff.differ import GraphDiffer
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph
from metaxy.models.field import FieldSpec
from metaxy.models.plan import FQFieldKey
from metaxy.models.types import FeatureKey, FieldKey


def test_load_snapshot_data_computes_proper_field_versions(
    graph: FeatureGraph, snapshot: SnapshotAssertion
):
    """Test that field versions are computed correctly from specs.

    This test validates the field version computation logic. Features defined
    in test functions use standalone spec loading since they can't be imported.
    The important validation is that:
    1. Fields are tracked separately in the snapshot data
    2. Field versions are computed correctly from specs
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
    parent_field1_version = graph.get_field_version(
        FQFieldKey(feature=FeatureKey(["parent"]), field=FieldKey(["field1"]))
    )
    parent_field2_version = graph.get_field_version(
        FQFieldKey(feature=FeatureKey(["parent"]), field=FieldKey(["field2"]))
    )

    # Field versions should be different from each other (different code versions)
    assert parent_field1_version != parent_field2_version

    with DuckDBMetadataStore() as store:
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

        # Verify field versions are computed correctly (captured via snapshot)
        assert parent_data["fields"] == snapshot(name="field_versions")
        assert parent_data["metaxy_feature_version"] == snapshot(name="feature_version")


def test_load_snapshot_data_fallback_when_graph_reconstruction_fails(
    graph: FeatureGraph, snapshot: SnapshotAssertion
):
    """Test field version computation when feature classes cannot be imported."""
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

    with DuckDBMetadataStore() as store:
        # Record snapshot
        result = SystemTableStorage(store).push_graph_snapshot()

        snapshot_version = result.snapshot_version

        _ = result.already_pushed

        # Load snapshot data - should load standalone specs
        snapshot_data = differ.load_snapshot_data(store, snapshot_version)

        # Verify data was loaded
        assert "test/feature" in snapshot_data
        feature_data = snapshot_data["test/feature"]
        assert "metaxy_feature_version" in feature_data
        assert "fields" in feature_data

        # Verify field versions are computed correctly (captured via snapshot)
        assert feature_data["fields"] == snapshot(name="field_versions")
        assert feature_data["metaxy_feature_version"] == snapshot(
            name="feature_version"
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

    with DuckDBMetadataStore() as store:
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
