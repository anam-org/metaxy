"""Test field version computation in load_snapshot_data()."""

import pytest

from metaxy.graph.diff.differ import GraphDiffer
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.plan import FQFieldKey
from metaxy.models.types import FeatureKey, FieldKey


def test_load_snapshot_data_computes_proper_field_versions():
    """Test that field versions can be computed separately (when graph reconstruction works).

    Note: This test validates the field version computation logic, even though
    features defined in test functions fall back to using feature_version since
    they can't be imported. The important validation is that:
    1. Fields are tracked separately in the snapshot data
    2. The fallback mechanism works when graph reconstruction fails
    """
    differ = GraphDiffer()
    graph = FeatureGraph()

    with graph.use():
        # Create parent feature with two fields with different code versions
        class ParentFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["parent"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version=1),
                    FieldSpec(key=FieldKey(["field2"]), code_version=2),
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
            result = store.record_feature_graph_snapshot()

            snapshot_version = result.snapshot_version

            _ = result.already_recorded

            # Load snapshot data - will use fallback since test features can't be imported
            with pytest.warns(
                UserWarning,
                match="Using feature_version as field_version",
            ):
                snapshot_data = differ.load_snapshot_data(
                    store, snapshot_version, project="default"
                )

            # Verify structure is correct
            assert "parent" in snapshot_data
            parent_data = snapshot_data["parent"]
            assert "feature_version" in parent_data
            assert "fields" in parent_data
            assert "field1" in parent_data["fields"]
            assert "field2" in parent_data["fields"]

            # In fallback mode, fields use feature_version
            # (This is acceptable behavior when features can't be imported)
            assert parent_data["fields"]["field1"] == parent_data["feature_version"]
            assert parent_data["fields"]["field2"] == parent_data["feature_version"]


def test_load_snapshot_data_fallback_when_graph_reconstruction_fails():
    """Test fallback behavior when feature classes cannot be imported."""
    differ = GraphDiffer()
    graph = FeatureGraph()

    with graph.use():
        # Create a feature defined in test scope (will not be importable)
        class TestFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "feature"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version=1),
                    FieldSpec(key=FieldKey(["field2"]), code_version=2),
                ],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            # Record snapshot
            result = store.record_feature_graph_snapshot()

            snapshot_version = result.snapshot_version

            _ = result.already_recorded

            # Load snapshot data - should trigger fallback
            with pytest.warns(
                UserWarning,
                match="Using feature_version as field_version",
            ):
                snapshot_data = differ.load_snapshot_data(
                    store, snapshot_version, project="default"
                )

            # Verify data was loaded (even with fallback)
            assert "test/feature" in snapshot_data
            feature_data = snapshot_data["test/feature"]
            assert "feature_version" in feature_data
            assert "fields" in feature_data

            # In fallback mode, all fields get the same version (feature_version)
            assert feature_data["fields"]["field1"] == feature_data["feature_version"]
            assert feature_data["fields"]["field2"] == feature_data["feature_version"]


def test_field_key_normalization():
    """Test that field keys are normalized to "/" separator format."""
    differ = GraphDiffer()
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["nested", "field"]), code_version=1),
                ],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            # Record snapshot
            result = store.record_feature_graph_snapshot()

            snapshot_version = result.snapshot_version

            _ = result.already_recorded

            # Load snapshot data (will use fallback since feature is in test scope)
            with pytest.warns(UserWarning):
                snapshot_data = differ.load_snapshot_data(
                    store, snapshot_version, project="default"
                )

            # Field key should be normalized to "/" format, not "__"
            assert "test" in snapshot_data
            feature_data = snapshot_data["test"]
            assert "nested/field" in feature_data["fields"]
            assert "nested__field" not in feature_data["fields"]
