"""Test that snapshot_id is stable across serialize/deserialize"""

# Import from conftest since it's in the same tests directory

from metaxy import FeatureDep, FeatureKey, FeatureSpec, FieldDep, FieldKey, FieldSpec
from metaxy._testing import TempFeatureModule
from metaxy.models.feature import FeatureGraph


def test_snapshot_id_stability_with_module():
    """Test that snapshot_id matches after serialize and reconstruct using real module"""

    # Create temp module with features
    temp_module = TempFeatureModule("test_snapshot_stability")

    parent_spec = FeatureSpec(
        key=FeatureKey(["test", "parent"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    child_spec = FeatureSpec(
        key=FeatureKey(["test", "child"]),
        deps=[FeatureDep(key=FeatureKey(["test", "parent"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["result"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "parent"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    temp_module.write_features(
        {"ParentFeature": parent_spec, "ChildFeature": child_spec}
    )

    try:
        # Get the graph with features
        graph1 = temp_module.get_graph()

        # Get original snapshot_id
        original_snapshot_id = graph1.snapshot_id
        print(f"\nOriginal snapshot_id: {original_snapshot_id}")

        # Serialize to snapshot dict
        snapshot_dict = graph1.to_snapshot()

        # Reconstruct from snapshot
        graph2 = FeatureGraph.from_snapshot(snapshot_dict)
        reconstructed_snapshot_id = graph2.snapshot_id
        print(f"Reconstructed snapshot_id: {reconstructed_snapshot_id}")

        # Verify they match
        assert original_snapshot_id == reconstructed_snapshot_id, (
            f"Snapshot IDs don't match!\n"
            f"Original: {original_snapshot_id}\n"
            f"Reconstructed: {reconstructed_snapshot_id}"
        )

        print("✓ Snapshot IDs match!")

    finally:
        temp_module.cleanup()
