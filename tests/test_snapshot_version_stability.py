"""Test that snapshot_version is stable across serialize/deserialize"""

# Import from conftest since it's in the same tests directory

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    TestingFeatureSpec,
)
from metaxy._testing import TempFeatureModule
from metaxy.models.feature import FeatureGraph


def test_snapshot_version_stability_with_module():
    """Test that snapshot_version matches after serialize and reconstruct using real module"""

    # Create temp module with features
    temp_module = TempFeatureModule("test_snapshot_stability")

    parent_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "parent"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    child_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "child"]),
        deps=[FeatureDep(key=FeatureKey(["test", "parent"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["result"]),
                code_version=1,
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
        graph1 = temp_module.graph

        # Get original snapshot_version
        original_snapshot_version = graph1.snapshot_version
        print(f"\nOriginal snapshot_version: {original_snapshot_version}")

        # Serialize to snapshot dict
        snapshot_dict = graph1.to_snapshot()

        # Reconstruct from snapshot
        graph2 = FeatureGraph.from_snapshot(snapshot_dict)
        reconstructed_snapshot_version = graph2.snapshot_version
        print(f"Reconstructed snapshot_version: {reconstructed_snapshot_version}")

        # Verify they match
        assert original_snapshot_version == reconstructed_snapshot_version, (
            f"Snapshot versions don't match!\n"
            f"Original: {original_snapshot_version}\n"
            f"Reconstructed: {reconstructed_snapshot_version}"
        )

        print("✓ Snapshot versions match!")

    finally:
        temp_module.cleanup()
