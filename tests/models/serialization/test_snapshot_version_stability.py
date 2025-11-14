"""Test that snapshot_version is stable across serialize/deserialize"""

# Import from conftest since it's in the same tests directory

import pytest

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    SampleFeatureSpec,
)
from metaxy._testing import TempFeatureModule
from metaxy.models.feature import FeatureGraph


@pytest.fixture
def temp_module():
    module = TempFeatureModule("test_snapshot_stability")
    parent_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "parent"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    child_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "child"]),
        deps=[FeatureDep(feature=FeatureKey(["test", "parent"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["result"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test", "parent"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    module.write_features({"ParentFeature": parent_spec, "ChildFeature": child_spec})

    try:
        yield module
    finally:
        module.cleanup()


def test_snapshot_version_stability_with_module(temp_module: TempFeatureModule):
    """Test that snapshot_version matches after serialize and reconstruct using real module"""
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
