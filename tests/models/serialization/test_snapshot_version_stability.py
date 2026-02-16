"""Test that project_version is stable across serialize/deserialize"""

# Import from conftest since it's in the same tests directory

import pytest
from metaxy_testing import TempFeatureModule
from metaxy_testing.models import SampleFeatureSpec

from metaxy import FeatureDep, FeatureKey, FieldDep, FieldKey, FieldSpec
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


def test_project_version_stability_with_module(temp_module: TempFeatureModule):
    """Test that project_version matches after serialize and reconstruct using real module"""
    # Get the graph with features
    graph1 = temp_module.graph

    # Get original project_version
    original_project_version = graph1.project_version
    print(f"\nOriginal project_version: {original_project_version}")

    # Serialize to snapshot dict
    snapshot_dict = graph1.to_snapshot()

    # Reconstruct from snapshot
    graph2 = FeatureGraph.from_snapshot(snapshot_dict)
    reconstructed_project_version = graph2.project_version
    print(f"Reconstructed project_version: {reconstructed_project_version}")

    # Verify they match
    assert original_project_version == reconstructed_project_version, (
        f"Snapshot versions don't match!\n"
        f"Original: {original_project_version}\n"
        f"Reconstructed: {reconstructed_project_version}"
    )

    print("✓ Snapshot versions match!")
