from metaxy.models.container import ContainerDep, ContainerSpec, SpecialContainerDep
from metaxy.models.feature import Feature, FeatureRegistry
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.types import ContainerKey, FeatureKey


def test_single_feature_data_version(snapshot, registry: FeatureRegistry):
    """Test feature with no dependencies."""

    class MyFeature(
        Feature, spec=FeatureSpec(key=FeatureKey(["my_feature"]), deps=None)
    ):
        pass

    assert MyFeature.data_version() == snapshot


def test_feature_with_multiple_containers(snapshot, registry: FeatureRegistry):
    """Test feature with multiple containers, each with different code versions."""

    class VideoFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["video"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
                ContainerSpec(key=ContainerKey(["audio"]), code_version=2),
                ContainerSpec(key=ContainerKey(["metadata"]), code_version=5),
            ],
        ),
    ):
        pass

    assert VideoFeature.data_version() == snapshot


def test_linear_dependency_chain(snapshot, registry: FeatureRegistry):
    """Test A -> B -> C linear dependency chain."""

    class FeatureA(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["a"]),
            deps=None,
            containers=[ContainerSpec(key=ContainerKey(["raw"]), code_version=1)],
        ),
    ):
        pass

    class FeatureB(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["b"]),
            deps=[FeatureDep(key=FeatureKey(["a"]))],
            containers=[ContainerSpec(key=ContainerKey(["processed"]), code_version=2)],
        ),
    ):
        pass

    class FeatureC(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["c"]),
            deps=[FeatureDep(key=FeatureKey(["b"]))],
            containers=[ContainerSpec(key=ContainerKey(["final"]), code_version=3)],
        ),
    ):
        pass

    versions = {
        "a": FeatureA.data_version(),
        "b": FeatureB.data_version(),
        "c": FeatureC.data_version(),
    }

    assert versions == snapshot


def test_diamond_dependency_graph(snapshot, registry: FeatureRegistry):
    """Test diamond dependency: A -> B, A -> C, B -> D, C -> D."""

    class Root(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["root"]),
            deps=None,
            containers=[ContainerSpec(key=ContainerKey(["data"]), code_version=1)],
        ),
    ):
        pass

    class BranchLeft(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["branch_left"]),
            deps=[FeatureDep(key=FeatureKey(["root"]))],
            containers=[
                ContainerSpec(key=ContainerKey(["left_processed"]), code_version=2)
            ],
        ),
    ):
        pass

    class BranchRight(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["branch_right"]),
            deps=[FeatureDep(key=FeatureKey(["root"]))],
            containers=[
                ContainerSpec(key=ContainerKey(["right_processed"]), code_version=3)
            ],
        ),
    ):
        pass

    class Merged(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["merged"]),
            deps=[
                FeatureDep(key=FeatureKey(["branch_left"])),
                FeatureDep(key=FeatureKey(["branch_right"])),
            ],
            containers=[ContainerSpec(key=ContainerKey(["fusion"]), code_version=4)],
        ),
    ):
        pass

    versions = {
        "root": Root.data_version(),
        "branch_left": BranchLeft.data_version(),
        "branch_right": BranchRight.data_version(),
        "merged": Merged.data_version(),
    }

    assert versions == snapshot


def test_specific_container_dependencies(snapshot, registry: FeatureRegistry):
    """Test feature with specific container-level dependencies."""

    class MultiContainer(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["multi"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
                ContainerSpec(key=ContainerKey(["audio"]), code_version=2),
                ContainerSpec(key=ContainerKey(["text"]), code_version=3),
            ],
        ),
    ):
        pass

    class Selective(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["selective"]),
            deps=[FeatureDep(key=FeatureKey(["multi"]))],
            containers=[
                # Only depends on frames
                ContainerSpec(
                    key=ContainerKey(["visual"]),
                    code_version=10,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["multi"]),
                            containers=[ContainerKey(["frames"])],
                        )
                    ],
                ),
                # Only depends on audio
                ContainerSpec(
                    key=ContainerKey(["audio_only"]),
                    code_version=11,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["multi"]),
                            containers=[ContainerKey(["audio"])],
                        )
                    ],
                ),
                # Depends on frames and text
                ContainerSpec(
                    key=ContainerKey(["mixed"]),
                    code_version=12,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["multi"]),
                            containers=[
                                ContainerKey(["frames"]),
                                ContainerKey(["text"]),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    versions = {
        "multi": MultiContainer.data_version(),
        "selective": Selective.data_version(),
    }

    assert versions == snapshot


def test_complex_multi_level_graph(snapshot, registry: FeatureRegistry):
    """Test complex graph with multiple levels and mixed dependency types."""

    # Level 0: Two root features
    class RawVideoData(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["raw_video"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
                ContainerSpec(key=ContainerKey(["audio"]), code_version=1),
            ],
        ),
    ):
        pass

    class RawMetadata(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["raw_metadata"]),
            deps=None,
            containers=[ContainerSpec(key=ContainerKey(["info"]), code_version=1)],
        ),
    ):
        pass

    # Level 1: Process video
    class ProcessedVideo(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["processed_video"]),
            deps=[FeatureDep(key=FeatureKey(["raw_video"]))],
            containers=[
                ContainerSpec(
                    key=ContainerKey(["enhanced_frames"]),
                    code_version=5,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["raw_video"]),
                            containers=[ContainerKey(["frames"])],
                        )
                    ],
                ),
                ContainerSpec(
                    key=ContainerKey(["normalized_audio"]),
                    code_version=3,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["raw_video"]),
                            containers=[ContainerKey(["audio"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Level 2: Analysis combining multiple sources
    class Analysis(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["analysis"]),
            deps=[
                FeatureDep(key=FeatureKey(["processed_video"])),
                FeatureDep(key=FeatureKey(["raw_metadata"])),
            ],
            containers=[
                # Depends on all upstream
                ContainerSpec(
                    key=ContainerKey(["full_analysis"]),
                    code_version=10,
                    deps=SpecialContainerDep.ALL,
                ),
                # Depends only on enhanced frames and metadata
                ContainerSpec(
                    key=ContainerKey(["visual_metadata"]),
                    code_version=7,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["processed_video"]),
                            containers=[ContainerKey(["enhanced_frames"])],
                        ),
                        ContainerDep(
                            feature_key=FeatureKey(["raw_metadata"]),
                            containers=SpecialContainerDep.ALL,
                        ),
                    ],
                ),
            ],
        ),
    ):
        pass

    versions = {
        "raw_video": RawVideoData.data_version(),
        "raw_metadata": RawMetadata.data_version(),
        "processed_video": ProcessedVideo.data_version(),
        "analysis": Analysis.data_version(),
    }

    assert versions == snapshot


def test_code_version_changes_propagate(snapshot, registry: FeatureRegistry):
    """Test that changing code version in upstream affects downstream."""

    class Base(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["base"]),
            deps=None,
            containers=[ContainerSpec(key=ContainerKey(["data"]), code_version=100)],
        ),
    ):
        pass

    class Derived(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["derived"]),
            deps=[FeatureDep(key=FeatureKey(["base"]))],
            containers=[ContainerSpec(key=ContainerKey(["processed"]), code_version=1)],
        ),
    ):
        pass

    versions = {
        "base": Base.data_version(),
        "derived": Derived.data_version(),
    }

    assert versions == snapshot


def test_multiple_containers_different_deps(snapshot, registry: FeatureRegistry):
    """Test feature where different containers have completely different dependency sets."""

    class FeatureX(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["x"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["x1"]), code_version=1),
                ContainerSpec(key=ContainerKey(["x2"]), code_version=2),
            ],
        ),
    ):
        pass

    class FeatureY(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["y"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["y1"]), code_version=3),
                ContainerSpec(key=ContainerKey(["y2"]), code_version=4),
            ],
        ),
    ):
        pass

    class FeatureZ(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["z"]),
            deps=[
                FeatureDep(key=FeatureKey(["x"])),
                FeatureDep(key=FeatureKey(["y"])),
            ],
            containers=[
                # First container only depends on FeatureX
                ContainerSpec(
                    key=ContainerKey(["z1"]),
                    code_version=20,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["x"]),
                            containers=SpecialContainerDep.ALL,
                        )
                    ],
                ),
                # Second container only depends on FeatureY
                ContainerSpec(
                    key=ContainerKey(["z2"]),
                    code_version=21,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["y"]),
                            containers=SpecialContainerDep.ALL,
                        )
                    ],
                ),
                # Third container depends on both
                ContainerSpec(
                    key=ContainerKey(["z3"]),
                    code_version=22,
                    deps=SpecialContainerDep.ALL,
                ),
            ],
        ),
    ):
        pass

    versions = {
        "x": FeatureX.data_version(),
        "y": FeatureY.data_version(),
        "z": FeatureZ.data_version(),
    }

    # Verify that different containers have different versions
    z_versions = versions["z"]
    assert len(set(z_versions.values())) == 3, (
        "All containers should have different versions"
    )

    assert versions == snapshot
