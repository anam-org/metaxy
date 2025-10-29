from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldDep, FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureKey, FieldKey


def test_single_feature_data_version(snapshot, graph: FeatureGraph):
    """Test feature with no dependencies."""

    class MyFeature(
        Feature, spec=FeatureSpec(key=FeatureKey(["my_feature"]), deps=None)
    ):
        pass

    assert MyFeature.data_version() == snapshot


def test_feature_with_multiple_fields(snapshot, graph: FeatureGraph):
    """Test feature with multiple fields, each with different code versions."""

    class VideoFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["video"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                FieldSpec(key=FieldKey(["metadata"]), code_version="5"),
            ],
        ),
    ):
        pass

    assert VideoFeature.data_version() == snapshot


def test_linear_dependency_chain(snapshot, graph: FeatureGraph):
    """Test A -> B -> C linear dependency chain."""

    class FeatureA(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["a"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["raw"]), code_version="1")],
        ),
    ):
        pass

    class FeatureB(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["b"]),
            deps=[FeatureDep(key=FeatureKey(["a"]))],
            fields=[FieldSpec(key=FieldKey(["processed"]), code_version="2")],
        ),
    ):
        pass

    class FeatureC(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["c"]),
            deps=[FeatureDep(key=FeatureKey(["b"]))],
            fields=[FieldSpec(key=FieldKey(["final"]), code_version="3")],
        ),
    ):
        pass

    versions = {
        "a": FeatureA.data_version(),
        "b": FeatureB.data_version(),
        "c": FeatureC.data_version(),
    }

    assert versions == snapshot


def test_diamond_dependency_graph(snapshot, graph: FeatureGraph):
    """Test diamond dependency: A -> B, A -> C, B -> D, C -> D."""

    class Root(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["root"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    class BranchLeft(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["branch_left"]),
            deps=[FeatureDep(key=FeatureKey(["root"]))],
            fields=[FieldSpec(key=FieldKey(["left_processed"]), code_version="2")],
        ),
    ):
        pass

    class BranchRight(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["branch_right"]),
            deps=[FeatureDep(key=FeatureKey(["root"]))],
            fields=[FieldSpec(key=FieldKey(["right_processed"]), code_version="3")],
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
            fields=[FieldSpec(key=FieldKey(["fusion"]), code_version="4")],
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


def test_specific_field_dependencies(snapshot, graph: FeatureGraph):
    """Test feature with specific field-level dependencies."""

    class MultiField(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["multi"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                FieldSpec(key=FieldKey(["text"]), code_version="3"),
            ],
        ),
    ):
        pass

    class Selective(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["selective"]),
            deps=[FeatureDep(key=FeatureKey(["multi"]))],
            fields=[
                # Only depends on frames
                FieldSpec(
                    key=FieldKey(["visual"]),
                    code_version="10",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["multi"]),
                            fields=[FieldKey(["frames"])],
                        )
                    ],
                ),
                # Only depends on audio
                FieldSpec(
                    key=FieldKey(["audio_only"]),
                    code_version="11",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["multi"]),
                            fields=[FieldKey(["audio"])],
                        )
                    ],
                ),
                # Depends on frames and text
                FieldSpec(
                    key=FieldKey(["mixed"]),
                    code_version="12",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["multi"]),
                            fields=[
                                FieldKey(["frames"]),
                                FieldKey(["text"]),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    versions = {
        "multi": MultiField.data_version(),
        "selective": Selective.data_version(),
    }

    assert versions == snapshot


def test_complex_multi_level_graph(snapshot, graph: FeatureGraph):
    """Test complex graph with multiple levels and mixed dependency types."""

    # Level 0: Two root features
    class RawVideoData(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["raw_video"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    class RawMetadata(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["raw_metadata"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["info"]), code_version="1")],
        ),
    ):
        pass

    # Level 1: Process video
    class ProcessedVideo(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["processed_video"]),
            deps=[FeatureDep(key=FeatureKey(["raw_video"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["enhanced_frames"]),
                    code_version="5",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["raw_video"]),
                            fields=[FieldKey(["frames"])],
                        )
                    ],
                ),
                FieldSpec(
                    key=FieldKey(["normalized_audio"]),
                    code_version="3",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["raw_video"]),
                            fields=[FieldKey(["audio"])],
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
            fields=[
                # Depends on all upstream
                FieldSpec(
                    key=FieldKey(["full_analysis"]),
                    code_version="10",
                    deps=SpecialFieldDep.ALL,
                ),
                # Depends only on enhanced frames and metadata
                FieldSpec(
                    key=FieldKey(["visual_metadata"]),
                    code_version="7",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["processed_video"]),
                            fields=[FieldKey(["enhanced_frames"])],
                        ),
                        FieldDep(
                            feature_key=FeatureKey(["raw_metadata"]),
                            fields=SpecialFieldDep.ALL,
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


def test_code_version_changes_propagate(snapshot, graph: FeatureGraph):
    """Test that changing code version in upstream affects downstream."""

    class Base(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["base"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="100")],
        ),
    ):
        pass

    class Derived(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["derived"]),
            deps=[FeatureDep(key=FeatureKey(["base"]))],
            fields=[FieldSpec(key=FieldKey(["processed"]), code_version="1")],
        ),
    ):
        pass

    versions = {
        "base": Base.data_version(),
        "derived": Derived.data_version(),
    }

    assert versions == snapshot


def test_multiple_fields_different_deps(snapshot, graph: FeatureGraph):
    """Test feature where different fields have completely different dependency sets."""

    class FeatureX(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["x"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["x1"]), code_version="1"),
                FieldSpec(key=FieldKey(["x2"]), code_version="2"),
            ],
        ),
    ):
        pass

    class FeatureY(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["y"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["y1"]), code_version="3"),
                FieldSpec(key=FieldKey(["y2"]), code_version="4"),
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
            fields=[
                # First field only depends on FeatureX
                FieldSpec(
                    key=FieldKey(["z1"]),
                    code_version="20",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["x"]),
                            fields=SpecialFieldDep.ALL,
                        )
                    ],
                ),
                # Second field only depends on FeatureY
                FieldSpec(
                    key=FieldKey(["z2"]),
                    code_version="21",
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["y"]),
                            fields=SpecialFieldDep.ALL,
                        )
                    ],
                ),
                # Third field depends on both
                FieldSpec(
                    key=FieldKey(["z3"]),
                    code_version="22",
                    deps=SpecialFieldDep.ALL,
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

    # Verify that different fields have different versions
    z_versions = versions["z"]
    assert len(set(z_versions.values())) == 3, (
        "All fields should have different versions"
    )

    assert versions == snapshot
