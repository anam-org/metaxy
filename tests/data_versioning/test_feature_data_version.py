from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.field import FieldDep, FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureKey, FieldKey


def test_single_feature_provenance(snapshot, graph: FeatureGraph):
    """Test feature with no dependencies."""

    class MyFeature(
        SampleFeature,
        spec=SampleFeatureSpec(key=FeatureKey(["my_feature"])),
    ):
        pass

    assert MyFeature.provenance_by_field() == snapshot


def test_feature_with_multiple_fields(snapshot, graph: FeatureGraph):
    """Test feature with multiple fields, each with different code versions."""

    class VideoFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                FieldSpec(key=FieldKey(["metadata"]), code_version="5"),
            ],
        ),
    ):
        pass

    assert VideoFeature.provenance_by_field() == snapshot


def test_linear_dependency_chain(snapshot, graph: FeatureGraph):
    """Test A -> B -> C linear dependency chain."""

    class FeatureA(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["a"]),
            fields=[FieldSpec(key=FieldKey(["raw"]), code_version="1")],
        ),
    ):
        pass

    class FeatureB(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["b"]),
            deps=[FeatureDep(feature=FeatureKey(["a"]))],
            fields=[FieldSpec(key=FieldKey(["processed"]), code_version="2")],
        ),
    ):
        pass

    class FeatureC(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["c"]),
            deps=[FeatureDep(feature=FeatureKey(["b"]))],
            fields=[FieldSpec(key=FieldKey(["final"]), code_version="3")],
        ),
    ):
        pass

    versions = {
        "a": FeatureA.provenance_by_field(),
        "b": FeatureB.provenance_by_field(),
        "c": FeatureC.provenance_by_field(),
    }

    assert versions == snapshot


def test_diamond_dependency_graph(snapshot, graph: FeatureGraph):
    """Test diamond dependency: A -> B, A -> C, B -> D, C -> D."""

    class Root(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["root"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    class BranchLeft(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["branch_left"]),
            deps=[FeatureDep(feature=FeatureKey(["root"]))],
            fields=[FieldSpec(key=FieldKey(["left_processed"]), code_version="2")],
        ),
    ):
        pass

    class BranchRight(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["branch_right"]),
            deps=[FeatureDep(feature=FeatureKey(["root"]))],
            fields=[FieldSpec(key=FieldKey(["right_processed"]), code_version="3")],
        ),
    ):
        pass

    class Merged(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["merged"]),
            deps=[
                FeatureDep(feature=FeatureKey(["branch_left"])),
                FeatureDep(feature=FeatureKey(["branch_right"])),
            ],
            fields=[FieldSpec(key=FieldKey(["fusion"]), code_version="4")],
        ),
    ):
        pass

    versions = {
        "root": Root.provenance_by_field(),
        "branch_left": BranchLeft.provenance_by_field(),
        "branch_right": BranchRight.provenance_by_field(),
        "merged": Merged.provenance_by_field(),
    }

    assert versions == snapshot


def test_specific_field_dependencies(snapshot, graph: FeatureGraph):
    """Test feature with specific field-level dependencies."""

    class MultiField(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["multi"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                FieldSpec(key=FieldKey(["text"]), code_version="3"),
            ],
        ),
    ):
        pass

    class Selective(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["selective"]),
            deps=[FeatureDep(feature=FeatureKey(["multi"]))],
            fields=[
                # Only depends on frames
                FieldSpec(
                    key=FieldKey(["visual"]),
                    code_version="10",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["multi"]),
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
                            feature=FeatureKey(["multi"]),
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
                            feature=FeatureKey(["multi"]),
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
        "multi": MultiField.provenance_by_field(),
        "selective": Selective.provenance_by_field(),
    }

    assert versions == snapshot


def test_complex_multi_level_graph(snapshot, graph: FeatureGraph):
    """Test complex graph with multiple levels and mixed dependency types."""

    # Level 0: Two root features
    class RawVideoData(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["raw_video"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    class RawMetadata(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["raw_metadata"]),
            fields=[FieldSpec(key=FieldKey(["info"]), code_version="1")],
        ),
    ):
        pass

    # Level 1: Process video
    class ProcessedVideo(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["processed_video"]),
            deps=[FeatureDep(feature=FeatureKey(["raw_video"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["enhanced_frames"]),
                    code_version="5",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["raw_video"]),
                            fields=[FieldKey(["frames"])],
                        )
                    ],
                ),
                FieldSpec(
                    key=FieldKey(["normalized_audio"]),
                    code_version="3",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["raw_video"]),
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
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["analysis"]),
            deps=[
                FeatureDep(feature=FeatureKey(["processed_video"])),
                FeatureDep(feature=FeatureKey(["raw_metadata"])),
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
                            feature=FeatureKey(["processed_video"]),
                            fields=[FieldKey(["enhanced_frames"])],
                        ),
                        FieldDep(
                            feature=FeatureKey(["raw_metadata"]),
                            fields=SpecialFieldDep.ALL,
                        ),
                    ],
                ),
            ],
        ),
    ):
        pass

    versions = {
        "raw_video": RawVideoData.provenance_by_field(),
        "raw_metadata": RawMetadata.provenance_by_field(),
        "processed_video": ProcessedVideo.provenance_by_field(),
        "analysis": Analysis.provenance_by_field(),
    }

    assert versions == snapshot


def test_code_version_changes_propagate(snapshot, graph: FeatureGraph):
    """Test that changing code version in upstream affects downstream."""

    class Base(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["base"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="100")],
        ),
    ):
        pass

    class Derived(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["derived"]),
            deps=[FeatureDep(feature=FeatureKey(["base"]))],
            fields=[FieldSpec(key=FieldKey(["processed"]), code_version="1")],
        ),
    ):
        pass

    versions = {
        "base": Base.provenance_by_field(),
        "derived": Derived.provenance_by_field(),
    }

    assert versions == snapshot


def test_multiple_fields_different_deps(snapshot, graph: FeatureGraph):
    """Test feature where different fields have completely different dependency sets."""

    class FeatureX(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["x"]),
            fields=[
                FieldSpec(key=FieldKey(["x1"]), code_version="1"),
                FieldSpec(key=FieldKey(["x2"]), code_version="2"),
            ],
        ),
    ):
        pass

    class FeatureY(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["y"]),
            fields=[
                FieldSpec(key=FieldKey(["y1"]), code_version="3"),
                FieldSpec(key=FieldKey(["y2"]), code_version="4"),
            ],
        ),
    ):
        pass

    class FeatureZ(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["z"]),
            deps=[
                FeatureDep(feature=FeatureKey(["x"])),
                FeatureDep(feature=FeatureKey(["y"])),
            ],
            fields=[
                # First field only depends on FeatureX
                FieldSpec(
                    key=FieldKey(["z1"]),
                    code_version="20",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["x"]),
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
                            feature=FeatureKey(["y"]),
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
        "x": FeatureX.provenance_by_field(),
        "y": FeatureY.provenance_by_field(),
        "z": FeatureZ.provenance_by_field(),
    }

    # Verify that different fields have different versions
    z_versions = versions["z"]
    assert len(set(z_versions.values())) == 3, "All fields should have different versions"

    assert versions == snapshot
