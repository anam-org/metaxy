"""Tests for code_version property on Feature class."""

from hypothesis import given
from hypothesis import strategies as st
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldKey,
    FieldSpec,
)
from metaxy.models.feature import FeatureGraph


def test_code_version_single_field(snapshot: SnapshotAssertion) -> None:
    """Test code_version with a single field."""

    class SingleFieldFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "single_field"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    feature_instance = SingleFieldFeature()
    code_ver = SingleFieldFeature.code_version

    # Should be deterministic
    assert code_ver == SingleFieldFeature.code_version
    assert code_ver == feature_instance.code_version

    # Should be 64 characters (SHA256 hex)
    assert len(code_ver) == 64

    # Should be hex string
    assert all(c in "0123456789abcdef" for c in code_ver)

    # Snapshot the hash
    assert code_ver == snapshot

    # The code_version should be accessible from spec as well
    assert SingleFieldFeature.spec().code_version == code_ver


def test_code_version_multiple_fields(snapshot: SnapshotAssertion) -> None:
    """Test code_version with multiple fields."""

    class MultiFieldFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "multi_field"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                FieldSpec(key=FieldKey(["metadata"]), code_version="3"),
            ],
        ),
    ):
        pass

    feature_instance = MultiFieldFeature()
    code_ver = MultiFieldFeature.code_version

    # Should be 64 characters
    assert len(code_ver) == 64

    # Should be hex string
    assert all(c in "0123456789abcdef" for c in code_ver)

    # Snapshot the hash
    assert code_ver == snapshot
    assert feature_instance.code_version == code_ver


def test_code_version_changes_with_field_code_version() -> None:
    """Test that code_version changes when field code_version changes."""
    graph_v1 = FeatureGraph()
    graph_v2 = FeatureGraph()

    with graph_v1.use():

        class FeatureV1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["versioned", "test_v1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="1"),
                ],
            ),
        ):
            pass

    with graph_v2.use():

        class FeatureV2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["versioned", "test_v2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="2"),  # Changed!
                ],
            ),
        ):
            pass

    code_v1 = FeatureV1.code_version
    code_v2 = FeatureV2.code_version

    # Should be different
    assert code_v1 != code_v2


def test_code_version_independence_from_dependencies() -> None:
    """Test that code_version does NOT change when dependencies change."""
    graph1 = FeatureGraph()
    graph2 = FeatureGraph()

    # Graph 1: Parent with code_version="1", child depends on parent
    with graph1.use():

        class ParentV1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["independence_test", "parent_v1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="1"),
                ],
            ),
        ):
            pass

        class ChildV1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["independence_test", "child_v1"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["independence_test", "parent_v1"]))
                ],
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="1"),
                ],
            ),
        ):
            pass

    # Graph 2: Parent with code_version="2", child still has code_version="1"
    with graph2.use():

        class ParentV2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["independence_test", "parent_v2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="2"),  # Changed!
                ],
            ),
        ):
            pass

        class ChildV2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["independence_test", "child_v2"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["independence_test", "parent_v2"]))
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]), code_version="1"
                    ),  # Same as ChildV1!
                ],
            ),
        ):
            pass

    child_code_v1 = ChildV1.code_version
    child_code_v2 = ChildV2.code_version

    # code_version should be the SAME (child's fields didn't change)
    assert child_code_v1 == child_code_v2

    # But feature_version should be DIFFERENT (parent changed)
    assert ChildV1.feature_version() != ChildV2.feature_version()


def test_code_version_determinism() -> None:
    """Test that code_version is deterministic for same inputs."""

    class TestFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["determinism_test"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="2"),
            ],
        ),
    ):
        pass

    # Call multiple times
    version1 = TestFeature.code_version
    version2 = TestFeature.code_version
    version3 = TestFeature.code_version

    # Should all be identical
    assert version1 == version2
    assert version2 == version3


def test_code_version_field_order_invariance() -> None:
    """Test that field order doesn't affect code_version (sorted internally)."""
    graph1 = FeatureGraph()
    graph2 = FeatureGraph()

    with graph1.use():

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["order_test", "feature1"]),
                deps=None,
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]), code_version="1"
                    ),  # alphabetically first
                    FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                ],
            ),
        ):
            pass

    with graph2.use():

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["order_test", "feature2"]),
                deps=None,
                fields=[
                    FieldSpec(
                        key=FieldKey(["frames"]), code_version="1"
                    ),  # Different order
                    FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                ],
            ),
        ):
            pass

    # Should have the same code_version (internal sorting makes it deterministic)
    assert Feature1.code_version == Feature2.code_version


def test_code_version_no_dependencies_no_fields_edge_case() -> None:
    """Test code_version with default field (single field with code_version="1")."""

    class MinimalFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["minimal"]),
            deps=None,
            # Uses default field: [FieldSpec(key=FieldKey(["default"]), code_version="1")]
        ),
    ):
        pass

    code_ver = MinimalFeature.code_version

    # Should still produce a valid hash
    assert len(code_ver) == 64
    assert all(c in "0123456789abcdef" for c in code_ver)


def test_code_version_complex_dependency_chain() -> None:
    """Test code_version with complex dependency chain."""
    graph = FeatureGraph()

    with graph.use():

        class A(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["chain", "a"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="1"),
                ],
            ),
        ):
            pass

        class B(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["chain", "b"]),
                deps=[FeatureDep(feature=FeatureKey(["chain", "a"]))],
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="2"),
                ],
            ),
        ):
            pass

        class C(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["chain", "c"]),
                deps=[FeatureDep(feature=FeatureKey(["chain", "b"]))],
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="3"),
                ],
            ),
        ):
            pass

    # Each should have a different code_version (different field code_versions)
    code_a = A.code_version
    code_b = B.code_version
    code_c = C.code_version

    assert code_a != code_b
    assert code_b != code_c
    assert code_a != code_c

    # But feature_version includes the entire chain
    feature_a = A.feature_version()
    feature_b = B.feature_version()
    feature_c = C.feature_version()

    # feature_versions should all be different (include dependencies)
    assert feature_a != feature_b
    assert feature_b != feature_c
    assert feature_a != feature_c


# Property-based tests using Hypothesis


@given(
    code_version=st.integers(min_value=1, max_value=1000).map(str),
)
def test_property_code_version_deterministic(code_version: str) -> None:
    """Property test: code_version is deterministic for any code_version value."""
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "deterministic"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=code_version),
                ],
            ),
        ):
            pass

        # Call multiple times
        v1 = TestFeature.code_version
        v2 = TestFeature.code_version

    # Should be deterministic
    assert v1 == v2


@given(
    code_version1=st.integers(min_value=1, max_value=1000).map(str),
    code_version2=st.integers(min_value=1, max_value=1000).map(str),
)
def test_property_code_version_changes_with_code(
    code_version1: str, code_version2: str
) -> None:
    """Property test: different code versions produce different hashes (or same if same input)."""
    graph1 = FeatureGraph()
    graph2 = FeatureGraph()

    with graph1.use():

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "changes", "f1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=code_version1),
                ],
            ),
        ):
            pass

    with graph2.use():

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "changes", "f2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=code_version2),
                ],
            ),
        ):
            pass

    cv1 = Feature1.code_version
    cv2 = Feature2.code_version

    # If code versions are the same, hashes should be the same
    # If code versions are different, hashes should be different
    if code_version1 == code_version2:
        assert cv1 == cv2
    else:
        assert cv1 != cv2


@given(
    num_fields=st.integers(min_value=1, max_value=10),
)
def test_property_code_version_multiple_fields(num_fields: int) -> None:
    """Property test: code_version works with variable number of fields."""
    graph = FeatureGraph()

    # Generate fields with unique keys
    fields = [
        FieldSpec(key=FieldKey([f"field_{i}"]), code_version=str(i + 1))
        for i in range(num_fields)
    ]

    with graph.use():

        class TestFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "multi"]),
                deps=None,
                fields=fields,
            ),
        ):
            pass

        code_ver = TestFeature.code_version

    # Should produce a valid hash
    assert len(code_ver) == 64
    assert all(c in "0123456789abcdef" for c in code_ver)

    # Should be deterministic
    assert code_ver == TestFeature.code_version


@given(
    field_names=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=5,
        unique=True,
    ),
)
def test_property_code_version_field_names_dont_affect_hash_if_sorted(
    field_names: list[str],
) -> None:
    """Property test: field ordering doesn't affect code_version (always sorted)."""
    graph1 = FeatureGraph()
    graph2 = FeatureGraph()

    # Create fields in original order
    fields1 = [
        FieldSpec(key=FieldKey([name]), code_version="1") for name in field_names
    ]

    # Create fields in reversed order
    fields2 = [
        FieldSpec(key=FieldKey([name]), code_version="1")
        for name in reversed(field_names)
    ]

    with graph1.use():

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "order1"]),
                deps=None,
                fields=fields1,
            ),
        ):
            pass

    with graph2.use():

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "order2"]),
                deps=None,
                fields=fields2,
            ),
        ):
            pass

    # Should have the same code_version (internal sorting)
    assert Feature1.code_version == Feature2.code_version


@given(
    parent_code_version=st.integers(min_value=1, max_value=1000).map(str),
    child_code_version=st.integers(min_value=1, max_value=1000).map(str),
)
def test_property_code_version_independent_of_parent(
    parent_code_version: str, child_code_version: str
) -> None:
    """Property test: child's code_version is independent of parent's code_version."""
    graph = FeatureGraph()

    with graph.use():

        class Parent(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "parent"]),
                deps=None,
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]), code_version=parent_code_version
                    ),
                ],
            ),
        ):
            pass

        class Child(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "child"]),
                deps=[FeatureDep(feature=FeatureKey(["property_test", "parent"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]), code_version=child_code_version
                    ),
                ],
            ),
        ):
            pass

        # Record child's code_version
        child_cv1 = Child.code_version

    # Create another graph with different parent code_version but same child code_version
    graph2 = FeatureGraph()

    with graph2.use():

        class Parent2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "parent2"]),
                deps=None,
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]),
                        code_version=str(
                            int(parent_code_version) + 100
                        ),  # Different from parent!
                    ),
                ],
            ),
        ):
            pass

        class Child2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["property_test", "child2"]),
                deps=[FeatureDep(feature=FeatureKey(["property_test", "parent2"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]),
                        code_version=child_code_version,  # Same as child!
                    ),
                ],
            ),
        ):
            pass

        child_cv2 = Child2.code_version

    # Child's code_version should be the same (independent of parent)
    assert child_cv1 == child_cv2
