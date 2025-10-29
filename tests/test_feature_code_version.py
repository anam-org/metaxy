"""Tests for Feature.code_version."""

from __future__ import annotations

from uuid import uuid4

from hypothesis import given
from hypothesis import strategies as st

from metaxy import Feature, FeatureDep, FeatureKey, FeatureSpec, FieldKey, FieldSpec


def _build_feature(
    *,
    key_suffix: str,
    fields: list[FieldSpec],
    deps: list[FeatureDep] | None = None,
) -> type[Feature]:
    feature_key = FeatureKey(["code_version", key_suffix])

    class _DynamicFeature(
        Feature,
        spec=FeatureSpec(
            key=feature_key,
            deps=deps,
            fields=fields,
        ),
    ):
        pass

    return _DynamicFeature


def test_code_version_single_field(graph) -> None:
    feature_cls = _build_feature(
        key_suffix="single_field",
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    value = feature_cls.code_version

    assert len(value) == 64
    assert all(c in "0123456789abcdef" for c in value)


def test_code_version_multiple_fields_order_invariant(graph) -> None:
    field_items = [
        (FieldKey(["frames"]), 1),
        (FieldKey(["audio"]), 2),
        (FieldKey(["metadata"]), 3),
    ]

    def build_fields(items: list[tuple[FieldKey, int]]) -> list[FieldSpec]:
        return [
            FieldSpec(key=key, code_version=code_version) for key, code_version in items
        ]

    ordered_feature = _build_feature(
        key_suffix="multi_ordered",
        fields=build_fields(field_items),
    )
    reversed_feature = _build_feature(
        key_suffix="multi_reversed",
        fields=build_fields(list(reversed(field_items))),
    )

    assert ordered_feature.code_version == reversed_feature.code_version


def test_code_version_changes_when_field_changes(graph) -> None:
    feature_v1 = _build_feature(
        key_suffix="field_change_v1",
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )
    feature_v2 = _build_feature(
        key_suffix="field_change_v2",
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
    )

    assert feature_v1.code_version != feature_v2.code_version


def test_code_version_ignores_dependency_versions(graph) -> None:
    parent_key = FeatureKey(["code_version", "parent"])

    class ParentV1(
        Feature,
        spec=FeatureSpec(
            key=parent_key,
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
    ):
        pass

    class ParentV2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["code_version", "parent_v2"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)],
        ),
    ):
        pass

    class ChildWithParentV1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["code_version", "child_v1"]),
            deps=[FeatureDep(key=parent_key)],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=7)],
        ),
    ):
        pass

    class ChildWithParentV2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["code_version", "child_v2"]),
            deps=[FeatureDep(key=ParentV2.spec.key)],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=7)],
        ),
    ):
        pass

    assert ChildWithParentV1.code_version == ChildWithParentV2.code_version


def test_code_version_deterministic_per_class(graph) -> None:
    feature_cls = _build_feature(
        key_suffix="deterministic",
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=3)],
    )

    first = feature_cls.code_version
    second = feature_cls().code_version
    third = feature_cls().code_version

    assert first == second == third


@given(
    st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=1,
        max_size=6,
    )
)
def test_code_version_consistent_across_field_order(
    graph, code_versions: list[int]
) -> None:
    field_items = [
        (FieldKey([f"field_{idx}"]), code_version)
        for idx, code_version in enumerate(code_versions)
    ]

    def build_fields(items: list[tuple[FieldKey, int]]) -> list[FieldSpec]:
        return [
            FieldSpec(key=field_key, code_version=code_version)
            for field_key, code_version in items
        ]

    unique_suffix = uuid4().hex
    ordered_feature = _build_feature(
        key_suffix=f"ordered_{unique_suffix}",
        fields=build_fields(field_items),
    )
    shuffled_feature = _build_feature(
        key_suffix=f"shuffled_{unique_suffix}",
        fields=build_fields(list(reversed(field_items))),
    )

    assert ordered_feature.code_version == shuffled_feature.code_version
