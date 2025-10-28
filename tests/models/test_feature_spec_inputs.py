from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.field import FieldDep


def test_feature_and_field_keys_accept_native_inputs() -> None:
    class UpstreamFeature(
        Feature, spec=FeatureSpec(key="coercion_upstream", deps=None)
    ):
        pass

    class DownstreamFeature(
        Feature,
        spec=FeatureSpec(
            key=["coercion", "downstream"],
            deps=[UpstreamFeature],
            fields=[
                FieldSpec(
                    key="result",
                    deps=[FieldDep(feature_key="coercion_upstream", fields=["result"])],
                )
            ],
        ),
    ):
        pass

    assert UpstreamFeature.spec.key == FeatureKey(["coercion_upstream"])
    assert DownstreamFeature.spec.key == FeatureKey(["coercion", "downstream"])

    dep = DownstreamFeature.spec.deps[0]
    assert dep.key == UpstreamFeature.spec.key

    field_spec = DownstreamFeature.spec.fields[0]
    assert field_spec.key == FieldKey(["result"])
    field_dep = field_spec.deps[0]
    assert field_dep.feature_key == FeatureKey(["coercion_upstream"])
    assert field_dep.fields[0] == FieldKey(["result"])


def test_feature_dep_coercion_from_multiple_sources() -> None:
    class NativeFeature(Feature, spec=FeatureSpec(key="coercion_native", deps=None)):
        pass

    dep_from_class = FeatureDep.model_validate(NativeFeature)
    dep_from_spec = FeatureDep.model_validate(NativeFeature.spec)
    dep_from_str = FeatureDep.model_validate("coercion_native")
    dep_from_list = FeatureDep.model_validate(["coercion", "native"])

    expected_key = FeatureKey(["coercion_native"])
    assert dep_from_class.key == expected_key
    assert dep_from_spec.key == expected_key
    assert dep_from_str.key == expected_key
    assert dep_from_list.key == FeatureKey(["coercion", "native"])
