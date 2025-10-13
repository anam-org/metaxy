from metaxy import Feature, FeatureKey, FeatureSpec


def test_single_feature_data_version():
    class MyFeature(
        Feature, spec=FeatureSpec(key=FeatureKey(["my_feature"]), deps=None)
    ): ...

    assert MyFeature.data_version() == {"default": "asd"}
