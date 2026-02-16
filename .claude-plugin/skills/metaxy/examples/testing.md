# Testing Metaxy Code

See full documentation: https://anam-org.github.io/metaxy/guide/concepts/testing/

## Graph Isolation

Always use isolated graphs in tests to avoid polluting the global feature registry.

```python
import pytest
import metaxy as mx


@pytest.fixture(autouse=True)
def isolated_graph():
    with mx.FeatureGraph().use():
        yield


def test_my_feature(isolated_graph):
    class TestFeature(
        mx.BaseFeature,
        spec=mx.FeatureSpec(key="test/feature", id_columns=["id"], fields=["value"]),
    ):
        pass

    # Feature is registered to isolated graph, not global
    assert mx.FeatureGraph.get_active().get_feature("test/feature") is not None
```

## Configuration Isolation

```python
import metaxy as mx
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore


def test_with_custom_config(tmp_path):
    store = DeltaMetadataStore(root_path=tmp_path / "delta_test")
    with mx.MetaxyConfig(stores={"test": store}).use() as config:
        store = config.get_store("test")
        # test with isolated config and store
```

## Combined Fixture

```python
import pytest
import metaxy as mx
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore


@pytest.fixture
def metaxy_env(tmp_path):
    with mx.FeatureGraph().use():
        store = DeltaMetadataStore(root_path=tmp_path / "delta_test")
        with mx.MetaxyConfig(stores={"test": store}).use() as config:
            yield config


def test_feature_workflow(metaxy_env):
    store = metaxy_env.get_store("test")
    # define features and test
```
