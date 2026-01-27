"""Pre-defined features for docstring examples.

These features have stable import paths so they work with to_snapshot()/from_snapshot().
"""

from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec

# Define spec at module level for stable reference
_MY_FEATURE_SPEC = FeatureSpec(key="my/feature", id_columns=["id"])


class MyFeature(BaseFeature, spec=None):
    """Pre-populated feature for docstring examples.

    Available in all docstring examples as `MyFeature`.
    Has key "my/feature" and id column "id".
    """

    id: str


# Set up class attributes (bypassing metaclass project detection)
MyFeature._spec = _MY_FEATURE_SPEC
MyFeature.project = "docs"


def register_doctest_fixtures(graph: FeatureGraph) -> None:
    """Register doctest fixtures to the given graph.

    Args:
        graph: The FeatureGraph to register fixtures to.
    """
    MyFeature.graph = graph
    graph.add_feature(MyFeature)


__all__ = ["MyFeature", "register_doctest_fixtures"]
