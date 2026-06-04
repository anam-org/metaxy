"""Feature graph builders shared by conftest and test_benchmark.

Returns fully assembled [`FeatureGraph`][metaxy.FeatureGraph] objects
wrapped in :class:`BenchGraph`, so tests do not assemble graphs themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import metaxy as mx

WIDE_FANIN = 2
WIDE_FIELDS = ("audio", "frames", "optical_flow", "embedding")


@dataclass
class BenchGraph:
    """A ready FeatureGraph with direct references to its features."""

    graph: mx.FeatureGraph
    roots: list[type[mx.BaseFeature]]
    leaf: type[mx.BaseFeature]


def build_simple_graph() -> BenchGraph:
    graph = mx.FeatureGraph()
    with graph.use():

        class SimpleRoot(
            mx.BaseFeature,
            spec=mx.FeatureSpec(
                key="bench/simple_root",
                fields=[
                    mx.FieldSpec(key="audio", code_version="1"),
                    mx.FieldSpec(key="frames", code_version="1"),
                ],
                id_columns=("sample_uid",),
            ),
        ):
            pass

        class SimpleLeaf(
            mx.BaseFeature,
            spec=mx.FeatureSpec(
                key="bench/simple_leaf",
                deps=[SimpleRoot],
                fields=[mx.FieldSpec(key="prediction", code_version="1")],
                id_columns=("sample_uid",),
            ),
        ):
            pass

    return BenchGraph(graph=graph, roots=[SimpleRoot], leaf=SimpleLeaf)


def _make_wide_root(i: int, fields: tuple[str, ...]) -> type[mx.BaseFeature]:
    class WideRoot(
        mx.BaseFeature,
        spec=mx.FeatureSpec(
            key=f"bench/wide_root_{i}",
            fields=[mx.FieldSpec(key=f, code_version="1") for f in fields],
            id_columns=("sample_uid",),
        ),
    ):
        pass

    WideRoot.__name__ = f"WideRoot_{i}"
    return WideRoot


def build_wide_graph(
    fanin: int = WIDE_FANIN, fields: tuple[str, ...] = WIDE_FIELDS
) -> BenchGraph:
    graph = mx.FeatureGraph()
    roots: list[type[mx.BaseFeature]] = []
    with graph.use():
        for i in range(fanin):
            roots.append(_make_wide_root(i, fields))

        class WideLeaf(
            mx.BaseFeature,
            spec=mx.FeatureSpec(
                key="bench/wide_leaf",
                deps=[mx.FeatureDep(feature=r) for r in roots],
                fields=[
                    mx.FieldSpec(key="prediction_a", code_version="1"),
                    mx.FieldSpec(key="prediction_b", code_version="1"),
                ],
                id_columns=("sample_uid",),
            ),
        ):
            pass

    return BenchGraph(graph=graph, roots=roots, leaf=WideLeaf)
