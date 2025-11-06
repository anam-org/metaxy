from collections.abc import Generator, Sequence
from dataclasses import dataclass, field

import narwhals as nw
import pytest
from narwhals.testing import assert_series_equal

from metaxy import BaseFeature, FeatureGraph, FeatureKey, FeatureSpec
from metaxy.provenance import ProvenanceTracker
from metaxy.provenance.polars import PolarsProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


@dataclass(frozen=True, kw_only=True)
class HashParams:
    algo: HashAlgorithm
    length: int


@dataclass(frozen=True, kw_only=True)
class Result:
    name: str
    value: nw.DataFrame

    def __eq__(self, value: object, /) -> bool:
        for col in self.value.columns:
            assert_series_equal(self.value.get_column(col), value.value.get_column(col))
        return True


@dataclass
class NarwhalsSeriesCollection:
    results: list[Result] = field(default_factory=list)

    def add_result(self, name: str, series: Sequence[nw.Series]):
        self.results.append(Result(name=name, value=series))

    def assert_all_equal(self):
        for result1, result2 in zip(self.results, self.results[1:]):
            assert result1 == result2, (
                f"Results for {result1.name} and {result2.name} are not equal:\n{result1.value}\n{result2.value}"
            )


@pytest.fixture(scope="module")
def features(graph: FeatureGraph) -> dict[str, type[BaseFeature]]:
    class Upstream(BaseFeature, spec=FeatureSpec(key="upstream", id_columns=["id"])):
        id: str

    class Downstream(
        BaseFeature,
        spec=FeatureSpec(key="downstream", id_columns=["id"], deps=[Upstream]),
    ):
        id: str

    return {
        "upstream": Upstream,
        "downstream": Downstream,
    }


def trackers(
    graph: FeatureGraph,
    features: dict[str, type[BaseFeature]],
) -> Generator[ProvenanceTracker, None, None]:
    plan = graph.get_feature_plan(FeatureKey("downstream"))
    yield PolarsProvenanceTracker(plan)
    yield PolarsProvenanceTracker(plan)


@pytest.fixture
def results() -> NarwhalsSeriesCollection:
    return NarwhalsSeriesCollection({})


@pytest.fixture
def df():
    return nw.DataFrame({"id": ["a", "b", "c"]})


@pytest.mark.parametrize("hash_length", [4, 16])
@pytest.mark.parametrize(
    "hash_algo",
    [
        HashAlgorithm.MD5,
        HashAlgorithm.SHA256,
        HashAlgorithm.XXHASH32,
        HashAlgorithm.XXHASH64,
    ],
)
def test_hash_string(
    df: nw.DataFrame,
    results: NarwhalsSeriesCollection,
    hash_length: int,
    hash_algo: HashAlgorithm,
    features: dict[str, type[BaseFeature]],
    graph: FeatureGraph,
):
    for tracker in trackers(graph, features):
        results.add_result(
            name=f"{hash_algo}_{hash_length}_{tracker.__class__.__name__}",
            result=tracker.hash_string_column(
                df, "id", "hash", hash_algo=hash_algo, hash_length=hash_length
            ),
        )

    assert results.all_equal()
