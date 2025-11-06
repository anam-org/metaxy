from collections.abc import Generator, Sequence
from dataclasses import dataclass, field

import narwhals as nw
import pytest
from narwhals.testing import assert_series_equal

from metaxy import BaseFeature, FeatureGraph, FeatureKey, FeatureSpec, FeatureDep
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

    def add_result(self, name: str, value: nw.DataFrame):
        self.results.append(Result(name=name, value=value))

    def assert_all_equal(self):
        for result1, result2 in zip(self.results, self.results[1:]):
            assert result1 == result2, (
                f"Results for {result1.name} and {result2.name} are not equal:\n{result1.value}\n{result2.value}"
            )


@pytest.fixture
def features(graph: FeatureGraph) -> dict[str, type[BaseFeature]]:
    class Upstream(BaseFeature, spec=FeatureSpec(key="upstream", id_columns=["id"])):
        id: str

    class Downstream(
        BaseFeature,
        spec=FeatureSpec(
            key="downstream",
            id_columns=["id"],
            deps=[FeatureDep(feature=FeatureKey("upstream"))],
        ),
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
    return NarwhalsSeriesCollection()


@pytest.fixture
def df():
    import polars as pl

    return nw.from_native(pl.DataFrame({"id": ["a", "b", "c"]}))


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
    hash_algo: HashAlgorithm,
    features: dict[str, type[BaseFeature]],
    graph: FeatureGraph,
):
    for tracker in trackers(graph, features):
        results.add_result(
            name=f"{hash_algo}_{tracker.__class__.__name__}",
            value=tracker.hash_string_column(
                df, "id", "hash", hash_algo=hash_algo
            ),
        )

    results.assert_all_equal()


def test_build_struct_column(
    df: nw.DataFrame,
    results: NarwhalsSeriesCollection,
    features: dict[str, type[BaseFeature]],
    graph: FeatureGraph,
):
    """Test that build_struct_column creates a struct column from existing columns."""
    # Add an extra column to create struct from
    df_with_extra = df.with_columns(nw.col("id").str.to_uppercase().alias("id_upper"))

    # Mapping from struct field names to column names
    field_columns = {"original": "id", "uppercase": "id_upper"}

    for tracker in trackers(graph, features):
        results.add_result(
            name=f"struct_{tracker.__class__.__name__}",
            value=tracker.build_struct_column(
                df_with_extra, "my_struct", field_columns
            ),
        )

    results.assert_all_equal()
