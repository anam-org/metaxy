"""Reproducible benchmark for the JOSS paper.

Measures the wall-clock time of ``resolve_update`` on the embedded DuckDB
metadata store for a simple two-feature graph, across a range of record
counts. The benchmark exercises the full pipeline: SQL-side per-record
hashing, join with existing target metadata, and diff computation.

Run:

    uv run python publications/2026-introducing-metaxy/benchmark.py

Optional flags:

    --records 1_000,100_000,1_000_000   comma-separated list of N values
    --repeats 3                         runs per N (median is reported)
    --database :memory:                 DuckDB path (default: tempfile)
    --hash xxhash64                     hash algorithm
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Global Metaxy configuration not initialized.*",
)
warnings.filterwarnings("ignore", message="AUTO_CREATE_TABLES is enabled.*")

import polars as pl

import metaxy as mx
from metaxy.config import MetaxyConfig
from metaxy.ext.duckdb import DuckDBMetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.versioning.types import HashAlgorithm
from metaxy_testing import add_metaxy_provenance_column

MetaxyConfig.set(MetaxyConfig())


# ---------------------------------------------------------------------------
# Feature graph
# ---------------------------------------------------------------------------


def build_graph() -> tuple[type[mx.BaseFeature], type[mx.BaseFeature], mx.FeatureGraph]:
    graph = mx.FeatureGraph()
    with graph.use():

        class Root(
            mx.BaseFeature,
            spec=mx.FeatureSpec(
                key="bench/root",
                fields=[
                    mx.FieldSpec(key="audio", code_version="1"),
                    mx.FieldSpec(key="frames", code_version="1"),
                ],
                id_columns=("sample_uid",),
            ),
        ):
            pass

        class Leaf(
            mx.BaseFeature,
            spec=mx.FeatureSpec(
                key="bench/leaf",
                deps=[Root],
                fields=[mx.FieldSpec(key="prediction", code_version="1")],
                id_columns=("sample_uid",),
            ),
        ):
            pass

    return Root, Leaf, graph


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def make_upstream_df(n: int, seed: int, root_feature: type[mx.BaseFeature]) -> pl.DataFrame:
    rng = random.Random(seed)
    audio_versions = [f"a{rng.randint(0, 2**20)}" for _ in range(n)]
    frames_versions = [f"f{rng.randint(0, 2**20)}" for _ in range(n)]
    df = pl.DataFrame(
        {
            "sample_uid": list(range(n)),
            METAXY_PROVENANCE_BY_FIELD: [
                {"audio": a, "frames": f} for a, f in zip(audio_versions, frames_versions, strict=True)
            ],
        },
        schema={
            "sample_uid": pl.UInt64,
            METAXY_PROVENANCE_BY_FIELD: pl.Struct({"audio": pl.Utf8, "frames": pl.Utf8}),
        },
    )
    return add_metaxy_provenance_column(df, root_feature)


def mutate_fraction(df: pl.DataFrame, fraction: float, seed: int, root_feature: type[mx.BaseFeature]) -> pl.DataFrame:
    """Return a copy of df with `fraction` of the `audio` field versions bumped."""
    n = df.height
    rng = random.Random(seed)
    mutated = {i for i in range(n) if rng.random() < fraction}
    prov = df[METAXY_PROVENANCE_BY_FIELD].to_list()
    for i in mutated:
        prov[i] = {**prov[i], "audio": prov[i]["audio"] + "_bumped"}
    out = df.with_columns(pl.Series(METAXY_PROVENANCE_BY_FIELD, prov))
    return add_metaxy_provenance_column(out.drop("metaxy_provenance"), root_feature)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class Result:
    n: int
    resolve_new_s: list[float]
    resolve_stale_s: list[float]

    def median_new(self) -> float:
        return statistics.median(self.resolve_new_s)

    def median_stale(self) -> float:
        return statistics.median(self.resolve_stale_s)


def bench(
    n: int,
    repeats: int,
    database: str,
    hash_algorithm: HashAlgorithm,
    mutation_fraction: float,
) -> Result:
    Root, Leaf, graph = build_graph()
    result = Result(n=n, resolve_new_s=[], resolve_stale_s=[])

    for run in range(repeats):
        run_db = database if database == ":memory:" else f"{database}.{n}.{run}.duckdb"
        if run_db != ":memory:" and Path(run_db).exists():
            Path(run_db).unlink()

        store = DuckDBMetadataStore(
            database=run_db,
            auto_create_tables=True,
            hash_algorithm=hash_algorithm,
        )
        with graph.use(), store.open("w") as s:
            upstream = make_upstream_df(n, seed=42 + run, root_feature=Root)
            s.write(Root, upstream)

            t0 = time.perf_counter()
            inc = s.resolve_update(Leaf)
            leaf_new = inc.new.to_polars()
            t1 = time.perf_counter()
            result.resolve_new_s.append(t1 - t0)

            s.write(Leaf, leaf_new)

            mutated_upstream = mutate_fraction(upstream, mutation_fraction, seed=99 + run, root_feature=Root)
            s.write(Root, mutated_upstream)

            t0 = time.perf_counter()
            inc2 = s.resolve_update(Leaf)
            _ = (inc2.new.to_polars(), inc2.stale.to_polars())
            t1 = time.perf_counter()
            result.resolve_stale_s.append(t1 - t0)

        if run_db != ":memory:":
            Path(run_db).unlink(missing_ok=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default="10_000,100_000,1_000_000,10_000_000",
        help="Comma-separated list of record counts to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--database", default=None, help="DuckDB path (default: tempdir)")
    parser.add_argument("--hash", default="xxhash64", choices=["xxhash32", "xxhash64", "md5"])
    parser.add_argument("--mutation-fraction", type=float, default=0.1)
    parser.add_argument("--json", action="store_true", help="Emit JSON results to stdout.")
    args = parser.parse_args()

    ns = [int(x.replace("_", "")) for x in args.records.split(",")]
    hash_algorithm = HashAlgorithm(args.hash)
    database = args.database if args.database is not None else str(Path(tempfile.mkdtemp(prefix="metaxy_bench_")) / "db")

    results: list[Result] = []
    print(f"DuckDB benchmark | hash={hash_algorithm.value} | repeats={args.repeats} | mutation_fraction={args.mutation_fraction}")
    print(f"{'N':>12}  {'resolve_new (s)':>18}  {'resolve_stale (s)':>18}  {'rows/s (new)':>18}")
    for n in ns:
        r = bench(n, args.repeats, database, hash_algorithm, args.mutation_fraction)
        results.append(r)
        print(f"{n:>12,}  {r.median_new():>18.3f}  {r.median_stale():>18.3f}  {n / r.median_new():>18,.0f}")

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "n": r.n,
                        "resolve_new_s": r.resolve_new_s,
                        "resolve_stale_s": r.resolve_stale_s,
                        "median_new_s": r.median_new(),
                        "median_stale_s": r.median_stale(),
                    }
                    for r in results
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
