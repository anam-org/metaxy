"""Reproducible benchmark for the JOSS paper.

Measures wall-clock time of [`resolve_update`][metaxy.MetadataStore.resolve_update]
on an embedded DuckDB metadata store across two feature graphs:

* ``simple``  Root(2 fields) -> Leaf(1 field)
* ``wide``    [Root_0, Root_1](4 fields each) -> Leaf(2 fields), joining both

Fixtures (see ``conftest.py``) parametrise the scenario and record count,
pre-build upstream DataFrames once per test, and provide a ``make_setup``
factory that turns a seeding function into a ``pytest-benchmark`` ``setup``
callback. Every timed round therefore starts from a cold DuckDB file in a
fresh temp directory.

Usage::

    uv run pytest publications/2026-introducing-metaxy/test_benchmark.py --benchmark-only
    uv run pytest ... --benchmark-only --benchmark-json=bench.json
    METAXY_BENCH_ROUNDS=3 uv run pytest ... --benchmark-only -k "simple and 10_000"
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import polars as pl
import pytest

import metaxy as mx
from metaxy.ext.duckdb import DuckDBMetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.utils import collect_to_polars
from metaxy.versioning.types import HashAlgorithm
from metaxy_testing import add_metaxy_provenance_column

from _graphs import BenchGraph  # pyright: ignore[reportMissingImports]

HASH = HashAlgorithm.XXHASH64
MUTATION_FRACTION = 0.1
ROUNDS = int(os.environ.get("METAXY_BENCH_ROUNDS", "10"))


# ---------------------------------------------------------------------------
# Synthetic data (public: used from fixtures in conftest.py)
# ---------------------------------------------------------------------------


def _field_names(root: type[mx.BaseFeature]) -> list[str]:
    return [str(f.key) for f in root.spec().fields]


def make_upstream_df(n: int, seed: int, root: type[mx.BaseFeature]) -> pl.DataFrame:
    rng = random.Random(seed)
    field_names = _field_names(root)
    versions = {f: [f"{f[0]}{rng.randint(0, 2**20)}" for _ in range(n)] for f in field_names}
    provenance = [{f: versions[f][i] for f in field_names} for i in range(n)]
    df = pl.DataFrame(
        {"sample_uid": list(range(n)), METAXY_PROVENANCE_BY_FIELD: provenance},
        schema={
            "sample_uid": pl.UInt64,
            METAXY_PROVENANCE_BY_FIELD: pl.Struct({f: pl.Utf8 for f in field_names}),
        },
    )
    return add_metaxy_provenance_column(df, root)


def _mutate_fraction(
    df: pl.DataFrame, fraction: float, seed: int, root: type[mx.BaseFeature]
) -> pl.DataFrame:
    rng = random.Random(seed)
    original_dtype = df.schema[METAXY_PROVENANCE_BY_FIELD]
    prov = df[METAXY_PROVENANCE_BY_FIELD].to_list()
    for i in range(df.height):
        if rng.random() < fraction:
            prov[i] = {**prov[i], "audio": prov[i]["audio"] + "_bumped"}
    out = df.with_columns(pl.Series(METAXY_PROVENANCE_BY_FIELD, prov, dtype=original_dtype))
    return add_metaxy_provenance_column(out.drop("metaxy_provenance"), root)


# ---------------------------------------------------------------------------
# Store helpers (fresh DuckDB file per round)
# ---------------------------------------------------------------------------


def _open_store(db_path: Path) -> DuckDBMetadataStore:
    return DuckDBMetadataStore(
        database=str(db_path), auto_create_tables=True, hash_algorithm=HASH
    )


def seed_new(
    db_path: Path, bench: BenchGraph, upstream_dfs: list[pl.DataFrame]
) -> None:
    """Populate store with upstream records so the leaf has ``new`` work to do."""
    with bench.graph.use(), _open_store(db_path).open("w") as s:
        for root, df in zip(bench.roots, upstream_dfs, strict=True):
            s.write(root, df)


def seed_stale(
    db_path: Path, bench: BenchGraph, upstream_dfs: list[pl.DataFrame]
) -> None:
    """Populate store, materialise the leaf, then bump upstream to force staleness."""
    with bench.graph.use(), _open_store(db_path).open("w") as s:
        for root, df in zip(bench.roots, upstream_dfs, strict=True):
            s.write(root, df)
        inc = s.resolve_update(bench.leaf)
        s.write(bench.leaf, collect_to_polars(inc.new))
        mutated = _mutate_fraction(
            upstream_dfs[0], MUTATION_FRACTION, seed=777, root=bench.roots[0]
        )
        s.write(bench.roots[0], mutated)


def _resolve_new(db_path: Path, bench: BenchGraph) -> None:
    with bench.graph.use(), _open_store(db_path).open("w") as s:
        inc = s.resolve_update(bench.leaf)
        _ = collect_to_polars(inc.new)


def _resolve_stale(db_path: Path, bench: BenchGraph) -> None:
    with bench.graph.use(), _open_store(db_path).open("w") as s:
        inc = s.resolve_update(bench.leaf)
        _ = (collect_to_polars(inc.new), collect_to_polars(inc.stale))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="resolve_new")
def test_resolve_new(benchmark: object, make_setup: object) -> None:
    benchmark.pedantic(  # type: ignore[attr-defined]
        _resolve_new,
        setup=make_setup(seed_new),  # type: ignore[operator]
        rounds=ROUNDS,
        iterations=1,
        warmup_rounds=0,
    )


@pytest.mark.benchmark(group="resolve_stale")
def test_resolve_stale(benchmark: object, make_setup: object) -> None:
    benchmark.pedantic(  # type: ignore[attr-defined]
        _resolve_stale,
        setup=make_setup(seed_stale),  # type: ignore[operator]
        rounds=ROUNDS,
        iterations=1,
        warmup_rounds=0,
    )
