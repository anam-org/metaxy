"""Fixtures for the publication benchmark.

Tests in this directory use the ``benchmark`` fixture from ``pytest-benchmark``
and are opt-in: they allocate multi-GB DuckDB files at N=10M and take over
an hour to finish end-to-end. They are skipped by default and only run
when ``--benchmark-only`` is passed explicitly.

Fixtures provide:

* ``bench``     fully assembled :class:`BenchGraph` for the current scenario
* ``n``         record count for the current parametrisation
* ``upstream``  pre-built upstream DataFrames for this ``(bench, n)``
* ``make_setup`` factory that wraps a seeding function into a
                ``pytest-benchmark`` ``setup`` callback (fresh DuckDB file
                per round + seeded state)

Scenario and N are parametrised once at fixture level, so tests are tiny
and contain no plumbing.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from metaxy.config import MetaxyConfig

from _graphs import BenchGraph, build_simple_graph, build_wide_graph  # pyright: ignore[reportMissingImports]


RECORDS: tuple[int, ...] = (10_000, 100_000, 1_000_000, 10_000_000)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip benchmark tests unless ``--benchmark-only`` is passed.

    The publication's resolve_update benchmarks are heavy (multi-GB DuckDB
    files, ~1-2 hours for a full run). We never want them executed as part
    of the regular test suite or the docs test pass; they must be requested
    explicitly via pytest-benchmark's ``--benchmark-only`` flag.
    """
    if config.getoption("--benchmark-only", default=False):
        return
    skip = pytest.mark.skip(reason="heavy benchmark; run with --benchmark-only")
    for item in items:
        if "benchmark" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def load_benchmark_config() -> None:
    """Load the publication's ``metaxy.toml`` so ``enable_map_datatype`` is active."""
    MetaxyConfig.load(config_file=Path(__file__).parent / "metaxy.toml")


@pytest.fixture(scope="session")
def simple_bench_graph() -> BenchGraph:
    return build_simple_graph()


@pytest.fixture(scope="session")
def wide_bench_graph() -> BenchGraph:
    return build_wide_graph()


@pytest.fixture(params=["simple", "wide"])
def bench(
    request: pytest.FixtureRequest,
    simple_bench_graph: BenchGraph,
    wide_bench_graph: BenchGraph,
) -> BenchGraph:
    """Parametrised :class:`BenchGraph` — one value per scenario."""
    return {"simple": simple_bench_graph, "wide": wide_bench_graph}[request.param]


@pytest.fixture(params=RECORDS, ids=lambda n: f"{n:_}")
def n(request: pytest.FixtureRequest) -> int:
    """Parametrised record count."""
    return request.param


@pytest.fixture
def upstream(bench: BenchGraph, n: int) -> list[pl.DataFrame]:
    """Pre-built upstream DataFrames for each root in ``bench`` at size ``n``.

    Generated once per test (outside the timed block), so every benchmark
    round reuses the same in-memory data.
    """
    # Local import to keep test_benchmark.py responsible for the helper.
    from test_benchmark import make_upstream_df  # pyright: ignore[reportMissingImports]

    with bench.graph.use():
        return [
            make_upstream_df(n, seed=42 + i, root=r)
            for i, r in enumerate(bench.roots)
        ]


SeedFn = Callable[[Path, BenchGraph, list[pl.DataFrame]], None]


@pytest.fixture
def make_setup(
    bench: BenchGraph, upstream: list[pl.DataFrame]
) -> Callable[[SeedFn], Callable[[], tuple[tuple[object, ...], dict[str, object]]]]:
    """Factory for ``pytest-benchmark`` ``setup`` callbacks.

    Given a ``seed_fn(db_path, bench, upstream)`` that populates the store,
    returns a zero-arg callable that:

    * allocates a fresh DuckDB file in a new temp directory (cold buffer
      pool and cold file-system page cache for this inode);
    * reseeds the store from scratch;
    * returns ``((db_path, bench), {})`` for the benchmarked target.
    """

    def factory(
        seed_fn: SeedFn,
    ) -> Callable[[], tuple[tuple[object, ...], dict[str, object]]]:
        def setup() -> tuple[tuple[object, ...], dict[str, object]]:
            db_path = Path(tempfile.mkdtemp(prefix="metaxy_bench_")) / "bench.duckdb"
            seed_fn(db_path, bench, upstream)
            return (db_path, bench), {}

        return setup

    return factory
