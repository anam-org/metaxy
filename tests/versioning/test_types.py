import metaxy.versioning.types as types_module
import polars as pl
from metaxy.versioning.types import PolarsLazyIncrement


def _increment() -> PolarsLazyIncrement:
    return PolarsLazyIncrement(
        new=pl.DataFrame({"value": [1]}).lazy(),
        stale=pl.DataFrame({"value": [2]}).lazy(),
        orphaned=pl.DataFrame({"value": [3]}).lazy(),
    )


def test_polars_143_disables_common_subplan_elimination(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def collect_all(frames, **kwargs):
        captured.update(kwargs)
        return [frame.collect() for frame in frames]

    monkeypatch.setattr(types_module.pl, "__version__", "1.43.0")
    monkeypatch.setattr(types_module.pl, "collect_all", collect_all)

    _increment().collect()

    optimizations = captured["optimizations"]
    assert isinstance(optimizations, pl.QueryOptFlags)
    assert optimizations.comm_subplan_elim is False


def test_polars_143_preserves_explicit_optimizations(monkeypatch) -> None:
    captured: dict[str, object] = {}
    requested = pl.QueryOptFlags(comm_subplan_elim=True)

    def collect_all(frames, **kwargs):
        captured.update(kwargs)
        return [frame.collect() for frame in frames]

    monkeypatch.setattr(types_module.pl, "__version__", "1.43.0")
    monkeypatch.setattr(types_module.pl, "collect_all", collect_all)

    _increment().collect(optimizations=requested)

    assert captured["optimizations"] is requested
