"""Render the benchmark plot for the JOSS paper.

Reads ``benchmark_results.json`` produced by ``pytest --benchmark-json``
and writes ``assets/benchmark.svg``: a log-log plot of wall-clock time vs
record count, with one line per (scenario, phase) and a shaded IQR band.

Usage::

    uv run python publications/2026-introducing-metaxy/make_plot.py
    uv run python publications/2026-introducing-metaxy/make_plot.py \\
        --input custom_results.json --output custom.svg
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

SCENARIO_COLORS = {"simple": "#1b9e77", "wide": "#d95f02"}
PHASE_STYLES = {"new": {"linestyle": "-", "marker": "o"}, "stale": {"linestyle": "--", "marker": "s"}}
PARAM_RE = re.compile(r"^test_resolve_(?P<phase>\w+)\[(?P<scenario>\w+)-(?P<n>[\d_]+)\]$")


@dataclass
class Point:
    n: int
    median_s: float
    q1_s: float
    q3_s: float


def _parse_results(path: Path) -> dict[tuple[str, str], list[Point]]:
    data = json.loads(path.read_text())
    series: dict[tuple[str, str], list[Point]] = {}
    for b in data["benchmarks"]:
        m = PARAM_RE.match(b["name"])
        if not m:
            continue
        key = (m["scenario"], m["phase"])
        n = int(m["n"].replace("_", ""))
        st = b["stats"]
        series.setdefault(key, []).append(
            Point(n=n, median_s=st["median"], q1_s=st["q1"], q3_s=st["q3"])
        )
    for v in series.values():
        v.sort(key=lambda p: p.n)
    return series


def _render(series: dict[tuple[str, str], list[Point]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for (scenario, phase), points in sorted(series.items()):
        xs = [p.n for p in points]
        ys = [p.median_s for p in points]
        q1 = [p.q1_s for p in points]
        q3 = [p.q3_s for p in points]
        color = SCENARIO_COLORS.get(scenario, "#666666")
        style = PHASE_STYLES[phase]
        ax.plot(
            xs,
            ys,
            color=color,
            label=f"{scenario} / {phase}",
            linewidth=1.6,
            **style,
        )
        ax.fill_between(xs, q1, q3, color=color, alpha=0.12, linewidth=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Record count $N$")
    ax.set_ylabel("resolve_update wall-clock (s)")
    ax.set_title("resolve_update on DuckDB (median, IQR shaded)")
    ax.grid(which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--input", type=Path, default=here / "benchmark_results.json")
    parser.add_argument("--output", type=Path, default=here / "assets" / "benchmark.svg")
    args = parser.parse_args()
    series = _parse_results(args.input)
    _render(series, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
