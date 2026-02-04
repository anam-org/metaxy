"""Analyze pytest timing data from JUnit XML output."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel
from tabulate import tabulate


class PatternRegistry(BaseModel):
    """Auto-discovered patterns from the project structure."""

    stores: set[str]
    hash_algorithms: set[str]
    technologies: set[str]
    directory_categories: dict[str, str]

    @classmethod
    def discover(cls, project_root: Path) -> PatternRegistry:
        stores = _parse_pyproject_markers(project_root / "pyproject.toml")
        stores |= _list_metadata_store_tests(project_root / "tests" / "metadata_stores")

        hash_algorithms = _parse_hash_algorithm_enum(project_root / "src" / "metaxy" / "versioning" / "types.py")

        technologies = _list_ext_technologies(project_root / "tests" / "ext")
        technologies |= stores

        directory_categories = _build_directory_categories(project_root / "tests")

        return cls(
            stores=stores,
            hash_algorithms=hash_algorithms,
            technologies=technologies,
            directory_categories=directory_categories,
        )


def _parse_pyproject_markers(path: Path) -> set[str]:
    """Extract marker names from pyproject.toml [tool.pytest.ini_options].markers."""
    if not path.is_file():
        return set()

    content = path.read_text()
    markers: set[str] = set()

    in_markers = False
    for line in content.splitlines():
        if line.strip().startswith("markers"):
            in_markers = True
            continue
        if in_markers:
            if line.strip().startswith("]"):
                break
            match = re.match(r'\s*"(\w+):', line)
            if match:
                markers.add(match.group(1))

    return markers


def _parse_hash_algorithm_enum(path: Path) -> set[str]:
    """Extract hash algorithm names from HashAlgorithm enum."""
    if not path.is_file():
        return set()

    content = path.read_text()
    algorithms: set[str] = set()

    for match in re.finditer(r'^\s+\w+\s*=\s*"(\w+)"', content, re.MULTILINE):
        algorithms.add(match.group(1))

    return algorithms


def _list_ext_technologies(tests_ext_path: Path) -> set[str]:
    """List subdirectory names from tests/ext/ as technologies."""
    if not tests_ext_path.is_dir():
        return set()

    return {d.name for d in tests_ext_path.iterdir() if d.is_dir() and not d.name.startswith("_")}


def _list_metadata_store_tests(metadata_stores_path: Path) -> set[str]:
    """Extract store names from test_<store>.py files."""
    if not metadata_stores_path.is_dir():
        return set()

    stores: set[str] = set()
    for f in metadata_stores_path.iterdir():
        if f.is_file() and f.name.startswith("test_") and f.suffix == ".py":
            name = f.stem.removeprefix("test_")
            if name not in {"basic_functionality", "metadata_store", "utils", "hash_algorithms"}:
                stores.add(name)

    return stores


def _build_directory_categories(tests_path: Path) -> dict[str, str]:
    """Build mapping from test module prefix to category."""
    categories: dict[str, str] = {}

    if not tests_path.is_dir():
        return categories

    for item in tests_path.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            prefix = f"tests.{item.name}"
            categories[prefix] = item.name

            for subitem in item.iterdir():
                if subitem.is_dir() and not subitem.name.startswith("_"):
                    subprefix = f"tests.{item.name}.{subitem.name}"
                    categories[subprefix] = subitem.name

    return categories


class Stats(BaseModel):
    """Timing statistics for a group of tests."""

    total: float
    count: int
    avg: float


class Summary(BaseModel):
    """Overall test run summary."""

    total_time: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errored: int


class TestResult(BaseModel):
    """Individual test result with extracted parameters."""

    name: str
    classname: str
    full_name: str
    time: float
    outcome: str
    stores: list[str]
    hash_algos: list[str]
    categories: list[str]
    technologies: list[str]


class TimingReport(BaseModel):
    """Complete timing analysis report."""

    summary: Summary
    slowest_tests: list[TestResult]
    by_store: dict[str, Stats]
    by_hash: dict[str, Stats]
    by_category: dict[str, Stats]
    by_technology: dict[str, Stats]
    by_directory: dict[str, Stats]

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds >= 60:
            return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
        return f"{seconds:.2f}s"

    def _stats_rows(self, stats: dict[str, Stats], header: str) -> list[list[str]]:
        """Convert stats dict to table rows with percentage."""
        total_time = self.summary.total_time
        fmt = self._format_time
        rows: list[list[str]] = []
        for key, st in stats.items():
            pct = (st.total / total_time * 100) if total_time else 0
            rows.append([key, f"{fmt(st.total)} ({pct:.1f}%)", str(st.count), f"{st.avg:.2f}s"])
        return rows

    def to_markdown(self) -> str:
        """Format as GitHub-flavored markdown."""
        lines: list[str] = []
        s = self.summary
        fmt = self._format_time

        lines.append(f"## Test Timing Analysis ({s.total_tests} tests in {fmt(s.total_time)})")
        lines.append("")

        def add_stats_section(header: str, summary_label: str, stats: dict[str, Stats]) -> None:
            if not stats:
                return
            lines.append("<details>")
            lines.append(f"<summary>{summary_label}</summary>")
            lines.append("")
            rows = self._stats_rows(stats, header)
            lines.append(tabulate(rows, headers=[header, "Time", "Tests", "Avg"], tablefmt="github"))
            lines.append("")
            lines.append("</details>")
            lines.append("")

        add_stats_section("Store", "Time by Store", self.by_store)
        add_stats_section("Algorithm", "Time by Hash Algorithm", self.by_hash)
        add_stats_section("Entity", "Time by Entity", self.by_technology)
        add_stats_section("Category", "Time by Category", self.by_category)
        add_stats_section("Directory", "Time by Directory", self.by_directory)

        lines.append("---")
        lines.append(
            "*Generated by [`mxtest-analyze-timings`](tests/metaxy_testing/src/metaxy_testing/analyze_timings.py)*"
        )

        return "\n".join(lines)

    def to_text(self) -> str:
        """Format as plain text for terminal output."""
        lines: list[str] = []
        s = self.summary
        fmt = self._format_time

        lines.append(f"Test Timing Analysis ({s.total_tests} tests in {fmt(s.total_time)})")
        lines.append("")

        def add_stats_section(title: str, header: str, stats: dict[str, Stats]) -> None:
            if not stats:
                return
            lines.append(title)
            rows = self._stats_rows(stats, header)
            lines.append(tabulate(rows, headers=[header, "Time", "Tests", "Avg"], tablefmt="github"))
            lines.append("")

        add_stats_section("By Store:", "Store", self.by_store)
        add_stats_section("By Hash Algorithm:", "Algorithm", self.by_hash)
        add_stats_section("By Entity:", "Entity", self.by_technology)
        add_stats_section("By Category:", "Category", self.by_category)
        add_stats_section("By Directory:", "Directory", self.by_directory)

        lines.append("Generated by mxtest-analyze-timings (tests/metaxy_testing/src/metaxy_testing/analyze_timings.py)")

        return "\n".join(lines)


def extract_bracket_params(test_name: str) -> list[str]:
    """Extract parameters from test name brackets like test_foo[param1-param2]."""
    match = re.search(r"\[([^\]]+)\]", test_name)
    if not match:
        return []

    params_str = match.group(1)
    return [p.strip() for p in re.split(r"[-,]", params_str) if p.strip()]


def classify_params(params: list[str], registry: PatternRegistry) -> tuple[list[str], list[str], list[str]]:
    """Classify extracted params into stores, hash algorithms, and technologies."""
    stores: list[str] = []
    hash_algos: list[str] = []
    technologies: list[str] = []

    for param in params:
        param_lower = param.lower()
        if param_lower in registry.stores:
            stores.append(param_lower)
        if param_lower in registry.hash_algorithms:
            hash_algos.append(param_lower)
        if param_lower in registry.technologies:
            technologies.append(param_lower)

    return stores, hash_algos, technologies


def get_categories_from_classname(classname: str, registry: PatternRegistry) -> list[str]:
    """Extract categories from classname based on directory structure."""
    categories: list[str] = []

    for prefix, category in sorted(registry.directory_categories.items(), key=lambda x: -len(x[0])):
        if classname.startswith(prefix):
            categories.append(category)
            break

    return categories


def parse_junit_xml(path: Path, registry: PatternRegistry) -> list[TestResult]:
    """Parse JUnit XML and extract test results with classified parameters."""
    tree = ET.parse(path)
    root = tree.getroot()

    results: list[TestResult] = []

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            name = testcase.get("name", "")
            classname = testcase.get("classname", "")
            time = float(testcase.get("time", "0"))

            outcome = "passed"
            if testcase.find("failure") is not None:
                outcome = "failed"
            elif testcase.find("skipped") is not None:
                outcome = "skipped"
            elif testcase.find("error") is not None:
                outcome = "error"

            full_name = f"{classname}::{name}"
            params = extract_bracket_params(name)
            stores, hash_algos, technologies = classify_params(params, registry)

            classname_lower = classname.lower()
            for tech in registry.technologies:
                if tech in classname_lower and tech not in technologies:
                    technologies.append(tech)

            categories = get_categories_from_classname(classname, registry)

            results.append(
                TestResult(
                    name=name,
                    classname=classname,
                    full_name=full_name,
                    time=time,
                    outcome=outcome,
                    stores=stores,
                    hash_algos=hash_algos,
                    categories=categories,
                    technologies=technologies,
                )
            )

    return results


def compute_stats(times: list[float]) -> Stats:
    """Compute timing statistics for a list of times."""
    total = sum(times)
    count = len(times)
    return Stats(total=total, count=count, avg=total / count if count else 0.0)


def compute_group_stats(results: list[TestResult], extractor: str) -> dict[str, Stats]:
    """Compute stats grouped by an attribute that can have multiple values per test."""
    groups: dict[str, list[float]] = {}

    for result in results:
        values = getattr(result, extractor)
        for value in values:
            groups.setdefault(value, []).append(result.time)

    return {k: compute_stats(v) for k, v in sorted(groups.items(), key=lambda x: -sum(x[1]))}


def build_timing_report(results: list[TestResult]) -> TimingReport:
    """Build complete timing report from test results."""
    passed = sum(1 for r in results if r.outcome == "passed")
    failed = sum(1 for r in results if r.outcome == "failed")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    errored = sum(1 for r in results if r.outcome == "error")
    total_time = sum(r.time for r in results)

    summary = Summary(
        total_time=total_time,
        total_tests=len(results),
        passed=passed,
        failed=failed,
        skipped=skipped,
        errored=errored,
    )

    slowest = sorted(results, key=lambda r: -r.time)[:50]

    directory_times: dict[str, list[float]] = {}
    for result in results:
        parts = result.classname.split(".")
        directory = "/".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else "unknown"
        directory_times.setdefault(directory, []).append(result.time)

    by_directory = {k: compute_stats(v) for k, v in sorted(directory_times.items(), key=lambda x: -sum(x[1]))}

    return TimingReport(
        summary=summary,
        slowest_tests=slowest,
        by_store=compute_group_stats(results, "stores"),
        by_hash=compute_group_stats(results, "hash_algos"),
        by_category=compute_group_stats(results, "categories"),
        by_technology=compute_group_stats(results, "technologies"),
        by_directory=by_directory,
    )


def find_project_root(start: Path) -> Path:
    """Find project root by looking for pyproject.toml."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pytest timing from JUnit XML")
    parser.add_argument("junit_xml", type=Path, help="Path to JUnit XML file")
    parser.add_argument(
        "-f", "--format", choices=["plain", "json", "markdown"], default="plain", help="Output format (default: plain)"
    )

    args = parser.parse_args()

    if not args.junit_xml.is_file():
        print(f"Error: {args.junit_xml} not found", file=sys.stderr)
        sys.exit(1)

    project_root = find_project_root(args.junit_xml)
    registry = PatternRegistry.discover(project_root)

    results = parse_junit_xml(args.junit_xml, registry)
    report = build_timing_report(results)

    match args.format:
        case "json":
            print(report.model_dump_json(indent=2))
        case "markdown":
            print(report.to_markdown())
        case _:
            print(report.to_text())


if __name__ == "__main__":
    main()
