"""Tests for the analyze_timings module."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from metaxy_testing.analyze_timings import (
    PatternRegistry,
    TimingReport,
    build_timing_report,
    parse_junit_xml,
)

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def sample_junit_xml(tmp_path: Path) -> Path:
    """Create a sample JUnit XML file for testing."""
    xml_content = dedent("""\
        <?xml version="1.0" encoding="utf-8"?>
        <testsuites>
            <testsuite name="pytest" errors="0" failures="1" skipped="1" tests="5" time="10.5">
                <testcase classname="tests.metadata_stores.test_duckdb" name="test_basic[duckdb-xxhash64]" time="2.5"/>
                <testcase classname="tests.metadata_stores.test_duckdb" name="test_basic[duckdb-sha256]" time="1.5"/>
                <testcase classname="tests.migrations.test_models" name="test_migration_flow" time="3.0"/>
                <testcase classname="tests.versioning.test_feature" name="test_version" time="0.5">
                    <skipped message="skipped"/>
                </testcase>
                <testcase classname="tests.ext.dagster.test_integration" name="test_dagster" time="3.0">
                    <failure message="AssertionError">Test failed</failure>
                </testcase>
            </testsuite>
        </testsuites>
    """)
    xml_path = tmp_path / "junit.xml"
    xml_path.write_text(xml_content)
    return xml_path


@pytest.fixture
def registry() -> PatternRegistry:
    return PatternRegistry.discover(PROJECT_ROOT)


@pytest.fixture
def report(sample_junit_xml: Path, registry: PatternRegistry) -> TimingReport:
    results = parse_junit_xml(sample_junit_xml, registry)
    return build_timing_report(results)


class TestPatternRegistry:
    def test_discover_from_project(self, registry: PatternRegistry) -> None:
        assert "duckdb" in registry.stores
        assert "clickhouse" in registry.stores
        assert "xxhash64" in registry.hash_algorithms
        assert "sha256" in registry.hash_algorithms
        assert "dagster" in registry.technologies
        assert len(registry.directory_categories) > 0


class TestReportGeneration:
    def test_summary(self, report: TimingReport) -> None:
        assert report.summary.total_tests == 5
        assert report.summary.passed == 3
        assert report.summary.failed == 1
        assert report.summary.skipped == 1

    def test_store_detection(self, report: TimingReport) -> None:
        assert "duckdb" in report.by_store
        assert report.by_store["duckdb"].count == 2

    def test_hash_detection(self, report: TimingReport) -> None:
        assert "xxhash64" in report.by_hash
        assert "sha256" in report.by_hash

    def test_category_detection(self, report: TimingReport) -> None:
        assert "metadata_stores" in report.by_category
        assert "migrations" in report.by_category

    def test_technology_detection(self, report: TimingReport) -> None:
        assert "dagster" in report.by_technology

    def test_slowest_tests_sorted(self, report: TimingReport) -> None:
        times = [t.time for t in report.slowest_tests]
        assert times == sorted(times, reverse=True)


class TestOutputFormats:
    def test_plain_output(self, report: TimingReport) -> None:
        output = report.to_text()
        assert "Test Timing Analysis" in output

    def test_markdown_output(self, report: TimingReport) -> None:
        output = report.to_markdown()
        assert "## Test Timing Analysis" in output

    def test_json_output(self, report: TimingReport) -> None:
        output = report.model_dump_json()
        assert "summary" in output
        assert "slowest_tests" in output
