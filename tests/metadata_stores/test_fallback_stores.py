"""Test fallback store behavior and warnings.

Tests that resolve_update correctly switches between native and Polars components
based on whether upstream features are in fallback stores, and that appropriate
warnings are issued.
"""

from typing import Any

import narwhals as nw
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    SampleFeatureSpec,
)
from metaxy._testing import HashAlgorithmCases, assert_all_results_equal
from metaxy._testing.pytest_helpers import skip_exception
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    InMemoryMetadataStore,
    MetadataStore,
)
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.provenance.types import HashAlgorithm

# ============= HELPERS =============


def get_available_store_types_for_fallback() -> list[str]:
    """Get store types that support native components and can be used for fallback tests.

    We test with DuckDB since it:
    1. Supports native components (IbisProvenanceByFieldCalculator)
    2. Can be easily instantiated with separate databases
    3. Supports multiple hash algorithms
    """
    return ["duckdb"]


def create_store_for_fallback(
    store_type: str,
    prefer_native: bool,
    hash_algorithm: HashAlgorithm,
    params: dict[str, Any],
    suffix: str = "",
    fallback_stores: list[MetadataStore] | None = None,
) -> MetadataStore:
    """Create a store instance for fallback testing.

    Args:
        store_type: "duckdb"
        prefer_native: Whether to prefer native components
        hash_algorithm: Hash algorithm to use
        params: Store-specific parameters
        suffix: Suffix to add to database filename (for creating distinct stores)
        fallback_stores: Optional list of fallback stores to configure
    """
    tmp_path = params.get("tmp_path")
    assert tmp_path is not None, f"tmp_path parameter required for {store_type}"

    if store_type == "duckdb":
        db_path = tmp_path / f"fallback_test_{suffix}_{hash_algorithm.value}.duckdb"
        extensions: list[str] = (
            ["hashfuncs"]
            if hash_algorithm in [HashAlgorithm.XXHASH32, HashAlgorithm.XXHASH64]
            else []
        )
        return DuckDBMetadataStore(
            db_path,
            hash_algorithm=hash_algorithm,
            extensions=extensions,  # pyright: ignore[reportArgumentType]
            prefer_native=prefer_native,
            fallback_stores=fallback_stores,
        )
    else:
        raise ValueError(f"Unknown store type for fallback test: {store_type}")


# ============= FEATURE DEFINITIONS =============


class RootFeature(
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["fallback_test", "root"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="1"),
        ],
    ),
):
    """Root feature with no dependencies."""

    pass


class DownstreamFeature(
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["fallback_test", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["fallback_test", "root"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["fallback_test", "root"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    ),
):
    """Downstream feature depending on RootFeature."""

    pass


# ============= TESTS =============


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("primary_store_type", get_available_store_types_for_fallback())
@pytest.mark.parametrize(
    "fallback_store_type", ["inmemory"]
)  # Parametrized for future expansion
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_fallback_store_warning_issued(
    store_params: dict[str, Any],
    hash_algorithm: HashAlgorithm,
    primary_store_type: str,
    fallback_store_type: str,
    caplog,
    snapshot,
):
    """Test that warning IS issued when upstream feature is in fallback store.

    This tests the core fallback scenario:
    - Root feature is in fallback_store (InMemory - no native components)
    - Downstream feature will be written to primary_store (DuckDB - has native)
    - resolve_update() should switch to Polars components and issue warning
    - Results should match snapshot across different configurations
    """
    graph = FeatureGraph()
    graph.add_feature(RootFeature)
    graph.add_feature(DownstreamFeature)

    # Create fallback store (InMemory - simple, no native components)
    fallback_store = InMemoryMetadataStore(hash_algorithm=hash_algorithm)

    # Prepare root data
    root_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"default": "hash1"},
                {"default": "hash2"},
                {"default": "hash3"},
            ],
        }
    )

    results = {}

    # Setup: Write root feature to fallback store
    with fallback_store, graph.use():
        fallback_store.write_metadata(RootFeature, root_data)

    # Test with prefer_native=True and prefer_native=False
    for prefer_native in [True, False]:
        # Create primary store with native component support and fallback configured
        primary_store = create_store_for_fallback(
            primary_store_type,
            prefer_native=prefer_native,
            hash_algorithm=hash_algorithm,
            params=store_params,
            suffix=f"primary_native{prefer_native}",
            fallback_stores=[fallback_store],
        )

        # Test: Resolve downstream feature with primary store that has fallback
        with primary_store, fallback_store, graph.use():
            import logging

            caplog.clear()
            with caplog.at_level(logging.WARNING):
                result = primary_store.resolve_update(DownstreamFeature)

            # Verify warning was issued only when prefer_native=True
            fallback_warnings = [
                record
                for record in caplog.records
                if "upstream dependencies in fallback stores" in record.message
                and "Falling back to in-memory Polars processing" in record.message
            ]

            if prefer_native:
                assert len(fallback_warnings) == 1, (
                    f"Expected exactly 1 fallback warning for {primary_store_type} with prefer_native=True, "
                    f"but got {len(fallback_warnings)}: {[r.message for r in fallback_warnings]}"
                )
            else:
                # prefer_native=False means we intentionally use Polars, not a fallback
                assert len(fallback_warnings) == 0, (
                    f"Unexpected fallback warning for {primary_store_type} with prefer_native=False"
                )

            # Collect field provenances for comparison
            added_sorted = (
                result.added.to_polars()
                if isinstance(result.added, nw.DataFrame)
                else result.added
            ).sort("sample_uid")
            versions = added_sorted["metaxy_provenance_by_field"].to_list()

            results[(primary_store_type, fallback_store_type, prefer_native)] = {
                "added": len(result.added),
                "changed": len(result.changed),
                "removed": len(result.removed),
                "versions": versions,
            }

    # All variants should produce identical results (including field provenances)
    assert_all_results_equal(results, snapshot)

    # Verify expected behavior
    if results:
        reference = list(results.values())[0]
        assert reference["added"] == 3
        assert reference["changed"] == 0
        assert reference["removed"] == 0
        assert len(reference["versions"]) == 3
        assert all(isinstance(v, dict) for v in reference["versions"])


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("store_type", get_available_store_types_for_fallback())
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_no_fallback_warning_when_all_local(
    store_params: dict[str, Any],
    hash_algorithm: HashAlgorithm,
    store_type: str,
    caplog,
):
    """Test that warning is NOT issued when all upstream is in the same store.

    This tests the normal case:
    - Both root and downstream features are in the same store
    - resolve_update() should use native components (no Polars fallback)
    - No warning should be issued
    """
    graph = FeatureGraph()
    graph.add_feature(RootFeature)
    graph.add_feature(DownstreamFeature)

    store = create_store_for_fallback(
        store_type,
        prefer_native=True,
        hash_algorithm=hash_algorithm,
        params=store_params,
        suffix="single",
    )

    with store, graph.use():
        # Write root feature to same store
        root_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        )
        store.write_metadata(RootFeature, root_data)

        # Resolve downstream feature - all upstream is local
        import logging

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result = store.resolve_update(DownstreamFeature)

        # Verify NO fallback warning was issued
        fallback_warnings = [
            record
            for record in caplog.records
            if "upstream dependencies in fallback stores" in record.message
            or "Falling back to in-memory Polars processing" in record.message
        ]

        assert len(fallback_warnings) == 0, (
            f"Unexpected fallback warning for {store_type} when all upstream is local: "
            f"{[r.message for r in fallback_warnings]}"
        )

        # Verify results are still correct
        assert len(result.added) == 3
        assert len(result.changed) == 0
        assert len(result.removed) == 0


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("primary_store_type", get_available_store_types_for_fallback())
@pytest.mark.parametrize("fallback_store_type", ["inmemory"])
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_fallback_store_switches_to_polars_components(
    store_params: dict[str, Any],
    hash_algorithm: HashAlgorithm,
    primary_store_type: str,
    fallback_store_type: str,
    caplog,
    snapshot,
):
    """Test that resolve_update switches from native to Polars components with fallback.

    This verifies the component selection logic:
    1. When all upstream is local → use native components (no warning)
    2. When upstream in fallback → switch to Polars components (with warning)
    3. Results should be identical in both cases
    4. All scenarios are snapshotted for regression detection
    """
    graph = FeatureGraph()
    graph.add_feature(RootFeature)
    graph.add_feature(DownstreamFeature)

    # Setup root feature data
    root_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"default": "hash1"},
                {"default": "hash2"},
                {"default": "hash3"},
            ],
        }
    )

    results: dict[Any, Any] = {}

    # Scenario 1: All local (native components)
    store_all_local = create_store_for_fallback(
        primary_store_type,
        prefer_native=True,
        hash_algorithm=hash_algorithm,
        params=store_params,
        suffix="all_local",
    )

    with store_all_local, graph.use():
        store_all_local.write_metadata(RootFeature, root_data)

        import logging

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result_local = store_all_local.resolve_update(DownstreamFeature)

        # Should be no warnings
        warnings_local = [r for r in caplog.records if "fallback" in r.message.lower()]
        assert len(warnings_local) == 0

        # Collect field provenances from result
        added_local = (
            result_local.added.to_polars()
            if isinstance(result_local.added, nw.DataFrame)
            else result_local.added
        )
        versions_local = added_local.sort("sample_uid")[
            "metaxy_provenance_by_field"
        ].to_list()

        results[(primary_store_type, fallback_store_type, "all_local")] = {
            "added": len(result_local.added),
            "changed": len(result_local.changed),
            "removed": len(result_local.removed),
            "versions": versions_local,
        }

    # Scenario 2: Upstream in fallback (Polars components)
    # Use InMemory for fallback (realistic - simple store from previous deployment)
    fallback_store = InMemoryMetadataStore(hash_algorithm=hash_algorithm)

    # Write root to fallback store
    with fallback_store, graph.use():
        fallback_store.write_metadata(RootFeature, root_data)

    # Create primary store with fallback configured
    primary_store = create_store_for_fallback(
        primary_store_type,
        prefer_native=True,
        hash_algorithm=hash_algorithm,
        params=store_params,
        suffix="with_fallback",
        fallback_stores=[fallback_store],
    )

    with primary_store, fallback_store, graph.use():
        import logging

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result_fallback = primary_store.resolve_update(DownstreamFeature)

        # Should have warning about fallback
        warnings_fallback = [
            r
            for r in caplog.records
            if "upstream dependencies in fallback stores" in r.message
        ]
        assert len(warnings_fallback) == 1

        # Collect field provenances from result
        added_fallback = (
            result_fallback.added.to_polars()
            if isinstance(result_fallback.added, nw.DataFrame)
            else result_fallback.added
        )
        versions_fallback = added_fallback.sort("sample_uid")[
            "metaxy_provenance_by_field"
        ].to_list()

        results[(primary_store_type, fallback_store_type, "with_fallback")] = {
            "added": len(result_fallback.added),
            "changed": len(result_fallback.changed),
            "removed": len(result_fallback.removed),
            "versions": versions_fallback,
        }

    # Compare results - versions should be identical
    assert_all_results_equal(results, snapshot)

    # Verify both scenarios have correct results
    assert len(results) == 2
    for scenario_key, result in results.items():
        result_versions = result["versions"]
        assert result["added"] == 3
        assert result["changed"] == 0
        assert result["removed"] == 0
        assert len(result_versions) == 3
        assert all(isinstance(v, dict) for v in result_versions)


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("store_type", get_available_store_types_for_fallback())
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_prefer_native_false_no_warning_even_without_fallback(
    store_params: dict[str, Any],
    hash_algorithm: HashAlgorithm,
    store_type: str,
    caplog,
):
    """Test that prefer_native=False doesn't issue fallback warning.

    When prefer_native=False, the store always uses Polars components,
    so there's no "fallback" from native to Polars - it's intentional.
    The warning should only be issued when we CAN'T use native due to fallback stores.
    """
    graph = FeatureGraph()
    graph.add_feature(RootFeature)
    graph.add_feature(DownstreamFeature)

    # Create store with prefer_native=False
    store = create_store_for_fallback(
        store_type,
        prefer_native=False,  # Explicitly disable native
        hash_algorithm=hash_algorithm,
        params=store_params,
        suffix="no_native",
    )

    with store, graph.use():
        root_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        )
        store.write_metadata(RootFeature, root_data)

        import logging

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result = store.resolve_update(DownstreamFeature)

        # Should be no warnings - prefer_native=False is intentional, not a fallback
        warnings = [r for r in caplog.records]
        assert len(warnings) == 0, (
            f"Unexpected warnings with prefer_native=False: {[r.message for r in warnings]}"
        )

        # Verify results are correct
        assert len(result.added) == 3
