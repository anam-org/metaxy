"""Test fallback store behavior and warnings.

Tests that resolve_update correctly switches between native and Polars components
based on whether upstream features are in fallback stores, and that appropriate
warnings are issued.
"""

from pathlib import Path
from typing import Any, Literal

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing import (
    HashAlgorithmCases,
    add_metaxy_provenance_column,
    assert_all_results_equal,
)
from metaxy_testing.models import SampleFeatureSpec
from metaxy_testing.pytest_helpers import skip_exception
from pytest_cases import parametrize_with_cases

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
)
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from metaxy.metadata_store.warnings import PolarsMaterializationWarning
from metaxy.models.feature import FeatureGraph
from metaxy.versioning.types import HashAlgorithm

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
    versioning_engine: Literal["auto", "native", "polars"],
    hash_algorithm: HashAlgorithm,
    tmp_path: Path,
    suffix: str = "",
    fallback_stores: list[MetadataStore] | None = None,
) -> MetadataStore:
    """Create a store instance for fallback testing.

    Args:
        store_type: "duckdb"
        versioning_engine: Versioning engine mode ("auto", "native", or "polars")
        hash_algorithm: Hash algorithm to use
        tmp_path: Temporary directory for database files
        suffix: Suffix to add to database filename (for creating distinct stores)
        fallback_stores: Optional list of fallback stores to configure
    """
    if store_type == "duckdb":
        db_path = tmp_path / f"fallback_test_{suffix}_{hash_algorithm.value}.duckdb"
        extensions: list[str] = (
            ["hashfuncs"] if hash_algorithm in [HashAlgorithm.XXHASH32, HashAlgorithm.XXHASH64] else []
        )
        return DuckDBMetadataStore(
            db_path,
            hash_algorithm=hash_algorithm,
            extensions=extensions,
            versioning_engine=versioning_engine,
            fallback_stores=fallback_stores,
        )
    else:
        raise ValueError(f"Unknown store type for fallback test: {store_type}")


# ============= FEATURE DEFINITIONS =============


@pytest.fixture
def features(graph: FeatureGraph):
    class RootFeature(
        BaseFeature,
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
        BaseFeature,
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

    return {
        "RootFeature": RootFeature,
        "DownstreamFeature": DownstreamFeature,
    }


@pytest.fixture
def RootFeature(features):
    return features["RootFeature"]


@pytest.fixture
def DownstreamFeature(features):
    return features["DownstreamFeature"]


@pytest.fixture
def fallback_store_type():
    return "deltalake"


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("primary_store_type", get_available_store_types_for_fallback())
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_fallback_store_warning_issued(
    tmp_path: Path,
    hash_algorithm: HashAlgorithm,
    primary_store_type: str,
    fallback_store_type: str,
    snapshot,
    RootFeature,
    DownstreamFeature,
):
    """Test that warning IS issued when upstream feature is in fallback store.

    This tests the core fallback scenario:
    - Root feature is in fallback_store
    - Downstream feature will be written to primary_store (DuckDB - has native)
    - resolve_update() should switch to Polars components and issue warning
    - Results should match snapshot across different configurations
    """
    fallback_store = DeltaMetadataStore(
        root_path=tmp_path / "delta_fallback",
        hash_algorithm=hash_algorithm,
    )

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
    with fallback_store:
        root_data = add_metaxy_provenance_column(root_data, RootFeature)
        fallback_store.write(RootFeature, root_data)

    # Test with versioning_engine="native" and versioning_engine="polars"
    for versioning_engine in ["native", "polars"]:
        # Create primary store with native component support and fallback configured
        primary_store = create_store_for_fallback(
            primary_store_type,
            versioning_engine=versioning_engine,  # ty: ignore[invalid-argument-type]
            hash_algorithm=hash_algorithm,
            tmp_path=tmp_path,
            suffix=f"primary_{versioning_engine}",
            fallback_stores=[fallback_store],
        )

        # Test: Resolve downstream feature with primary store that has fallback
        with primary_store, fallback_store:
            if versioning_engine == "native":
                # Should warn when falling back from native to Polars
                with pytest.warns(
                    PolarsMaterializationWarning,
                    match="Using Polars for resolving the increment instead",
                ):
                    result = primary_store.resolve_update(DownstreamFeature)
            else:
                # versioning_engine="polars" means we intentionally use Polars, not a fallback
                # No warning should be issued
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("error", PolarsMaterializationWarning)
                    result = primary_store.resolve_update(DownstreamFeature)

            # Collect field provenances for comparison
            added_sorted = (result.new.to_polars() if isinstance(result.new, nw.DataFrame) else result.added).sort(
                "sample_uid"
            )
            versions = added_sorted["metaxy_provenance_by_field"].to_list()

            results[(primary_store_type, fallback_store_type, versioning_engine)] = {
                "added": len(result.new),
                "changed": len(result.stale),
                "removed": len(result.orphaned),
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
    tmp_path: Path,
    hash_algorithm: HashAlgorithm,
    store_type: str,
    RootFeature,
    DownstreamFeature,
):
    """Test that warning is NOT issued when all upstream is in the same store.

    This tests the normal case:
    - Both root and downstream features are in the same store
    - resolve_update() should use native components (no Polars fallback)
    - No warning should be issued
    """
    store = create_store_for_fallback(
        store_type,
        versioning_engine="native",
        hash_algorithm=hash_algorithm,
        tmp_path=tmp_path,
        suffix="single",
    )

    with store:
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
        root_data = add_metaxy_provenance_column(root_data, RootFeature)
        store.write(RootFeature, root_data)

        # Resolve downstream feature - all upstream is local
        # No warning should be issued
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", PolarsMaterializationWarning)
            result = store.resolve_update(DownstreamFeature)

        # Verify results are still correct
        assert len(result.new) == 3
        assert len(result.stale) == 0
        assert len(result.orphaned) == 0


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("primary_store_type", get_available_store_types_for_fallback())
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_fallback_store_switches_to_polars_components(
    tmp_path: Path,
    hash_algorithm: HashAlgorithm,
    primary_store_type: str,
    fallback_store_type: str,
    snapshot,
    RootFeature,
    DownstreamFeature,
):
    """Test that resolve_update switches from native to Polars components with fallback.

    This verifies the component selection logic:
    1. When all upstream is local → use native components (no warning)
    2. When upstream in fallback → switch to Polars components (with warning)
    3. Results should be identical in both cases
    4. All scenarios are snapshotted for regression detection
    """
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
        versioning_engine="native",
        hash_algorithm=hash_algorithm,
        tmp_path=tmp_path,
        suffix="all_local",
    )

    with store_all_local:
        root_data_with_prov = add_metaxy_provenance_column(root_data, RootFeature)
        store_all_local.write(RootFeature, root_data_with_prov)

        # Should be no warnings
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", PolarsMaterializationWarning)
            result_local = store_all_local.resolve_update(DownstreamFeature)

        # Collect field provenances from result
        added_local = result_local.new.to_polars() if isinstance(result_local.new, nw.DataFrame) else result_local.added
        versions_local = added_local.sort("sample_uid")["metaxy_provenance_by_field"].to_list()

        results[(primary_store_type, fallback_store_type, "all_local")] = {
            "added": len(result_local.new),
            "changed": len(result_local.stale),
            "removed": len(result_local.orphaned),
            "versions": versions_local,
        }

    # Scenario 2: Upstream in fallback (Polars components)
    fallback_store = DeltaMetadataStore(
        root_path=tmp_path / "delta_fallback",
        hash_algorithm=hash_algorithm,
    )

    # Write root to fallback store
    with fallback_store:
        root_data_with_prov = add_metaxy_provenance_column(root_data, RootFeature)
        fallback_store.write(RootFeature, root_data_with_prov)

    # Create primary store with fallback configured
    primary_store = create_store_for_fallback(
        primary_store_type,
        versioning_engine="native",
        hash_algorithm=hash_algorithm,
        tmp_path=tmp_path,
        suffix="with_fallback",
        fallback_stores=[fallback_store],
    )

    with primary_store, fallback_store:
        # Should have warning about fallback
        with pytest.warns(
            PolarsMaterializationWarning,
            match="Using Polars for resolving the increment instead",
        ):
            result_fallback = primary_store.resolve_update(DownstreamFeature)

        # Collect field provenances from result
        added_fallback = (
            result_fallback.new.to_polars() if isinstance(result_fallback.new, nw.DataFrame) else result_fallback.added
        )
        versions_fallback = added_fallback.sort("sample_uid")["metaxy_provenance_by_field"].to_list()

        results[(primary_store_type, fallback_store_type, "with_fallback")] = {
            "added": len(result_fallback.new),
            "changed": len(result_fallback.stale),
            "removed": len(result_fallback.orphaned),
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
def test_versioning_engine_polars_no_warning_even_without_fallback(
    tmp_path: Path,
    hash_algorithm: HashAlgorithm,
    store_type: str,
    RootFeature,
    DownstreamFeature,
):
    """Test that versioning_engine="polars" doesn't issue fallback warning.

    When versioning_engine="polars", the store always uses Polars components,
    so there's no "fallback" from native to Polars - it's intentional.
    The warning should only be issued when we CAN'T use native due to fallback stores.
    """
    # Create store with versioning_engine="polars"
    store = create_store_for_fallback(
        store_type,
        versioning_engine="polars",  # Explicitly use Polars
        hash_algorithm=hash_algorithm,
        tmp_path=tmp_path,
        suffix="polars_engine",
    )

    with store:
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
        root_data = add_metaxy_provenance_column(root_data, RootFeature)
        store.write(RootFeature, root_data)

        # Should be no warnings - versioning_engine="polars" is intentional, not a fallback
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", PolarsMaterializationWarning)
            result = store.resolve_update(DownstreamFeature)

        # Verify results are correct
        assert len(result.new) == 3


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("store_type", get_available_store_types_for_fallback())
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_versioning_engine_native_no_error_when_data_is_local_despite_fallback_configured(
    tmp_path: Path,
    hash_algorithm: HashAlgorithm,
    store_type: str,
    RootFeature,
    DownstreamFeature,
):
    """Test that versioning_engine="native" works when data is local even with fallback configured.

    When fallback stores are configured but the data exists in the primary store,
    versioning_engine="native" should work without errors or warnings because the
    data is actually local (native format).
    """
    fallback_store = DeltaMetadataStore(
        root_path=tmp_path / "delta_fallback",
        hash_algorithm=hash_algorithm,
    )

    # Create primary store with fallback configured and versioning_engine="native"
    primary_store = create_store_for_fallback(
        store_type,
        versioning_engine="native",
        hash_algorithm=hash_algorithm,
        tmp_path=tmp_path,
        suffix="local_data_test",
        fallback_stores=[fallback_store],
    )

    with primary_store, fallback_store:
        # Write data to PRIMARY store (not fallback)
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
        root_data = add_metaxy_provenance_column(root_data, RootFeature)
        primary_store.write(RootFeature, root_data)

        # Data is local, so versioning_engine="native" should work without errors or warnings
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", PolarsMaterializationWarning)
            result = primary_store.resolve_update(DownstreamFeature)

        # Verify results are correct
        assert len(result.new) == 3


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@pytest.mark.parametrize("store_type", get_available_store_types_for_fallback())
@skip_exception(HashAlgorithmNotSupportedError, "not supported")
def test_versioning_engine_native_warns_when_fallback_actually_used(
    tmp_path: Path,
    hash_algorithm: HashAlgorithm,
    store_type: str,
    RootFeature,
    DownstreamFeature,
):
    """Test that versioning_engine="native" warns (but doesn't error) when fallback is accessed.

    When fallback stores are configured AND actually used (data only exists in fallback),
    versioning_engine="native" should issue a warning but not raise an error, because
    the implementation mismatch is due to legitimate fallback access.
    """
    fallback_store = DeltaMetadataStore(
        root_path=tmp_path / "delta_fallback",
        hash_algorithm=hash_algorithm,
    )

    # Write data ONLY to fallback store
    with fallback_store:
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
        root_data = add_metaxy_provenance_column(root_data, RootFeature)
        fallback_store.write(RootFeature, root_data)

    # Create primary store with fallback configured and versioning_engine="native"
    primary_store = create_store_for_fallback(
        store_type,
        versioning_engine="native",
        hash_algorithm=hash_algorithm,
        tmp_path=tmp_path,
        suffix="fallback_access_test",
        fallback_stores=[fallback_store],
    )

    with primary_store, fallback_store:
        # Data is ONLY in fallback, so reading will cause implementation mismatch
        # Should warn but NOT raise VersioningEngineMismatchError
        with pytest.warns(
            PolarsMaterializationWarning,
            match="Using Polars for resolving the increment instead",
        ):
            result = primary_store.resolve_update(DownstreamFeature)

        # Should NOT raise VersioningEngineMismatchError
        assert len(result.new) == 3


# NOTE: There is no test for samples with wrong implementation because:
# 1. The code at base.py:336 correctly raises VersioningEngineMismatchError for samples
#    with wrong implementation WITHOUT checking fallback_stores (unlike line 314 which does check)
# 2. This is the correct behavior: samples come from user argument, not from fallback stores
# 3. Testing this is complex because samples need proper metadata columns before the
#    implementation check, and creating native (Ibis) samples in tests is non-trivial


def test_fallback_stores_opened_on_demand_when_reading(tmp_path, graph: FeatureGraph) -> None:
    """Test that fallback stores are opened on demand when reading metadata.

    This tests the fix for a bug where fallback stores were not opened when
    the primary store tried to read from them, resulting in StoreNotOpenError
    being raised (or worse, being masked as FeatureNotFoundError).
    """
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    class TestFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["fallback_open_test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    dev_path = tmp_path / "dev"
    branch_path = tmp_path / "branch"

    # Create stores with fallback chain
    branch_store = DeltaMetadataStore(root_path=branch_path)
    dev_store = DeltaMetadataStore(root_path=dev_path, fallback_stores=[branch_store])

    # Write data to the fallback (branch) store
    with branch_store.open(mode="w"):
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        branch_store.write(TestFeature, metadata)

    # Now open only the dev store and try to read - it should find data in fallback
    # The fallback store should be opened on demand
    with dev_store:
        result = dev_store.read(TestFeature)
        assert result is not None
        collected = result.collect()
        assert len(collected) == 3


def test_get_store_metadata_respects_fallback_stores(tmp_path, graph: FeatureGraph) -> None:
    """Test that get_store_metadata returns metadata from fallback store when feature is only there.

    This tests the fix for issue #549: MetadataStore.get_store_metadata should respect
    fallback stores, returning metadata from the fallback store where the feature is
    actually found when it doesn't exist in the current store.
    """
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    class TestFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["get_store_metadata_fallback_test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    primary_path = tmp_path / "primary"
    fallback_path = tmp_path / "fallback"

    # Create stores with fallback chain
    fallback_store = DeltaMetadataStore(root_path=fallback_path)
    primary_store = DeltaMetadataStore(root_path=primary_path, fallback_stores=[fallback_store])

    # Write data to the fallback store only
    with fallback_store.open(mode="w"):
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        fallback_store.write(TestFeature, metadata)

    # Now open only the primary store and call get_store_metadata
    # It should return metadata showing both primary and resolved_from (fallback) stores
    with primary_store:
        store_metadata = primary_store.get_store_metadata(TestFeature)

        # Should include info about the queried store (primary)
        assert "name" in store_metadata
        assert "display" in store_metadata
        assert "primary" in store_metadata["display"]

        # Should include resolved_from with info about where the feature was found
        assert "resolved_from" in store_metadata
        resolved_from = store_metadata["resolved_from"]
        assert "name" in resolved_from
        assert "type" in resolved_from
        assert "display" in resolved_from
        assert "fallback" in resolved_from["display"]

        # Store-specific metadata (uri) should be inside resolved_from
        assert "uri" in resolved_from
        assert "fallback" in resolved_from["uri"]

    # Test check_fallback=False - should return only queried store info when feature not found
    with primary_store:
        store_metadata_no_fallback = primary_store.get_store_metadata(TestFeature, check_fallback=False)
        # Should still return the queried store's info
        assert "name" in store_metadata_no_fallback
        assert "display" in store_metadata_no_fallback
        # But no resolved_from since feature wasn't found
        assert "resolved_from" not in store_metadata_no_fallback


def test_get_store_metadata_prefers_current_store(tmp_path, graph: FeatureGraph) -> None:
    """Test that get_store_metadata returns metadata from current store when feature exists there.

    Even when a fallback store has the same feature, get_store_metadata should return
    metadata from the current store (not the fallback).
    """
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    class TestFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["get_store_metadata_prefer_current_test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    primary_path = tmp_path / "primary"
    fallback_path = tmp_path / "fallback"

    # Create stores with fallback chain
    fallback_store = DeltaMetadataStore(root_path=fallback_path)
    primary_store = DeltaMetadataStore(root_path=primary_path, fallback_stores=[fallback_store])

    metadata = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"default": "h1"},
                {"default": "h2"},
                {"default": "h3"},
            ],
        }
    )

    # Write data to both stores
    with fallback_store.open(mode="w"):
        fallback_store.write(TestFeature, metadata)

    with primary_store.open(mode="w"):
        primary_store.write(TestFeature, metadata)

    # get_store_metadata should return metadata from primary (current) store
    with primary_store:
        store_metadata = primary_store.get_store_metadata(TestFeature)

        assert store_metadata is not None
        # resolved_from should point to primary store since feature exists there
        assert "resolved_from" in store_metadata
        resolved_from = store_metadata["resolved_from"]
        assert "uri" in resolved_from
        # Should be the primary store's path, not the fallback
        assert "primary" in resolved_from["uri"]
        assert "fallback" not in resolved_from["uri"]


def test_read_with_store_info(tmp_path, graph: FeatureGraph) -> None:
    """Test read with_store_info returns the resolved store.

    When with_store_info=True, read should return a tuple of
    (LazyFrame, MetadataStore) where the MetadataStore is the store that
    actually contained the feature (which may be a fallback store).
    """
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    class TestFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["read_with_store_info_test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    primary_path = tmp_path / "primary"
    fallback_path = tmp_path / "fallback"

    # Create stores with fallback chain
    fallback_store = DeltaMetadataStore(root_path=fallback_path, name="fallback")
    primary_store = DeltaMetadataStore(root_path=primary_path, name="primary", fallback_stores=[fallback_store])

    # Write data to the fallback store only
    with fallback_store.open(mode="w"):
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        fallback_store.write(TestFeature, metadata)

    # Read from primary with with_store_info=True - should resolve to fallback
    with primary_store:
        df, resolved_store = primary_store.read(TestFeature, with_store_info=True)
        assert resolved_store.name == "fallback"
        assert resolved_store is not primary_store
        # Verify the data is correct
        collected = df.collect()
        assert len(collected) == 3

    # Now write to primary store too
    with primary_store.open(mode="w"):
        primary_store.write(TestFeature, metadata)

    # Read from primary with with_store_info=True - should resolve to primary
    with primary_store:
        df, resolved_store = primary_store.read(TestFeature, with_store_info=True)
        assert resolved_store.name == "primary"
        assert resolved_store is primary_store
        collected = df.collect()
        assert len(collected) == 3

    # Verify default behavior (with_store_info=False) still works
    with primary_store:
        df = primary_store.read(TestFeature)
        # Should return just LazyFrame, not tuple
        assert hasattr(df, "collect")
        collected = df.collect()
        assert len(collected) == 3
