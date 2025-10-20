"""Test resolve_update across different stores and hash algorithms.

Tests that resolve_update (which orchestrates joiner/calculator/diff components)
produces consistent results across store types and hash algorithms.

NOTE: Currently both InMemory and DuckDB stores use Polars components for computation
(DuckDB's _can_compute_native() returns False). When native SQL computation is implemented
for DuckDB, these tests will automatically cover both native and Polars paths.

The computation path (native vs polars) is determined by:
1. Store's _can_compute_native() method
2. Whether all upstream data is local (all_local) or in fallback stores (has_fallback)

Current paths:
- InMemory: Always Polars (no native implementation)
- DuckDB: Always Polars (_can_compute_native() = False, pending native SQL implementation)
"""

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from metaxy import (
    ContainerDep,
    ContainerKey,
    ContainerSpec,
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    InMemoryMetadataStore,
    MetadataStore,
)
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.feature import FeatureRegistry

from .conftest import HashAlgorithmCases  # type: ignore[import-not-found]

# ============= STORE CONFIGURATION =============


def get_available_store_types() -> list[str]:
    """Get list of available store types from StoreCases.

    Dynamically discovers which store types are available by inspecting
    the StoreCases class for case methods. This allows different branches
    to have different store types without hardcoding the list.
    """
    from .conftest import StoreCases

    store_types = []
    for attr_name in dir(StoreCases):
        if attr_name.startswith("case_") and not attr_name.startswith("case__"):
            # Extract store type name from case method (e.g., "case_duckdb" -> "duckdb")
            store_type = attr_name[5:]  # Remove "case_" prefix
            store_types.append(store_type)

    return store_types


def assert_all_results_equal(results: dict, snapshot=None) -> None:
    """Compare all results from different store type combinations.

    Ensures all variants produce identical results, then optionally snapshots all results.

    Args:
        results: Dict mapping store_type to result data
        snapshot: Optional syrupy snapshot fixture to record all results

    Raises:
        AssertionError: If any variants produce different results
    """
    if not results:
        return

    # Get all result values as a list
    all_results = list(results.items())
    reference_key, reference_result = all_results[0]

    # Compare each result to the reference
    for key, result in all_results[1:]:
        assert result == reference_result, (
            f"{key} produced different results than {reference_key}:\n"
            f"Expected: {reference_result}\n"
            f"Got: {result}"
        )

    # Snapshot ALL results if snapshot provided
    if snapshot is not None:
        assert results == snapshot


def create_store(
    store_type: str,
    prefer_native: bool,
    hash_algorithm: HashAlgorithm,
    params: dict[str, Any],
) -> MetadataStore:
    """Create a store instance for testing.

    Args:
        store_type: "inmemory", "duckdb", "sqlite", or "clickhouse"
        prefer_native: Whether to prefer native components
        hash_algorithm: Hash algorithm to use
        params: Store-specific parameters dict containing:
            - tmp_path: Temporary directory for file-based stores (duckdb, sqlite)
            - clickhouse_db: Connection string for ClickHouse (optional)

    Returns:
        Configured metadata store instance
    """
    if store_type == "inmemory":
        return InMemoryMetadataStore(
            hash_algorithm=hash_algorithm, prefer_native=prefer_native
        )
    elif store_type == "duckdb":
        tmp_path = params.get("tmp_path")
        if tmp_path is None:
            raise ValueError("tmp_path parameter required for duckdb store type")
        db_path = (
            tmp_path
            / f"test_{store_type}_{hash_algorithm.value}_{prefer_native}.duckdb"
        )
        extensions: list[str] = (
            ["hashfuncs"]
            if hash_algorithm in [HashAlgorithm.XXHASH32, HashAlgorithm.XXHASH64]
            else []
        )
        return DuckDBMetadataStore(
            db_path,
            hash_algorithm=hash_algorithm,
            extensions=extensions,  # type: ignore[arg-type]
            prefer_native=prefer_native,
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# ============= FEATURE LIBRARY =============
# Define all features once, then add them to registries on demand


# Root features (no dependencies)
class RootA(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["resolve", "root_a"]),
        deps=None,
        containers=[
            ContainerSpec(key=ContainerKey(["default"]), code_version=1),
        ],
    ),
):
    """Root feature with single default container."""

    pass


class MultiContainerRoot(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["resolve", "multi_root"]),
        deps=None,
        containers=[
            ContainerSpec(key=ContainerKey(["train"]), code_version=1),
            ContainerSpec(key=ContainerKey(["test"]), code_version=1),
        ],
    ),
):
    """Root feature with multiple containers."""

    pass


# Intermediate features (depend on roots)
class BranchB(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["resolve", "branch_b"]),
        deps=[FeatureDep(key=FeatureKey(["resolve", "root_a"]))],
        containers=[
            ContainerSpec(
                key=ContainerKey(["default"]),
                code_version=1,
                deps=[
                    ContainerDep(
                        feature_key=FeatureKey(["resolve", "root_a"]),
                        containers=[ContainerKey(["default"])],
                    )
                ],
            ),
        ],
    ),
):
    """Branch depending on RootA."""

    pass


class BranchC(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["resolve", "branch_c"]),
        deps=[FeatureDep(key=FeatureKey(["resolve", "root_a"]))],
        containers=[
            ContainerSpec(
                key=ContainerKey(["default"]),
                code_version=1,
                deps=[
                    ContainerDep(
                        feature_key=FeatureKey(["resolve", "root_a"]),
                        containers=[ContainerKey(["default"])],
                    )
                ],
            ),
        ],
    ),
):
    """Another branch depending on RootA."""

    pass


# Leaf features (converge multiple branches)
class LeafSimple(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["resolve", "leaf_simple"]),
        deps=[FeatureDep(key=FeatureKey(["resolve", "root_a"]))],
        containers=[
            ContainerSpec(
                key=ContainerKey(["default"]),
                code_version=1,
                deps=[
                    ContainerDep(
                        feature_key=FeatureKey(["resolve", "root_a"]),
                        containers=[ContainerKey(["default"])],
                    )
                ],
            ),
        ],
    ),
):
    """Simple leaf depending on single root."""

    pass


class LeafDiamond(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["resolve", "leaf_diamond"]),
        deps=[
            FeatureDep(key=FeatureKey(["resolve", "branch_b"])),
            FeatureDep(key=FeatureKey(["resolve", "branch_c"])),
        ],
        containers=[
            ContainerSpec(
                key=ContainerKey(["default"]),
                code_version=1,
                deps=[
                    ContainerDep(
                        feature_key=FeatureKey(["resolve", "branch_b"]),
                        containers=[ContainerKey(["default"])],
                    ),
                    ContainerDep(
                        feature_key=FeatureKey(["resolve", "branch_c"]),
                        containers=[ContainerKey(["default"])],
                    ),
                ],
            ),
        ],
    ),
):
    """Diamond leaf converging two branches."""

    pass


class LeafMultiContainer(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["resolve", "leaf_multi"]),
        deps=[FeatureDep(key=FeatureKey(["resolve", "multi_root"]))],
        containers=[
            ContainerSpec(
                key=ContainerKey(["default"]),
                code_version=1,
                deps=[
                    ContainerDep(
                        feature_key=FeatureKey(["resolve", "multi_root"]),
                        containers=[
                            ContainerKey(["train"]),
                            ContainerKey(["test"]),
                        ],
                    )
                ],
            ),
        ],
    ),
):
    """Leaf depending on multi-container root."""

    pass


# ============= REGISTRY FIXTURES =============


class RegistryCases:
    """Different feature registry configurations with various dependency graphs."""

    def case_simple_chain(self) -> tuple[FeatureRegistry, list[type[Feature]]]:
        """Simple dependency chain: A → B."""
        registry = FeatureRegistry()
        registry.add_feature(RootA)
        registry.add_feature(LeafSimple)
        return (registry, [RootA, LeafSimple])

    def case_diamond_graph(self) -> tuple[FeatureRegistry, list[type[Feature]]]:
        """Diamond dependency graph: A → B, A → C, B → D, C → D."""
        registry = FeatureRegistry()
        registry.add_feature(RootA)
        registry.add_feature(BranchB)
        registry.add_feature(BranchC)
        registry.add_feature(LeafDiamond)
        return (registry, [RootA, BranchB, BranchC, LeafDiamond])

    def case_multi_container(self) -> tuple[FeatureRegistry, list[type[Feature]]]:
        """Feature with multiple containers: A (train + test) → B."""
        registry = FeatureRegistry()
        registry.add_feature(MultiContainerRoot)
        registry.add_feature(LeafMultiContainer)
        return (registry, [MultiContainerRoot, LeafMultiContainer])


@pytest.fixture
def simple_chain_registry():
    """Simple chain registry fixture for tests that don't need parametrization."""
    cases = RegistryCases()
    registry, features = cases.case_simple_chain()

    with registry.use():
        # Return dict with named features for backwards compatibility
        # features = [RootA, LeafSimple]
        yield {"UpstreamA": features[0], "DownstreamB": features[1]}


# ============= TESTS =============


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@parametrize_with_cases("registry_config", cases=RegistryCases)
def test_resolve_update_no_upstream(
    tmp_path: Path,
    registry_config: tuple[FeatureRegistry, list[type[Feature]]],
    hash_algorithm: HashAlgorithm,
    snapshot,
    recwarn,
):
    """Test resolve_update for a feature with no upstream dependencies.

    Verifies that prefer_native=True and prefer_native=False produce identical results
    across all store types and different feature graphs.
    """
    registry, features = registry_config

    # Test the first feature (root/upstream with no dependencies)
    root_feature = features[0]

    # Iterate over store types and prefer_native variants
    results = {}

    for store_type in get_available_store_types():
        for prefer_native in [True, False]:
            store = create_store(
                store_type,
                prefer_native,
                hash_algorithm,
                params={"tmp_path": tmp_path},
            )

            # Try to open store - will fail validation if hash algorithm not supported
            try:
                with store, registry.use():
                    # Drop all feature metadata to ensure clean state between prefer_native variants
                    for feature in features:
                        store.drop_feature_metadata(feature)

                    # Root feature has no dependencies - resolve_update should return empty result
                    # since there's no metadata yet
                    recwarn.clear()
                    result = store.resolve_update(root_feature)

                    # Assert no warning when prefer_native=True for stores that support native components
                    # Check if store actually supports native components
                    if prefer_native and store._supports_native_components():
                        prefer_native_warnings = [
                            w
                            for w in recwarn
                            if "prefer_native=True but using Polars" in str(w.message)
                        ]
                        assert len(prefer_native_warnings) == 0, (
                            f"Unexpected warning with prefer_native=True for {store_type}: "
                            f"{[str(w.message) for w in prefer_native_warnings]}"
                        )

                    # Collect results
                    results[(store_type, prefer_native)] = {
                        "added": len(result.added),
                        "changed": len(result.changed),
                        "removed": len(result.removed),
                    }
            except HashAlgorithmNotSupportedError:
                # Hash algorithm not supported by this store - skip
                continue

    # All variants should produce identical results
    assert_all_results_equal(results, snapshot)

    # Verify expected behavior: no samples
    if results:
        reference = list(results.values())[0]
        assert reference == {"added": 0, "changed": 0, "removed": 0}


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@parametrize_with_cases("registry_config", cases=RegistryCases)
def test_resolve_update_with_upstream(
    tmp_path: Path,
    registry_config: tuple[FeatureRegistry, list[type[Feature]]],
    hash_algorithm: HashAlgorithm,
    snapshot,
    recwarn,
):
    """Test resolve_update calculates data versions from upstream.

    Verifies that prefer_native=True and prefer_native=False produce identical results
    across all store types and different feature graphs.
    """
    registry, features = registry_config

    # Get root feature (first) and a downstream feature (last)
    root_feature = features[0]
    downstream_feature = features[-1]

    # Helper to create data_version dicts for a feature's containers
    def make_data_versions(feature: type[Feature], prefix: str) -> list[dict]:
        containers = [c.key for c in feature.spec.containers]
        if len(containers) == 1:
            container_name = "_".join(containers[0])
            return [{container_name: f"{prefix}_{i}"} for i in [1, 2, 3]]
        else:
            return [
                {"_".join(ck): f"{prefix}_{i}_{ck}" for ck in containers}
                for i in [1, 2, 3]
            ]

    # Create upstream data for root feature
    upstream_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": ["a1", "a2", "a3"],
            "data_version": make_data_versions(root_feature, "manual"),
        }
    )

    # Iterate over store types and prefer_native variants
    results = {}

    for store_type in get_available_store_types():
        for prefer_native in [True, False]:
            store = create_store(
                store_type,
                prefer_native,
                hash_algorithm,
                params={"tmp_path": tmp_path},
            )

            # Try to open store - will fail validation if hash algorithm not supported
            try:
                with store, registry.use():
                    # Drop all feature metadata to ensure clean state between prefer_native variants
                    for feature in features:
                        store.drop_feature_metadata(feature)

                    recwarn.clear()

                    # Write metadata for all features except the last one
                    # This simulates a realistic scenario where all upstream features have been computed
                    for i, feature in enumerate(features[:-1]):
                        if i == 0:
                            # Root feature - use manual data
                            store.write_metadata(feature, upstream_data)
                        else:
                            # Intermediate features - resolve and write their data versions
                            result = store.resolve_update(feature)
                            if len(result.added) > 0:
                                # Write metadata with computed data versions
                                feature_data = pl.DataFrame(
                                    {
                                        "sample_id": [1, 2, 3],
                                        "value": [f"f{i}_1", f"f{i}_2", f"f{i}_3"],
                                    }
                                ).with_columns(
                                    pl.Series(
                                        "data_version",
                                        result.added.sort("sample_id")[
                                            "data_version"
                                        ].to_list(),
                                    )
                                )
                                store.write_metadata(feature, feature_data)

                    # Now resolve update for the final downstream feature
                    result = store.resolve_update(downstream_feature)

                    # Assert no warning when prefer_native=True for stores that support native components
                    if prefer_native and store._supports_native_components():
                        prefer_native_warnings = [
                            w
                            for w in recwarn
                            if "prefer_native=True but using Polars" in str(w.message)
                        ]
                        assert len(prefer_native_warnings) == 0, (
                            f"Unexpected warning with prefer_native=True for {store_type}: "
                            f"{[str(w.message) for w in prefer_native_warnings]}"
                        )

                    # Collect data versions
                    added_sorted = result.added.sort("sample_id")
                    versions = added_sorted["data_version"].to_list()

                    results[(store_type, prefer_native)] = {
                        "added": len(result.added),
                        "changed": len(result.changed),
                        "removed": len(result.removed),
                        "versions": versions,
                    }
            except HashAlgorithmNotSupportedError:
                # Hash algorithm not supported by this store - skip
                continue

    # All variants should produce identical results
    assert_all_results_equal(results, snapshot)

    # Verify expected behavior
    if results:
        reference = list(results.values())[0]
        assert reference["added"] == 3, "All 3 samples should be added"
        assert reference["changed"] == 0
        assert reference["removed"] == 0
        assert all(isinstance(v, dict) for v in reference["versions"])
        # Verify versions contain expected container keys
        downstream_containers = [c.key for c in downstream_feature.spec.containers]
        for version_dict in reference["versions"]:
            for container_key in downstream_containers:
                container_name = "_".join(container_key)
                assert container_name in version_dict


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@parametrize_with_cases("registry_config", cases=RegistryCases)
def test_resolve_update_detects_changes(
    tmp_path: Path,
    registry_config: tuple[FeatureRegistry, list[type[Feature]]],
    hash_algorithm: HashAlgorithm,
    snapshot,
    recwarn,
):
    """Test resolve_update detects changed data versions.

    Verifies that prefer_native=True and prefer_native=False produce identical results
    across all store types and different feature graphs.
    """
    registry, features = registry_config

    # Get root feature (first) and a downstream feature (last)
    root_feature = features[0]
    downstream_feature = features[-1]

    # Determine which containers to use
    root_containers = [c.key for c in root_feature.spec.containers]

    # Create data version dicts for containers
    def make_data_versions(version_prefix: str, sample_ids: list[int]) -> list[dict]:
        if len(root_containers) == 1:
            container_name = "_".join(root_containers[0])
            return [{container_name: f"{version_prefix}{i}"} for i in sample_ids]
        else:
            return [
                {"_".join(ck): f"{version_prefix}{i}_{ck}" for ck in root_containers}
                for i in sample_ids
            ]

    # Initial upstream data
    initial_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": ["a1", "a2", "a3"],
            "data_version": make_data_versions("v", [1, 2, 3]),
        }
    )

    # Changed upstream data (change sample 2)
    changed_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": ["a1", "a2_CHANGED", "a3"],  # Changed
            "data_version": make_data_versions(
                "v_new", [1, 2, 3]
            ),  # All changed to new version
        }
    )

    # Iterate over store types and prefer_native variants
    results = {}

    for store_type in get_available_store_types():
        for prefer_native in [True, False]:
            store = create_store(
                store_type,
                prefer_native,
                hash_algorithm,
                params={"tmp_path": tmp_path},
            )

            # Try to open store - will fail validation if hash algorithm not supported
            try:
                with store, registry.use():
                    # Drop all feature metadata to ensure clean state between prefer_native variants
                    for feature in features:
                        store.drop_feature_metadata(feature)

                    recwarn.clear()

                    # Write metadata for all features except the last one with initial data
                    for i, feature in enumerate(features[:-1]):
                        if i == 0:
                            # Root feature - use initial data
                            store.write_metadata(feature, initial_data)
                        else:
                            # Intermediate features - resolve and write their data versions
                            result = store.resolve_update(feature)
                            if len(result.added) > 0:
                                feature_data = pl.DataFrame(
                                    {
                                        "sample_id": [1, 2, 3],
                                        "value": [f"f{i}_1", f"f{i}_2", f"f{i}_3"],
                                    }
                                ).with_columns(
                                    pl.Series(
                                        "data_version",
                                        result.added.sort("sample_id")[
                                            "data_version"
                                        ].to_list(),
                                    )
                                )
                                store.write_metadata(feature, feature_data)

                    # First resolve - get initial data versions for downstream
                    result1 = store.resolve_update(downstream_feature)
                    initial_versions = result1.added.sort("sample_id")

                    # Write downstream feature with these versions
                    downstream_data = pl.DataFrame(
                        {
                            "sample_id": [1, 2, 3],
                            "feature_data": ["b1", "b2", "b3"],
                        }
                    ).with_columns(
                        pl.Series(
                            "data_version", initial_versions["data_version"].to_list()
                        )
                    )
                    store.write_metadata(downstream_feature, downstream_data)

                    # Now change upstream root feature and propagate through intermediate features
                    store.write_metadata(root_feature, changed_data)

                    # Update intermediate features with new data versions
                    for i, feature in enumerate(features[1:-1], start=1):
                        result = store.resolve_update(feature)
                        if len(result.changed) > 0 or len(result.added) > 0:
                            # Combine changed and added
                            updated = (
                                pl.concat([result.changed, result.added])
                                if len(result.added) > 0
                                else result.changed
                            )
                            feature_data = pl.DataFrame(
                                {
                                    "sample_id": [1, 2, 3],
                                    "value": [
                                        f"f{i}_1_new",
                                        f"f{i}_2_new",
                                        f"f{i}_3_new",
                                    ],
                                }
                            ).join(
                                updated.select(["sample_id", "data_version"]),
                                on="sample_id",
                                how="left",
                            )
                            store.write_metadata(feature, feature_data)

                    # Resolve update again for downstream - should detect changes
                    result2 = store.resolve_update(downstream_feature)

                    # Assert no warning when prefer_native=True for stores that support native components
                    if prefer_native and store._supports_native_components():
                        prefer_native_warnings = [
                            w
                            for w in recwarn
                            if "prefer_native=True but using Polars" in str(w.message)
                        ]
                        assert len(prefer_native_warnings) == 0, (
                            f"Unexpected warning with prefer_native=True for {store_type}: "
                            f"{[str(w.message) for w in prefer_native_warnings]}"
                        )

                    # Collect results
                    changed_sample_ids = (
                        result2.changed["sample_id"].to_list()
                        if len(result2.changed) > 0
                        else []
                    )

                    # Check if any versions changed
                    version_changes = []
                    for sample_id in changed_sample_ids:
                        new_version = result2.changed.filter(
                            pl.col("sample_id") == sample_id
                        )["data_version"][0]
                        old_version = initial_versions.filter(
                            pl.col("sample_id") == sample_id
                        )["data_version"][0]
                        version_changes.append(new_version != old_version)

                    results[(store_type, prefer_native)] = {
                        "added": len(result2.added),
                        "changed": len(result2.changed),
                        "removed": len(result2.removed),
                        "changed_sample_ids": sorted(changed_sample_ids),
                        "any_version_changed": any(version_changes)
                        if version_changes
                        else False,
                    }
            except HashAlgorithmNotSupportedError:
                # Hash algorithm not supported by this store - skip
                continue

    # All variants should produce identical results
    assert_all_results_equal(results, snapshot)

    # Verify expected behavior
    if results:
        reference = list(results.values())[0]
        assert reference["added"] == 0, "No new samples"
        assert reference["changed"] >= 1, "Should detect at least 1 changed sample"
        assert reference["removed"] == 0, "No removed samples"
        assert len(reference["changed_sample_ids"]) >= 1, (
            "At least one sample should change"
        )
        assert reference["any_version_changed"], "Data version should have changed"
