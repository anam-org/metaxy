"""Test resolve_update across different stores and hash algorithms.

Tests that resolve_update (which orchestrates joiner/calculator/diff components)
produces consistent results across store types and hash algorithms.

With Narwhals integration, all stores use a unified Narwhals-based path that works
with any backend (Polars in-memory, Ibis/SQL for databases). Backend-specific
optimizations (e.g., staying in SQL vs pulling to memory) are handled transparently.
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
    TestingFeatureSpec,
)
from metaxy._testing import HashAlgorithmCases, assert_all_results_equal
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    InMemoryMetadataStore,
    MetadataStore,
)
from metaxy.metadata_store.clickhouse import ClickHouseMetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.feature import FeatureGraph

# ============= STORE CONFIGURATION =============


def get_available_store_types() -> list[str]:
    """Get list of available store types from StoreCases.

    Dynamically discovers which store types are available by inspecting
    the StoreCases class for case methods. This allows different branches
    to have different store types without hardcoding the list.
    """
    from .conftest import StoreCases  # type: ignore[import-not-found]

    store_types = []
    for attr_name in dir(StoreCases):
        if attr_name.startswith("case_") and not attr_name.startswith("case__"):
            # Extract store type name from case method (e.g., "case_duckdb" -> "duckdb")
            store_type = attr_name[5:]  # Remove "case_" prefix
            store_types.append(store_type)

    return store_types


def create_store(
    store_type: str,
    prefer_native: bool,
    hash_algorithm: HashAlgorithm,
    params: dict[str, Any],
) -> MetadataStore:
    """Create a store instance for testing.

    Args:
        store_type: "inmemory", "duckdb", or "clickhouse"
        prefer_native: Whether to prefer native data version calculations
        hash_algorithm: Hash algorithm to use
        params: Store-specific parameters dict containing:
            - tmp_path: Temporary directory for file-based stores (duckdb)
            - clickhouse_db: Connection string for ClickHouse (optional)

    Returns:
        Configured metadata store instance
    """
    tmp_path = params.get("tmp_path")

    if store_type == "inmemory":
        return InMemoryMetadataStore(
            hash_algorithm=hash_algorithm, prefer_native=prefer_native
        )
    elif store_type == "duckdb":
        assert tmp_path is not None, f"tmp_path parameter required for {store_type}"
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
            extensions=extensions,  # pyright: ignore[reportArgumentType]
            prefer_native=prefer_native,
        )
    elif store_type == "duckdb_ducklake":
        assert tmp_path is not None, f"tmp_path parameter required for {store_type}"
        db_path = (
            tmp_path
            / f"test_{store_type}_{hash_algorithm.value}_{prefer_native}.duckdb"
        )
        metadata_path = (
            tmp_path
            / f"test_{store_type}_{hash_algorithm.value}_{prefer_native}_catalog.duckdb"
        )
        storage_dir = (
            tmp_path
            / f"test_{store_type}_{hash_algorithm.value}_{prefer_native}_storage"
        )

        ducklake_config = {
            "alias": "integration_lake",
            "metadata_backend": {"type": "duckdb", "path": str(metadata_path)},
            "storage_backend": {"type": "local", "path": str(storage_dir)},
        }

        extensions_ducklake: list[str] = ["json"]
        if hash_algorithm in [HashAlgorithm.XXHASH32, HashAlgorithm.XXHASH64]:
            extensions_ducklake.append("hashfuncs")

        return DuckDBMetadataStore(
            db_path,
            hash_algorithm=hash_algorithm,
            extensions=extensions_ducklake,  # pyright: ignore[reportArgumentType]
            prefer_native=prefer_native,
            ducklake=ducklake_config,
        )
    elif store_type == "clickhouse":
        clickhouse_db = params.get("clickhouse_db")
        if clickhouse_db is None:
            raise ValueError(
                "clickhouse_db parameter required for clickhouse store type"
            )
        # ClickHouse uses the same database but tables will be unique per feature
        # However, prefer_native variants need isolation since they write to same tables
        # We solve this by using the same connection - the fixture provides a clean database per test
        # Each test gets its own database, so prefer_native variants within the same test share tables
        # This is the root cause of duplicates - we need to drop tables between variants
        return ClickHouseMetadataStore(
            clickhouse_db,
            hash_algorithm=hash_algorithm,
            prefer_native=prefer_native,
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# ============= FEATURE LIBRARY =============
# Define all features once, then add them to registries on demand


# Root features (no dependencies)
class RootA(
    Feature,
    spec=TestingFeatureSpec(
        key=FeatureKey(["resolve", "root_a"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    ),
):
    """Root feature with single default field."""

    pass


class MultiFieldRoot(
    Feature,
    spec=TestingFeatureSpec(
        key=FeatureKey(["resolve", "multi_root"]),
        fields=[
            FieldSpec(key=FieldKey(["train"]), code_version=1),
            FieldSpec(key=FieldKey(["test"]), code_version=1),
        ],
    ),
):
    """Root feature with multiple fields."""

    pass


# Intermediate features (depend on roots)
class BranchB(
    Feature,
    spec=TestingFeatureSpec(
        key=FeatureKey(["resolve", "branch_b"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "root_a"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature=FeatureKey(["resolve", "root_a"]),
                        fields=[FieldKey(["default"])],
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
    spec=TestingFeatureSpec(
        key=FeatureKey(["resolve", "branch_c"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "root_a"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature=FeatureKey(["resolve", "root_a"]),
                        fields=[FieldKey(["default"])],
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
    spec=TestingFeatureSpec(
        key=FeatureKey(["resolve", "leaf_simple"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "root_a"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature=FeatureKey(["resolve", "root_a"]),
                        fields=[FieldKey(["default"])],
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
    spec=TestingFeatureSpec(
        key=FeatureKey(["resolve", "leaf_diamond"]),
        deps=[
            FeatureDep(feature=FeatureKey(["resolve", "branch_b"]), columns=()),
            FeatureDep(feature=FeatureKey(["resolve", "branch_c"]), columns=()),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature=FeatureKey(["resolve", "branch_b"]),
                        fields=[FieldKey(["default"])],
                    ),
                    FieldDep(
                        feature=FeatureKey(["resolve", "branch_c"]),
                        fields=[FieldKey(["default"])],
                    ),
                ],
            ),
        ],
    ),
):
    """Diamond leaf converging two branches."""

    pass


class LeafMultiField(
    Feature,
    spec=TestingFeatureSpec(
        key=FeatureKey(["resolve", "leaf_multi"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "multi_root"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature=FeatureKey(["resolve", "multi_root"]),
                        fields=[
                            FieldKey(["train"]),
                            FieldKey(["test"]),
                        ],
                    )
                ],
            ),
        ],
    ),
):
    """Leaf depending on multi-field root."""

    pass


# ============= REGISTRY FIXTURES =============


class FeatureGraphCases:
    """Different feature graph configurations with various dependency graphs."""

    def case_simple_chain(self) -> tuple[FeatureGraph, list[type[Feature]]]:
        """Simple dependency chain: A → B."""
        graph = FeatureGraph()
        graph.add_feature(RootA)
        graph.add_feature(LeafSimple)
        return (graph, [RootA, LeafSimple])

    def case_diamond_graph(self) -> tuple[FeatureGraph, list[type[Feature]]]:
        """Diamond dependency graph: A → B, A → C, B → D, C → D."""
        graph = FeatureGraph()
        graph.add_feature(RootA)
        graph.add_feature(BranchB)
        graph.add_feature(BranchC)
        graph.add_feature(LeafDiamond)
        return (graph, [RootA, BranchB, BranchC, LeafDiamond])

    def case_multi_field(self) -> tuple[FeatureGraph, list[type[Feature]]]:
        """Feature with multiple fields: A (train + test) → B."""
        graph = FeatureGraph()
        graph.add_feature(MultiFieldRoot)
        graph.add_feature(LeafMultiField)
        return (graph, [MultiFieldRoot, LeafMultiField])


@pytest.fixture
def simple_chain_graph():
    """Simple chain graph fixture for tests that don't need parametrization."""
    cases = FeatureGraphCases()
    graph, features = cases.case_simple_chain()

    with graph.use():
        # Return dict with named features for backwards compatibility
        # features = [RootA, LeafSimple]
        yield {"UpstreamA": features[0], "DownstreamB": features[1]}


# ============= TESTS =============


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@parametrize_with_cases("graph_config", cases=FeatureGraphCases)
@pytest.mark.parametrize("prefer_native", [True, False])
@pytest.mark.parametrize("use_native_samples", [True, False])
def test_resolve_update_no_upstream(
    store_params: dict[str, Any],
    graph_config: tuple[FeatureGraph, list[type[Feature]]],
    hash_algorithm: HashAlgorithm,
    prefer_native: bool,
    use_native_samples: bool,
    snapshot,
    caplog,
):
    """Test resolve_update for root features with samples parameter.

    Root features have no upstream dependencies, so users must provide samples
    with manually computed data_version. This test verifies:
    1. ValueError raised when samples not provided
    2. Correct diff results when samples provided
    3. Warning issued when Polars samples provided to SQL store (use_native_samples=False)
    4. No warning when samples match store backend (use_native_samples=True)
    5. Results are consistent across all stores and prefer_native settings
    """
    import narwhals as nw
    import polars as pl

    graph, features = graph_config

    # Test the first feature (root/upstream with no dependencies)
    root_feature = features[0]

    # 1. Source of truth: Polars DataFrame with sample data
    source_samples = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "data_version": [
                {"field_a": "hash1"},
                {"field_a": "hash2"},
                {"field_a": "hash3"},
            ],
        }
    )

    # Collect results from all stores
    results = {}

    for store_type in get_available_store_types():
        # Skip duckdb_ducklake - it's a storage format that shouldn't affect
        # data version computations. Tested separately in test_duckdb.py::test_duckdb_ducklake_integration
        if store_type == "duckdb_ducklake":
            continue

        store = create_store(
            store_type,
            prefer_native,
            hash_algorithm,
            params=store_params,
        )

        try:
            with store, graph.use():
                # For ClickHouse, drop feature metadata to ensure clean state
                if store_type == "clickhouse":
                    store.drop_feature_metadata(root_feature)

                # First verify that resolve_update raises ValueError without samples
                try:
                    store.resolve_update(root_feature)
                    pytest.fail("Expected ValueError for root feature without samples")
                except ValueError as e:
                    assert "root feature" in str(e)

                # 2. Adapt samples based on use_native_samples parameter
                temp_table_name: str | None = None
                if use_native_samples and store._supports_native_components():
                    # Convert to Ibis for SQL stores to match backend
                    # Write samples to temp table and read back as Ibis table
                    from metaxy.metadata_store.ibis import IbisMetadataStore

                    assert isinstance(store, IbisMetadataStore), (
                        "Native stores must be Ibis-based"
                    )
                    temp_table_name = f"_test_samples_{root_feature.spec().key[-1]}"
                    store.conn.create_table(
                        temp_table_name, source_samples.to_arrow(), overwrite=True
                    )
                    ibis_samples = store.conn.table(temp_table_name)
                    adapted_samples = nw.from_native(ibis_samples)
                    should_warn = False
                else:
                    # Use Polars directly (either in-memory store or testing warning path)
                    adapted_samples = nw.from_native(source_samples)
                    # Should warn if SQL store but using Polars samples
                    should_warn = store._supports_native_components()

                # 3. Call resolve_update with adapted samples and capture logs
                import logging

                with caplog.at_level(logging.WARNING):
                    result = store.resolve_update(root_feature, samples=adapted_samples)

                # 4. Verify warning behavior matches expectation
                polars_warnings = [
                    record
                    for record in caplog.records
                    if "Polars-backed but store uses native SQL backend"
                    in record.message
                ]
                if should_warn:
                    assert len(polars_warnings) == 1, (
                        f"Expected Polars fallback warning for {store_type} with "
                        f"use_native_samples={use_native_samples}, but got {len(polars_warnings)} warnings"
                    )
                else:
                    assert len(polars_warnings) == 0, (
                        f"Unexpected Polars fallback warning for {store_type} with "
                        f"use_native_samples={use_native_samples}: {[r.message for r in polars_warnings]}"
                    )

                caplog.clear()

                # 5. Record results for comparison - include data versions
                # Convert Narwhals to Polars for manipulation
                added_sorted = (
                    result.added.to_polars()
                    if isinstance(result.added, nw.DataFrame)
                    else result.added
                ).sort("sample_uid")
                versions = added_sorted["data_version"].to_list()

                results[(store_type, prefer_native, use_native_samples)] = {
                    "added": len(result.added),
                    "changed": len(result.changed),
                    "removed": len(result.removed),
                    "versions": versions,
                }

                # Clean up temp table for SQL stores if created
                if (
                    temp_table_name is not None
                    and use_native_samples
                    and store._supports_native_components()
                ):
                    from metaxy.metadata_store.ibis import IbisMetadataStore

                    assert isinstance(store, IbisMetadataStore), (
                        "Native stores must be Ibis-based"
                    )
                    store.conn.drop_table(temp_table_name)

        except HashAlgorithmNotSupportedError:
            # Hash algorithm not supported by this store - continue with other stores
            continue

    # 6. Ensure results from all stores are correct and consistent
    if results:
        # All variants should produce identical results (including data versions)
        assert_all_results_equal(results, snapshot)

        # Verify expected behavior: all samples are new (added)
        reference = list(results.values())[0]
        assert reference["added"] == 3
        assert reference["changed"] == 0
        assert reference["removed"] == 0

        # Verify versions structure
        assert len(reference["versions"]) == 3
        assert all(isinstance(v, dict) for v in reference["versions"])
        # All versions should match the source (user provided them manually)
        assert reference["versions"] == [
            {"field_a": "hash1"},
            {"field_a": "hash2"},
            {"field_a": "hash3"},
        ]


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@parametrize_with_cases("graph_config", cases=FeatureGraphCases)
def test_resolve_update_with_upstream(
    store_params: dict[str, Any],
    graph_config: tuple[FeatureGraph, list[type[Feature]]],
    hash_algorithm: HashAlgorithm,
    snapshot,
    recwarn,
):
    """Test resolve_update calculates data versions from upstream.

    Verifies that prefer_native=True and prefer_native=False produce identical results
    across all store types and different feature graphs.
    """
    graph, features = graph_config

    # Get root feature (first) and a downstream feature (last)
    root_feature = features[0]
    downstream_feature = features[-1]

    # Helper to create data_version dicts for a feature's fields
    def make_data_versions(feature: type[Feature], prefix: str) -> list[dict[str, Any]]:
        fields = [c.key for c in feature.spec().fields]
        if len(fields) == 1:
            field_name = "_".join(fields[0])
            return [{field_name: f"{prefix}_{i}"} for i in [1, 2, 3]]
        else:
            return [
                {"_".join(ck): f"{prefix}_{i}_{ck}" for ck in fields} for i in [1, 2, 3]
            ]

    # Create upstream data for root feature
    upstream_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "value": ["a1", "a2", "a3"],
            "data_version": make_data_versions(root_feature, "manual"),
        }
    )

    # Iterate over store types and prefer_native variants
    results = {}

    for store_type in get_available_store_types():
        # Skip duckdb_ducklake - it's a storage format that shouldn't affect
        # data version computations. Tested separately in test_duckdb.py::test_duckdb_ducklake_integration
        if store_type == "duckdb_ducklake":
            continue

        for prefer_native in [True, False]:
            store = create_store(
                store_type,
                prefer_native,
                hash_algorithm,
                params=store_params,
            )

            # Try to open store - will fail validation if hash algorithm not supported
            try:
                with store, graph.use():
                    # For ClickHouse, drop feature metadata to ensure clean state between prefer_native variants
                    # since they share the same database (unlike DuckDB which uses separate files)
                    if store_type == "clickhouse":
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
                                        "sample_uid": [1, 2, 3],
                                        "value": [f"f{i}_1", f"f{i}_2", f"f{i}_3"],
                                    }
                                ).with_columns(
                                    pl.Series(
                                        "data_version",
                                        result.added.sort("sample_uid")[
                                            "data_version"
                                        ].to_list(),
                                    )
                                )
                                store.write_metadata(feature, feature_data)

                    # Now resolve update for the final downstream feature
                    result = store.resolve_update(downstream_feature)

                    # Assert no warning when prefer_native=True for stores that support native data version calculations
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
                    # Convert Narwhals to Polars for manipulation
                    added_sorted = (
                        result.added.to_polars()
                        if isinstance(result.added, nw.DataFrame)
                        else result.added
                    ).sort("sample_uid")
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
        # Verify versions contain expected field keys
        downstream_fields = [c.key for c in downstream_feature.spec().fields]
        for version_dict in reference["versions"]:
            for field_key in downstream_fields:
                field_name = "_".join(field_key)
                assert field_name in version_dict


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@parametrize_with_cases("graph_config", cases=FeatureGraphCases)
def test_resolve_update_detects_changes(
    store_params: dict[str, Any],
    graph_config: tuple[FeatureGraph, list[type[Feature]]],
    hash_algorithm: HashAlgorithm,
    snapshot,
    recwarn,
):
    """Test resolve_update detects changed data versions.

    Verifies that prefer_native=True and prefer_native=False produce identical results
    across all store types and different feature graphs.
    """
    graph, features = graph_config

    # Get root feature (first) and a downstream feature (last)
    root_feature = features[0]
    downstream_feature = features[-1]

    # Determine which fields to use
    root_fields = [c.key for c in root_feature.spec().fields]

    # Create data version dicts for fields
    def make_data_versions(
        version_prefix: str, sample_uids: list[int]
    ) -> list[dict[str, Any]]:
        if len(root_fields) == 1:
            field_name = "_".join(root_fields[0])
            return [{field_name: f"{version_prefix}{i}"} for i in sample_uids]
        else:
            return [
                {"_".join(ck): f"{version_prefix}{i}_{ck}" for ck in root_fields}
                for i in sample_uids
            ]

    # Initial upstream data
    initial_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "value": ["a1", "a2", "a3"],
            "data_version": make_data_versions("v", [1, 2, 3]),
        }
    )

    # Changed upstream data (change sample 2)
    changed_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "value": ["a1", "a2_CHANGED", "a3"],  # Changed
            "data_version": make_data_versions(
                "v_new", [1, 2, 3]
            ),  # All changed to new version
        }
    )

    # Iterate over store types and prefer_native variants
    results = {}

    for store_type in get_available_store_types():
        # Skip duckdb_ducklake - it's a storage format that shouldn't affect
        # data version computations. Tested separately in test_duckdb.py::test_duckdb_ducklake_integration
        if store_type == "duckdb_ducklake":
            continue

        for prefer_native in [True, False]:
            store = create_store(
                store_type,
                prefer_native,
                hash_algorithm,
                params=store_params,
            )

            # Try to open store - will fail validation if hash algorithm not supported
            try:
                with store, graph.use():
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
                                # Convert Narwhals to Polars for manipulation
                                added_df = (
                                    result.added.to_polars()
                                    if isinstance(result.added, nw.DataFrame)
                                    else result.added
                                )
                                feature_data = pl.DataFrame(
                                    {
                                        "sample_uid": [1, 2, 3],
                                        "value": [f"f{i}_1", f"f{i}_2", f"f{i}_3"],
                                    }
                                ).with_columns(
                                    pl.Series(
                                        "data_version",
                                        added_df.sort("sample_uid")[
                                            "data_version"
                                        ].to_list(),
                                    )
                                )
                                store.write_metadata(feature, feature_data)

                    # First resolve - get initial data versions for downstream
                    result1 = store.resolve_update(downstream_feature)
                    # Convert to Polars for easier manipulation in test
                    initial_versions_nw = (
                        result1.added.to_polars()
                        if isinstance(result1.added, nw.DataFrame)
                        else result1.added
                    )
                    initial_versions = initial_versions_nw.sort("sample_uid")

                    # Write downstream feature with these versions
                    downstream_data = pl.DataFrame(
                        {
                            "sample_uid": [1, 2, 3],
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
                            # Convert Narwhals to Polars and combine changed and added
                            changed_pl = (
                                result.changed.to_polars()
                                if isinstance(result.changed, nw.DataFrame)
                                else result.changed
                            )
                            added_pl = (
                                result.added.to_polars()
                                if isinstance(result.added, nw.DataFrame)
                                else result.added
                            )
                            updated = (
                                pl.concat([changed_pl, added_pl])
                                if len(added_pl) > 0
                                else changed_pl
                            )
                            feature_data = pl.DataFrame(
                                {
                                    "sample_uid": [1, 2, 3],
                                    "value": [
                                        f"f{i}_1_new",
                                        f"f{i}_2_new",
                                        f"f{i}_3_new",
                                    ],
                                }
                            ).join(
                                updated.select(["sample_uid", "data_version"]),
                                on="sample_uid",
                                how="left",
                            )
                            store.write_metadata(feature, feature_data)

                    # Resolve update again for downstream - should detect changes
                    result2 = store.resolve_update(downstream_feature)

                    # Assert no warning when prefer_native=True for stores that support native data version calculations
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

                    # Collect results - convert Narwhals to Polars for easier manipulation
                    changed_df_pl = (
                        result2.changed.to_polars()
                        if isinstance(result2.changed, nw.DataFrame)
                        else result2.changed
                    )

                    if len(changed_df_pl) > 0:
                        changed_sample_uids = changed_df_pl["sample_uid"].to_list()
                    else:
                        changed_sample_uids = []

                    # Check if any versions changed
                    version_changes = []
                    for sample_uid in changed_sample_uids:
                        new_version = changed_df_pl.filter(
                            pl.col("sample_uid") == sample_uid
                        )["data_version"][0]
                        old_version = initial_versions.filter(
                            pl.col("sample_uid") == sample_uid
                        )["data_version"][0]
                        version_changes.append(new_version != old_version)

                    results[(store_type, prefer_native)] = {
                        "added": len(result2.added),
                        "changed": len(result2.changed),
                        "removed": len(result2.removed),
                        "changed_sample_uids": sorted(changed_sample_uids),
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
        assert len(reference["changed_sample_uids"]) >= 1, (
            "At least one sample should change"
        )
        assert reference["any_version_changed"], "Data version should have changed"


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
@parametrize_with_cases("graph_config", cases=FeatureGraphCases)
def test_resolve_update_feature_version_change_idempotency(
    store_params: dict[str, Any],
    graph_config: tuple[FeatureGraph, list[type[Feature]]],
    hash_algorithm: HashAlgorithm,
    snapshot,
):
    """Test that resolve_update is idempotent after writing changed metadata."""
    graph, features = graph_config

    # Use simple chain for clarity (RootA -> LeafSimple)
    if len(graph.features_by_key) != 2:
        pytest.skip("Test requires simple chain graph")

    # Identify root and leaf features
    root_keys = [k for k, v in graph.features_by_key.items() if not v.spec().deps]
    leaf_keys = [k for k, v in graph.features_by_key.items() if v.spec().deps]

    if len(root_keys) != 1 or len(leaf_keys) != 1:
        pytest.skip("Test requires exactly 1 root and 1 leaf")

    root_key = root_keys[0]
    leaf_key = leaf_keys[0]
    RootFeature = graph.features_by_key[root_key]
    LeafFeature = graph.features_by_key[leaf_key]

    # Get initial feature versions
    initial_leaf_version = LeafFeature.feature_version()

    results = {}

    for store_type in get_available_store_types():
        # Skip duckdb_ducklake - it's a storage format that shouldn't affect
        # data version computations. Tested separately in test_duckdb.py::test_duckdb_ducklake_integration
        if store_type == "duckdb_ducklake":
            continue

        for prefer_native in [True, False]:
            store = create_store(
                store_type,
                prefer_native,
                hash_algorithm,
                params=store_params,
            )

            try:
                with store, graph.use():
                    # For ClickHouse, drop feature metadata to ensure clean state between prefer_native variants
                    if store_type == "clickhouse":
                        for feature in features:
                            store.drop_feature_metadata(feature)

                    # === PHASE 1: Write root metadata (upstream) ===
                    # Write root metadata - handle multiple fields
                    root_field_names = {
                        field.key[0] for field in RootFeature.spec().fields
                    }
                    root_data_version_dicts = []
                    for i in range(1, 4):
                        dv = {
                            field_name: f"v1_{i}_{field_name}"
                            for field_name in root_field_names
                        }
                        root_data_version_dicts.append(dv)

                    root_data_v1 = pl.DataFrame(
                        {
                            "sample_uid": [1, 2, 3],
                            "data_version": root_data_version_dicts,
                        },
                        schema={
                            "sample_uid": pl.UInt32,
                            "data_version": pl.Struct(
                                {fn: pl.Utf8 for fn in root_field_names}
                            ),
                        },
                    )
                    store.write_metadata(RootFeature, root_data_v1)

                    # === PHASE 2: First resolve_update (should detect 3 new samples) ===
                    result_first = store.resolve_update(LeafFeature)

                    first_run_stats = {
                        "added": len(result_first.added),
                        "changed": len(result_first.changed),
                        "removed": len(result_first.removed),
                    }

                    # === PHASE 3: Write the computed metadata ===
                    if len(result_first.added) > 0:
                        store.write_metadata(LeafFeature, result_first.added)
                    if len(result_first.changed) > 0:
                        store.write_metadata(LeafFeature, result_first.changed)

                    # === PHASE 4: Second resolve_update (should be idempotent - no changes) ===
                    result_second = store.resolve_update(LeafFeature)

                    second_run_stats = {
                        "added": len(result_second.added),
                        "changed": len(result_second.changed),
                        "removed": len(result_second.removed),
                    }

                    # Store results for comparison and snapshot
                    variant_key = f"{store_type}_native={prefer_native}"
                    results[variant_key] = {
                        "feature_version": initial_leaf_version,
                        "first_run": first_run_stats,
                        "second_run": second_run_stats,
                    }

            except HashAlgorithmNotSupportedError:
                # Hash algorithm not supported by this store - skip
                continue

    # All variants should produce identical results
    assert_all_results_equal(results, snapshot)

    # Verify expected behavior
    if results:
        reference = list(results.values())[0]

        # First run should detect new samples (leaf hasn't been computed yet)
        assert reference["first_run"]["added"] == 3, "Should detect 3 new samples"
        assert reference["first_run"]["changed"] == 0, "No changes yet"
        assert reference["first_run"]["removed"] == 0, "No removed samples"

        # Second run should be idempotent (no changes)
        assert reference["second_run"]["added"] == 0, (
            "Second run should detect 0 new samples (idempotent)"
        )
        assert reference["second_run"]["changed"] == 0, (
            "Second run should detect 0 changes (idempotent) - this was the bug!"
        )
        assert reference["second_run"]["removed"] == 0, "No removed samples"
