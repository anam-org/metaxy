"""Test resolve_update across different stores and feature graph configurations.

This test suite verifies that resolve_update produces correct and consistent results
across different metadata store backends, hash algorithms, and feature graph topologies.

The tests use the parametric metadata generation utilities from metaxy._testing.parametric
to avoid manual data construction and ensure correctness.
"""

import warnings
from collections.abc import Mapping
from typing import Any

import polars.testing as pl_testing
import pytest
import pytest_cases
from hypothesis.errors import NonInteractiveExampleWarning
from pytest_cases import parametrize_with_cases

from metaxy import (
    BaseFeature,
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    SampleFeatureSpec,
)
from metaxy._testing import HashAlgorithmCases
from metaxy._testing.parametric import (
    downstream_metadata_strategy,
    feature_metadata_strategy,
)
from metaxy.config import MetaxyConfig
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    InMemoryMetadataStore,
    MetadataStore,
)
from metaxy.metadata_store.clickhouse import ClickHouseMetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.plan import FeaturePlan
from metaxy.provenance.types import HashAlgorithm, LazyIncrement

# Type alias for feature plan output
FeaturePlanOutput = tuple[
    FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan
]


def verify_lazy_sql_frames(store: MetadataStore, lazy_result: LazyIncrement):
    """Verify that lazy frames from IbisMetadataStore contain SQL and haven't been materialized.

    Args:
        store: The metadata store instance
        lazy_result: LazyIncrement with lazy frames to verify
    """
    from metaxy.metadata_store.ibis import IbisMetadataStore

    if not isinstance(store, IbisMetadataStore):
        return  # Only check for Ibis-based stores

    # Check each lazy frame in the result
    for frame_name in ["added", "changed", "removed"]:
        lazy_frame = getattr(lazy_result, frame_name)

        # Get the native lazy frame
        native_frame = lazy_frame.to_native()

        # For Ibis-based stores, the native frame should be an Ibis expression
        # that can be compiled to SQL
        import ibis

        if isinstance(native_frame, ibis.expr.types.Table):
            # Verify we can get SQL from it
            try:
                sql = store.conn.compile(native_frame)
                assert isinstance(sql, str), (
                    f"{frame_name} lazy frame should have SQL representation"
                )
                # Check that it's actually SQL (contains SELECT, FROM, etc.)
                assert any(keyword in sql.upper() for keyword in ["SELECT", "FROM"]), (
                    f"{frame_name} should contain SQL keywords"
                )
            except Exception as e:
                raise AssertionError(
                    f"Failed to get SQL from {frame_name} lazy frame: {e}"
                )


def get_available_store_types() -> list[str]:
    """Get list of available store types from StoreCases.

    Dynamically discovers which store types are available by inspecting
    the StoreCases class for case methods. This allows different branches
    to have different store types without hardcoding the list while still
    allowing us to limit default parametrization to a fast subset.
    """
    from .conftest import StoreCases  # type: ignore[import-not-found]

    store_types = []
    for attr_name in dir(StoreCases):
        if attr_name.startswith("case_") and not attr_name.startswith("case__"):
            # Extract store type name from case method (e.g., "case_duckdb" -> "duckdb")
            store_type = attr_name[5:]  # Remove "case_" prefix
            store_types.append(store_type)

    preferred_order = ["inmemory", "duckdb"]
    preferred_store_types = [st for st in preferred_order if st in store_types]

    # Fall back to all discovered stores if preferred ones are not available
    return preferred_store_types or store_types


def create_store(
    store_type: str,
    prefer_native: bool,
    hash_algorithm: HashAlgorithm,
    params: dict[str, Any],
) -> MetadataStore:
    """Create a store instance for testing.

    Args:
        store_type: "inmemory", "duckdb", or "clickhouse"
        prefer_native: Whether to prefer native field provenance calculations
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
    spec=SampleFeatureSpec(
        key=FeatureKey(["resolve", "root_a"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="1"),
        ],
    ),
):
    """Root feature with single default field."""

    pass


class MultiFieldRoot(
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["resolve", "multi_root"]),
        fields=[
            FieldSpec(key=FieldKey(["train"]), code_version="1"),
            FieldSpec(key=FieldKey(["test"]), code_version="1"),
        ],
    ),
):
    """Root feature with multiple fields."""

    pass


# Intermediate features (depend on roots)
class BranchB(
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["resolve", "branch_b"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "root_a"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
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
    spec=SampleFeatureSpec(
        key=FeatureKey(["resolve", "branch_c"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "root_a"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
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
    spec=SampleFeatureSpec(
        key=FeatureKey(["resolve", "leaf_simple"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "root_a"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
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
    spec=SampleFeatureSpec(
        key=FeatureKey(["resolve", "leaf_diamond"]),
        deps=[
            FeatureDep(feature=FeatureKey(["resolve", "branch_b"]), columns=()),
            FeatureDep(feature=FeatureKey(["resolve", "branch_c"]), columns=()),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
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
    spec=SampleFeatureSpec(
        key=FeatureKey(["resolve", "leaf_multi"]),
        deps=[FeatureDep(feature=FeatureKey(["resolve", "multi_root"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
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
    """Different feature graph topologies for testing."""

    def case_simple_chain(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Simple two-feature chain: Root -> Leaf.

        Returns:
            (graph, upstream_features, leaf_plan)
        """

        class RootFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["value"],
            ),
        ):
            pass

        class LeafFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="leaf",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["result"],
            ),
        ):
            pass

        leaf_plan = graph.get_feature_plan(LeafFeature.spec().key)
        upstream_features = {RootFeature.spec().key: RootFeature}

        return graph, upstream_features, leaf_plan

    def case_diamond_graph(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Diamond dependency graph: Root -> BranchA,BranchB -> Leaf.

        Returns:
            (graph, upstream_features, leaf_plan)
        """

        class RootFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["value"],
            ),
        ):
            pass

        class BranchAFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="branch_a",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["a_result"],
            ),
        ):
            pass

        class BranchBFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="branch_b",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["b_result"],
            ),
        ):
            pass

        class LeafFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="leaf",
                deps=[
                    FeatureDep(feature=BranchAFeature),
                    FeatureDep(feature=BranchBFeature),
                ],
                fields=["final_result"],
            ),
        ):
            pass

        leaf_plan = graph.get_feature_plan(LeafFeature.spec().key)
        upstream_features = {
            RootFeature.spec().key: RootFeature,
            BranchAFeature.spec().key: BranchAFeature,
            BranchBFeature.spec().key: BranchBFeature,
        }

        return graph, upstream_features, leaf_plan

    def case_multi_field(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Features with multiple fields to test field-level provenance.

        Returns:
            (graph, upstream_features, leaf_plan)
        """

        class RootFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="root",
                fields=["field_a", "field_b", "field_c"],
            ),
        ):
            pass

        class LeafFeature(
            Feature,
            spec=SampleFeatureSpec(
                key="leaf",
                deps=[FeatureDep(feature=RootFeature)],
                fields=["result_x", "result_y"],
            ),
        ):
            pass

        leaf_plan = graph.get_feature_plan(LeafFeature.spec().key)
        upstream_features = {RootFeature.spec().key: RootFeature}

        return graph, upstream_features, leaf_plan


class RootFeatureCases:
    """Root features (no upstream dependencies) for testing sample-based resolve_update."""

    def case_simple_root(self, graph: FeatureGraph) -> type[BaseFeature]:
        """Single-field root feature."""

        class SimpleRoot(
            Feature,
            spec=SampleFeatureSpec(
                key="simple_root",
                fields=["value"],
            ),
        ):
            pass

        return SimpleRoot

    def case_multi_field_root(self, graph: FeatureGraph) -> type[BaseFeature]:
        """Multi-field root feature."""

        class MultiFieldRoot(
            Feature,
            spec=SampleFeatureSpec(
                key="multi_root",
                fields=["field_a", "field_b", "field_c"],
            ),
        ):
            pass

        return MultiFieldRoot


# ============= STORE CONFIGURATIONS =============


class StoreCases:
    """Different metadata store backend configurations."""

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_inmemory(self, hash_algorithm: HashAlgorithm) -> MetadataStore:
        """In-memory Polars-based store."""
        try:
            return InMemoryMetadataStore(hash_algorithm=hash_algorithm)
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by InMemoryMetadataStore"
            )

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_duckdb(self, hash_algorithm: HashAlgorithm, tmp_path) -> MetadataStore:
        """DuckDB SQL-based store."""
        store = DuckDBMetadataStore(
            tmp_path / "test.duckdb",
            hash_algorithm=hash_algorithm,
            extensions=["hashfuncs"],
            prefer_native=True,
        )
        # Check if hash algorithm is supported before opening
        try:
            with store:
                pass  # Just validate
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by DuckDBMetadataStore"
            )
        return store

    @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    def case_clickhouse(
        self, hash_algorithm: HashAlgorithm, clickhouse_db: str
    ) -> MetadataStore:
        """ClickHouse SQL-based store."""
        store = ClickHouseMetadataStore(
            connection_string=clickhouse_db,
            hash_algorithm=hash_algorithm,
            prefer_native=True,
        )
        # Check if hash algorithm is supported before opening
        try:
            with store:
                pass  # Just validate
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {hash_algorithm} not supported by ClickHouseMetadataStore"
            )
        return store


class TruncationCases:
    """Hash truncation length configurations."""

    def case_none(self):
        """No truncation."""
        return None

    def case_16(self):
        """Truncate to 16 characters."""
        return 16


# ============= FIXTURES =============


@pytest_cases.fixture
@parametrize_with_cases("hash_truncation_length", cases=TruncationCases)
def metaxy_config(hash_truncation_length: int | None):
    """Configure hash truncation length for tests."""
    old = MetaxyConfig.get()
    cfg_struct = old.model_dump()
    cfg_struct["hash_truncation_length"] = hash_truncation_length
    new = MetaxyConfig.model_validate(cfg_struct)
    MetaxyConfig.set(new)
    yield new
    MetaxyConfig.set(old)


# ============= TEST: ROOT FEATURES (NO UPSTREAM) =============


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def test_resolve_update_root_feature_requires_samples(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    root_feature: type[BaseFeature],
):
    """Test that resolve_update raises ValueError for root features without samples.

    Root features have no upstream dependencies, so provenance cannot be computed
    from upstream metadata. Users must provide samples with manually computed
    provenance_by_field.
    """
    with store:
        with pytest.raises(ValueError, match="root feature"):
            store.resolve_update(root_feature, lazy=True)


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def test_resolve_update_root_feature_with_samples(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    root_feature: type[BaseFeature],
    graph: FeatureGraph,
):
    """Test resolve_update for root features with provided samples.

    When samples are provided, resolve_update should:
    - Accept the samples with user-provided provenance_by_field
    - Return all samples as "added" (first write)
    - Compute correct metaxy_provenance from the field provenances
    """
    # Generate sample data using the parametric strategy
    feature_spec = root_feature.spec()
    feature_version = root_feature.feature_version()
    snapshot_version = graph.snapshot_version

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        samples_df = feature_metadata_strategy(
            feature_spec=feature_spec,
            feature_version=feature_version,
            snapshot_version=snapshot_version,
            min_rows=5,
            max_rows=10,
        ).example()

    # Select only required columns for samples (drop system columns that will be added)
    id_columns = list(feature_spec.id_columns)
    samples_df = samples_df.select(id_columns + ["metaxy_provenance_by_field"])

    # Convert to Narwhals
    import narwhals as nw

    samples_nw = nw.from_native(samples_df.lazy())

    # Call resolve_update with samples
    with store, graph.use():
        try:
            increment = store.resolve_update(
                root_feature, samples=samples_nw, lazy=True
            ).collect()
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        # Verify all samples are added (first write)
        assert len(increment.added) == len(samples_df)
        assert len(increment.changed) == 0
        assert len(increment.removed) == 0

        # Verify provenance_by_field structure matches input
        added_df = increment.added.lazy().collect().to_polars().sort(id_columns)
        samples_sorted = samples_df.sort(id_columns)

        pl_testing.assert_series_equal(
            added_df["metaxy_provenance_by_field"],
            samples_sorted["metaxy_provenance_by_field"],
            check_names=False,
        )


# ============= TEST: DOWNSTREAM FEATURES (WITH UPSTREAM) =============


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_downstream_feature(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test resolve_update for downstream features with upstream dependencies.

    This test verifies that:
    - Upstream metadata is correctly joined
    - Field provenance is computed from upstream dependencies
    - Results match the golden reference implementation
    - Behavior is consistent across all store backends
    """
    graph, upstream_features, child_plan = feature_plan_config

    # Get feature versions
    child_key = child_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    feature_versions = {
        feat_key.to_string(): feat_class.feature_version()
        for feat_key, feat_class in upstream_features.items()
    }
    feature_versions[child_key.to_string()] = child_version

    # Generate test data using golden reference strategy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=5,
            max_rows=15,
        ).example()

    # Write upstream metadata to store
    with store, graph.use():
        try:
            # Write all upstream data (includes transitive dependencies)
            for feat_key_str, upstream_df in upstream_data.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Call resolve_update to compute child metadata
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        # Get computed metadata
        added_df = increment.added.lazy().collect().to_polars()

        # Sort both for comparison
        id_columns = list(child_plan.feature.id_columns)
        added_sorted = added_df.sort(id_columns)
        golden_sorted = golden_downstream.sort(id_columns)

        # Compare provenance columns
        common_columns = [
            col for col in added_sorted.columns if col in golden_sorted.columns
        ]
        added_selected = added_sorted.select(common_columns)
        golden_selected = golden_sorted.select(common_columns)

        pl_testing.assert_frame_equal(
            added_selected,
            golden_selected,
            check_row_order=True,
            check_column_order=False,
        )


# ============= TEST: INCREMENTAL UPDATES =============


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_detects_changes(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test that resolve_update correctly detects added/changed/removed samples.

    This test:
    1. Writes initial upstream metadata
    2. Computes and writes child metadata
    3. Modifies upstream metadata (add/change/remove samples)
    4. Verifies resolve_update detects all changes correctly
    """
    graph, upstream_features, child_plan = feature_plan_config

    # Get feature versions
    child_key = child_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    feature_versions = {
        feat_key.to_string(): feat_class.feature_version()
        for feat_key, feat_class in upstream_features.items()
    }
    feature_versions[child_key.to_string()] = child_version

    # Generate initial test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        initial_upstream, initial_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=10,
            max_rows=10,
        ).example()

    # Write initial data
    with store, graph.use():
        try:
            # Write all upstream data (includes transitive dependencies)
            for feat_key_str, upstream_df in initial_upstream.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Write initial child metadata
            store.write_metadata(ChildFeature, initial_downstream)

            # Generate new upstream data (simulating changes)
            # The easiest way to test change detection is to generate completely new data
            modified_upstream, modified_downstream = downstream_metadata_strategy(
                child_plan,
                feature_versions=feature_versions,
                snapshot_version=graph.snapshot_version,
                hash_algorithm=store.hash_algorithm,
                min_rows=8,  # Different number of rows
                max_rows=8,
            ).example()

            # Write modified upstream data
            for feat_key_str, upstream_df in modified_upstream.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Call resolve_update - should detect all changes
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        # Verify changes were detected
        # With completely new data (8 samples vs 10 initial), we expect:
        # - Some samples added (new ones)
        # - Some samples removed (old ones not in new data)
        # - Possibly some changed (if IDs overlap but provenance differs)

        total_changes = (
            len(increment.added) + len(increment.changed) + len(increment.removed)
        )
        assert total_changes > 0, (
            "Expected resolve_update to detect some changes when upstream data changes"
        )


# ============= TEST: LAZY EXECUTION =============


@parametrize_with_cases("store", cases=StoreCases)
def test_resolve_update_lazy_execution(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    graph: FeatureGraph,
):
    """Test resolve_update with lazy=True returns lazy frames with correct implementation.

    This test verifies that:
    - lazy=True returns LazyIncrement with lazy frames
    - Lazy frames have correct implementation (POLARS for InMemory, IBIS for SQL stores)
    - Lazy frames can be collected to produce results
    - Results match eager execution
    """

    # Create a feature graph with multiple parents (realistic scenario)
    class Parent1(
        Feature,
        spec=SampleFeatureSpec(
            key="parent1",
            fields=["field_a", "field_b"],
        ),
    ):
        pass

    class Parent2(
        Feature,
        spec=SampleFeatureSpec(
            key="parent2",
            fields=["field_c"],
        ),
    ):
        pass

    class Child(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[
                FeatureDep(feature=Parent1),
                FeatureDep(feature=Parent2),
            ],
            fields=["result_x", "result_y"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(Child.spec().key)

    # Get feature versions
    feature_versions = {
        "parent1": Parent1.feature_version(),
        "parent2": Parent2.feature_version(),
        "child": Child.feature_version(),
    }

    # Generate test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=10,
            max_rows=10,
        ).example()

    with store, graph.use():
        try:
            # Write upstream metadata
            for feat_key_str, upstream_df in upstream_data.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # Call resolve_update with lazy=True
            lazy_increment = store.resolve_update(
                Child,
                target_version=Child.feature_version(),
                snapshot_version=graph.snapshot_version,
                lazy=True,
            )

            # Verify we got a LazyIncrement
            from metaxy.provenance.types import LazyIncrement

            assert isinstance(lazy_increment, LazyIncrement), (
                f"Expected LazyIncrement with lazy=True, got {type(lazy_increment)}"
            )

            # Verify lazy frames have correct implementation
            expected_impl = store.native_implementation()

            for frame_name in ["added", "changed", "removed"]:
                lazy_frame = getattr(lazy_increment, frame_name)
                actual_impl = lazy_frame.implementation

                assert actual_impl == expected_impl, (
                    f"Expected {frame_name} to have implementation {expected_impl}, "
                    f"but got {actual_impl} for store type {type(store).__name__}"
                )

            # Collect lazy result
            eager_increment = lazy_increment.collect()

            # Also get eager result for comparison
            eager_increment_direct = store.resolve_update(
                Child,
                target_version=Child.feature_version(),
                snapshot_version=graph.snapshot_version,
                lazy=False,
            )

            # Verify both approaches produce same results
            id_columns = list(child_plan.feature.id_columns)

            # Compare added frames
            lazy_added = (
                eager_increment.added.lazy().collect().to_polars().sort(id_columns)
            )
            eager_added = (
                eager_increment_direct.added.lazy()
                .collect()
                .to_polars()
                .sort(id_columns)
            )

            pl_testing.assert_frame_equal(lazy_added, eager_added)

        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )


# ============= TEST: IDEMPOTENCY =============


@parametrize_with_cases("store", cases=StoreCases)
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def test_resolve_update_idempotency(
    store: MetadataStore,
    metaxy_config: MetaxyConfig,
    feature_plan_config: FeaturePlanOutput,
):
    """Test that calling resolve_update multiple times is idempotent.

    After writing the increment from first resolve_update, calling it again
    should return empty increments (no changes).
    """
    graph, upstream_features, child_plan = feature_plan_config

    child_key = child_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    feature_versions = {
        feat_key.to_string(): feat_class.feature_version()
        for feat_key, feat_class in upstream_features.items()
    }
    feature_versions[child_key.to_string()] = child_version

    # Generate test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=store.hash_algorithm,
            min_rows=5,
            max_rows=10,
        ).example()

    with store, graph.use():
        try:
            # Write all upstream data (includes transitive dependencies)
            for feat_key_str, upstream_df in upstream_data.items():
                feat_key = FeatureKey([feat_key_str])
                feat_class = graph.features_by_key[feat_key]
                store.write_metadata(feat_class, upstream_df)

            # First resolve_update
            increment1 = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

            assert len(increment1.added) > 0, (
                "Expected resolve_update to detect added samples"
            )

            # Write the increment
            added_df = increment1.added.lazy().collect().to_polars()
            store.write_metadata(ChildFeature, added_df)

            # Second resolve_update - should be empty
            increment2 = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        # Verify second call is idempotent
        assert len(increment2.added) == 0, (
            "Second resolve_update should have no added samples"
        )
        assert len(increment2.changed) == 0, (
            "Second resolve_update should have no changed samples"
        )
        assert len(increment2.removed) == 0, (
            "Second resolve_update should have no removed samples"
        )
