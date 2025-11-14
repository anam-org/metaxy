"""Test that users can manually override metaxy_data_version_by_field and metaxy_data_version columns.

This test demonstrates the key capability where users can provide their own data versioning
(e.g., from content hashes, timestamps, version numbers) instead of relying on the
automatically computed provenance values.

Key behavior:
- Users write Parent metadata with custom metaxy_data_version_by_field values
- Users write Child metadata that depends on Parent
- When Parent's data_version changes for some samples, resolve_update on Child detects it
- Only samples with changed data_version appear in the increment
- metaxy_provenance columns are still computed/tracked separately for audit purposes
"""

from __future__ import annotations

from pathlib import Path

import narwhals as nw
import polars as pl

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
    SampleFeatureSpec,
)
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)


class TestDataVersionOverride:
    """Test suite for manual data version override functionality."""

    def test_manual_data_version_override_triggers_downstream_update(
        self, tmp_path: Path
    ) -> None:
        """Test that manually overriding data_version triggers downstream recomputation.

        Workflow:
        1. Create Parent (root) and Child (depends on Parent)
        2. Write Parent metadata with custom data_version values (e.g., content hashes)
        3. Write Child metadata
        4. Update Parent's data_version for some samples (simulating content change)
        5. Verify resolve_update on Child detects only changed samples
        """
        graph = FeatureGraph()
        with graph.use():

            class ParentFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["parent"]),
                    fields=[
                        FieldSpec(key=FieldKey(["content"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class ChildFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["child"]),
                    deps=[FeatureDep(feature=ParentFeature)],
                    fields=[
                        FieldSpec(key=FieldKey(["derived"]), code_version="1"),
                    ],
                ),
            ):
                pass

            db_path = tmp_path / "test_override.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                # Step 1: Write initial Parent metadata with custom data_version
                # Simulating content-based versioning (e.g., file hash, checksum)
                parent_df_v1 = pl.DataFrame(
                    {
                        "sample_uid": ["sample_a", "sample_b", "sample_c"],
                        "content": ["content_a_v1", "content_b_v1", "content_c_v1"],
                        # Custom data versions (e.g., content hashes)
                        METAXY_DATA_VERSION_BY_FIELD: [
                            {"content": "hash_content_a_v1"},
                            {"content": "hash_content_b_v1"},
                            {"content": "hash_content_c_v1"},
                        ],
                        METAXY_DATA_VERSION: [
                            "hash_content_a_v1",
                            "hash_content_b_v1",
                            "hash_content_c_v1",
                        ],
                        # Provenance will be auto-computed separately for audit
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"content": "prov_a_v1"},
                            {"content": "prov_b_v1"},
                            {"content": "prov_c_v1"},
                        ],
                        METAXY_PROVENANCE: ["prov_a_v1", "prov_b_v1", "prov_c_v1"],
                    }
                )
                store.write_metadata(ParentFeature, nw.from_native(parent_df_v1))

                # Step 2: Verify Parent data was written correctly
                parent_current = (
                    store.read_metadata(ParentFeature).collect().to_polars()
                )
                assert len(parent_current) == 3
                assert set(parent_current["sample_uid"]) == {
                    "sample_a",
                    "sample_b",
                    "sample_c",
                }

                # Verify custom data_version values are preserved
                assert sorted(parent_current[METAXY_DATA_VERSION].to_list()) == sorted(
                    [
                        "hash_content_a_v1",
                        "hash_content_b_v1",
                        "hash_content_c_v1",
                    ]
                )

                # Step 3: Write initial Child metadata
                # Use resolve_update to compute correct data_version values based on Parent
                child_increment_v1 = store.resolve_update(ChildFeature, lazy=False)

                # Add user columns to the computed increment
                child_to_write = child_increment_v1.added.to_polars().with_columns(
                    derived=pl.Series(["derived_a", "derived_b", "derived_c"])
                )
                store.write_metadata(ChildFeature, nw.from_native(child_to_write))

                # Step 4: Verify initial state - no updates needed
                increment_before_change = store.resolve_update(ChildFeature, lazy=False)
                assert len(increment_before_change.added) == 0
                assert len(increment_before_change.changed) == 0
                assert len(increment_before_change.removed) == 0

                # Step 5: Update Parent's data_version for sample_b only
                # This simulates the content changing (new file hash)
                parent_df_v2 = pl.DataFrame(
                    {
                        "sample_uid": ["sample_b"],
                        "content": [
                            "content_b_v2"
                        ],  # Content changed (but we could keep it same)
                        # NEW content hash - this is what matters
                        METAXY_DATA_VERSION_BY_FIELD: [
                            {"content": "hash_content_b_v2"}  # Changed!
                        ],
                        METAXY_DATA_VERSION: [
                            "hash_content_b_v2"  # Changed!
                        ],
                        # Provenance also changes but data_version is what's compared
                        METAXY_PROVENANCE_BY_FIELD: [{"content": "prov_b_v2"}],
                        METAXY_PROVENANCE: ["prov_b_v2"],
                        # Must provide these to avoid auto-computation
                        METAXY_FEATURE_VERSION: [ParentFeature.feature_version()],
                        METAXY_SNAPSHOT_VERSION: [graph.snapshot_version],
                    }
                )
                store.write_metadata(ParentFeature, nw.from_native(parent_df_v2))

                # Step 6: Verify Parent now has 2 versions of sample_b
                all_parent_records = (
                    store.read_metadata(ParentFeature, current_only=False)
                    .collect()
                    .to_polars()
                )
                sample_b_versions = all_parent_records.filter(
                    pl.col("sample_uid") == "sample_b"
                )
                assert len(sample_b_versions) == 2

                # Step 7: Resolve update on Child - should detect sample_b needs update
                increment_after_change = store.resolve_update(ChildFeature, lazy=False)

                # Convert to Polars for easier assertions
                added = increment_after_change.added
                changed = increment_after_change.changed
                removed = increment_after_change.removed

                # No new samples
                assert len(added) == 0

                # Only sample_b should be in changed
                assert len(changed) == 1
                assert changed["sample_uid"][0] == "sample_b"

                # No removed samples
                assert len(removed) == 0

    def test_data_version_vs_provenance_independence(self, tmp_path: Path) -> None:
        """Test that data_version and provenance can be set independently.

        This demonstrates that:
        - Provenance tracks what data was used (for audit/lineage)
        - Data version tracks when to recompute (for change detection)
        - They serve different purposes and can diverge
        """
        graph = FeatureGraph()
        with graph.use():

            class RootFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

            db_path = tmp_path / "test_independence.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                # Write metadata where data_version differs from provenance
                # Use case: data_version = semantic version, provenance = content hash
                df = pl.DataFrame(
                    {
                        "sample_uid": ["s1", "s2"],
                        "value": [100, 200],
                        # Semantic versioning for data_version
                        METAXY_DATA_VERSION_BY_FIELD: [
                            {"value": "v1.0.0"},
                            {"value": "v1.0.0"},
                        ],
                        METAXY_DATA_VERSION: ["v1.0.0", "v1.0.0"],
                        # Content hashes for provenance (different values)
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"value": "sha256_abc123"},
                            {"value": "sha256_def456"},
                        ],
                        METAXY_PROVENANCE: ["sha256_abc123", "sha256_def456"],
                    }
                )
                store.write_metadata(RootFeature, nw.from_native(df))

                # Read back and verify both columns are preserved independently
                result = (
                    store.read_metadata(RootFeature)
                    .collect()
                    .to_polars()
                    .sort("sample_uid")
                )

                # Data versions are semantic versions
                assert result[METAXY_DATA_VERSION].to_list() == ["v1.0.0", "v1.0.0"]

                # Provenance values are content hashes (different from data_version)
                assert result[METAXY_PROVENANCE].to_list() == [
                    "sha256_abc123",
                    "sha256_def456",
                ]

                # Verify they are truly independent
                assert result[METAXY_DATA_VERSION][0] != result[METAXY_PROVENANCE][0]

    def test_partial_data_version_update_selective_recomputation(
        self, tmp_path: Path
    ) -> None:
        """Test that only samples with changed data_version trigger downstream updates.

        This is the core value proposition: fine-grained change detection.
        """
        graph = FeatureGraph()
        with graph.use():

            class UpstreamFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["upstream"]),
                    fields=[
                        FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                        FieldSpec(key=FieldKey(["field2"]), code_version="1"),
                    ],
                ),
            ):
                pass

            class DownstreamFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["downstream"]),
                    deps=[FeatureDep(feature=UpstreamFeature)],
                    fields=[
                        FieldSpec(key=FieldKey(["result"]), code_version="1"),
                    ],
                ),
            ):
                pass

            db_path = tmp_path / "test_selective.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                # Write upstream with 5 samples
                upstream_df = pl.DataFrame(
                    {
                        "sample_uid": ["s1", "s2", "s3", "s4", "s5"],
                        "field1": [1, 2, 3, 4, 5],
                        "field2": [10, 20, 30, 40, 50],
                        # Custom data versions (e.g., from file timestamps)
                        METAXY_DATA_VERSION_BY_FIELD: [
                            {"field1": "ts_1000", "field2": "ts_1000"},
                            {"field1": "ts_1001", "field2": "ts_1001"},
                            {"field1": "ts_1002", "field2": "ts_1002"},
                            {"field1": "ts_1003", "field2": "ts_1003"},
                            {"field1": "ts_1004", "field2": "ts_1004"},
                        ],
                        METAXY_DATA_VERSION: [
                            "ts_1000",
                            "ts_1001",
                            "ts_1002",
                            "ts_1003",
                            "ts_1004",
                        ],
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"field1": "p1", "field2": "p1"},
                            {"field1": "p2", "field2": "p2"},
                            {"field1": "p3", "field2": "p3"},
                            {"field1": "p4", "field2": "p4"},
                            {"field1": "p5", "field2": "p5"},
                        ],
                    }
                )
                store.write_metadata(UpstreamFeature, nw.from_native(upstream_df))

                # Write downstream for all 5 samples
                # Use resolve_update to compute correct data_version values based on Upstream
                downstream_increment = store.resolve_update(
                    DownstreamFeature, lazy=False
                )

                # Add user columns to the computed increment
                downstream_to_write = (
                    downstream_increment.added.to_polars().with_columns(
                        result=pl.Series([11, 22, 33, 44, 55])
                    )
                )
                store.write_metadata(
                    DownstreamFeature, nw.from_native(downstream_to_write)
                )

                # Verify no changes initially
                increment = store.resolve_update(DownstreamFeature, lazy=False)
                assert len(increment.added) == 0
                assert len(increment.changed) == 0

                # Update data_version for ONLY s2 and s4
                upstream_update = pl.DataFrame(
                    {
                        "sample_uid": ["s2", "s4"],
                        "field1": [2, 4],  # Same values, but new data_version
                        "field2": [20, 40],
                        # New timestamps for these samples only
                        METAXY_DATA_VERSION_BY_FIELD: [
                            {"field1": "ts_2000", "field2": "ts_2000"},  # Updated
                            {"field1": "ts_2001", "field2": "ts_2001"},  # Updated
                        ],
                        METAXY_DATA_VERSION: [
                            "ts_2000",  # Changed from ts_1001
                            "ts_2001",  # Changed from ts_1003
                        ],
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"field1": "p2_new", "field2": "p2_new"},
                            {"field1": "p4_new", "field2": "p4_new"},
                        ],
                        METAXY_FEATURE_VERSION: [UpstreamFeature.feature_version()] * 2,
                        METAXY_SNAPSHOT_VERSION: [graph.snapshot_version] * 2,
                    }
                )
                store.write_metadata(UpstreamFeature, nw.from_native(upstream_update))

                # Resolve downstream updates
                increment = store.resolve_update(DownstreamFeature, lazy=False)

                # Only s2 and s4 should be in changed
                assert len(increment.added) == 0
                assert len(increment.changed) == 2
                changed_samples = set(increment.changed["sample_uid"].to_list())
                assert changed_samples == {"s2", "s4"}
                assert len(increment.removed) == 0

    def test_root_feature_with_custom_data_version(self, tmp_path: Path) -> None:
        """Test that root features can provide custom data_version directly.

        Root features have no upstream, so users must provide both:
        - provenance (what was computed)
        - data_version (when to recompute)
        """
        graph = FeatureGraph()
        with graph.use():

            class RootFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root"]),
                    fields=[
                        FieldSpec(key=FieldKey(["data"]), code_version="1"),
                    ],
                ),
            ):
                pass

            db_path = tmp_path / "test_root.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                # Resolve update with custom data_version for root feature
                samples = nw.from_native(
                    pl.DataFrame(
                        {
                            "sample_uid": ["x", "y"],
                            # User-provided data versions (e.g., from source system)
                            METAXY_DATA_VERSION_BY_FIELD: [
                                {"data": "source_v1"},
                                {"data": "source_v1"},
                            ],
                            METAXY_DATA_VERSION: ["source_v1", "source_v1"],
                            # Provenance for audit
                            METAXY_PROVENANCE_BY_FIELD: [
                                {"data": "computed_hash_x"},
                                {"data": "computed_hash_y"},
                            ],
                            METAXY_PROVENANCE: ["computed_hash_x", "computed_hash_y"],
                        }
                    )
                )

                # First resolution - everything is new
                increment = store.resolve_update(
                    RootFeature, samples=samples, lazy=False
                )
                assert len(increment.added) == 2
                assert len(increment.changed) == 0

                # Write the metadata
                result_df = nw.from_native(
                    pl.concat(
                        [
                            increment.added.to_polars(),
                            increment.changed.to_polars(),
                        ]
                    ).with_columns(data=pl.lit("computed_value"))
                )
                store.write_metadata(RootFeature, result_df)

                # Resolve again with same data_version - no changes
                increment2 = store.resolve_update(
                    RootFeature, samples=samples, lazy=False
                )
                assert len(increment2.added) == 0
                assert len(increment2.changed) == 0

                # Update data_version for sample x
                samples_updated = nw.from_native(
                    pl.DataFrame(
                        {
                            "sample_uid": ["x", "y"],
                            # New version for x only
                            METAXY_DATA_VERSION_BY_FIELD: [
                                {"data": "source_v2"},  # Changed
                                {"data": "source_v1"},  # Same
                            ],
                            METAXY_DATA_VERSION: [
                                "source_v2",  # Changed
                                "source_v1",  # Same
                            ],
                            METAXY_PROVENANCE_BY_FIELD: [
                                {"data": "computed_hash_x_new"},
                                {"data": "computed_hash_y"},
                            ],
                            METAXY_PROVENANCE: [
                                "computed_hash_x_new",
                                "computed_hash_y",
                            ],
                        }
                    )
                )

                # Resolve with updated data_version
                increment3 = store.resolve_update(
                    RootFeature, samples=samples_updated, lazy=False
                )
                assert len(increment3.added) == 0
                assert len(increment3.changed) == 1
                assert increment3.changed["sample_uid"][0] == "x"

    def test_data_version_fallback_to_provenance(self, tmp_path: Path) -> None:
        """Test that data_version defaults to provenance when not provided.

        This ensures backward compatibility with existing code that doesn't
        use custom data versioning.
        """
        graph = FeatureGraph()
        with graph.use():

            class SimpleFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["simple"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

            db_path = tmp_path / "test_fallback.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                # Write without data_version columns (old behavior)
                df = pl.DataFrame(
                    {
                        "sample_uid": ["a", "b"],
                        "value": [1, 2],
                        # Only provide provenance
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"value": "prov_a"},
                            {"value": "prov_b"},
                        ],
                    }
                )
                store.write_metadata(SimpleFeature, nw.from_native(df))

                # Read back - data_version should equal provenance
                result = store.read_metadata(SimpleFeature).collect().to_polars()

                # When data_version is not provided, it should default to provenance
                # (this is handled by write_metadata)
                assert METAXY_DATA_VERSION in result.columns
                assert METAXY_DATA_VERSION_BY_FIELD in result.columns
