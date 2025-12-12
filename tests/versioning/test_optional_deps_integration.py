"""Integration tests for optional dependencies (FeatureDep.optional=True).

This module tests:
- Join behavior with optional vs required dependencies
- Provenance calculation with optional dependencies (including NULL handling)
- Mixed required and optional dependencies scenarios
"""

from __future__ import annotations

from collections.abc import Callable

import narwhals as nw
import polars as pl
from syrupy.assertion import SnapshotAssertion

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.field import FieldDep, FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class TestJoinBehavior:
    """Test join behavior with optional vs required dependencies."""

    def test_all_required_deps_use_inner_join(
        self, graph: FeatureGraph, snapshot: SnapshotAssertion
    ) -> None:
        """Test that all required deps use inner join (existing behavior)."""

        # Create upstream features
        class UpstreamA(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_a"]),
                fields=[FieldSpec(key=FieldKey(["data_a"]), code_version="1")],
            ),
        ):
            pass

        class UpstreamB(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream_b"]),
                fields=[FieldSpec(key=FieldKey(["data_b"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["upstream_a"]), optional=False),
                    FeatureDep(feature=FeatureKey(["upstream_b"]), optional=False),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["upstream_a"]),
                                fields=[FieldKey(["data_a"])],
                            ),
                            FieldDep(
                                feature=FeatureKey(["upstream_b"]),
                                fields=[FieldKey(["data_b"])],
                            ),
                        ],
                    ),
                ],
            ),
        ):
            pass

        # Verify features registered in graph
        assert FeatureKey(["upstream_a"]) in graph.features_by_key
        assert FeatureKey(["upstream_b"]) in graph.features_by_key
        assert FeatureKey(["downstream"]) in graph.features_by_key
        # Reference classes to suppress unused warnings (they're used via graph registration)
        del UpstreamA, UpstreamB, Downstream

        plan = graph.get_feature_plan(FeatureKey(["downstream"]))
        engine = PolarsVersioningEngine(plan)

        # Create upstream metadata with partial overlap
        upstream_a = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "data_a": ["a1", "a2", "a3"],
                    "metaxy_provenance_by_field": [
                        {"data_a": "hash_a1"},
                        {"data_a": "hash_a2"},
                        {"data_a": "hash_a3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_a": "hash_a1"},
                        {"data_a": "hash_a2"},
                        {"data_a": "hash_a3"},
                    ],
                    "metaxy_provenance": ["prov_a1", "prov_a2", "prov_a3"],
                }
            ).lazy()
        )

        upstream_b = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [2, 3, 4],  # Only 2 and 3 overlap
                    "data_b": ["b2", "b3", "b4"],
                    "metaxy_provenance_by_field": [
                        {"data_b": "hash_b2"},
                        {"data_b": "hash_b3"},
                        {"data_b": "hash_b4"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_b": "hash_b2"},
                        {"data_b": "hash_b3"},
                        {"data_b": "hash_b4"},
                    ],
                    "metaxy_provenance": ["prov_b2", "prov_b3", "prov_b4"],
                }
            ).lazy()
        )

        # Join with inner join - should only have overlapping samples (2, 3)
        joined = engine.join(
            {
                FeatureKey(["upstream_a"]): engine.feature_transformers_by_key[
                    FeatureKey(["upstream_a"])
                ].transform(upstream_a),
                FeatureKey(["upstream_b"]): engine.feature_transformers_by_key[
                    FeatureKey(["upstream_b"])
                ].transform(upstream_b),
            }
        )

        result = joined.collect().to_polars()

        # Should only have samples 2 and 3 (inner join)
        assert sorted(result["sample_uid"].to_list()) == [2, 3]

        # Both data columns should be present with no NULLs
        assert "data_a" in result.columns
        assert "data_b" in result.columns
        assert result["data_a"].null_count() == 0
        assert result["data_b"].null_count() == 0

        # Snapshot the result
        assert result.to_dicts() == snapshot

    def test_optional_dep_left_join_full_match(
        self,
        standard_upstream_setup: dict[str, type[SampleFeature]],
        versioning_engine_for: Callable[[str], PolarsVersioningEngine],
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test optional dep with left join when all samples match."""
        # Use standard setup from fixture
        _ = standard_upstream_setup

        engine = versioning_engine_for("downstream")

        # Create upstream metadata - all samples match
        required = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "data_req": ["r1", "r2", "r3"],
                    "metaxy_provenance_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                    ],
                    "metaxy_provenance": ["prov_r1", "prov_r2", "prov_r3"],
                }
            ).lazy()
        )

        optional = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],  # All match
                    "data_opt": ["o1", "o2", "o3"],
                    "metaxy_provenance_by_field": [
                        {"data_opt": "hash_o1"},
                        {"data_opt": "hash_o2"},
                        {"data_opt": "hash_o3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_opt": "hash_o1"},
                        {"data_opt": "hash_o2"},
                        {"data_opt": "hash_o3"},
                    ],
                    "metaxy_provenance": ["prov_o1", "prov_o2", "prov_o3"],
                }
            ).lazy()
        )

        # Join with left join for optional dep
        joined = engine.join(
            {
                FeatureKey(["required"]): engine.feature_transformers_by_key[
                    FeatureKey(["required"])
                ].transform(required),
                FeatureKey(["optional"]): engine.feature_transformers_by_key[
                    FeatureKey(["optional"])
                ].transform(optional),
            }
        )

        result = joined.collect().to_polars()

        # Should have all samples from required (1, 2, 3)
        assert sorted(result["sample_uid"].to_list()) == [1, 2, 3]

        # Both data columns should be present with no NULLs (all match)
        assert "data_req" in result.columns
        assert "data_opt" in result.columns
        assert result["data_req"].null_count() == 0
        assert result["data_opt"].null_count() == 0

        # Snapshot the result
        assert result.to_dicts() == snapshot

    def test_optional_dep_left_join_no_match(
        self,
        standard_upstream_setup: dict[str, type[SampleFeature]],
        versioning_engine_for: Callable[[str], PolarsVersioningEngine],
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test optional dep with left join when no samples match (all NULLs)."""
        # Use standard setup from fixture
        _ = standard_upstream_setup

        engine = versioning_engine_for("downstream")

        # Create upstream metadata - no overlap
        required = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "data_req": ["r1", "r2", "r3"],
                    "metaxy_provenance_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                    ],
                    "metaxy_provenance": ["prov_r1", "prov_r2", "prov_r3"],
                }
            ).lazy()
        )

        optional = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [4, 5, 6],  # No overlap
                    "data_opt": ["o4", "o5", "o6"],
                    "metaxy_provenance_by_field": [
                        {"data_opt": "hash_o4"},
                        {"data_opt": "hash_o5"},
                        {"data_opt": "hash_o6"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_opt": "hash_o4"},
                        {"data_opt": "hash_o5"},
                        {"data_opt": "hash_o6"},
                    ],
                    "metaxy_provenance": ["prov_o4", "prov_o5", "prov_o6"],
                }
            ).lazy()
        )

        # Join with left join for optional dep
        joined = engine.join(
            {
                FeatureKey(["required"]): engine.feature_transformers_by_key[
                    FeatureKey(["required"])
                ].transform(required),
                FeatureKey(["optional"]): engine.feature_transformers_by_key[
                    FeatureKey(["optional"])
                ].transform(optional),
            }
        )

        result = joined.collect().to_polars()

        # Should have all samples from required (1, 2, 3)
        assert sorted(result["sample_uid"].to_list()) == [1, 2, 3]

        # Required data should be present with no NULLs
        assert "data_req" in result.columns
        assert result["data_req"].null_count() == 0

        # Optional data should be present but all NULL
        assert "data_opt" in result.columns
        assert result["data_opt"].null_count() == 3  # All NULL

        # Snapshot the result
        assert result.to_dicts() == snapshot

    def test_optional_dep_left_join_partial_match(
        self,
        standard_upstream_setup: dict[str, type[SampleFeature]],
        versioning_engine_for: Callable[[str], PolarsVersioningEngine],
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test optional dep with left join when some samples match."""
        # Use standard setup from fixture
        _ = standard_upstream_setup

        engine = versioning_engine_for("downstream")

        # Create upstream metadata - partial overlap
        required = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3, 4],
                    "data_req": ["r1", "r2", "r3", "r4"],
                    "metaxy_provenance_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                        {"data_req": "hash_r4"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                        {"data_req": "hash_r4"},
                    ],
                    "metaxy_provenance": ["prov_r1", "prov_r2", "prov_r3", "prov_r4"],
                }
            ).lazy()
        )

        optional = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [2, 3],  # Only 2 and 3 overlap
                    "data_opt": ["o2", "o3"],
                    "metaxy_provenance_by_field": [
                        {"data_opt": "hash_o2"},
                        {"data_opt": "hash_o3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_opt": "hash_o2"},
                        {"data_opt": "hash_o3"},
                    ],
                    "metaxy_provenance": ["prov_o2", "prov_o3"],
                }
            ).lazy()
        )

        # Join with left join for optional dep
        joined = engine.join(
            {
                FeatureKey(["required"]): engine.feature_transformers_by_key[
                    FeatureKey(["required"])
                ].transform(required),
                FeatureKey(["optional"]): engine.feature_transformers_by_key[
                    FeatureKey(["optional"])
                ].transform(optional),
            }
        )

        result = joined.collect().to_polars()

        # Should have all samples from required (1, 2, 3, 4)
        assert sorted(result["sample_uid"].to_list()) == [1, 2, 3, 4]

        # Required data should be present with no NULLs
        assert "data_req" in result.columns
        assert result["data_req"].null_count() == 0

        # Optional data should be present with some NULLs
        assert "data_opt" in result.columns
        assert result["data_opt"].null_count() == 2  # Samples 1 and 4 have NULL

        # Snapshot the result
        assert result.to_dicts() == snapshot

    def test_multiple_optional_deps(
        self, graph: FeatureGraph, snapshot: SnapshotAssertion
    ) -> None:
        """Test multiple optional dependencies."""

        # Create upstream features
        class RequiredUpstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["required"]),
                fields=[FieldSpec(key=FieldKey(["data_req"]), code_version="1")],
            ),
        ):
            pass

        class OptionalA(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["optional_a"]),
                fields=[FieldSpec(key=FieldKey(["data_a"]), code_version="1")],
            ),
        ):
            pass

        class OptionalB(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["optional_b"]),
                fields=[FieldSpec(key=FieldKey(["data_b"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["required"]), optional=False),
                    FeatureDep(feature=FeatureKey(["optional_a"]), optional=True),
                    FeatureDep(feature=FeatureKey(["optional_b"]), optional=True),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["required"]),
                                fields=[FieldKey(["data_req"])],
                            ),
                            FieldDep(
                                feature=FeatureKey(["optional_a"]),
                                fields=[FieldKey(["data_a"])],
                            ),
                            FieldDep(
                                feature=FeatureKey(["optional_b"]),
                                fields=[FieldKey(["data_b"])],
                            ),
                        ],
                    ),
                ],
            ),
        ):
            pass

        # Verify features registered in graph
        assert FeatureKey(["required"]) in graph.features_by_key
        assert FeatureKey(["optional_a"]) in graph.features_by_key
        assert FeatureKey(["optional_b"]) in graph.features_by_key
        assert FeatureKey(["downstream"]) in graph.features_by_key
        del RequiredUpstream, OptionalA, OptionalB, Downstream

        plan = graph.get_feature_plan(FeatureKey(["downstream"]))
        engine = PolarsVersioningEngine(plan)

        # Create upstream metadata with different overlaps
        required = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3, 4],
                    "data_req": ["r1", "r2", "r3", "r4"],
                    "metaxy_provenance_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                        {"data_req": "hash_r4"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                        {"data_req": "hash_r3"},
                        {"data_req": "hash_r4"},
                    ],
                    "metaxy_provenance": ["prov_r1", "prov_r2", "prov_r3", "prov_r4"],
                }
            ).lazy()
        )

        optional_a = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],  # Only 1, 2
                    "data_a": ["a1", "a2"],
                    "metaxy_provenance_by_field": [
                        {"data_a": "hash_a1"},
                        {"data_a": "hash_a2"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_a": "hash_a1"},
                        {"data_a": "hash_a2"},
                    ],
                    "metaxy_provenance": ["prov_a1", "prov_a2"],
                }
            ).lazy()
        )

        optional_b = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [2, 3, 4],  # Only 2, 3, 4
                    "data_b": ["b2", "b3", "b4"],
                    "metaxy_provenance_by_field": [
                        {"data_b": "hash_b2"},
                        {"data_b": "hash_b3"},
                        {"data_b": "hash_b4"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_b": "hash_b2"},
                        {"data_b": "hash_b3"},
                        {"data_b": "hash_b4"},
                    ],
                    "metaxy_provenance": ["prov_b2", "prov_b3", "prov_b4"],
                }
            ).lazy()
        )

        # Join with left joins for optional deps
        joined = engine.join(
            {
                FeatureKey(["required"]): engine.feature_transformers_by_key[
                    FeatureKey(["required"])
                ].transform(required),
                FeatureKey(["optional_a"]): engine.feature_transformers_by_key[
                    FeatureKey(["optional_a"])
                ].transform(optional_a),
                FeatureKey(["optional_b"]): engine.feature_transformers_by_key[
                    FeatureKey(["optional_b"])
                ].transform(optional_b),
            }
        )

        result = joined.collect().to_polars()

        # Should have all samples from required (1, 2, 3, 4)
        assert sorted(result["sample_uid"].to_list()) == [1, 2, 3, 4]

        # Required data should be present with no NULLs
        assert result["data_req"].null_count() == 0

        # Optional A: NULL for samples 3, 4
        assert result["data_a"].null_count() == 2

        # Optional B: NULL for sample 1
        assert result["data_b"].null_count() == 1

        # Snapshot the result
        assert result.to_dicts() == snapshot

    def test_mixed_required_and_optional_deps(
        self, graph: FeatureGraph, snapshot: SnapshotAssertion
    ) -> None:
        """Test mixed required and optional dependencies."""

        # Create upstream features
        class RequiredA(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["required_a"]),
                fields=[FieldSpec(key=FieldKey(["data_a"]), code_version="1")],
            ),
        ):
            pass

        class RequiredB(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["required_b"]),
                fields=[FieldSpec(key=FieldKey(["data_b"]), code_version="1")],
            ),
        ):
            pass

        class Optional(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["optional"]),
                fields=[FieldSpec(key=FieldKey(["data_opt"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["required_a"]), optional=False),
                    FeatureDep(feature=FeatureKey(["optional"]), optional=True),
                    FeatureDep(feature=FeatureKey(["required_b"]), optional=False),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["required_a"]),
                                fields=[FieldKey(["data_a"])],
                            ),
                            FieldDep(
                                feature=FeatureKey(["optional"]),
                                fields=[FieldKey(["data_opt"])],
                            ),
                            FieldDep(
                                feature=FeatureKey(["required_b"]),
                                fields=[FieldKey(["data_b"])],
                            ),
                        ],
                    ),
                ],
            ),
        ):
            pass

        # Verify features registered in graph
        assert FeatureKey(["required_a"]) in graph.features_by_key
        assert FeatureKey(["required_b"]) in graph.features_by_key
        assert FeatureKey(["optional"]) in graph.features_by_key
        assert FeatureKey(["downstream"]) in graph.features_by_key
        del RequiredA, RequiredB, Optional, Downstream

        plan = graph.get_feature_plan(FeatureKey(["downstream"]))
        engine = PolarsVersioningEngine(plan)

        # Create upstream metadata
        required_a = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "data_a": ["a1", "a2", "a3"],
                    "metaxy_provenance_by_field": [
                        {"data_a": "hash_a1"},
                        {"data_a": "hash_a2"},
                        {"data_a": "hash_a3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_a": "hash_a1"},
                        {"data_a": "hash_a2"},
                        {"data_a": "hash_a3"},
                    ],
                    "metaxy_provenance": ["prov_a1", "prov_a2", "prov_a3"],
                }
            ).lazy()
        )

        required_b = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [2, 3, 4],  # Inner join will limit to 2, 3
                    "data_b": ["b2", "b3", "b4"],
                    "metaxy_provenance_by_field": [
                        {"data_b": "hash_b2"},
                        {"data_b": "hash_b3"},
                        {"data_b": "hash_b4"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_b": "hash_b2"},
                        {"data_b": "hash_b3"},
                        {"data_b": "hash_b4"},
                    ],
                    "metaxy_provenance": ["prov_b2", "prov_b3", "prov_b4"],
                }
            ).lazy()
        )

        optional = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [2],  # Only sample 2
                    "data_opt": ["o2"],
                    "metaxy_provenance_by_field": [{"data_opt": "hash_o2"}],
                    "metaxy_data_version_by_field": [{"data_opt": "hash_o2"}],
                    "metaxy_provenance": ["prov_o2"],
                }
            ).lazy()
        )

        # Join - required deps use inner join, optional use left join
        joined = engine.join(
            {
                FeatureKey(["required_a"]): engine.feature_transformers_by_key[
                    FeatureKey(["required_a"])
                ].transform(required_a),
                FeatureKey(["optional"]): engine.feature_transformers_by_key[
                    FeatureKey(["optional"])
                ].transform(optional),
                FeatureKey(["required_b"]): engine.feature_transformers_by_key[
                    FeatureKey(["required_b"])
                ].transform(required_b),
            }
        )

        result = joined.collect().to_polars()

        # Should only have samples 2 and 3 (inner join on required_a and required_b)
        assert sorted(result["sample_uid"].to_list()) == [2, 3]

        # Required data should be present with no NULLs
        assert result["data_a"].null_count() == 0
        assert result["data_b"].null_count() == 0

        # Optional data: has value for sample 2, NULL for sample 3
        assert result["data_opt"].null_count() == 1

        # Snapshot the result
        assert result.to_dicts() == snapshot


class TestProvenanceWithOptionalDeps:
    """Test provenance computation with optional dependencies."""

    def test_provenance_with_optional_dep_null_handling(
        self,
        standard_upstream_setup: dict[str, type[SampleFeature]],
        versioning_engine_for: Callable[[str], PolarsVersioningEngine],
        hash_algo: HashAlgorithm,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test that provenance correctly handles NULL from optional deps using __NULL__ sentinel."""
        # Use standard setup from fixture
        _ = standard_upstream_setup

        engine = versioning_engine_for("downstream")

        # Create upstream metadata - partial overlap
        required = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "data_req": ["r1", "r2"],
                    "metaxy_provenance_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"data_req": "hash_r1"},
                        {"data_req": "hash_r2"},
                    ],
                    "metaxy_provenance": ["prov_r1", "prov_r2"],
                }
            ).lazy()
        )

        optional = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [2],  # Only sample 2 has optional data
                    "data_opt": ["o2"],
                    "metaxy_provenance_by_field": [{"data_opt": "hash_o2"}],
                    "metaxy_data_version_by_field": [{"data_opt": "hash_o2"}],
                    "metaxy_provenance": ["prov_o2"],
                }
            ).lazy()
        )

        upstream = {
            FeatureKey(["required"]): required,
            FeatureKey(["optional"]): optional,
        }

        # Compute provenance
        result = engine.load_upstream_with_provenance(
            upstream=upstream,
            hash_algo=hash_algo,
            filters={},
        )

        result_df = result.collect().to_polars()

        # Should have samples 1 and 2
        assert sorted(result_df["sample_uid"].to_list()) == [1, 2]

        # Both samples should have provenance (deterministic even with NULL)
        assert result_df["metaxy_provenance"].null_count() == 0
        assert result_df["metaxy_provenance_by_field"].null_count() == 0

        # Provenance should differ between sample 1 (NULL optional) and sample 2 (has optional)
        prov1 = result_df.filter(pl.col("sample_uid") == 1)["metaxy_provenance"][0]
        prov2 = result_df.filter(pl.col("sample_uid") == 2)["metaxy_provenance"][0]
        assert prov1 != prov2

        # Snapshot the provenance
        provenance_data = sorted(
            [
                {
                    "sample_uid": result_df["sample_uid"][i],
                    "field_provenance": result_df["metaxy_provenance_by_field"][i],
                    "sample_provenance": result_df["metaxy_provenance"][i],
                }
                for i in range(len(result_df))
            ],
            key=lambda x: x["sample_uid"],
        )
        assert provenance_data == snapshot

    def test_provenance_is_deterministic_with_nulls(
        self,
        standard_upstream_setup: dict[str, type[SampleFeature]],
        versioning_engine_for: Callable[[str], PolarsVersioningEngine],
        hash_algo: HashAlgorithm,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test that provenance with NULLs from optional deps is deterministic."""
        # Use standard setup from fixture
        _ = standard_upstream_setup

        engine = versioning_engine_for("downstream")

        # Create upstream metadata - no optional data
        required = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1],
                    "data_req": ["r1"],
                    "metaxy_provenance_by_field": [{"data_req": "hash_r1"}],
                    "metaxy_data_version_by_field": [{"data_req": "hash_r1"}],
                    "metaxy_provenance": ["prov_r1"],
                }
            ).lazy()
        )

        optional = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [99],  # No overlap
                    "data_opt": ["o99"],
                    "metaxy_provenance_by_field": [{"data_opt": "hash_o99"}],
                    "metaxy_data_version_by_field": [{"data_opt": "hash_o99"}],
                    "metaxy_provenance": ["prov_o99"],
                }
            ).lazy()
        )

        upstream = {
            FeatureKey(["required"]): required,
            FeatureKey(["optional"]): optional,
        }

        # Compute provenance twice
        result1 = engine.load_upstream_with_provenance(
            upstream=upstream,
            hash_algo=hash_algo,
            filters={},
        )

        result2 = engine.load_upstream_with_provenance(
            upstream=upstream,
            hash_algo=hash_algo,
            filters={},
        )

        result1_df = result1.collect().to_polars()
        result2_df = result2.collect().to_polars()

        # Provenance should be identical
        assert result1_df["metaxy_provenance"][0] == result2_df["metaxy_provenance"][0]
        assert (
            result1_df["metaxy_provenance_by_field"][0]
            == result2_df["metaxy_provenance_by_field"][0]
        )

        # Snapshot the provenance
        assert {
            "sample_uid": result1_df["sample_uid"][0],
            "field_provenance": result1_df["metaxy_provenance_by_field"][0],
            "sample_provenance": result1_df["metaxy_provenance"][0],
        } == snapshot

    def test_provenance_differs_with_vs_without_optional_data(
        self,
        standard_upstream_setup: dict[str, type[SampleFeature]],
        versioning_engine_for: Callable[[str], PolarsVersioningEngine],
        hash_algo: HashAlgorithm,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test that provenance differs when optional data is present vs absent."""
        # Use standard setup from fixture
        _ = standard_upstream_setup

        engine = versioning_engine_for("downstream")

        # Scenario 1: Optional data is present
        required_with = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1],
                    "data_req": ["r1"],
                    "metaxy_provenance_by_field": [{"data_req": "hash_r1"}],
                    "metaxy_data_version_by_field": [{"data_req": "hash_r1"}],
                    "metaxy_provenance": ["prov_r1"],
                }
            ).lazy()
        )

        optional_with = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1],  # Has optional data
                    "data_opt": ["o1"],
                    "metaxy_provenance_by_field": [{"data_opt": "hash_o1"}],
                    "metaxy_data_version_by_field": [{"data_opt": "hash_o1"}],
                    "metaxy_provenance": ["prov_o1"],
                }
            ).lazy()
        )

        # Scenario 2: Optional data is absent
        required_without = required_with
        optional_without = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [99],  # No overlap
                    "data_opt": ["o99"],
                    "metaxy_provenance_by_field": [{"data_opt": "hash_o99"}],
                    "metaxy_data_version_by_field": [{"data_opt": "hash_o99"}],
                    "metaxy_provenance": ["prov_o99"],
                }
            ).lazy()
        )

        # Compute provenance with optional data
        result_with = engine.load_upstream_with_provenance(
            upstream={
                FeatureKey(["required"]): required_with,
                FeatureKey(["optional"]): optional_with,
            },
            hash_algo=hash_algo,
            filters={},
        )

        # Compute provenance without optional data
        result_without = engine.load_upstream_with_provenance(
            upstream={
                FeatureKey(["required"]): required_without,
                FeatureKey(["optional"]): optional_without,
            },
            hash_algo=hash_algo,
            filters={},
        )

        result_with_df = result_with.collect().to_polars()
        result_without_df = result_without.collect().to_polars()

        # Provenance should differ
        prov_with = result_with_df["metaxy_provenance"][0]
        prov_without = result_without_df["metaxy_provenance"][0]
        assert prov_with != prov_without

        # Snapshot both
        assert {
            "with_optional": {
                "sample_uid": result_with_df["sample_uid"][0],
                "field_provenance": result_with_df["metaxy_provenance_by_field"][0],
                "sample_provenance": prov_with,
            },
            "without_optional": {
                "sample_uid": result_without_df["sample_uid"][0],
                "field_provenance": result_without_df["metaxy_provenance_by_field"][0],
                "sample_provenance": prov_without,
            },
        } == snapshot
