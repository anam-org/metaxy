"""Unit tests for PolarsVersioningEngine."""

from __future__ import annotations

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing.models import SampleFeature
from metaxy.models.feature import FeatureGraph
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


def test_polars_engine_initialization(graph: FeatureGraph, simple_features: dict[str, type[SampleFeature]]) -> None:
    """Test PolarsVersioningEngine can be initialized."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)
    assert engine.plan is plan


def test_compute_provenance_single_upstream(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test compute_provenance with single upstream feature."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    # Materialize for inspection
    result_df = result.collect()

    # Verify provenance columns exist
    assert "metaxy_provenance_by_field" in result_df.columns
    assert "metaxy_provenance" in result_df.columns

    # Verify provenance_by_field is a struct
    result_pl = result_df.to_native()
    assert isinstance(result_pl.schema["metaxy_provenance_by_field"], pl.Struct)

    # Verify provenance_by_field has the expected field
    first_prov = result_df["metaxy_provenance_by_field"][0]
    assert "default" in first_prov

    # Snapshot the provenance values
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "sample_provenance": result_df["metaxy_provenance"][i],
            }
        )

    assert provenance_data == snapshot


def test_compute_provenance_multiple_upstreams(
    multi_upstream_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    upstream_audio_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test compute_provenance with multiple upstream features."""
    from metaxy.models.types import FeatureKey

    feature = multi_upstream_features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {
        FeatureKey(["video"]): upstream_video_metadata,
        FeatureKey(["audio"]): upstream_audio_metadata,
    }

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Verify provenance columns exist
    assert "metaxy_provenance_by_field" in result_df.columns
    assert "metaxy_provenance" in result_df.columns

    # Verify provenance_by_field has both fields
    result_pl = result_df.to_native()
    provenance_schema = result_pl.schema["metaxy_provenance_by_field"]
    field_names = {f.name for f in provenance_schema.fields}
    assert field_names == {"fusion", "analysis"}

    # Different code versions should produce different field hashes
    first_prov = result_df["metaxy_provenance_by_field"][0]
    assert first_prov["fusion"] != first_prov["analysis"]

    # Snapshot the results
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "sample_provenance": result_df["metaxy_provenance"][i],
            }
        )

    assert provenance_data == snapshot


def test_compute_provenance_selective_field_deps(
    selective_field_dep_features: dict[str, type[SampleFeature]],
    upstream_metadata_multi_field: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test compute_provenance with selective field-level dependencies."""
    from metaxy.models.types import FeatureKey

    feature = selective_field_dep_features["SelectiveFeature"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["multi_field"]): upstream_metadata_multi_field}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Verify provenance_by_field has all three fields
    result_pl = result_df.to_native()
    provenance_schema = result_pl.schema["metaxy_provenance_by_field"]
    field_names = {f.name for f in provenance_schema.fields}
    assert field_names == {"visual", "audio_only", "mixed"}

    # All fields should have different hashes (different deps + code versions)
    first_prov = result_df["metaxy_provenance_by_field"][0]
    hashes = [first_prov["visual"], first_prov["audio_only"], first_prov["mixed"]]
    assert len(set(hashes)) == 3

    # Snapshot the results
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "sample_provenance": result_df["metaxy_provenance"][i],
            }
        )

    assert provenance_data == snapshot


def test_resolve_increment_no_current(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test resolve_increment_with_provenance when no current metadata exists."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    added_lazy, changed_lazy, removed_lazy, _ = engine.resolve_increment_with_provenance(
        current=None,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    # Materialize lazy frames
    added = added_lazy.collect()

    # When current is None, changed and removed should be None
    assert changed_lazy is None
    assert removed_lazy is None

    # All samples should be added
    assert len(added) == 3

    # Verify added has the correct columns
    assert "sample_uid" in added.columns
    assert "metaxy_provenance" in added.columns


def test_resolve_increment_with_changes(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test resolve_increment_with_provenance identifies added, changed, removed."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # First, compute expected provenance
    expected = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )
    expected_df = expected.collect()

    # Create current metadata with:
    # - sample 1: unchanged
    # - sample 2: changed (different provenance)
    # - sample 4: removed (not in upstream)
    # - sample 3: added (not in current)
    current = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 4],
                "metaxy_provenance_by_field": [
                    expected_df["metaxy_provenance_by_field"][0],  # Same as expected
                    {"default": "different_hash"},  # Different
                    {"default": "removed_hash"},  # Will be removed
                ],
                "metaxy_provenance": [
                    expected_df["metaxy_provenance"][0],  # Same
                    "different_provenance",  # Different
                    "removed_provenance",  # Removed
                ],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = engine.resolve_increment_with_provenance(
        current=current,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    # Materialize lazy frames
    added = added_lazy.collect()
    assert changed_lazy is not None
    assert removed_lazy is not None
    changed = changed_lazy.collect()
    removed = removed_lazy.collect()

    # Verify counts
    assert len(added) == 1  # sample 3
    assert len(changed) == 1  # sample 2
    assert len(removed) == 1  # sample 4

    # Verify sample IDs
    assert added["sample_uid"][0] == 3
    assert changed["sample_uid"][0] == 2
    assert removed["sample_uid"][0] == 4


def test_resolve_increment_all_unchanged(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test resolve_increment_with_provenance when all samples are unchanged."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # Compute expected provenance
    expected = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    # Current is same as expected (use lazy version for consistency)
    current = expected

    added_lazy, changed_lazy, removed_lazy, _ = engine.resolve_increment_with_provenance(
        current=current,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    # Materialize lazy frames
    added = added_lazy.collect()
    assert changed_lazy is not None
    assert removed_lazy is not None
    changed = changed_lazy.collect()
    removed = removed_lazy.collect()

    # Nothing should have changed
    assert len(added) == 0
    assert len(changed) == 0
    assert len(removed) == 0


@pytest.mark.parametrize(
    "hash_algo",
    [HashAlgorithm.XXHASH64, HashAlgorithm.SHA256, HashAlgorithm.MD5],
    ids=["xxhash64", "sha256", "md5"],
)
def test_compute_provenance_different_algorithms_snapshot(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    hash_algo: HashAlgorithm,
    snapshot,
) -> None:
    """Test compute_provenance produces correct hashes for different algorithms."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=hash_algo,
        filters={},
    )

    result_df = result.collect()

    # Extract field hashes for snapshot
    field_hashes = [result_df["metaxy_provenance_by_field"][i]["default"] for i in range(len(result_df))]

    assert field_hashes == snapshot


def test_prepare_upstream_applies_filters(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test prepare_upstream correctly applies filters."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # Filter to only include sample_uid > 1
    filters = {FeatureKey(["video"]): [nw.col("sample_uid") > 1]}

    prepared = engine.prepare_upstream(
        upstream=upstream,
        filters=filters,
    )

    prepared_df = prepared.collect()

    # Only samples 2 and 3 should remain
    assert len(prepared_df) == 2
    assert set(prepared_df["sample_uid"].to_list()) == {2, 3}
