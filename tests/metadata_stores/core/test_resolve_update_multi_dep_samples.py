"""Tests for sample-scoped resolve_update on multi-dependency features."""

from __future__ import annotations

from typing import Any

import polars as pl
from metaxy import FeatureDep, FeatureGraph
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.field import FieldDep, FieldSpec
from metaxy.models.types import FieldKey
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from pytest_cases import parametrize_with_cases

from metaxy_testing import add_metaxy_provenance_column
from tests.metadata_stores.conftest import BasicStoreCases


def _provider_metadata(field_name: str, versions: dict[str, str]) -> pl.DataFrame:
    sample_ids = list(versions)
    field_versions = list(versions.values())
    return pl.DataFrame(
        {
            "sample_uid": sample_ids,
            METAXY_PROVENANCE_BY_FIELD: [{field_name: version} for version in field_versions],
            METAXY_DATA_VERSION_BY_FIELD: [{field_name: version} for version in field_versions],
        }
    )


def _write_provider(
    store: MetadataStore,
    feature: type[SampleFeature],
    *,
    field_name: str,
    versions: dict[str, str],
) -> None:
    provider_df = _provider_metadata(field_name, versions)
    store.write(feature, add_metaxy_provenance_column(provider_df, feature))


def _create_multi_dep_features(
    graph: FeatureGraph,
    *,
    suffix: str,
) -> tuple[type[SampleFeature], type[SampleFeature], type[SampleFeature]]:
    with graph.use():

        class ProviderA(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=f"provider_a_{suffix}",
                fields=[FieldSpec(key=FieldKey(["a_payload"]), code_version="1")],
            ),
        ):
            pass

        class ProviderB(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=f"provider_b_{suffix}",
                fields=[FieldSpec(key=FieldKey(["b_payload"]), code_version="1")],
            ),
        ):
            pass

        class FusedFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=f"fused_{suffix}",
                deps=[
                    FeatureDep(feature=ProviderA),
                    FeatureDep(feature=ProviderB),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["fused_payload"]),
                        code_version="1",
                        deps=[
                            FieldDep(feature=ProviderA.spec().key, fields=[FieldKey(["a_payload"])]),
                            FieldDep(feature=ProviderB.spec().key, fields=[FieldKey(["b_payload"])]),
                        ],
                    )
                ],
            ),
        ):
            pass

    return ProviderA, ProviderB, FusedFeature


class TestResolveUpdateMultiDepSamples:
    """Tests for non-root resolve_update calls that provide samples."""

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_samples_detect_staleness_across_all_upstreams(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ) -> None:
        """A scoped sample still detects changes in every declared dependency."""
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        provider_a, provider_b, fused_feature = _create_multi_dep_features(graph, suffix="stale")

        with graph.use(), store.open("w"):
            _write_provider(
                store,
                provider_a,
                field_name="a_payload",
                versions={"s1": "a_v1_s1", "s2": "a_v1_s2"},
            )
            _write_provider(
                store,
                provider_b,
                field_name="b_payload",
                versions={"s1": "b_v1_s1", "s2": "b_v1_s2"},
            )
            store.write(fused_feature, store.resolve_update(fused_feature).new)

            _write_provider(
                store,
                provider_b,
                field_name="b_payload",
                versions={"s1": "b_v2_s1", "s2": "b_v2_s2"},
            )

            increment = store.resolve_update(
                fused_feature,
                samples=pl.DataFrame({"sample_uid": ["s1", "s2"]}),
            )

            assert increment.new.shape[0] == 0
            assert increment.stale["sample_uid"].sort().to_list() == ["s1", "s2"]

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_samples_only_scope_the_requested_ids(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ) -> None:
        """The sample frame limits results without becoming the provenance source."""
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        provider_a, provider_b, fused_feature = _create_multi_dep_features(graph, suffix="scope")

        with graph.use(), store.open("w"):
            _write_provider(
                store,
                provider_a,
                field_name="a_payload",
                versions={"s1": "a_v1_s1", "s2": "a_v1_s2", "s3": "a_v1_s3"},
            )
            _write_provider(
                store,
                provider_b,
                field_name="b_payload",
                versions={"s1": "b_v1_s1", "s2": "b_v1_s2", "s3": "b_v1_s3"},
            )
            store.write(fused_feature, store.resolve_update(fused_feature).new)

            _write_provider(
                store,
                provider_b,
                field_name="b_payload",
                versions={"s1": "b_v2_s1", "s2": "b_v2_s2", "s3": "b_v2_s3"},
            )

            increment = store.resolve_update(
                fused_feature,
                samples=pl.DataFrame({"sample_uid": ["s1", "s3"]}),
            )

            assert increment.new.shape[0] == 0
            assert increment.stale["sample_uid"].sort().to_list() == ["s1", "s3"]
            assert increment.orphaned.shape[0] == 0

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_skip_comparison_respects_sample_scope(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ) -> None:
        """skip_comparison returns the expected rows for the requested sample scope."""
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        provider_a, provider_b, fused_feature = _create_multi_dep_features(graph, suffix="skip")

        with graph.use(), store.open("w"):
            _write_provider(
                store,
                provider_a,
                field_name="a_payload",
                versions={"s1": "a_v1_s1", "s2": "a_v1_s2", "s3": "a_v1_s3"},
            )
            _write_provider(
                store,
                provider_b,
                field_name="b_payload",
                versions={"s1": "b_v1_s1", "s2": "b_v1_s2", "s3": "b_v1_s3"},
            )

            increment = store.resolve_update(
                fused_feature,
                samples=pl.DataFrame({"sample_uid": ["s1"]}),
                skip_comparison=True,
            )

            assert increment.new["sample_uid"].to_list() == ["s1"]
            assert increment.stale.shape[0] == 0
            assert increment.orphaned.shape[0] == 0

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_root_feature_samples_behavior_is_unchanged(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ) -> None:
        """Root features still use the caller-provided provenance directly."""
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        with graph.use():

            class RootFeature(
                SampleFeature,
                spec=SampleFeatureSpec(
                    key="root_samples_unchanged",
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

        with graph.use(), store.open("w"):
            increment = store.resolve_update(
                RootFeature,
                samples=pl.DataFrame(
                    {
                        "sample_uid": ["s1", "s2"],
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"value": "hash1"},
                            {"value": "hash2"},
                        ],
                    }
                ),
            )

            assert increment.new.shape[0] == 2
