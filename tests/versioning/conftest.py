"""Fixtures for versioning tests, especially optional dependencies integration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy import FeatureDep, FeatureKey, FieldDep, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.models.feature import FeatureGraph
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


@pytest.fixture
def make_upstream_feature(
    graph: FeatureGraph,
) -> Callable[[str, str], type[SampleFeature]]:
    """Factory fixture to create upstream features dynamically.

    Args:
        graph: The feature graph (injected by pytest)

    Returns:
        A callable that creates a feature class given a key and field name
    """

    def _make_feature(feature_key: str, field_key: str) -> type[SampleFeature]:
        """Create an upstream feature with the given key and field.

        Args:
            feature_key: The feature key (e.g., "required", "optional_a")
            field_key: The field key (e.g., "data_req", "data_a")
        """

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey([feature_key]),
                fields=[FieldSpec(key=FieldKey([field_key]), code_version="1")],
            ),
        ):
            pass

        # Verify registration
        assert FeatureKey([feature_key]) in graph.features_by_key
        return UpstreamFeature

    return _make_feature


@pytest.fixture
def make_downstream_feature(
    graph: FeatureGraph,
) -> Callable[
    [str, list[tuple[str, bool]], list[tuple[str, str]]], type[SampleFeature]
]:
    """Factory fixture to create downstream features with dependencies.

    Args:
        graph: The feature graph (injected by pytest)

    Returns:
        A callable that creates a downstream feature
    """

    def _make_feature(
        feature_key: str,
        deps: list[tuple[str, bool]],
        field_deps: list[tuple[str, str]],
    ) -> type[SampleFeature]:
        """Create a downstream feature with the given dependencies.

        Args:
            feature_key: The feature key
            deps: List of (feature_key, optional) tuples
            field_deps: List of (feature_key, field_key) tuples for field dependencies
        """

        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey([feature_key]),
                deps=[
                    FeatureDep(feature=FeatureKey([fk]), optional=opt)
                    for fk, opt in deps
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["default"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey([fk]),
                                fields=[FieldKey([field_k])],
                            )
                            for fk, field_k in field_deps
                        ],
                    ),
                ],
            ),
        ):
            pass

        # Verify registration
        assert FeatureKey([feature_key]) in graph.features_by_key
        return DownstreamFeature

    return _make_feature


@pytest.fixture
def make_metadata_df() -> Callable[[list[int], str, dict[int, str]], nw.LazyFrame[Any]]:
    """Factory fixture to create metadata DataFrames with provenance.

    Returns:
        A callable that creates a lazy DataFrame with metaxy metadata columns
    """

    def _make_df(
        sample_uids: list[int],
        field_key: str,
        data_values: dict[int, str],
    ) -> nw.LazyFrame[Any]:
        """Create a metadata DataFrame.

        Args:
            sample_uids: List of sample UIDs
            field_key: The field key for the data column
            data_values: Dict mapping sample_uid to data value
        """
        data = {
            "sample_uid": sample_uids,
            field_key: [data_values[uid] for uid in sample_uids],
            "metaxy_provenance_by_field": [
                {field_key: f"hash_{data_values[uid]}"} for uid in sample_uids
            ],
            "metaxy_data_version_by_field": [
                {field_key: f"hash_{data_values[uid]}"} for uid in sample_uids
            ],
            "metaxy_provenance": [f"prov_{data_values[uid]}" for uid in sample_uids],
        }
        return nw.from_native(pl.DataFrame(data).lazy())

    return _make_df


@pytest.fixture
def versioning_engine_for(
    graph: FeatureGraph,
) -> Callable[[str], PolarsVersioningEngine]:
    """Factory fixture to create a PolarsVersioningEngine for a feature.

    Args:
        graph: The feature graph (injected by pytest)

    Returns:
        A callable that creates an engine for a given feature key
    """

    def _make_engine(feature_key: str) -> PolarsVersioningEngine:
        """Create a versioning engine for the given feature.

        Args:
            feature_key: The feature key to create an engine for
        """
        plan = graph.get_feature_plan(FeatureKey([feature_key]))
        return PolarsVersioningEngine(plan)

    return _make_engine


@pytest.fixture
def standard_upstream_setup(
    make_upstream_feature: Callable[[str, str], type[SampleFeature]],
    make_downstream_feature: Callable[
        [str, list[tuple[str, bool]], list[tuple[str, str]]], type[SampleFeature]
    ],
) -> dict[str, type[SampleFeature]]:
    """Create a standard setup with required and optional upstream features.

    Creates:
    - RequiredUpstream with field "data_req"
    - OptionalUpstream with field "data_opt"
    - Downstream with both as dependencies

    Returns:
        Dictionary with keys: required, optional, downstream
    """
    required = make_upstream_feature("required", "data_req")
    optional = make_upstream_feature("optional", "data_opt")
    downstream = make_downstream_feature(
        "downstream",
        [("required", False), ("optional", True)],
        [("required", "data_req"), ("optional", "data_opt")],
    )

    return {
        "required": required,
        "optional": optional,
        "downstream": downstream,
    }


@pytest.fixture
def hash_algo() -> HashAlgorithm:
    """Return the default hash algorithm for testing."""
    return HashAlgorithm.XXHASH64
