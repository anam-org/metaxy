"""Shared fixtures for per-store test modules.

Provides parametrized fixtures that pytest-cases generates, since
@parametrize_with_cases doesn't work on inherited class methods across modules.
"""

from pytest_cases import fixture, parametrize_with_cases

from metaxy import BaseFeature
from tests.metadata_stores.shared.resolve_update import (
    FeatureGraphCases,
    FeaturePlanOutput,
    OptionalDependencyCases,
    RootFeatureCases,
)
from tests.metadata_stores.shared.versioning import FeaturePlanCases, FeaturePlanSequence


@fixture
@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def root_feature(root_feature: type[BaseFeature]) -> type[BaseFeature]:
    return root_feature


@fixture
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def feature_plan_config(feature_plan_config: FeaturePlanOutput) -> FeaturePlanOutput:
    return feature_plan_config


@fixture
@parametrize_with_cases("feature_plan_sequence", cases=FeaturePlanCases)
def feature_plan_sequence(feature_plan_sequence: FeaturePlanSequence) -> FeaturePlanSequence:
    return feature_plan_sequence


@fixture
@parametrize_with_cases("optional_dep_config", cases=OptionalDependencyCases)
def optional_dep_config(optional_dep_config: FeaturePlanOutput) -> FeaturePlanOutput:
    return optional_dep_config
