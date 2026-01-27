"""Testing infrastructure for Metaxy examples and runbooks.

This module contains testing utilities organized into:
- runbook: Runbook system for testing and documenting examples
- metaxy_project: Project helpers for creating and managing temporary Metaxy projects
- pytest_helpers: Testing helpers for pytest tests
- models: Testing-specific model implementations
"""

# Runbook system
# Metaxy project helpers
from metaxy_testing.metaxy_project import (
    COVERAGE_ENV_VARS,
    ExternalMetaxyProject,
    HashAlgorithmCases,
    MetaxyProject,
    TempFeatureModule,
    TempMetaxyProject,
    _get_coverage_env,
    assert_all_results_equal,
    env_override,
)
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from metaxy_testing.pytest_helpers import (
    add_metaxy_provenance_column,
    add_metaxy_system_columns,
)
from metaxy_testing.runbook import (
    ApplyPatchStep,
    AssertOutputStep,
    BaseStep,
    CommandExecuted,
    GraphPushed,
    PatchApplied,
    Runbook,
    RunbookExecutionState,
    RunbookRunner,
    RunCommandStep,
    SavedRunbookResult,
    Scenario,
    StepType,
)

# Module path for Ray test features - defined as string to avoid importing
# the module at load time (which would register features to the global graph)
RAY_FEATURES_MODULE = "metaxy_testing.ray_features"

__all__ = [
    # Module paths
    "RAY_FEATURES_MODULE",
    # Runbook system
    "Runbook",
    "Scenario",
    "BaseStep",
    "RunCommandStep",
    "ApplyPatchStep",
    "AssertOutputStep",
    "StepType",
    "RunbookRunner",
    # Runbook execution state
    "RunbookExecutionState",
    "SavedRunbookResult",
    "GraphPushed",
    "PatchApplied",
    "CommandExecuted",
    # Metaxy project helpers
    "TempFeatureModule",
    "HashAlgorithmCases",
    "MetaxyProject",
    "ExternalMetaxyProject",
    "TempMetaxyProject",
    "assert_all_results_equal",
    "env_override",
    "COVERAGE_ENV_VARS",
    "_get_coverage_env",
    # Pytest helpers
    "add_metaxy_provenance_column",
    "add_metaxy_system_columns",
    # Testing models
    "SampleFeatureSpec",
    "SampleFeature",
]
