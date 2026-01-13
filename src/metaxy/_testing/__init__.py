"""Testing infrastructure for Metaxy examples and runbooks.

This is a private module (_testing) containing testing utilities organized into:
- runbook: Runbook system for testing and documenting examples
- metaxy_project: Project helpers for creating and managing temporary Metaxy projects
- pytest_helpers: Testing helpers for pytest tests
- models: Testing-specific model implementations
"""

# Runbook system
# Metaxy project helpers
from metaxy._testing.metaxy_project import (
    ExternalMetaxyProject,
    HashAlgorithmCases,
    MetaxyProject,
    TempFeatureModule,
    TempMetaxyProject,
    assert_all_results_equal,
)
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.pytest_helpers import (
    add_metaxy_provenance_column,
    add_metaxy_system_columns,
)
from metaxy._testing.runbook import (
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

__all__ = [
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
    # Pytest helpers
    "add_metaxy_provenance_column",
    "add_metaxy_system_columns",
    # Testing models
    "SampleFeatureSpec",
    "SampleFeature",
]
