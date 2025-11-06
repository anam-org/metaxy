"""Testing infrastructure for Metaxy examples and runbooks.

This is a private module (_testing) containing testing utilities organized into:
- runbook: Runbook system for testing and documenting examples
- metaxy_project: Project helpers for creating and managing temporary Metaxy projects
- helpers: Test data helpers for creating properly formatted test DataFrames
"""

# Test data helpers
from metaxy._testing.helpers import add_metaxy_provenance_column

# Metaxy project helpers
from metaxy._testing.metaxy_project import (
    ExternalMetaxyProject,
    HashAlgorithmCases,
    MetaxyProject,
    TempFeatureModule,
    TempMetaxyProject,
    assert_all_results_equal,
)

# Runbook system
from metaxy._testing.runbook import (
    ApplyPatchStep,
    AssertOutputStep,
    BaseStep,
    Runbook,
    RunbookRunner,
    RunCommandStep,
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
    # Metaxy project helpers
    "TempFeatureModule",
    "HashAlgorithmCases",
    "MetaxyProject",
    "ExternalMetaxyProject",
    "TempMetaxyProject",
    "assert_all_results_equal",
    # Test data helpers
    "add_metaxy_provenance_column",
]
