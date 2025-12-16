"""Runbook system for testing and documenting Metaxy examples.

This module provides:
- Pydantic models for `.example.yaml` runbook files (Runbook, Scenario, Step types)
- RunbookRunner for executing runbooks with automatic patch management
- Context manager for running examples in tests
"""

from __future__ import annotations

import os
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

# ============================================================================
# Runbook Models
# ============================================================================


class StepType(str, Enum):
    """Type of step in a runbook scenario."""

    RUN_COMMAND = "run_command"
    APPLY_PATCH = "apply_patch"
    ASSERT_OUTPUT = "assert_output"


class BaseStep(BaseModel, ABC):
    """Base class for runbook steps.

    Each step represents an action in testing or documenting an example.
    """

    model_config = ConfigDict(frozen=True)

    description: str | None = None
    """Optional human-readable description of this step."""

    @abstractmethod
    def step_type(self) -> StepType:
        """Return the step type for this step."""
        raise NotImplementedError


class RunCommandStep(BaseStep):
    """Run a command or Python module.

    This step executes a shell command or Python module and optionally captures
    the output for later assertions.

    Examples:
        >>> # Run a Python module
        >>> RunCommandStep(
        ...     type="run_command",
        ...     command="python -m example_recompute.setup_data",
        ... )

        >>> # Run metaxy CLI
        >>> RunCommandStep(
        ...     type="run_command",
        ...     command="metaxy list features",
        ...     capture_output=True,
        ... )
    """

    type: Literal[StepType.RUN_COMMAND] = StepType.RUN_COMMAND
    command: str
    """The command to execute (e.g., 'python -m module_name' or 'metaxy list features')."""

    env: dict[str, str] | None = None
    """Environment variables to set for this command."""

    capture_output: bool = False
    """Whether to capture stdout/stderr for assertions."""

    timeout: float = 30.0
    """Timeout in seconds for the command."""

    def step_type(self) -> StepType:
        return StepType.RUN_COMMAND


class ApplyPatchStep(BaseStep):
    """Apply a git patch file to modify example code.

    This step applies a patch to transition between code versions, demonstrating
    code evolution. Patches are applied temporarily during test execution.

    The patch_path is relative to the example directory (where .example.yaml lives).

    Examples:
        >>> # Apply a patch to update algorithm
        >>> ApplyPatchStep(
        ...     type="apply_patch",
        ...     patch_path="patches/01_update_algorithm.patch",
        ...     description="Update parent feature embedding algorithm to v2",
        ... )
    """

    type: Literal[StepType.APPLY_PATCH] = StepType.APPLY_PATCH
    patch_path: str
    """Path to patch file relative to example directory."""

    def step_type(self) -> StepType:
        return StepType.APPLY_PATCH


class AssertOutputStep(BaseStep):
    """Assert on the output of the previous command.

    This step validates that the previous RunCommandStep produced the expected
    output. Supports substring matching and regex patterns.

    Examples:
        >>> # Assert specific strings appear in output
        >>> AssertOutputStep(
        ...     type="assert_output",
        ...     contains=["Pipeline STAGE=1", "✅ Stage 1 pipeline complete!"],
        ... )

        >>> # Assert returncode is 0
        >>> AssertOutputStep(
        ...     type="assert_output",
        ...     returncode=0,
        ... )
    """

    type: Literal[StepType.ASSERT_OUTPUT] = StepType.ASSERT_OUTPUT
    contains: list[str] | None = None
    """List of substrings that must appear in stdout."""

    not_contains: list[str] | None = None
    """List of substrings that must NOT appear in stdout."""

    matches_regex: str | None = None
    """Regex pattern that stdout must match."""

    returncode: int | None = None
    """Expected return code (default: 0 if not specified)."""

    def step_type(self) -> StepType:
        return StepType.ASSERT_OUTPUT


class Scenario(BaseModel):
    """A scenario represents a sequence of steps to test an example.

    Scenarios are the main unit of testing. Each scenario has a name and a list
    of steps that are executed in order.

    Examples:
        >>> Scenario(
        ...     name="Initial run",
        ...     description="First pipeline run with initial feature definitions",
        ...     steps=[
        ...         RunCommandStep(command="python -m example.setup_data"),
        ...         RunCommandStep(command="python -m example.pipeline", capture_output=True),
        ...         AssertOutputStep(contains=["Pipeline complete!"]),
        ...     ],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Name of this scenario (e.g., 'Initial run', 'Idempotent rerun')."""

    description: str | None = None
    """Optional human-readable description of what this scenario tests."""

    steps: list[
        Annotated[
            RunCommandStep | ApplyPatchStep | AssertOutputStep,
            PydanticField(discriminator="type"),
        ]
    ]
    """Ordered list of steps to execute in this scenario."""


class Runbook(BaseModel):
    """Top-level runbook model for an example.

    A runbook defines how to test and document an example. It contains metadata
    about the example and one or more scenarios that test different aspects.

    The runbook file should be named `.example.yaml` and placed in the example
    directory alongside metaxy.toml.

    Examples:
        >>> Runbook(
        ...     name="Recompute Example",
        ...     description="Demonstrates automatic recomputation",
        ...     package_name="example_recompute",
        ...     scenarios=[...],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Human-readable name of the example."""

    description: str | None = None
    """Description of what this example demonstrates."""

    package_name: str
    """Python package name (e.g., 'example_recompute')."""

    scenarios: list[Scenario]
    """List of test scenarios for this example."""

    @classmethod
    def from_yaml_file(cls, path: Path) -> Runbook:
        """Load a runbook from a YAML file.

        Args:
            path: Path to the .example.yaml file.

        Returns:
            Parsed Runbook instance.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml_file(self, path: Path) -> None:
        """Save this runbook to a YAML file.

        Args:
            path: Path where the .example.yaml file should be written.
        """
        import yaml

        with open(path, "w") as f:
            # Use model_dump with mode='json' to get JSON-serializable data
            data = self.model_dump(mode="json")
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


# ============================================================================
# Runbook Runner
# ============================================================================


class CommandResult:
    """Result of running a command."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RunbookRunner:
    """Runner for executing example runbooks.

    This class handles:
    - Loading runbooks from YAML
    - Setting up test environments
    - Executing commands
    - Applying/reverting patches
    - Validating assertions
    """

    def __init__(
        self,
        runbook: Runbook,
        example_dir: Path,
        override_db_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
    ):
        """Initialize the runbook runner.

        Args:
            runbook: The runbook to execute.
            example_dir: Directory containing the example code and .example.yaml.
            override_db_path: Optional path to database (for METAXY_STORES__DEV__CONFIG__DATABASE).
            env_overrides: Additional environment variable overrides.
        """
        self.runbook = runbook
        self.example_dir = example_dir
        self.override_db_path = override_db_path
        self.env_overrides = env_overrides or {}
        self.last_result: CommandResult | None = None
        self.applied_patches: list[str] = []

    def get_base_env(self) -> dict[str, str]:
        """Get base environment with test-specific overrides.

        Returns:
            Environment dict with test database and other overrides.
        """
        env = os.environ.copy()

        # Override database path if provided
        if self.override_db_path:
            env["METAXY_STORES__DEV__CONFIG__DATABASE"] = str(self.override_db_path)

        # Apply additional overrides
        env.update(self.env_overrides)

        return env

    def run_command(
        self,
        step: RunCommandStep,
        scenario_name: str,
    ) -> CommandResult:
        """Execute a command step.

        Args:
            step: The RunCommandStep to execute.
            scenario_name: Name of the current scenario (for logging).

        Returns:
            CommandResult with returncode and output.
        """
        env = self.get_base_env()

        # Apply step-specific environment variables
        if step.env:
            env.update(step.env)

        # Execute the command
        result = subprocess.run(
            step.command,
            shell=True,
            capture_output=step.capture_output,
            text=True,
            timeout=step.timeout,
            env=env,
            cwd=self.example_dir,
        )

        # Store for assertions
        self.last_result = CommandResult(
            returncode=result.returncode,
            stdout=result.stdout if step.capture_output else "",
            stderr=result.stderr if step.capture_output else "",
        )

        return self.last_result

    def apply_patch(self, step: ApplyPatchStep, scenario_name: str) -> None:
        """Apply a patch file.

        Args:
            step: The ApplyPatchStep to execute.
            scenario_name: Name of the current scenario (for logging).

        Raises:
            RuntimeError: If patch application fails.
        """
        patch_path_abs = self.example_dir / step.patch_path

        if not patch_path_abs.exists():
            raise FileNotFoundError(
                f"Patch file not found: {patch_path_abs} (resolved from {step.patch_path})"
            )

        # Apply the patch using patch command (works without git)
        # -p1: strip one level from paths (a/ and b/ prefixes)
        # -i: specify input file
        # --no-backup-if-mismatch: don't create .orig backup files
        result = subprocess.run(
            ["patch", "-p1", "-i", step.patch_path, "--no-backup-if-mismatch"],
            capture_output=True,
            text=True,
            cwd=self.example_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to apply patch {step.patch_path}:\n{result.stderr}"
            )

        # Track applied patches for cleanup (use relative path)
        self.applied_patches.append(step.patch_path)

    def revert_patches(self) -> None:
        """Revert all applied patches in reverse order."""
        for patch_path in reversed(self.applied_patches):
            # Use patch -R to reverse the patch
            subprocess.run(
                ["patch", "-R", "-p1", "-i", patch_path, "--no-backup-if-mismatch"],
                capture_output=True,
                cwd=self.example_dir,
            )
        self.applied_patches.clear()

    def assert_output(self, step: AssertOutputStep, scenario_name: str) -> None:
        """Validate assertions on the last command result.

        Args:
            step: The AssertOutputStep to execute.
            scenario_name: Name of the current scenario (for logging).

        Raises:
            AssertionError: If any assertion fails.
        """
        if self.last_result is None:
            raise RuntimeError(
                "No command result available for assertion. "
                "AssertOutputStep must follow a RunCommandStep with capture_output=True."
            )

        # Check return code
        expected_returncode = step.returncode if step.returncode is not None else 0
        assert self.last_result.returncode == expected_returncode, (
            f"Expected returncode {expected_returncode}, "
            f"got {self.last_result.returncode}\n"
            f"stderr: {self.last_result.stderr}"
        )

        # Check contains assertions
        if step.contains:
            for substring in step.contains:
                assert substring in self.last_result.stdout, (
                    f"Expected substring not found in stdout: {substring!r}\n"
                    f"stdout: {self.last_result.stdout}"
                )

        # Check not_contains assertions
        if step.not_contains:
            for substring in step.not_contains:
                assert substring not in self.last_result.stdout, (
                    f"Unexpected substring found in stdout: {substring!r}\n"
                    f"stdout: {self.last_result.stdout}"
                )

        # Check regex match
        if step.matches_regex:
            assert re.search(step.matches_regex, self.last_result.stdout), (
                f"Regex pattern not matched: {step.matches_regex!r}\n"
                f"stdout: {self.last_result.stdout}"
            )

    def run_scenario(self, scenario: Scenario) -> None:
        """Execute a single scenario.

        Args:
            scenario: The scenario to execute.
        """
        for step in scenario.steps:
            if isinstance(step, RunCommandStep):
                self.run_command(step, scenario.name)
            elif isinstance(step, ApplyPatchStep):
                self.apply_patch(step, scenario.name)
            elif isinstance(step, AssertOutputStep):
                self.assert_output(step, scenario.name)
            else:
                raise ValueError(f"Unknown step type: {type(step)}")

    def run(self) -> None:
        """Execute all scenarios in the runbook."""
        try:
            for scenario in self.runbook.scenarios:
                self.run_scenario(scenario)
        finally:
            # Always revert patches on completion or error
            if self.applied_patches:
                self.revert_patches()

    @classmethod
    def from_yaml_file(
        cls,
        yaml_path: Path,
        override_db_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> RunbookRunner:
        """Create a runner from a YAML runbook file.

        Args:
            yaml_path: Path to the .example.yaml file.
            override_db_path: Optional path to test database.
            env_overrides: Additional environment variable overrides.

        Returns:
            Configured RunbookRunner instance.
        """
        runbook = Runbook.from_yaml_file(yaml_path)
        example_dir = yaml_path.parent
        return cls(
            runbook=runbook,
            example_dir=example_dir,
            override_db_path=override_db_path,
            env_overrides=env_overrides,
        )

    @classmethod
    @contextmanager
    def runner_for_project(
        cls,
        example_dir: Path,
        override_db_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> Iterator[RunbookRunner]:
        """Context manager for running an example runbook in tests.

        This handles cleanup even if the runbook execution fails.

        Args:
            example_dir: Directory containing .example.yaml.
            override_db_path: Optional path to test database.
            env_overrides: Additional environment variable overrides.

        Yields:
            RunbookRunner instance ready to execute.
        """
        yaml_path = example_dir / ".example.yaml"
        runner = cls.from_yaml_file(
            yaml_path=yaml_path,
            override_db_path=override_db_path,
            env_overrides=env_overrides,
        )

        try:
            yield runner
        finally:
            # Ensure patches are reverted
            if runner.applied_patches:
                runner.revert_patches()
