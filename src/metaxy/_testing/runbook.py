"""Runbook system for testing and documenting Metaxy examples.

This module provides:
- Pydantic models for `.example.yaml` runbook files (Runbook, Scenario, Step types)
- RunbookRunner for executing runbooks with automatic patch management
- Context manager for running examples in tests
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr
from pydantic import Field as PydanticField

# ============================================================================
# Runbook Execution State Models
# ============================================================================


@dataclass(frozen=True)
class GraphPushed:
    """Event recorded when a graph snapshot is pushed."""

    snapshot_version: str
    timestamp: datetime
    scenario_name: str | None = None


@dataclass(frozen=True)
class PatchApplied:
    """Event recorded when a patch is applied."""

    patch_path: str
    before_snapshot: str | None
    after_snapshot: str | None
    timestamp: datetime
    scenario_name: str | None = None


RunbookEvent = GraphPushed | PatchApplied


@dataclass(frozen=True)
class RunbookExecutionState:
    """Captured state from a runbook execution."""

    events: list[RunbookEvent]

    @property
    def patch_snapshots(self) -> dict[str, tuple[str | None, str | None]]:
        """Extract patch snapshots from the event stream."""
        result = {}
        for event in self.events:
            if isinstance(event, PatchApplied):
                result[event.patch_path] = (event.before_snapshot, event.after_snapshot)
        return result

    @property
    def latest_snapshot(self) -> str | None:
        """Get the most recent snapshot version."""
        for event in reversed(self.events):
            if isinstance(event, GraphPushed):
                return event.snapshot_version
        return None


# ============================================================================
# Runbook Models
# ============================================================================


class StepType(str, Enum):
    """Type of step in a runbook scenario."""

    RUN_COMMAND = "run_command"
    APPLY_PATCH = "apply_patch"
    ASSERT_OUTPUT = "assert_output"


class BaseStep(BaseModel, ABC):
    """Base class for runbook steps."""

    model_config = ConfigDict(frozen=True)

    description: str | None = None

    @abstractmethod
    def step_type(self) -> StepType:
        raise NotImplementedError


class RunCommandStep(BaseStep):
    """Run a command or Python module."""

    type: Literal[StepType.RUN_COMMAND] = StepType.RUN_COMMAND
    command: str
    env: dict[str, str] | None = None
    capture_output: bool = False
    timeout: float = 30.0

    def step_type(self) -> StepType:
        return StepType.RUN_COMMAND


class ApplyPatchStep(BaseStep):
    """Apply a git patch file to modify example code."""

    type: Literal[StepType.APPLY_PATCH] = StepType.APPLY_PATCH
    patch_path: str
    push_graph: bool = True

    def step_type(self) -> StepType:
        return StepType.APPLY_PATCH


class AssertOutputStep(BaseStep):
    """Assert on the output of the previous command."""

    type: Literal[StepType.ASSERT_OUTPUT] = StepType.ASSERT_OUTPUT
    contains: list[str] | None = None
    not_contains: list[str] | None = None
    matches_regex: str | None = None
    returncode: int | None = None

    def step_type(self) -> StepType:
        return StepType.ASSERT_OUTPUT


class Scenario(BaseModel):
    """A scenario represents a sequence of steps to test an example."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None
    steps: list[
        Annotated[
            RunCommandStep | ApplyPatchStep | AssertOutputStep,
            PydanticField(discriminator="type"),
        ]
    ]


class Runbook(BaseModel):
    """Top-level runbook model for an example."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None
    package_name: str
    scenarios: list[Scenario]
    auto_push_graph: bool = True

    _execution_state: RunbookExecutionState | None = PrivateAttr(default=None)

    def set_execution_state(self, state: RunbookExecutionState) -> None:
        self._execution_state = state

    @property
    def execution_state(self) -> RunbookExecutionState | None:
        return self._execution_state

    @property
    def patch_snapshots(self) -> dict[str, tuple[str | None, str | None]]:
        if self._execution_state is None:
            return {}
        return self._execution_state.patch_snapshots

    @classmethod
    def from_yaml_file(cls, path: Path) -> Runbook:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml_file(self, path: Path) -> None:
        import yaml

        with open(path, "w") as f:
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
    """Runner for executing example runbooks."""

    def __init__(
        self,
        runbook: Runbook,
        example_dir: Path,
        env_overrides: dict[str, str] | None = None,
    ):
        self.runbook = runbook
        self.example_dir = example_dir
        self.env_overrides = env_overrides or {}
        self.last_result: CommandResult | None = None
        self._initial_graph_pushed = False
        self._last_pushed_snapshot: str | None = None
        self._events: list[RunbookEvent] = []
        self._current_scenario: str | None = None
        self._patch_stack = ExitStack()

        from metaxy._testing.metaxy_project import ExternalMetaxyProject

        self.project = ExternalMetaxyProject(example_dir, require_config=True)

    def get_latest_snapshot_version(self) -> str | None:
        """Get the latest snapshot version from the metadata store."""
        import os

        from metaxy.config import MetaxyConfig
        from metaxy.metadata_store.system.storage import SystemTableStorage

        old_env = {k: os.environ.get(k) for k in self.env_overrides}

        try:
            os.environ.update(self.env_overrides)
            config = MetaxyConfig.load(self.project.project_dir / "metaxy.toml")
            store = config.get_store()
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        with store:
            storage = SystemTableStorage(store)
            snapshots_df = storage.read_graph_snapshots()

            if snapshots_df.height == 0:
                return None

            return snapshots_df["metaxy_snapshot_version"][0]

    def push_graph_snapshot(self) -> str | None:
        """Push the current graph snapshot using metaxy graph push."""
        if not self.runbook.auto_push_graph:
            return None

        result = self.project.push_graph(env=self.env_overrides)

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to push graph snapshot:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        snapshot_version = result.stdout.strip()

        if snapshot_version:
            self._events.append(
                GraphPushed(
                    timestamp=datetime.now(),
                    scenario_name=self._current_scenario,
                    snapshot_version=snapshot_version,
                )
            )
            return snapshot_version

        return None

    def run_command(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        timeout: float = 30.0,
    ) -> CommandResult:
        """Execute a command."""
        merged_env = self.env_overrides.copy()
        if env:
            merged_env.update(env)

        result = self.project.run_command(
            command,
            env=merged_env,
            capture_output=capture_output,
            timeout=timeout,
        )

        self.last_result = CommandResult(
            returncode=result.returncode,
            stdout=result.stdout if capture_output else "",
            stderr=result.stderr if capture_output else "",
        )

        return self.last_result

    @contextmanager
    def apply_patch(
        self, patch_path: str, *, push_graph: bool = True
    ) -> Iterator[None]:
        """Apply a patch file as a context manager.

        The patch is reverted when exiting the context.

        Args:
            patch_path: Path to patch file relative to example directory.
            push_graph: Whether to push graph snapshot after applying.

        Yields:
            None

        Example:
            with runner.apply_patch("patches/01_update.patch"):
                runner.run_command("python pipeline.py")
        """
        patch_path_abs = self.example_dir / patch_path

        if not patch_path_abs.exists():
            raise FileNotFoundError(
                f"Patch file not found: {patch_path_abs} (resolved from {patch_path})"
            )

        before_snapshot = self._last_pushed_snapshot

        # Apply the patch
        result = subprocess.run(
            ["patch", "-p1", "-i", patch_path, "--no-backup-if-mismatch"],
            capture_output=True,
            text=True,
            cwd=self.example_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to apply patch {patch_path}:\n{result.stderr}")

        after_snapshot = None
        if push_graph:
            after_snapshot = self.push_graph_snapshot()
            if after_snapshot:
                self._last_pushed_snapshot = after_snapshot

        self._events.append(
            PatchApplied(
                timestamp=datetime.now(),
                scenario_name=self._current_scenario,
                patch_path=patch_path,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
            )
        )

        try:
            yield
        finally:
            # Revert the patch
            subprocess.run(
                ["patch", "-R", "-p1", "-i", patch_path, "--no-backup-if-mismatch"],
                capture_output=True,
                cwd=self.example_dir,
            )

    def _run_step(self, step: BaseStep) -> None:
        """Execute a single step."""
        if isinstance(step, RunCommandStep):
            self.run_command(
                step.command,
                env=step.env,
                capture_output=step.capture_output,
                timeout=step.timeout,
            )
        elif isinstance(step, ApplyPatchStep):
            # For runbook execution, patches stack (don't revert until run completes)
            self._patch_stack.enter_context(
                self.apply_patch(step.patch_path, push_graph=step.push_graph)
            )
        elif isinstance(step, AssertOutputStep):
            self._assert_output(step)
        else:
            raise ValueError(f"Unknown step type: {type(step)}")

    def _assert_output(self, step: AssertOutputStep) -> None:
        """Validate assertions on the last command result."""
        if self.last_result is None:
            raise RuntimeError(
                "No command result available for assertion. "
                "AssertOutputStep must follow a RunCommandStep with capture_output=True."
            )

        expected_returncode = step.returncode if step.returncode is not None else 0
        assert self.last_result.returncode == expected_returncode, (
            f"Expected returncode {expected_returncode}, "
            f"got {self.last_result.returncode}\n"
            f"stderr: {self.last_result.stderr}"
        )

        if step.contains:
            for substring in step.contains:
                assert substring in self.last_result.stdout, (
                    f"Expected substring not found in stdout: {substring!r}\n"
                    f"stdout: {self.last_result.stdout}"
                )

        if step.not_contains:
            for substring in step.not_contains:
                assert substring not in self.last_result.stdout, (
                    f"Unexpected substring found in stdout: {substring!r}\n"
                    f"stdout: {self.last_result.stdout}"
                )

        if step.matches_regex:
            assert re.search(step.matches_regex, self.last_result.stdout), (
                f"Regex pattern not matched: {step.matches_regex!r}\n"
                f"stdout: {self.last_result.stdout}"
            )

    def run_scenario(self, scenario: Scenario) -> None:
        """Execute a single scenario."""
        self._current_scenario = scenario.name

        if not self._initial_graph_pushed:
            initial_snapshot = self.push_graph_snapshot()
            if initial_snapshot:
                self._last_pushed_snapshot = initial_snapshot
            self._initial_graph_pushed = True

        for step in scenario.steps:
            self._run_step(step)

    def run(self) -> None:
        """Execute all scenarios in the runbook."""
        with self._patch_stack:
            for scenario in self.runbook.scenarios:
                self.run_scenario(scenario)

            state = RunbookExecutionState(events=self._events.copy())
            self.runbook.set_execution_state(state)

    @classmethod
    def from_yaml_file(
        cls,
        yaml_path: Path,
        env_overrides: dict[str, str] | None = None,
    ) -> RunbookRunner:
        """Create a runner from a YAML runbook file."""
        runbook = Runbook.from_yaml_file(yaml_path)
        example_dir = yaml_path.parent
        return cls(
            runbook=runbook,
            example_dir=example_dir,
            env_overrides=env_overrides,
        )

    @classmethod
    @contextmanager
    def runner_for_project(
        cls,
        example_dir: Path,
        env_overrides: dict[str, str] | None = None,
        work_dir: Path | None = None,
    ) -> Iterator[RunbookRunner]:
        """Context manager for running an example runbook in tests.

        Copies the example to a temporary work directory so patches don't
        modify the original source code.
        """
        temp_dir_obj = None
        if work_dir is None:
            temp_dir_obj = tempfile.TemporaryDirectory()
            work_dir = Path(temp_dir_obj.name) / example_dir.name

        try:

            def ignore_patterns(directory, files):
                return [
                    f
                    for f in files
                    if f in ("__pycache__", ".venv") or f.endswith(".pyc")
                ]

            shutil.copytree(
                example_dir, work_dir, dirs_exist_ok=True, ignore=ignore_patterns
            )

            yaml_path = work_dir / ".example.yaml"
            runner = cls.from_yaml_file(
                yaml_path=yaml_path,
                env_overrides=env_overrides,
            )

            yield runner
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()
