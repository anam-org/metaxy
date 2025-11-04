"""Core logic for loading runbooks and applying patches.

This module provides utilities for:
- Loading runbooks from .example.yaml files
- Applying patches to show code evolution
- Reading source files at different stages
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from metaxy._testing import Runbook


class RunbookLoader:
    """Loader for example runbooks and associated files."""

    def __init__(self, examples_dir: Path):
        """Initialize the runbook loader.

        Args:
            examples_dir: Path to the examples directory.
        """
        self.examples_dir = examples_dir
        self._runbook_cache: dict[str, Runbook] = {}

    def get_example_dir(self, example_name: str) -> Path:
        """Get the directory for a specific example.

        Args:
            example_name: Name of the example (e.g., "recompute" for "example-recompute").

        Returns:
            Path to the example directory.

        Raises:
            FileNotFoundError: If example directory doesn't exist.
        """
        # Support both "recompute" and "example-recompute" formats
        if not example_name.startswith("example-"):
            example_name = f"example-{example_name}"

        example_dir = self.examples_dir / example_name

        if not example_dir.exists():
            raise FileNotFoundError(
                f"Example directory not found: {example_dir}. "
                f"Available examples: {self.list_examples()}"
            )

        return example_dir

    def list_examples(self) -> list[str]:
        """List all available examples.

        Returns:
            List of example names (without "example-" prefix).
        """
        examples = []
        for path in self.examples_dir.glob("example-*"):
            if path.is_dir() and (path / ".example.yaml").exists():
                # Remove "example-" prefix
                examples.append(path.name.replace("example-", ""))
        return sorted(examples)

    def load_runbook(self, example_name: str) -> Runbook:
        """Load runbook for an example.

        Args:
            example_name: Name of the example.

        Returns:
            Parsed Runbook instance.

        Raises:
            FileNotFoundError: If runbook file doesn't exist.
        """
        if example_name in self._runbook_cache:
            return self._runbook_cache[example_name]

        example_dir = self.get_example_dir(example_name)
        runbook_path = example_dir / ".example.yaml"

        if not runbook_path.exists():
            raise FileNotFoundError(f"Runbook not found: {runbook_path}")

        runbook = Runbook.from_yaml_file(runbook_path)
        self._runbook_cache[example_name] = runbook
        return runbook

    def read_file(
        self, example_name: str, file_path: str, patches: list[str] | None = None
    ) -> str:
        """Read a source file, optionally applying patches.

        Args:
            example_name: Name of the example.
            file_path: Path to file relative to example directory.
            patches: List of patch paths to apply (relative to example directory).

        Returns:
            File content, possibly after applying patches.

        Raises:
            FileNotFoundError: If file or patch doesn't exist.
            RuntimeError: If patch application fails.
        """
        example_dir = self.get_example_dir(example_name)
        source_file = example_dir / file_path

        if not source_file.exists():
            raise FileNotFoundError(
                f"File not found: {source_file} (resolved from {file_path})"
            )

        # If no patches, just read the file directly
        if not patches:
            return source_file.read_text()

        # Apply patches in a temporary directory
        return self._apply_patches_and_read(example_dir, file_path, patches)

    def _apply_patches_and_read(
        self, example_dir: Path, file_path: str, patches: list[str]
    ) -> str:
        """Apply patches in a temporary copy and read the result.

        Args:
            example_dir: Example directory.
            file_path: Path to file relative to example directory.
            patches: List of patch paths to apply.

        Returns:
            File content after applying patches.

        Raises:
            FileNotFoundError: If patch doesn't exist.
            RuntimeError: If patch application fails.
        """
        # Create a temporary directory with a copy of the example
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_example = Path(tmpdir) / "example"

            # Copy the entire example directory to preserve structure
            # This is necessary because patches may reference multiple files
            import shutil

            shutil.copytree(example_dir, tmp_example, symlinks=False)

            # Apply each patch
            for patch_path in patches:
                patch_file = example_dir / patch_path
                if not patch_file.exists():
                    raise FileNotFoundError(f"Patch not found: {patch_file}")

                # Copy patch to temp dir (git apply expects relative path)
                tmp_patch = tmp_example / patch_path
                tmp_patch.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(patch_file, tmp_patch)

                # Apply the patch
                result = subprocess.run(
                    ["git", "apply", "--unsafe-paths", patch_path],
                    capture_output=True,
                    text=True,
                    cwd=tmp_example,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to apply patch {patch_path}: {result.stderr}"
                    )

            # Read the modified file
            modified_file = tmp_example / file_path
            if not modified_file.exists():
                raise FileNotFoundError(
                    f"File not found after applying patches: {modified_file}"
                )

            return modified_file.read_text()

    def read_patch(self, example_name: str, patch_path: str) -> str:
        """Read a patch file.

        Args:
            example_name: Name of the example.
            patch_path: Path to patch relative to example directory.

        Returns:
            Patch content.

        Raises:
            FileNotFoundError: If patch doesn't exist.
        """
        example_dir = self.get_example_dir(example_name)
        patch_file = example_dir / patch_path

        if not patch_file.exists():
            raise FileNotFoundError(f"Patch not found: {patch_file}")

        return patch_file.read_text()

    def get_scenarios(self, example_name: str) -> list[dict[str, Any]]:
        """Get scenarios from a runbook.

        Args:
            example_name: Name of the example.

        Returns:
            List of scenario dictionaries.
        """
        runbook = self.load_runbook(example_name)
        # Convert Pydantic models to dicts for backwards compatibility
        return [scenario.model_dump(mode="python") for scenario in runbook.scenarios]

    def get_scenario_patches(self, example_name: str) -> dict[str, list[str]]:
        """Extract patches applied in each scenario.

        Args:
            example_name: Name of the example.

        Returns:
            Dictionary mapping scenario name to list of patch paths applied up to that point.
        """
        runbook = self.load_runbook(example_name)
        patches_by_scenario: dict[str, list[str]] = {}
        accumulated_patches: list[str] = []

        for scenario in runbook.scenarios:
            scenario_patches = accumulated_patches.copy()

            # Look for apply_patch steps in this scenario
            from metaxy._testing import ApplyPatchStep

            for step in scenario.steps:
                if isinstance(step, ApplyPatchStep):
                    accumulated_patches.append(step.patch_path)
                    scenario_patches.append(step.patch_path)

            patches_by_scenario[scenario.name] = scenario_patches

        return patches_by_scenario
