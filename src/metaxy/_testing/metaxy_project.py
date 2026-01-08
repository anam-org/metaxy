import importlib
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_spec import (
    FeatureSpecWithIDColumns,
)
from metaxy.versioning.types import HashAlgorithm

DEFAULT_ID_COLUMNS = ["sample_uid"]


@contextmanager
def env_override(overrides: dict[str, str | None]):
    """Context manager to temporarily override environment variables.

    Args:
        overrides: Dict mapping env var names to values. None values will unset the var.

    Yields:
        None

    Example:
        ```py
        with env_override({"FOO": "bar", "BAZ": None}):
            # FOO is set to "bar", BAZ is unset
            do_something()
        # Original values restored
        ```
    """
    old_values = {}
    for key in overrides:
        old_values[key] = os.environ.get(key)

    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


__all__ = [
    "TempFeatureModule",
    "assert_all_results_equal",
    "HashAlgorithmCases",
    "MetaxyProject",
    "ExternalMetaxyProject",
    "TempMetaxyProject",  # Backward compatibility alias
    "DEFAULT_ID_COLUMNS",
    "env_override",
]


class TempFeatureModule:
    """Helper to create temporary Python modules with feature definitions.

    This allows features to be importable by historical graph reconstruction.
    The same import path (e.g., 'temp_features.Upstream') can be used across
    different feature versions by overwriting the module file.
    """

    def __init__(self, module_name: str = "temp_test_features"):
        self.temp_dir = tempfile.mkdtemp(prefix="metaxy_test_")
        self.module_name = module_name
        self.module_path = Path(self.temp_dir) / f"{module_name}.py"

        # Add to sys.path so module can be imported
        sys.path.insert(0, self.temp_dir)

    def write_features(self, feature_specs: dict[str, FeatureSpecWithIDColumns]):
        """Write feature classes to the module file.

        Args:
            feature_specs: Dict mapping class names to FeatureSpec objects
        """
        code_lines = [
            "# Auto-generated test feature module",
            "from metaxy import BaseFeature as Feature, FeatureSpec, FieldSpec, FieldKey, FeatureDep, FeatureKey, FieldDep, SpecialFieldDep",
            "from metaxy._testing.models import SampleFeatureSpec",
            "from metaxy.models.feature import FeatureGraph",
            "",
            "# Use a dedicated graph for this temp module",
            "_graph = FeatureGraph()",
            "",
        ]

        for class_name, spec in feature_specs.items():
            # Generate the spec definition
            spec_dict = spec.model_dump(mode="python")
            spec_class_name = spec.__class__.__name__
            spec_repr = self._generate_spec_repr(
                spec_dict, spec_class_name=spec_class_name
            )

            code_lines.extend(
                [
                    f"# Define {class_name} in the temp graph context",
                    "with _graph.use():",
                    f"    class {class_name}(",
                    "        Feature,",
                    f"        spec={spec_repr}",
                    "    ):",
                    "        pass",
                    "",
                ]
            )

        # Write the file
        self.module_path.write_text("\n".join(code_lines))

        # Reload module if it was already imported
        if self.module_name in sys.modules:
            importlib.reload(sys.modules[self.module_name])

    def _generate_spec_repr(
        self, spec_dict: dict[str, Any], spec_class_name: str = "FeatureSpec"
    ) -> str:
        """Generate FeatureSpec constructor call from dict.

        Args:
            spec_dict: Dictionary representation of the spec
            spec_class_name: Name of the spec class to use (e.g., "SampleFeatureSpec", "FeatureSpec")
        """
        # This is a simple representation - could be made more robust
        parts = []

        # key
        key = spec_dict["key"]
        parts.append(f"key=FeatureKey({key!r})")

        # deps
        deps = spec_dict.get("deps") or []
        deps_repr = [f"FeatureDep(feature=FeatureKey({d['feature']!r}))" for d in deps]
        parts.append(f"deps=[{', '.join(deps_repr)}]")

        # fields
        fields = spec_dict.get("fields", [])
        if fields:
            field_reprs = []
            for c in fields:
                c_parts = [
                    f"key=FieldKey({c['key']!r})",
                    f"code_version={c['code_version']!r}",
                ]

                # Handle deps
                deps_val = c.get("deps")
                if deps_val == "__METAXY_ALL_DEP__":
                    c_parts.append("deps=SpecialFieldDep.ALL")
                elif isinstance(deps_val, list) and deps_val:
                    # Field deps (list of FieldDep)
                    cdeps: list[str] = []
                    for cd in deps_val:
                        fields_val = cd.get("fields")
                        if fields_val == "__METAXY_ALL_DEP__":
                            cdeps.append(
                                f"FieldDep(feature=FeatureKey({cd['feature']!r}), fields=SpecialFieldDep.ALL)"
                            )
                        else:
                            # Build list of FieldKey objects
                            field_keys = [f"FieldKey({k!r})" for k in fields_val]
                            cdeps.append(
                                f"FieldDep(feature=FeatureKey({cd['feature']!r}), fields=[{', '.join(field_keys)}])"
                            )
                    c_parts.append(f"deps=[{', '.join(cdeps)}]")

                field_reprs.append(f"FieldSpec({', '.join(c_parts)})")

            parts.append(f"fields=[{', '.join(field_reprs)}]")

        # Note: id_columns is handled by the concrete spec class (SampleFeatureSpec has default)
        # so we don't need to include it here explicitly

        return f"{spec_class_name}({', '.join(parts)})"

    @property
    def graph(self) -> FeatureGraph:
        """Get the FeatureGraph from the temp module.

        Returns:
            The _graph instance from the imported module
        """
        # Import the module to get its _graph
        module = importlib.import_module(self.module_name)
        return module._graph

    def cleanup(self):
        """Remove temp directory and module from sys.path.

        NOTE: Don't call this until the test session is completely done,
        as historical graph loading may need to import from these modules.
        """
        if self.temp_dir in sys.path:
            sys.path.remove(self.temp_dir)

        # Remove from sys.modules
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]

        # Delete temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


def assert_all_results_equal(results: dict[str, Any], snapshot=None) -> None:
    """Compare all results from different store type combinations.

    Ensures all variants produce identical results, then optionally snapshots all results.

    Args:
        results: Dict mapping store_type to result data
        snapshot: Optional syrupy snapshot fixture to record all results

    Raises:
        AssertionError: If any variants produce different results
    """
    if not results:
        return

    # Get all result values as a list
    all_results = list(results.items())
    reference_key, reference_result = all_results[0]

    # Compare each result to the reference
    for key, result in all_results[1:]:
        assert result == reference_result, (
            f"{key} produced different results than {reference_key}:\n"
            f"Expected: {reference_result}\n"
            f"Got: {result}"
        )

    # Snapshot ALL results if snapshot provided
    # Sort by keys to ensure deterministic ordering across test runs
    if snapshot is not None:
        sorted_results = dict(sorted(results.items()))
        assert sorted_results == snapshot


class HashAlgorithmCases:
    """Test cases for different hash algorithms."""

    def case_xxhash64(self) -> HashAlgorithm:
        """xxHash64 algorithm."""
        return HashAlgorithm.XXHASH64

    def case_xxhash32(self) -> HashAlgorithm:
        """xxHash32 algorithm."""
        return HashAlgorithm.XXHASH32

    def case_wyhash(self) -> HashAlgorithm:
        """WyHash algorithm."""
        return HashAlgorithm.WYHASH

    def case_sha256(self) -> HashAlgorithm:
        """SHA256 algorithm."""
        return HashAlgorithm.SHA256

    def case_md5(self) -> HashAlgorithm:
        """MD5 algorithm."""
        return HashAlgorithm.MD5


class MetaxyProject:
    """Base class for Metaxy projects.

    Provides common functionality for running CLI commands with proper
    environment setup and accessing project configuration.
    """

    def __init__(self, project_dir: Path):
        """Initialize a Metaxy project.

        Args:
            project_dir: Path to project directory containing metaxy.toml
        """
        self.project_dir = Path(project_dir)

    def run_cli(
        self,
        args: list[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
        **kwargs,
    ):
        """Run CLI command with proper environment setup.

        Args:
            args: CLI command arguments (e.g., ["graph", "push"])
            check: If True (default), raises CalledProcessError on non-zero exit
            env: Optional dict of additional environment variables
            **kwargs: Additional arguments to pass to subprocess.run()

        Returns:
            subprocess.CompletedProcess: Result of the CLI command

        Raises:
            subprocess.CalledProcessError: If check=True and command fails

        Example:
            ```py
            result = project.run_cli(["graph", "history", "--limit", "5"])
            print(result.stdout)
            ```
        """
        # Start with current environment
        cmd_env = os.environ.copy()

        # Add project directory (and src/ subdirectory if it exists) to PYTHONPATH
        # so modules can be imported. These are prepended to take precedence.
        paths_to_add = [str(self.project_dir)]
        src_dir = self.project_dir / "src"
        if src_dir.is_dir():
            paths_to_add.insert(0, str(src_dir))

        pythonpath = os.pathsep.join(paths_to_add)
        if "PYTHONPATH" in cmd_env:
            pythonpath = f"{pythonpath}{os.pathsep}{cmd_env['PYTHONPATH']}"
        cmd_env["PYTHONPATH"] = pythonpath

        # Apply additional env overrides
        if env:
            cmd_env.update(env)

        # Run CLI command
        try:
            result = subprocess.run(
                [sys.executable, "-m", "metaxy.cli.app", *args],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                env=cmd_env,
                check=check,
                **kwargs,
            )
        except subprocess.CalledProcessError as e:
            # Re-raise with stderr output for better debugging
            error_msg = f"CLI command failed: {' '.join(str(a) for a in args)}\n"
            error_msg += f"Exit code: {e.returncode}\n"
            if e.stdout:
                error_msg += f"STDOUT:\n{e.stdout}\n"
            if e.stderr:
                error_msg += f"STDERR:\n{e.stderr}\n"
            raise RuntimeError(error_msg) from e

        return result

    @cached_property
    def config(self) -> MetaxyConfig:
        """Load configuration from project's metaxy.toml."""
        return MetaxyConfig.load(self.project_dir / "metaxy.toml")

    @cached_property
    def stores(self) -> dict[str, MetadataStore]:
        """Get all configured stores from project config."""
        return {k: self.config.get_store(k) for k in self.config.stores}


class ExternalMetaxyProject(MetaxyProject):
    """Helper for working with existing Metaxy projects.

    Use this class to interact with pre-existing projects like examples,
    running CLI commands and accessing their configuration.

    Example:
        ```py
        project = ExternalMetaxyProject(Path("examples/example-migration"))
        result = project.run_cli(["graph", "push"], env={"STAGE": "1"})
        assert result.returncode == 0
        print(project.package_name)  # "example_migration"
        ```
    """

    def __init__(self, project_dir: Path, require_config: bool = True):
        """Initialize an external Metaxy project.

        Args:
            project_dir: Path to existing project directory (may contain metaxy.toml)
            require_config: If True, requires metaxy.toml to exist (default: True)
        """
        super().__init__(project_dir)
        if require_config and not (self.project_dir / "metaxy.toml").exists():
            raise ValueError(
                f"No metaxy.toml found in {self.project_dir}. "
                "ExternalMetaxyProject requires an existing project configuration."
            )
        self._venv_path: Path | None = None
        self._venv_python: Path | None = None

    def setup_venv(self, venv_path: Path, install_metaxy_from: Path | None = None):
        """Create a virtual environment and install the project.

        Args:
            venv_path: Path where the venv should be created
            install_metaxy_from: Optional path to metaxy source to install (defaults to current)

        Returns:
            Path to the Python interpreter in the venv

        Example:
            ```py
            project = ExternalMetaxyProject(Path("tests/fixtures/test-project"))
            with tempfile.TemporaryDirectory() as tmpdir:
                project.setup_venv(Path(tmpdir) / "venv")
                result = project.run_in_venv("python", "-c", "import test_metaxy_project")
            ```
        """
        import os
        import subprocess

        # Create venv using uv
        subprocess.run(
            ["uv", "venv", str(venv_path), "--python", str(sys.executable)], check=True
        )

        # Install metaxy using the venv's pip directly
        if install_metaxy_from is None:
            # Default to metaxy package location (get the repo root)
            # metaxy.__file__ -> .../src/metaxy/__init__.py
            # .parent -> .../src/metaxy
            # .parent -> .../src
            # .parent -> repo root
            import metaxy

            install_metaxy_from = Path(metaxy.__file__).parent.parent.parent  # ty: ignore[invalid-argument-type]

        # Set VIRTUAL_ENV to activate the venv
        venv_env = os.environ.copy()
        venv_env["VIRTUAL_ENV"] = str(venv_path)
        # Remove PYTHONHOME if set (can interfere with venv)
        venv_env.pop("PYTHONHOME", None)

        # Use uv pip to install packages into the venv
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-e",
                str(install_metaxy_from),
            ],
            env=venv_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to install metaxy from {install_metaxy_from}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        # Install the project itself using uv pip
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-e",
                str(self.project_dir),
            ],
            env=venv_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to install project from {self.project_dir}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        self._venv_path = venv_path

    def run_in_venv(
        self, *args, check: bool = True, env: dict[str, str] | None = None, **kwargs
    ):
        """Run a command in the configured venv.

        Args:
            *args: Command and arguments (e.g., "python", "-c", "print('hello')")
            check: If True (default), raises CalledProcessError on non-zero exit
            env: Optional dict of additional environment variables
            **kwargs: Additional arguments to pass to subprocess.run()

        Returns:
            subprocess.CompletedProcess: Result of the command

        Raises:
            RuntimeError: If setup_venv() hasn't been called yet
            subprocess.CalledProcessError: If check=True and command fails

        Example:
            ```py
            project.setup_venv(Path("/tmp/venv"))
            result = project.run_in_venv("python", "-m", "my_module")
            ```
        """
        import subprocess

        if self._venv_path is None:
            raise RuntimeError("No venv configured. Call setup_venv() first.")

        # Start with current environment
        import os

        cmd_env = os.environ.copy()

        # Set VIRTUAL_ENV to activate the venv
        cmd_env["VIRTUAL_ENV"] = str(self._venv_path)
        # Remove PYTHONHOME if set (can interfere with venv)
        cmd_env.pop("PYTHONHOME", None)

        # Apply additional env overrides
        if env:
            cmd_env.update(env)

        # Run command with venv python
        result = subprocess.run(
            ["uv", "run", "--active", *args],
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            env=cmd_env,
            check=check,
            **kwargs,
        )

        return result

    @cached_property
    def package_name(self) -> str:
        """Get the Python package name from pyproject.toml.

        Converts the project name (e.g., "example-migration") to a valid
        Python module name (e.g., "example_migration") by replacing hyphens
        with underscores.

        Returns:
            The Python package/module name

        Raises:
            FileNotFoundError: If pyproject.toml doesn't exist
            ValueError: If pyproject.toml doesn't contain project.name
        """
        pyproject_path = self.project_dir / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError(
                f"No pyproject.toml found in {self.project_dir}. "
                "Cannot determine package name."
            )

        # Parse TOML to get project name
        import tomli

        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)

        project_name = pyproject.get("project", {}).get("name")
        if not project_name:
            raise ValueError(
                f"No project.name found in {pyproject_path}. "
                "Cannot determine package name."
            )

        # Convert project name to valid Python package name (replace hyphens with underscores)
        return project_name.replace("-", "_")

    def run_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run an arbitrary shell command in the project context.

        Handles:
        - PYTHONPATH setup to include project directory
        - PATH setup to prefer current Python executable's directory

        Args:
            command: Shell command to execute.
            env: Optional dict of additional environment variables.
            capture_output: Whether to capture stdout/stderr (default: True).
            timeout: Optional timeout in seconds.
            check: If True, raises CalledProcessError on non-zero exit (default: False).

        Returns:
            subprocess.CompletedProcess: Result of the command.

        Example:
            ```py
            result = project.run_command("python -m example.pipeline")
            print(result.stdout)
            ```
        """
        # Start with current environment
        cmd_env = os.environ.copy()

        # Add project directory (and src/ subdirectory if it exists) to PYTHONPATH
        # so modules can be imported. These are prepended to take precedence.
        paths_to_add = [str(self.project_dir)]
        src_dir = self.project_dir / "src"
        if src_dir.is_dir():
            paths_to_add.insert(0, str(src_dir))

        pythonpath = os.pathsep.join(paths_to_add)
        if "PYTHONPATH" in cmd_env:
            pythonpath = f"{pythonpath}{os.pathsep}{cmd_env['PYTHONPATH']}"
        cmd_env["PYTHONPATH"] = pythonpath

        # Prepend Python executable's directory to PATH so "python" resolves correctly
        python_dir = str(Path(sys.executable).parent)
        current_path = cmd_env.get("PATH", "")
        cmd_env["PATH"] = (
            f"{python_dir}{os.pathsep}{current_path}" if current_path else python_dir
        )

        # Apply additional env overrides
        if env:
            cmd_env.update(env)

        # Run command
        return subprocess.run(
            command,
            shell=True,
            cwd=str(self.project_dir),
            capture_output=capture_output,
            text=True,
            env=cmd_env,
            timeout=timeout,
            check=check,
        )

    def push_graph(
        self, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Push the current graph snapshot.

        Args:
            env: Optional dict of additional environment variables.

        Returns:
            subprocess.CompletedProcess: Result of the push command.
        """
        return self.run_cli(["graph", "push"], env=env)

    @property
    def graph(self) -> FeatureGraph:
        """Load the feature graph from entrypoints.

        Forces a fresh reload of all entrypoint modules to capture any source changes.

        Returns:
            FeatureGraph with all features loaded from entrypoints.
        """
        # Build paths to add to sys.path
        paths_to_add: list[str] = []
        src_dir = self.project_dir / "src"
        if src_dir.is_dir():
            paths_to_add.append(str(src_dir))
        paths_to_add.append(str(self.project_dir))

        # Track which paths we actually add (weren't already there)
        added_paths: list[str] = []
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
                added_paths.append(path)

        try:
            graph = FeatureGraph()
            with graph.use():
                for entrypoint in self.config.entrypoints:
                    # Force reload to pick up any source file changes
                    # First, invalidate any cached modules in the package hierarchy
                    self._invalidate_module_cache(entrypoint)
                    importlib.import_module(entrypoint)
            return graph
        finally:
            # Clean up sys.path
            for path in added_paths:
                if path in sys.path:
                    sys.path.remove(path)

    def _invalidate_module_cache(self, entrypoint: str) -> None:
        """Remove entrypoint and its parent/sibling modules from sys.modules.

        This ensures importlib.import_module() loads fresh code from disk
        rather than returning cached modules with stale code.
        """
        # Remove the entrypoint itself and any parent packages
        parts = entrypoint.split(".")
        modules_to_remove = []
        for i in range(len(parts)):
            module_name = ".".join(parts[: i + 1])
            if module_name in sys.modules:
                modules_to_remove.append(module_name)

        # Get the package root (first component)
        package_root = parts[0]

        # Remove ALL modules in the package hierarchy
        # This ensures sibling modules (like example_overview.features) are also invalidated
        for mod_name in list(sys.modules.keys()):
            if mod_name == package_root or mod_name.startswith(package_root + "."):
                if mod_name not in modules_to_remove:
                    modules_to_remove.append(mod_name)

        for mod_name in modules_to_remove:
            del sys.modules[mod_name]


class TempMetaxyProject(MetaxyProject):
    """Helper for creating temporary Metaxy projects.

    Provides a context manager API for dynamically creating feature modules
    and running CLI commands with proper entrypoint configuration.

    Example:
        ```py
        project = TempMetaxyProject(tmp_path)

        def features():
            from metaxy import BaseFeature as Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

            class MyFeature(Feature, spec=FeatureSpec(
                key=FeatureKey(["my_feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
            )):
                pass

        with project.with_features(features):
            result = project.run_cli("graph", "push")
            assert result.returncode == 0
        ```
    """

    def __init__(self, tmp_path: Path, config_content: str | None = None):
        """Initialize a temporary Metaxy project.

        Args:
            tmp_path: Temporary directory path (usually from pytest tmp_path fixture)
            config_content: Optional custom configuration content for metaxy.toml.
                If not provided, uses default DuckDB configuration.
        """
        super().__init__(tmp_path)
        self.project_dir.mkdir(exist_ok=True)
        self._feature_modules: list[str] = []
        self._module_counter = 0
        self._custom_config = config_content
        self._write_config()

    def _write_config(self):
        """Write metaxy.toml configuration file."""
        if self._custom_config is not None:
            # Use custom config content
            config_content = self._custom_config
        else:
            # Default DuckDB store configuration
            dev_db_path = self.project_dir / "metadata.duckdb"
            staging_db_path = self.project_dir / "metadata_staging.duckdb"
            config_content = f'''project = "test"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{dev_db_path}"

[stores.staging]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.staging.config]
database = "{staging_db_path}"
'''
        (self.project_dir / "metaxy.toml").write_text(config_content)

    def with_features(self, features_func, module_name: str | None = None):
        """Context manager that sets up features for the duration of the block.

        Extracts source code from features_func (skipping the function definition line),
        writes it to a Python module file, and tracks it for METAXY_ENTRYPOINTS__N
        environment variable configuration.

        Args:
            features_func: Function containing feature class definitions.
                All imports must be inside the function body.
            module_name: Optional module name. If not provided, generates
                "features_N" based on number of existing modules.

        Yields:
            str: The module name that was created

        Example:
            ```py
            def my_features():
                from metaxy import BaseFeature as Feature, FeatureSpec, FeatureKey

                class MyFeature(Feature, spec=...):
                    pass

            with project.with_features(my_features) as module:
                print(module)  # "features_0"
                result = project.run_cli(["graph", "push"])
            ```
        """

        @contextmanager
        def _context():
            # Generate module name if not provided
            nonlocal module_name
            if module_name is None:
                module_name = f"features_{self._module_counter}"
                self._module_counter += 1

            # Extract source code from function
            source = inspect.getsource(features_func)

            # Remove function definition line and dedent
            lines = source.split("\n")
            # Find the first line that's not a decorator or function def
            body_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("def ") and ":" in line:
                    body_start = i + 1
                    break

            body_lines = lines[body_start:]
            dedented = textwrap.dedent("\n".join(body_lines))

            # Write to file in project directory
            feature_file = self.project_dir / f"{module_name}.py"
            feature_file.write_text(dedented)

            # Track this module
            self._feature_modules.append(module_name)

            try:
                yield module_name
            finally:
                # Cleanup: remove from tracking (file stays for debugging)
                if module_name in self._feature_modules:
                    self._feature_modules.remove(module_name)

        return _context()

    def run_cli(
        self,
        args: list[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> subprocess.CompletedProcess[str]:
        """Run CLI command with current feature modules loaded.

        Automatically sets METAXY_ENTRYPOINT_0, METAXY_ENTRYPOINT_1, etc.
        based on active with_features() context managers.

        Args:
            args: CLI command arguments (e.g., ["graph", "push"])
            check: If True (default), raises CalledProcessError on non-zero exit
            env: Optional dict of additional environment variables
            **kwargs: Additional arguments to pass to subprocess.run()

        Returns:
            subprocess.CompletedProcess: Result of the CLI command

        Raises:
            subprocess.CalledProcessError: If check=True and command fails

        Example:
            ```py
            result = project.run_cli(["graph", "history", "--limit", "5"])
            print(result.stdout)
            ```
        """
        # Start with current environment
        cmd_env = os.environ.copy()

        # Add project directory to PYTHONPATH so modules can be imported
        pythonpath = str(self.project_dir)
        if "PYTHONPATH" in cmd_env:
            pythonpath = f"{pythonpath}{os.pathsep}{cmd_env['PYTHONPATH']}"
        cmd_env["PYTHONPATH"] = pythonpath

        # Set entrypoints for all tracked modules
        # Use METAXY_ENTRYPOINT_0, METAXY_ENTRYPOINT_1, etc. (single underscore for list indexing)
        for idx, module_name in enumerate(self._feature_modules):
            cmd_env[f"METAXY_ENTRYPOINT_{idx}"] = module_name

        # Apply additional env overrides
        if env:
            cmd_env.update(env)

        # Run CLI command
        try:
            result = subprocess.run(
                [sys.executable, "-m", "metaxy.cli.app", *args],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                env=cmd_env,
                check=check,
                **kwargs,
            )
        except subprocess.CalledProcessError as e:
            # Re-raise with stderr output for better debugging
            error_msg = f"CLI command failed: {' '.join(str(a) for a in args)}\n"
            error_msg += f"Exit code: {e.returncode}\n"
            if e.stdout:
                error_msg += f"STDOUT:\n{e.stdout}\n"
            if e.stderr:
                error_msg += f"STDERR:\n{e.stderr}\n"
            raise RuntimeError(error_msg) from e

        return result

    @property
    def entrypoints(self):
        return [f"METAXY_ENTRYPOINT_{idx}" for idx in range(len(self._feature_modules))]

    @property
    def graph(self) -> FeatureGraph:
        """Load features from the project's feature modules into a graph.

        Returns:
            FeatureGraph with all features from tracked modules loaded
        """
        import importlib
        import sys

        graph = FeatureGraph()

        # Ensure project dir is in sys.path
        project_dir_str = str(self.project_dir)
        was_in_path = project_dir_str in sys.path
        if not was_in_path:
            sys.path.insert(0, project_dir_str)

        try:
            with graph.use():
                # Import feature modules directly
                for module_name in self._feature_modules:
                    # Import or reload the module
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                    else:
                        importlib.import_module(module_name)
        finally:
            # Clean up sys.path if we added it
            if not was_in_path and project_dir_str in sys.path:
                sys.path.remove(project_dir_str)

        return graph
