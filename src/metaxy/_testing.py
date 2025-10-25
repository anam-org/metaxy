import importlib
import sys
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any

from metaxy import (
    FeatureSpec,
)
from metaxy.config import MetaxyConfig
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph


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

    def write_features(self, feature_specs: dict[str, FeatureSpec]):
        """Write feature classes to the module file.

        Args:
            feature_specs: Dict mapping class names to FeatureSpec objects
        """
        code_lines = [
            "# Auto-generated test feature module",
            "from metaxy import Feature, FeatureSpec, FieldSpec, FieldKey, FeatureDep, FeatureKey, FieldDep, SpecialFieldDep",
            "from metaxy.models.feature import FeatureGraph",
            "",
            "# Use a dedicated graph for this temp module",
            "_graph = FeatureGraph()",
            "",
        ]

        for class_name, spec in feature_specs.items():
            # Generate the spec definition
            spec_dict = spec.model_dump(mode="python")
            spec_repr = self._generate_spec_repr(spec_dict)

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

    def _generate_spec_repr(self, spec_dict: dict[str, Any]) -> str:
        """Generate FeatureSpec constructor call from dict."""
        # This is a simple representation - could be made more robust
        parts = []

        # key
        key = spec_dict["key"]
        parts.append(f"key=FeatureKey({key!r})")

        # deps
        deps = spec_dict.get("deps")
        if deps is None:
            parts.append("deps=None")
        else:
            deps_repr = [f"FeatureDep(key=FeatureKey({d['key']!r}))" for d in deps]
            parts.append(f"deps=[{', '.join(deps_repr)}]")

        # fields
        fields = spec_dict.get("fields", [])
        if fields:
            field_reprs = []
            for c in fields:
                c_parts = [
                    f"key=FieldKey({c['key']!r})",
                    f"code_version={c['code_version']}",
                ]

                # Handle deps
                deps_val = c.get("deps")
                if deps_val == "__METAXY_ALL_DEP__":
                    c_parts.append("deps=SpecialFieldDep.ALL")
                elif isinstance(deps_val, list) and deps_val:
                    # Field deps (list of FieldDep)
                    cdeps: list[str] = []  # type: ignore[misc]
                    for cd in deps_val:
                        fields_val = cd.get("fields")
                        if fields_val == "__METAXY_ALL_DEP__":
                            cdeps.append(  # type: ignore[arg-type]
                                f"FieldDep(feature_key=FeatureKey({cd['feature_key']!r}), fields=SpecialFieldDep.ALL)"
                            )
                        else:
                            # Build list of FieldKey objects
                            field_keys = [f"FieldKey({k!r})" for k in fields_val]
                            cdeps.append(
                                f"FieldDep(feature_key=FeatureKey({cd['feature_key']!r}), fields=[{', '.join(field_keys)}])"
                            )
                    c_parts.append(f"deps=[{', '.join(cdeps)}]")

                field_reprs.append(f"FieldSpec({', '.join(c_parts)})")  # type: ignore[arg-type]

            parts.append(f"fields=[{', '.join(field_reprs)}]")

        return f"FeatureSpec({', '.join(parts)})"

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
    if snapshot is not None:
        assert results == snapshot


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


class TempMetaxyProject:
    """Helper for creating temporary Metaxy projects.

    Provides a context manager API for dynamically creating feature modules
    and running CLI commands with proper entrypoint configuration.

    Example:
        >>> project = TempMetaxyProject(tmp_path)
        >>>
        >>> def features():
        ...     from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey
        ...
        ...     class MyFeature(Feature, spec=FeatureSpec(
        ...         key=FeatureKey(["my_feature"]),
        ...         deps=None,
        ...         fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
        ...     )):
        ...         pass
        >>>
        >>> with project.with_features(features):
        ...     result = project.run_cli("graph", "push")
        ...     assert result.returncode == 0
    """

    def __init__(self, tmp_path: Path):
        """Initialize a temporary Metaxy project.

        Args:
            tmp_path: Temporary directory path (usually from pytest tmp_path fixture)
        """
        self.project_dir = tmp_path
        self.project_dir.mkdir(exist_ok=True)
        self._feature_modules: list[str] = []
        self._write_config()

    def _write_config(self):
        """Write basic metaxy.toml with DuckDB store configuration."""
        dev_db_path = self.project_dir / "metadata.duckdb"
        staging_db_path = self.project_dir / "metadata_staging.duckdb"
        config_content = f'''store = "dev"

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
            >>> def my_features():
            ...     from metaxy import Feature, FeatureSpec, FeatureKey
            ...
            ...     class MyFeature(Feature, spec=...):
            ...         pass
            >>>
            >>> with project.with_features(my_features) as module:
            ...     print(module)  # "features_0"
            ...     result = project.run_cli("graph", "push")
        """
        import inspect
        import textwrap
        from contextlib import contextmanager

        @contextmanager
        def _context():
            # Generate module name if not provided
            nonlocal module_name
            if module_name is None:
                module_name = f"features_{len(self._feature_modules)}"

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
        self, *args, check: bool = True, env: dict[str, str] | None = None, **kwargs
    ):
        """Run CLI command with current feature modules loaded.

        Automatically sets METAXY_ENTRYPOINT_0, METAXY_ENTRYPOINT_1, etc.
        based on active with_features() context managers.

        Args:
            *args: CLI command arguments (e.g., "graph", "push")
            check: If True (default), raises CalledProcessError on non-zero exit
            env: Optional dict of additional environment variables
            **kwargs: Additional arguments to pass to subprocess.run()

        Returns:
            subprocess.CompletedProcess: Result of the CLI command

        Raises:
            subprocess.CalledProcessError: If check=True and command fails

        Example:
            >>> result = project.run_cli("graph", "history", "--limit", "5")
            >>> print(result.stdout)
        """
        import os
        import subprocess

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
        result = subprocess.run(
            [sys.executable, "-m", "metaxy.cli.app", *args],
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            env=cmd_env,
            check=check,
            **kwargs,
        )

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

    @cached_property
    def config(self) -> MetaxyConfig:
        return MetaxyConfig.load(self.project_dir / "metaxy.toml")

    @cached_property
    def stores(self) -> dict[str, MetadataStore]:
        return {k: self.config.get_store(k) for k in self.config.stores}
