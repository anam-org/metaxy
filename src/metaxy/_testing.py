import importlib
import sys
import tempfile
from pathlib import Path

from metaxy import (
    Feature,
    FeatureSpec,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
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

    def _generate_spec_repr(self, spec_dict: dict) -> str:
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

    def get_graph(self) -> FeatureGraph:
        """Get the graph from the temp module.

        Creates a new graph and re-registers the features from the module.
        This ensures we get fresh features after module reloading.
        """
        # Force reload to get latest class definitions
        if self.module_name in sys.modules:
            module = importlib.reload(sys.modules[self.module_name])
        else:
            module = importlib.import_module(self.module_name)

        # Create fresh graph and register features from module
        fresh_graph = FeatureGraph()

        # Find and register all Feature subclasses in the module
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Feature)
                and obj is not Feature
                and hasattr(obj, "spec")
            ):
                fresh_graph.add_feature(obj)

        return fresh_graph

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


def assert_all_results_equal(results: dict, snapshot=None) -> None:
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
