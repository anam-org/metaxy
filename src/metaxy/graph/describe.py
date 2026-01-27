"""Graph description utilities for analyzing feature graphs."""

from typing import Any

from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey


def describe_graph(
    graph: FeatureGraph,
    project: str | None = None,
) -> dict[str, Any]:
    """Generate comprehensive description of a feature graph.

    Analyzes the graph structure and provides metrics including:

    - Feature count (optionally filtered by project)

    - Graph depth (longest dependency chain)

    - Root features (features with no dependencies)

    - Leaf features (features with no dependents)

    - Feature breakdown by project (if multi-project)

    Args:
        graph: The FeatureGraph to analyze
        project: Optional project filter to analyze only features from a specific project

    Returns:
        Dictionary containing graph metrics and analysis:
        {
            "metaxy_snapshot_version": str,
            "total_features": int,
            "filtered_features": int,  # If project filter applied
            "graph_depth": int,
            "root_features": list[str],
            "leaf_features": list[str],
            "projects": dict[str, int],  # Project -> feature count
        }

    Example:
        ```py
        graph = FeatureGraph.get_active()
        info = describe_graph(graph, project="my_project")
        print(f"Graph has {info['filtered_features']} features from my_project")
        ```
    """
    # Get all features, optionally filtered by project
    if project is not None:
        filtered_features = {
            key: defn for key, defn in graph.feature_definitions_by_key.items() if defn.project == project
        }
    else:
        filtered_features = graph.feature_definitions_by_key

    # Calculate graph depth (longest dependency chain)
    def get_feature_depth(
        feature_key: FeatureKey,
        visited: set[FeatureKey] | None = None,
    ) -> int:
        """Calculate the depth of a feature in the dependency tree."""
        if visited is None:
            visited = set()

        if feature_key in visited:
            return 0  # Avoid cycles

        visited.add(feature_key)

        defn = graph.feature_definitions_by_key.get(feature_key)
        if defn is None:
            return 1

        deps = defn.spec.deps
        if not deps:
            return 1

        max_dep_depth = 0
        for dep in deps:
            dep_depth = get_feature_depth(dep.feature, visited.copy())
            max_dep_depth = max(max_dep_depth, dep_depth)

        return max_dep_depth + 1

    # Calculate metrics for filtered features
    max_depth = 0
    for feature_key in filtered_features:
        depth = get_feature_depth(feature_key)
        max_depth = max(max_depth, depth)

    # Find root features (no dependencies) in filtered set
    root_features = [key.to_string() for key, defn in filtered_features.items() if not defn.spec.deps]

    # Find leaf features (no dependents) in filtered set
    # Build a set of all dependency keys for efficient lookup
    all_dep_keys = {dep.feature for defn in filtered_features.values() if defn.spec.deps for dep in defn.spec.deps}
    leaf_features = [key.to_string() for key in filtered_features if key not in all_dep_keys]

    # Calculate project breakdown
    projects: dict[str, int] = {}
    for defn in graph.feature_definitions_by_key.values():
        project_name = defn.project
        projects[project_name] = projects.get(project_name, 0) + 1

    # Build result
    result: dict[str, Any] = {
        "metaxy_snapshot_version": graph.snapshot_version,
        "total_features": len(graph.feature_definitions_by_key),
        "graph_depth": max_depth,
        "root_features": sorted(root_features),
        "leaf_features": sorted(leaf_features),
        "projects": projects,
    }

    # Add filtered count if project filter was applied
    if project is not None:
        result["filtered_features"] = len(filtered_features)
        result["filter_project"] = project

    return result


def get_feature_dependencies(
    graph: FeatureGraph,
    feature_key: FeatureKey,
    recursive: bool = False,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Get dependencies of a specific feature.

    Args:
        graph: The FeatureGraph to analyze
        feature_key: The feature to analyze
        recursive: If True, recursively get all upstream dependencies
        max_depth: Maximum recursion depth (None for unlimited)

    Returns:
        Dictionary containing dependency information:
        {
            "direct_dependencies": list[str],
            "all_dependencies": list[str],  # If recursive=True
            "dependency_tree": dict,  # Nested structure if recursive=True
        }
    """
    defn = graph.feature_definitions_by_key.get(feature_key)
    if defn is None:
        raise ValueError(f"Feature {feature_key.to_string()} not found in graph")

    # Get direct dependencies
    direct_deps = []
    deps = defn.spec.deps
    if deps:
        direct_deps = [dep.feature.to_string() for dep in deps]

    result: dict[str, Any] = {
        "direct_dependencies": direct_deps,
    }

    if recursive:
        # Build full dependency tree
        def build_dep_tree(
            key: FeatureKey,
            current_depth: int = 0,
            visited: set[FeatureKey] | None = None,
        ) -> dict[str, Any]:
            if visited is None:
                visited = set()

            if key in visited:
                return {"circular": True, "key": key.to_string()}

            if max_depth is not None and current_depth >= max_depth:
                return {"truncated": True, "key": key.to_string()}

            visited.add(key)

            feature_defn = graph.feature_definitions_by_key.get(key)
            if feature_defn is None:
                return {"key": key.to_string(), "dependencies": []}

            spec_deps = feature_defn.spec.deps
            if not spec_deps:
                return {"key": key.to_string(), "dependencies": []}

            deps = []
            for dep in spec_deps:
                dep_tree = build_dep_tree(
                    dep.feature,
                    current_depth + 1,
                    visited.copy(),
                )
                deps.append(dep_tree)

            return {
                "key": key.to_string(),
                "project": feature_defn.project,
                "dependencies": deps,
            }

        tree = build_dep_tree(feature_key)
        result["dependency_tree"] = tree

        # Collect all unique dependencies
        all_deps = set()

        def collect_deps(node: dict[str, Any]) -> None:
            if "dependencies" in node:
                for dep in node["dependencies"]:
                    if "key" in dep:
                        all_deps.add(dep["key"])
                        collect_deps(dep)

        collect_deps(tree)
        result["all_dependencies"] = sorted(all_deps)

    return result


def get_feature_dependents(
    graph: FeatureGraph,
    feature_key: FeatureKey,
    recursive: bool = False,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Get features that depend on a specific feature (downstream).

    Args:
        graph: The FeatureGraph to analyze
        feature_key: The feature to analyze
        recursive: If True, recursively get all downstream dependents
        max_depth: Maximum recursion depth (None for unlimited)

    Returns:
        Dictionary containing dependent information:
        {
            "direct_dependents": list[str],
            "all_dependents": list[str],  # If recursive=True
            "dependent_tree": dict,  # Nested structure if recursive=True
        }
    """
    # Find direct dependents
    direct_dependents = []
    for other_key, other_defn in graph.feature_definitions_by_key.items():
        deps = other_defn.spec.deps
        if deps:
            for dep in deps:
                if dep.feature == feature_key:
                    direct_dependents.append(other_key.to_string())
                    break

    result: dict[str, Any] = {
        "direct_dependents": sorted(direct_dependents),
    }

    if recursive:
        # Build full dependent tree
        def build_dependent_tree(
            key: FeatureKey,
            current_depth: int = 0,
            visited: set[FeatureKey] | None = None,
        ) -> dict[str, Any]:
            if visited is None:
                visited = set()

            if key in visited:
                return {"circular": True, "key": key.to_string()}

            if max_depth is not None and current_depth >= max_depth:
                return {"truncated": True, "key": key.to_string()}

            visited.add(key)

            # Find features that depend on this one
            dependents = []
            for other_key, other_defn in graph.feature_definitions_by_key.items():
                deps = other_defn.spec.deps
                if deps:
                    for dep in deps:
                        if dep.feature == key:
                            dep_tree = build_dependent_tree(
                                other_key,
                                current_depth + 1,
                                visited.copy(),
                            )
                            dependents.append(dep_tree)
                            break

            defn = graph.feature_definitions_by_key.get(key)
            return {
                "key": key.to_string(),
                "project": defn.project if defn else None,
                "dependents": dependents,
            }

        tree = build_dependent_tree(feature_key)
        result["dependent_tree"] = tree

        # Collect all unique dependents
        all_dependents = set()

        def collect_dependents(node: dict[str, Any]) -> None:
            if "dependents" in node:
                for dep in node["dependents"]:
                    if "key" in dep:
                        all_dependents.add(dep["key"])
                        collect_dependents(dep)

        collect_dependents(tree)
        result["all_dependents"] = sorted(all_dependents)

    return result
