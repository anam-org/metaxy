"""Graph description utilities for analyzing feature graphs."""

from typing import Any

from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey


def _get_feature_depth(
    graph: FeatureGraph,
    feature_key: FeatureKey,
    visited: set[FeatureKey] | None = None,
) -> int:
    """Calculate the depth of a feature in the dependency tree."""
    if visited is None:
        visited = set()

    if feature_key in visited:
        return 0  # Avoid cycles

    visited.add(feature_key)

    feature_cls = graph.features_by_key.get(feature_key)
    if feature_cls is None:
        return 1

    deps = feature_cls.spec().deps
    if not deps:
        return 1

    max_dep_depth = 0
    for dep in deps:
        dep_depth = _get_feature_depth(graph, dep.feature, visited.copy())
        max_dep_depth = max(max_dep_depth, dep_depth)

    return max_dep_depth + 1


def _find_root_features(
    filtered_features: dict[FeatureKey, Any],
) -> list[str]:
    """Find root features (no dependencies) in filtered set."""
    return [
        key.to_string() for key, cls in filtered_features.items() if not cls.spec().deps
    ]


def _is_leaf_feature(
    feature_key: FeatureKey,
    filtered_features: dict[FeatureKey, Any],
) -> bool:
    """Check if a feature is a leaf (no dependents in the filtered set)."""
    for other_key, other_cls in filtered_features.items():
        if other_key == feature_key:
            continue
        deps = other_cls.spec().deps
        if not deps:
            continue
        for dep in deps:
            if dep.feature == feature_key:
                return False
    return True


def _find_leaf_features(
    filtered_features: dict[FeatureKey, Any],
) -> list[str]:
    """Find leaf features (no dependents) in filtered set."""
    return [
        feature_key.to_string()
        for feature_key in filtered_features
        if _is_leaf_feature(feature_key, filtered_features)
    ]


def _calculate_project_breakdown(graph: FeatureGraph) -> dict[str, int]:
    """Calculate feature count per project."""
    projects: dict[str, int] = {}
    for cls in graph.features_by_key.values():
        project_name = cls.project
        projects[project_name] = projects.get(project_name, 0) + 1
    return projects


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
            key: cls
            for key, cls in graph.features_by_key.items()
            if cls.project == project
        }
    else:
        filtered_features = graph.features_by_key

    # Calculate metrics for filtered features
    max_depth = 0
    for feature_key in filtered_features:
        depth = _get_feature_depth(graph, feature_key)
        max_depth = max(max_depth, depth)

    # Build result
    result: dict[str, Any] = {
        "metaxy_snapshot_version": graph.snapshot_version,
        "total_features": len(graph.features_by_key),
        "graph_depth": max_depth,
        "root_features": sorted(_find_root_features(filtered_features)),
        "leaf_features": sorted(_find_leaf_features(filtered_features)),
        "projects": _calculate_project_breakdown(graph),
    }

    # Add filtered count if project filter was applied
    if project is not None:
        result["filtered_features"] = len(filtered_features)
        result["filter_project"] = project

    return result


def _build_dependency_tree(
    graph: FeatureGraph,
    key: FeatureKey,
    max_depth: int | None,
    current_depth: int = 0,
    visited: set[FeatureKey] | None = None,
) -> dict[str, Any]:
    """Build a dependency tree for a feature."""
    if visited is None:
        visited = set()

    if key in visited:
        return {"circular": True, "key": key.to_string()}

    if max_depth is not None and current_depth >= max_depth:
        return {"truncated": True, "key": key.to_string()}

    visited.add(key)

    cls = graph.features_by_key.get(key)
    if cls is None:
        return {"key": key.to_string(), "dependencies": []}

    spec_deps = cls.spec().deps
    if not spec_deps:
        return {"key": key.to_string(), "dependencies": []}

    deps = [
        _build_dependency_tree(
            graph, dep.feature, max_depth, current_depth + 1, visited.copy()
        )
        for dep in spec_deps
    ]

    return {
        "key": key.to_string(),
        "project": cls.project,
        "dependencies": deps,
    }


def _collect_deps_from_tree(node: dict[str, Any], all_deps: set[str]) -> None:
    """Collect all unique dependencies from a dependency tree."""
    if "dependencies" in node:
        for dep in node["dependencies"]:
            if "key" in dep:
                all_deps.add(dep["key"])
                _collect_deps_from_tree(dep, all_deps)


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
    feature_cls = graph.features_by_key.get(feature_key)
    if feature_cls is None:
        raise ValueError(f"Feature {feature_key.to_string()} not found in graph")

    # Get direct dependencies
    direct_deps = []
    deps = feature_cls.spec().deps
    if deps:
        direct_deps = [dep.feature.to_string() for dep in deps]

    result: dict[str, Any] = {
        "direct_dependencies": direct_deps,
    }

    if recursive:
        tree = _build_dependency_tree(graph, feature_key, max_depth)
        result["dependency_tree"] = tree

        all_deps: set[str] = set()
        _collect_deps_from_tree(tree, all_deps)
        result["all_dependencies"] = sorted(all_deps)

    return result


def _find_direct_dependents(
    graph: FeatureGraph,
    feature_key: FeatureKey,
) -> list[str]:
    """Find all features that directly depend on a given feature."""
    direct_dependents = []
    for other_key, other_cls in graph.features_by_key.items():
        deps = other_cls.spec().deps
        if not deps:
            continue
        for dep in deps:
            if dep.feature == feature_key:
                direct_dependents.append(other_key.to_string())
                break
    return direct_dependents


def _build_dependent_tree(
    graph: FeatureGraph,
    key: FeatureKey,
    max_depth: int | None,
    current_depth: int = 0,
    visited: set[FeatureKey] | None = None,
) -> dict[str, Any]:
    """Build a dependent tree for a feature."""
    if visited is None:
        visited = set()

    if key in visited:
        return {"circular": True, "key": key.to_string()}

    if max_depth is not None and current_depth >= max_depth:
        return {"truncated": True, "key": key.to_string()}

    visited.add(key)

    # Find features that depend on this one
    dependents = []
    for other_key, other_cls in graph.features_by_key.items():
        deps = other_cls.spec().deps
        if not deps:
            continue
        for dep in deps:
            if dep.feature == key:
                dep_tree = _build_dependent_tree(
                    graph, other_key, max_depth, current_depth + 1, visited.copy()
                )
                dependents.append(dep_tree)
                break

    cls = graph.features_by_key.get(key)
    return {
        "key": key.to_string(),
        "project": cls.project if cls else None,
        "dependents": dependents,
    }


def _collect_dependents_from_tree(
    node: dict[str, Any], all_dependents: set[str]
) -> None:
    """Collect all unique dependents from a dependent tree."""
    if "dependents" in node:
        for dep in node["dependents"]:
            if "key" in dep:
                all_dependents.add(dep["key"])
                _collect_dependents_from_tree(dep, all_dependents)


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
    result: dict[str, Any] = {
        "direct_dependents": sorted(_find_direct_dependents(graph, feature_key)),
    }

    if recursive:
        tree = _build_dependent_tree(graph, feature_key, max_depth)
        result["dependent_tree"] = tree

        all_dependents: set[str] = set()
        _collect_dependents_from_tree(tree, all_dependents)
        result["all_dependents"] = sorted(all_dependents)

    return result
