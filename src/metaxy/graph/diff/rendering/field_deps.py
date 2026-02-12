"""Field-level dependency rendering as Mermaid subgraph diagrams.

Renders feature graphs with subgraphs per feature and field-to-field edges,
using snapshot data to reconstruct FeatureSpec → FeaturePlan → field_dependencies.
"""

from typing import Any

from metaxy.graph.utils import sanitize_mermaid_id
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.plan import FeaturePlan

_SUBGRAPH_COLOR = "#4C78A8"


def resolve_field_dependencies_from_snapshot(
    graph_snapshot: dict[str, Any],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Resolve field-level lineage from a graph snapshot.

    Parses each feature's serialized feature_spec into a FeatureSpec,
    builds a FeaturePlan for downstream features, and uses
    FeaturePlan.field_dependencies for full resolution.

    Returns:
        Mapping of feature_key_str → {field_key_str → {upstream_feature_key_str → [upstream_field_key_strs]}}
    """
    specs_by_key: dict[str, FeatureSpec] = {}
    for feature_key_str, feature_data in graph_snapshot.items():
        raw_spec = feature_data.get("feature_spec")
        if raw_spec is None:
            continue
        specs_by_key[feature_key_str] = FeatureSpec.model_validate(raw_spec)

    result: dict[str, dict[str, dict[str, list[str]]]] = {}

    for feature_key_str, spec in specs_by_key.items():
        if not spec.deps:
            result[feature_key_str] = {}
            continue

        parent_specs = [
            specs_by_key[dep.feature.to_string()] for dep in spec.deps if dep.feature.to_string() in specs_by_key
        ]

        plan = FeaturePlan(
            feature=spec,
            deps=parent_specs if parent_specs else None,
            feature_deps=spec.deps if parent_specs else None,
        )

        field_deps_raw = plan.field_dependencies
        serialized: dict[str, dict[str, list[str]]] = {}
        for field_key, upstream_map in field_deps_raw.items():
            upstream_serialized: dict[str, list[str]] = {}
            for upstream_feature_key, upstream_field_keys in upstream_map.items():
                upstream_serialized[upstream_feature_key.to_string()] = [fk.to_string() for fk in upstream_field_keys]
            serialized[field_key.to_string()] = upstream_serialized
        result[feature_key_str] = serialized

    return result


def _topological_sort_keys(
    graph_snapshot: dict[str, Any],
    feature_filter: list[str] | None,
) -> list[str]:
    """Topological sort of feature keys (dependencies first).

    Only includes features present in feature_filter (if provided).
    """
    all_keys = set(graph_snapshot.keys())
    if feature_filter is not None:
        all_keys = all_keys & set(feature_filter)

    deps_map: dict[str, list[str]] = {}
    for key in all_keys:
        feature_data = graph_snapshot[key]
        raw_spec = feature_data.get("feature_spec", {})
        raw_deps = raw_spec.get("deps", [])
        deps_map[key] = [
            d["feature"] if isinstance(d["feature"], str) else "/".join(d["feature"])
            for d in raw_deps
            if (d["feature"] if isinstance(d["feature"], str) else "/".join(d["feature"])) in all_keys
        ]

    visited: set[str] = set()
    result: list[str] = []

    def visit(key: str) -> None:
        if key in visited or key not in all_keys:
            return
        visited.add(key)
        for dep_key in sorted(deps_map.get(key, []), key=str.lower):
            visit(dep_key)
        result.append(key)

    for key in sorted(all_keys, key=str.lower):
        visit(key)

    return result


def render_field_deps_mermaid(
    graph_snapshot: dict[str, Any],
    features: list[str] | None = None,
    direction: str = "TB",
) -> str:
    """Render field-level dependency graph as Mermaid markup with subgraphs.

    Each feature becomes a subgraph containing its field nodes.
    Edges connect upstream field nodes to downstream field nodes.

    Args:
        graph_snapshot: Snapshot dict {feature_key_str → {metaxy_feature_version, fields, feature_spec}}.
        features: Optional list of feature key strings to include. None means all.
        direction: Mermaid flowchart direction (TB, LR, etc.).

    Returns:
        Mermaid flowchart string.
    """
    sorted_keys = _topological_sort_keys(graph_snapshot, features)
    field_deps = resolve_field_dependencies_from_snapshot(graph_snapshot)

    lines: list[str] = [f"flowchart {direction}"]

    # Build subgraphs
    for idx, feature_key_str in enumerate(sorted_keys):
        feature_data = graph_snapshot[feature_key_str]
        fields_dict = feature_data.get("fields", {})
        subgraph_id = sanitize_mermaid_id(feature_key_str)

        lines.append(f'    subgraph {subgraph_id}["{feature_key_str}"]')
        for field_name in sorted(fields_dict.keys()):
            node_id = _field_node_id(feature_key_str, field_name)
            lines.append(f'        {node_id}["{field_name}"]')
        lines.append("    end")

    # Build cross-subgraph field-to-field edges
    for feature_key_str in sorted_keys:
        feature_field_deps = field_deps.get(feature_key_str, {})
        for field_name in sorted(feature_field_deps.keys()):
            upstream_map = feature_field_deps[field_name]
            for upstream_feature in sorted(upstream_map.keys()):
                if features is not None and upstream_feature not in features:
                    continue
                for upstream_field in sorted(upstream_map[upstream_feature]):
                    src = _field_node_id(upstream_feature, upstream_field)
                    dst = _field_node_id(feature_key_str, field_name)
                    lines.append(f"    {src} --> {dst}")

    # Add subgraph border styling
    for feature_key_str in sorted_keys:
        subgraph_id = sanitize_mermaid_id(feature_key_str)
        lines.append(f"    style {subgraph_id} stroke:{_SUBGRAPH_COLOR},stroke-width:2px")

    return "\n".join(lines)


def _field_node_id(feature_key_str: str, field_name: str) -> str:
    """Generate a unique Mermaid node ID for a field within a feature."""
    return sanitize_mermaid_id(f"{feature_key_str}_{field_name}")
