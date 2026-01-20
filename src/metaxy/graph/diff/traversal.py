"""Graph traversal utilities."""

from collections import deque

from metaxy.graph.diff.models import GraphData, GraphNode
from metaxy.models.types import FeatureKey


class GraphWalker:
    """Traverses and filters graph data structures.

    Provides various traversal strategies:
    - Topological sort (dependencies first)
    - BFS from starting node
    - Subgraph extraction with up/down filtering
    """

    def __init__(self, graph_data: GraphData):
        """Initialize walker with graph data.

        Args:
            graph_data: Graph structure to traverse
        """
        self.graph_data = graph_data

    def topological_sort(self, nodes_to_include: set[str] | None = None) -> list[GraphNode]:
        """Get nodes in topological order (dependencies first).

        Uses stable alphabetical ordering when multiple nodes are at the same level.
        This ensures deterministic output for diff comparisons.

        Args:
            nodes_to_include: Optional set of feature key strings to include.
                            If None, includes all nodes.

        Returns:
            List of nodes sorted so dependencies appear before dependents
        """
        if nodes_to_include is None:
            nodes_to_include = set(self.graph_data.nodes.keys())

        visited = set()
        result = []

        def visit(key_str: str):
            if key_str in visited or key_str not in nodes_to_include:
                return
            visited.add(key_str)

            node = self.graph_data.nodes[key_str]

            # Visit dependencies first, in sorted order for determinism
            sorted_deps = sorted(
                (dep_key.to_string() for dep_key in node.dependencies),
                key=str.lower,  # Case-insensitive sort
            )
            for dep_key_str in sorted_deps:
                if dep_key_str in nodes_to_include:
                    visit(dep_key_str)

            result.append(node)

        # Visit all nodes in sorted order for deterministic traversal
        for key_str in sorted(nodes_to_include, key=str.lower):
            visit(key_str)

        return result

    def bfs_from(self, start_key: FeatureKey, max_depth: int | None = None) -> list[GraphNode]:
        """BFS traversal starting from a node.

        Args:
            start_key: Feature key to start from
            max_depth: Maximum depth to traverse (None = unlimited)

        Returns:
            List of nodes in BFS order
        """
        start_key_str = start_key.to_string()
        if start_key_str not in self.graph_data.nodes:
            return []

        visited = set()
        result = []
        queue = deque([(start_key_str, 0)])  # (key_str, depth)

        while queue:
            key_str, depth = queue.popleft()

            if key_str in visited:
                continue

            if max_depth is not None and depth > max_depth:
                continue

            visited.add(key_str)
            node = self.graph_data.nodes[key_str]
            result.append(node)

            # Add dependencies
            for dep_key in node.dependencies:
                dep_key_str = dep_key.to_string()
                if dep_key_str not in visited and dep_key_str in self.graph_data.nodes:
                    queue.append((dep_key_str, depth + 1))

        return result

    def extract_subgraph(
        self,
        focus_key: FeatureKey,
        up: int | None = None,
        down: int | None = None,
    ) -> GraphData:
        """Extract a subgraph centered on a focus node.

        Args:
            focus_key: Feature to focus on
            up: Number of upstream levels (dependencies) to include.
                None = all, 0 = none
            down: Number of downstream levels (dependents) to include.
                None = all, 0 = none

        Returns:
            New GraphData with filtered nodes and edges

        Raises:
            ValueError: If focus_key not found in graph
        """
        focus_key_str = focus_key.to_string()
        if focus_key_str not in self.graph_data.nodes:
            raise ValueError(f"Feature '{focus_key_str}' not found in graph")

        # Start with focus node
        nodes_to_include = {focus_key_str}

        # Add upstream (dependencies)
        if up != 0:
            max_up = None if up is None or up < 0 else up
            upstream = self._get_upstream(focus_key_str, max_levels=max_up)
            nodes_to_include.update(upstream)

        # Add downstream (dependents)
        if down != 0:
            max_down = None if down is None or down < 0 else down
            downstream = self._get_downstream(focus_key_str, max_levels=max_down)
            nodes_to_include.update(downstream)

        # Filter nodes and edges
        filtered_nodes = {k: v for k, v in self.graph_data.nodes.items() if k in nodes_to_include}

        filtered_edges = [
            edge
            for edge in self.graph_data.edges
            if edge.from_key.to_string() in nodes_to_include and edge.to_key.to_string() in nodes_to_include
        ]

        return GraphData(
            nodes=filtered_nodes,
            edges=filtered_edges,
            snapshot_version=self.graph_data.snapshot_version,
            old_snapshot_version=self.graph_data.old_snapshot_version,
        )

    def _get_upstream(self, start_key_str: str, max_levels: int | None = None) -> set[str]:
        """Get upstream features (dependencies) recursively.

        Args:
            start_key_str: Feature key string to start from
            max_levels: Maximum levels to traverse (None = unlimited)

        Returns:
            Set of upstream feature key strings
        """
        upstream = set()

        def visit(key_str: str, level: int):
            if key_str not in self.graph_data.nodes:
                return

            node = self.graph_data.nodes[key_str]

            for dep_key in node.dependencies:
                dep_key_str = dep_key.to_string()
                if dep_key_str not in upstream and dep_key_str in self.graph_data.nodes:
                    upstream.add(dep_key_str)
                    # Only recurse if we haven't reached max level
                    if max_levels is None or level + 1 < max_levels:
                        visit(dep_key_str, level + 1)

        visit(start_key_str, 0)
        return upstream

    def _get_downstream(self, start_key_str: str, max_levels: int | None = None) -> set[str]:
        """Get downstream features (dependents) recursively.

        Args:
            start_key_str: Feature key string to start from
            max_levels: Maximum levels to traverse (None = unlimited)

        Returns:
            Set of downstream feature key strings
        """
        # Build reverse dependency map (feature -> dependents)
        dependents_map: dict[str, list[str]] = {}
        for node in self.graph_data.nodes.values():
            for dep_key in node.dependencies:
                dep_key_str = dep_key.to_string()
                if dep_key_str not in dependents_map:
                    dependents_map[dep_key_str] = []
                dependents_map[dep_key_str].append(node.key.to_string())

        downstream = set()

        def visit(key_str: str, level: int):
            if key_str not in dependents_map:
                return

            for dependent_key_str in dependents_map[key_str]:
                if dependent_key_str not in downstream:
                    downstream.add(dependent_key_str)
                    # Only recurse if we haven't reached max level
                    if max_levels is None or level + 1 < max_levels:
                        visit(dependent_key_str, level + 1)

        visit(start_key_str, 0)
        return downstream

    def get_root_nodes(self) -> list[GraphNode]:
        """Get all root nodes (nodes with no dependencies).

        Returns:
            List of root nodes
        """
        return [node for node in self.graph_data.nodes.values() if not node.dependencies]
