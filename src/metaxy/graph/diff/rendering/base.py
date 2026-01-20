"""Base classes and configuration for graph rendering."""

from dataclasses import dataclass, field

from metaxy.graph.diff.models import GraphData, NodeStatus
from metaxy.graph.diff.rendering.theme import Theme
from metaxy.graph.diff.traversal import GraphWalker
from metaxy.graph.utils import format_feature_key, format_field_key, format_hash
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey, FieldKey


@dataclass(kw_only=True)
class RenderConfig:
    """Configuration for graph rendering.

    Controls what information is displayed and how it's formatted.
    """

    # What to show
    show_fields: bool = field(
        default=True,
        metadata={"help": "Show field-level details within features"},
    )

    show_feature_versions: bool = field(
        default=True,
        metadata={"help": "Show feature version hashes"},
    )

    show_field_versions: bool = field(
        default=True,
        metadata={"help": "Show field version hashes (requires --show-fields)"},
    )

    show_code_versions: bool = field(
        default=False,
        metadata={"help": "Show feature and field code versions"},
    )

    show_snapshot_version: bool = field(
        default=True,
        metadata={"help": "Show graph snapshot version in output"},
    )

    # Display options
    hash_length: int = field(
        default=8,
        metadata={"help": "Number of characters to show for version hashes (0 for full)"},
    )

    direction: str = field(
        default="TB",
        metadata={"help": "Graph layout direction: TB (top-bottom) or LR (left-right)"},
    )

    # Filtering options
    feature: str | None = field(
        default=None,
        metadata={"help": "Focus on a specific feature (e.g., 'video/files' or 'video__files')"},
    )

    up: int | None = field(
        default=None,
        metadata={"help": "Number of dependency levels to render upstream (default: all)"},
    )

    down: int | None = field(
        default=None,
        metadata={"help": "Number of dependency levels to render downstream (default: all)"},
    )

    project: str | None = field(
        default=None,
        metadata={"help": "Filter nodes by project (show only features from this project)"},
    )

    show_projects: bool = field(
        default=True,
        metadata={"help": "Show project names in feature nodes"},
    )

    title: str | None = field(
        default=None,
        metadata={
            "help": "Custom title for the graph. Defaults to 'Feature Graph' or 'Feature Graph Changes' for diffs."
        },
    )

    def get_feature_key(self) -> FeatureKey | None:
        """Parse feature string into FeatureKey.

        Returns:
            FeatureKey if feature is set, None otherwise
        """
        if self.feature is None:
            return None

        # Support both formats: "video__files" or "video/files"
        if "/" in self.feature:
            return FeatureKey(self.feature.split("/"))
        else:
            return FeatureKey(self.feature.split("__"))

    @classmethod
    def minimal(cls, show_projects: bool = True) -> "RenderConfig":
        """Preset: minimal information (structure only)."""
        return cls(
            show_fields=True,
            show_feature_versions=False,
            show_field_versions=False,
            show_code_versions=False,
            show_snapshot_version=False,
            show_projects=show_projects,
        )

    @classmethod
    def default(cls) -> "RenderConfig":
        """Preset: default information level (balanced)."""
        return cls(
            show_fields=True,
            show_feature_versions=True,
            show_field_versions=True,
            show_code_versions=False,
            show_snapshot_version=True,
            hash_length=8,
        )

    @classmethod
    def verbose(cls, show_projects: bool = True) -> "RenderConfig":
        """Preset: maximum information (everything)."""
        return cls(
            show_fields=True,
            show_feature_versions=True,
            show_field_versions=True,
            show_code_versions=True,
            show_snapshot_version=True,
            hash_length=0,  # Full hashes
            show_projects=show_projects,
        )


class BaseRenderer:
    """Base class for graph renderers.

    Provides common utilities for formatting keys and hashes.
    Uses unified GraphData model and Theme system.
    """

    def __init__(
        self,
        graph: FeatureGraph | None = None,
        config: RenderConfig | None = None,
        graph_data: GraphData | None = None,
        theme: Theme | None = None,
    ):
        """Initialize renderer.

        Args:
            graph: FeatureGraph (converted to GraphData automatically)
            config: Render configuration
            graph_data: GraphData (alternative to feature graph)
            theme: Color theme (uses default if None)

        Note:
            Either graph or graph_data must be provided.
            If both are provided, an error is raised.
        """
        if graph_data is None and graph is None:
            raise ValueError("Either graph or graph_data must be provided")

        # Prefer graph_data if provided, otherwise convert from graph
        if graph_data is not None:
            self.graph_data: GraphData = graph_data
        else:
            # graph is not None (validated above)
            assert graph is not None
            self.graph_data = GraphData.from_feature_graph(graph)

        self.config = config or RenderConfig()
        self.theme = theme or Theme.default()
        self.walker = GraphWalker(self.graph_data)

    def _format_hash(self, hash_str: str | None) -> str:
        """Format hash according to config.

        Args:
            hash_str: Full hash string (or None for removed nodes)

        Returns:
            Truncated hash if hash_length > 0, otherwise full hash
        """
        if hash_str is None:
            return "none"
        return format_hash(hash_str, length=self.config.hash_length)

    def _get_status_color(self, status: NodeStatus) -> str:
        """Get color for a given status.

        Args:
            status: Node or field status

        Returns:
            Color string for Rich markup
        """
        if status == NodeStatus.ADDED:
            return self.theme.added_color
        elif status == NodeStatus.REMOVED:
            return self.theme.removed_color
        elif status == NodeStatus.CHANGED:
            return self.theme.changed_color
        elif status == NodeStatus.UNCHANGED:
            return self.theme.unchanged_color
        else:
            return self.theme.feature_color

    def _format_version_transition(self, old_version: str | None, new_version: str | None) -> str:
        """Format version transition for diff display.

        Args:
            old_version: Old version hash
            new_version: New version hash

        Returns:
            Formatted string like "old... → new..."
        """
        old_str = self._format_hash(old_version)
        new_str = self._format_hash(new_version)
        return (
            f"[{self.theme.old_version_color}]{old_str}[/{self.theme.old_version_color}]... → "
            f"[{self.theme.new_version_color}]{new_str}[/{self.theme.new_version_color}]..."
        )

    def _format_feature_key(self, key: FeatureKey) -> str:
        """Format feature key for display.

        Uses / separator instead of __ for better readability.

        Args:
            key: Feature key

        Returns:
            Formatted string like "my/feature/key"
        """
        return format_feature_key(key)

    def _format_field_key(self, key: FieldKey) -> str:
        """Format field key for display.

        Args:
            key: Field key

        Returns:
            Formatted string like "field_name"
        """
        return format_field_key(key)

    def _get_filtered_graph_data(self) -> GraphData:
        """Get filtered graph data based on config filters.

        Returns:
            GraphData with only filtered nodes and edges
        """
        graph_data = self.graph_data

        # Apply project filter if specified
        if self.config.project is not None:
            filtered_nodes = {}
            for key, node in graph_data.nodes.items():
                # Include node if it matches the project or if it's a parent of a matching node
                if node.project == self.config.project:
                    filtered_nodes[key] = node
                else:
                    # Check if this node is a parent of any node in the project
                    for other_node in graph_data.nodes.values():
                        if other_node.project == self.config.project and node.key in other_node.dependencies:
                            filtered_nodes[key] = node
                            break

            # Filter edges to only include those between filtered nodes
            filtered_edges = []
            filtered_keys = set(filtered_nodes.keys())
            for edge in graph_data.edges:
                from_key_str = edge.from_key.to_string()
                to_key_str = edge.to_key.to_string()
                if from_key_str in filtered_keys and to_key_str in filtered_keys:
                    filtered_edges.append(edge)

            # Create new graph data with filtered nodes and edges
            graph_data = GraphData(
                nodes=filtered_nodes,
                edges=filtered_edges,
                snapshot_version=graph_data.snapshot_version,
                old_snapshot_version=graph_data.old_snapshot_version,
            )

        # Apply feature focus filter if specified
        focus_key = self.config.get_feature_key()
        if focus_key is not None:
            # Use walker to extract subgraph
            return self.walker.extract_subgraph(
                focus_key=focus_key,
                up=self.config.up,
                down=self.config.down,
            )

        return graph_data

    def render(self) -> str:
        """Render the graph and return string output.

        Returns:
            Rendered graph as string
        """
        raise NotImplementedError
