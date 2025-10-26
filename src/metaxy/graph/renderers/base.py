"""Base classes and configuration for graph rendering."""

from dataclasses import dataclass, field

from metaxy.graph import utils
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey, FieldKey


@dataclass
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
        metadata={
            "help": "Number of characters to show for version hashes (0 for full)"
        },
    )

    direction: str = field(
        default="TB",
        metadata={"help": "Graph layout direction: TB (top-bottom) or LR (left-right)"},
    )

    # Filtering options
    feature: str | None = field(
        default=None,
        metadata={
            "help": "Focus on a specific feature (e.g., 'video/files' or 'video__files')"
        },
    )

    up: int | None = field(
        default=None,
        metadata={
            "help": "Number of dependency levels to render upstream (default: all)"
        },
    )

    down: int | None = field(
        default=None,
        metadata={
            "help": "Number of dependency levels to render downstream (default: all)"
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
    def minimal(cls) -> "RenderConfig":
        """Preset: minimal information (structure only)."""
        return cls(
            show_fields=False,
            show_feature_versions=False,
            show_field_versions=False,
            show_code_versions=False,
            show_snapshot_version=False,
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
    def verbose(cls) -> "RenderConfig":
        """Preset: maximum information (everything)."""
        return cls(
            show_fields=True,
            show_feature_versions=True,
            show_field_versions=True,
            show_code_versions=True,
            show_snapshot_version=True,
            hash_length=0,  # Full hashes
        )


class GraphRenderer:
    """Base class for graph renderers.

    Provides common utilities for formatting keys and hashes.
    """

    def __init__(self, graph: FeatureGraph, config: RenderConfig):
        self.graph = graph
        self.config = config

    def _format_hash(self, hash_str: str) -> str:
        """Format hash according to config.

        Args:
            hash_str: Full hash string

        Returns:
            Truncated hash if hash_length > 0, otherwise full hash
        """
        return utils.format_hash(hash_str, length=self.config.hash_length)

    def _format_feature_key(self, key: FeatureKey) -> str:
        """Format feature key for display.

        Uses / separator instead of __ for better readability.

        Args:
            key: Feature key

        Returns:
            Formatted string like "my/feature/key"
        """
        return utils.format_feature_key(key)

    def _format_field_key(self, key: FieldKey) -> str:
        """Format field key for display.

        Args:
            key: Field key

        Returns:
            Formatted string like "field_name"
        """
        return utils.format_field_key(key)

    def _get_filtered_features(self) -> set[FeatureKey]:
        """Get features to render based on config filters.

        Returns:
            Set of feature keys to include in rendering
        """
        focus_key = self.config.get_feature_key()

        # If no focus feature specified, include all features
        if focus_key is None:
            return set(self.graph.features_by_key.keys())

        # Start with focus feature
        features_to_render = {focus_key}

        # Add upstream dependencies (up)
        # If up is not specified, don't include upstream (default to 0)
        # If up is 0, explicitly don't include upstream
        # If up > 0, include that many levels
        # If up < 0, include all upstream (unlimited)
        up_levels = self.config.up if self.config.up is not None else 0
        if up_levels != 0:
            max_up = None if up_levels < 0 else up_levels
            upstream = self._get_upstream_features(focus_key, max_levels=max_up)
            features_to_render.update(upstream)

        # Add downstream dependents (down)
        # If down is not specified, don't include downstream (default to 0)
        # If down is 0, explicitly don't include downstream
        # If down > 0, include that many levels
        # If down < 0, include all downstream (unlimited)
        down_levels = self.config.down if self.config.down is not None else 0
        if down_levels != 0:
            max_down = None if down_levels < 0 else down_levels
            downstream = self._get_downstream_features(focus_key, max_levels=max_down)
            features_to_render.update(downstream)

        return features_to_render

    def _get_upstream_features(
        self, start_key: FeatureKey, max_levels: int | None = None
    ) -> set[FeatureKey]:
        """Get all upstream dependencies of a feature.

        Args:
            start_key: Feature to start from
            max_levels: Maximum levels to traverse (None = unlimited)
                      1 means direct dependencies only
                      2 means dependencies of dependencies, etc.

        Returns:
            Set of upstream feature keys
        """
        upstream = set()

        def visit(key: FeatureKey, level: int):
            feature_cls = self.graph.features_by_key.get(key)
            if not feature_cls or not feature_cls.spec.deps:
                return

            for dep in feature_cls.spec.deps:
                if dep.key in self.graph.features_by_key and dep.key not in upstream:
                    upstream.add(dep.key)
                    # Only recurse if we haven't reached the max level
                    if max_levels is None or level + 1 < max_levels:
                        visit(dep.key, level + 1)

        visit(start_key, 0)
        return upstream

    def _get_downstream_features(
        self, start_key: FeatureKey, max_levels: int | None = None
    ) -> set[FeatureKey]:
        """Get all downstream dependents of a feature.

        Args:
            start_key: Feature to start from
            max_levels: Maximum levels to traverse (None = unlimited)
                      1 means direct dependents only
                      2 means dependents of dependents, etc.

        Returns:
            Set of downstream feature keys
        """
        downstream = set()

        # Build reverse dependency map
        dependents_map = {}  # feature_key -> list of features that depend on it
        for key, feature_cls in self.graph.features_by_key.items():
            if feature_cls.spec.deps:
                for dep in feature_cls.spec.deps:
                    if dep.key not in dependents_map:
                        dependents_map[dep.key] = []
                    dependents_map[dep.key].append(key)

        def visit(key: FeatureKey, level: int):
            if key not in dependents_map:
                return

            for dependent_key in dependents_map[key]:
                if dependent_key not in downstream:
                    downstream.add(dependent_key)
                    # Only recurse if we haven't reached the max level
                    if max_levels is None or level + 1 < max_levels:
                        visit(dependent_key, level + 1)

        visit(start_key, 0)
        return downstream

    def _get_topological_order(self) -> list[FeatureKey]:
        """Get features in topological order (dependencies first).

        Only includes features that pass the filter criteria.

        Returns:
            List of feature keys sorted so dependencies appear before dependents
        """
        features_to_include = self._get_filtered_features()
        visited = set()
        result = []

        def visit(key: FeatureKey):
            if key in visited or key not in features_to_include:
                return
            visited.add(key)

            # Visit dependencies first (only if they're also in features_to_include)
            feature_cls = self.graph.features_by_key[key]
            if feature_cls.spec.deps:
                for dep in feature_cls.spec.deps:
                    if dep.key in features_to_include:
                        visit(dep.key)

            result.append(key)

        # Visit all features that should be included
        for key in features_to_include:
            visit(key)

        return result

    def _is_root_feature(self, key: FeatureKey) -> bool:
        """Check if feature is a root (has no dependencies).

        Args:
            key: Feature key

        Returns:
            True if feature has no dependencies
        """
        feature_cls = self.graph.features_by_key[key]
        return not feature_cls.spec.deps

    def render(self) -> str:
        """Render the graph and return string output.

        Returns:
            Rendered graph as string
        """
        raise NotImplementedError
