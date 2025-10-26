"""Graph visualization and rendering utilities."""

from metaxy.graph.diff import GraphData
from metaxy.graph.diff.rendering import (
    BaseRenderer,
    CardsRenderer,
    GraphvizRenderer,
    MermaidRenderer,
    RenderConfig,
    TerminalRenderer,
)

# Backward compatibility aliases
TerminalCardsRenderer = CardsRenderer
GraphRenderer = BaseRenderer

__all__ = [
    "BaseRenderer",
    "RenderConfig",
    "GraphData",
    "TerminalRenderer",
    "CardsRenderer",
    "MermaidRenderer",
    "GraphvizRenderer",
    # Backward compatibility
    "TerminalCardsRenderer",
    "GraphRenderer",
]
