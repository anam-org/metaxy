"""Theme system for graph rendering."""

from pydantic import BaseModel, Field


class Theme(BaseModel):
    """Color theme for graph rendering.

    Unified theme for all rendering backends (terminal, Mermaid, Graphviz).
    All colors use hex format (e.g., "#FF5733") for consistency.
    Rich terminal supports hex via markup: [#FF0000]text[/#FF0000]
    """

    # Normal mode colors
    feature_color: str = Field(default="#00FFFF", description="Feature node color (cyan)")
    field_color: str = Field(default="#5F87AF", description="Field color (steel blue)")
    version_color: str = Field(default="#FFFF00", description="Version info color (yellow)")
    edge_color: str = Field(default="#808080", description="Edge/dependency color (gray)")
    snapshot_color: str = Field(default="#FF00FF", description="Snapshot info color (magenta)")

    # Diff mode - node/edge colors
    added_color: str = Field(default="#00FF00", description="Added items (green)")
    removed_color: str = Field(default="#FF0000", description="Removed items (red)")
    changed_color: str = Field(default="#FFAA00", description="Changed items (orange)")
    unchanged_color: str = Field(default="#808080", description="Unchanged items (gray)")

    # Version transition colors (for showing old→new in diffs)
    old_version_color: str = Field(default="#FF0000", description="Old version color (red)")
    new_version_color: str = Field(default="#00FF00", description="New version color (green)")

    @classmethod
    def default(cls) -> "Theme":
        """Create default theme."""
        return cls()
