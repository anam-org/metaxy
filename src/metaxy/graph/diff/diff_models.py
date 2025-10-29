"""Graph diff models for migration system.

Provides GraphDiff with struct serialization for storage in migration tables.
"""

from typing import Any

from pydantic import Field

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey


def _normalize_code_version(value: Any) -> str | None:
    """Normalize code_version for backward compatibility.

    Args:
        value: Code version value (int, str, or None)

    Returns:
        Normalized code version (str or None)

    Raises:
        ValueError: If value is an empty string
        TypeError: If value is not int, str, or None
    """
    if isinstance(value, int):
        return str(value) if value != 0 else None
    elif value is None:
        return None
    elif isinstance(value, str):
        if not value:  # empty string
            raise ValueError("code_version cannot be an empty string")
        return value
    else:
        raise TypeError(f"code_version must be str, int, or None, got {type(value)}")


class FieldChange(FrozenBaseModel):
    """Represents a change in a field between two snapshots."""

    field_key: FieldKey
    old_version: str | None = None  # None if field was added
    new_version: str | None = None  # None if field was removed
    old_code_version: str | None = None
    new_code_version: str | None = None

    @property
    def is_added(self) -> bool:
        """Check if field was added."""
        return self.old_version is None

    @property
    def is_removed(self) -> bool:
        """Check if field was removed."""
        return self.new_version is None

    @property
    def is_changed(self) -> bool:
        """Check if field version changed."""
        return (
            self.old_version is not None
            and self.new_version is not None
            and self.old_version != self.new_version
        )


class NodeChange(FrozenBaseModel):
    """Represents a change in a node/feature between two snapshots."""

    feature_key: FeatureKey
    old_version: str | None = None  # None if node was added
    new_version: str | None = None  # None if node was removed
    old_code_version: str | None = None
    new_code_version: str | None = None
    added_fields: list[FieldChange] = Field(default_factory=list)
    removed_fields: list[FieldChange] = Field(default_factory=list)
    changed_fields: list[FieldChange] = Field(default_factory=list)

    @property
    def is_added(self) -> bool:
        """Check if node was added."""
        return self.old_version is None

    @property
    def is_removed(self) -> bool:
        """Check if node was removed."""
        return self.new_version is None

    @property
    def is_changed(self) -> bool:
        """Check if node version changed."""
        return (
            self.old_version is not None
            and self.new_version is not None
            and self.old_version != self.new_version
        )

    @property
    def field_changes(self) -> list[FieldChange]:
        """Get all field changes (added + removed + changed).

        Backward compatibility property for old API.
        """
        return self.added_fields + self.removed_fields + self.changed_fields

    @property
    def has_field_changes(self) -> bool:
        """Check if node has any field changes.

        Backward compatibility property for old API.
        """
        return bool(self.added_fields or self.removed_fields or self.changed_fields)


class AddedNode(FrozenBaseModel):
    """Represents a node that was added in the diff."""

    feature_key: FeatureKey
    version: str
    code_version: str | None = None
    fields: list[dict[str, Any]] = Field(
        default_factory=list
    )  # {key, version, code_version}
    dependencies: list[FeatureKey] = Field(default_factory=list)


class RemovedNode(FrozenBaseModel):
    """Represents a node that was removed in the diff."""

    feature_key: FeatureKey
    version: str
    code_version: str | None = None
    fields: list[dict[str, Any]] = Field(
        default_factory=list
    )  # {key, version, code_version}
    dependencies: list[FeatureKey] = Field(default_factory=list)


class GraphDiff(FrozenBaseModel):
    """Result of comparing two graph snapshots.

    Stores changes between two graph states for migration generation.
    """

    from_snapshot_version: str
    to_snapshot_version: str
    added_nodes: list[AddedNode] = Field(default_factory=list)
    removed_nodes: list[RemovedNode] = Field(default_factory=list)
    changed_nodes: list[NodeChange] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if diff contains any changes."""
        return bool(self.added_nodes or self.removed_nodes or self.changed_nodes)

    def to_struct(self) -> dict[str, Any]:
        """Serialize to struct (native Python types for storage).

        Returns:
            Dict with structure compatible with Polars struct type
        """
        added_nodes_list = []
        for node in self.added_nodes:
            fields_list = []
            for field in node.fields:
                fields_list.append(
                    {
                        "key": field["key"]
                        if isinstance(field["key"], str)
                        else field["key"].to_string(),
                        "version": field.get("version", ""),
                        "code_version": field.get("code_version", "") or "",
                    }
                )

            added_nodes_list.append(
                {
                    "key": node.feature_key.to_string(),
                    "version": node.version,
                    "code_version": node.code_version or "",
                    "fields": fields_list,
                    "dependencies": [dep.to_string() for dep in node.dependencies],
                }
            )

        removed_nodes_list = []
        for node in self.removed_nodes:
            fields_list = []
            for field in node.fields:
                fields_list.append(
                    {
                        "key": field["key"]
                        if isinstance(field["key"], str)
                        else field["key"].to_string(),
                        "version": field.get("version", ""),
                        "code_version": field.get("code_version", "") or "",
                    }
                )

            removed_nodes_list.append(
                {
                    "key": node.feature_key.to_string(),
                    "version": node.version,
                    "code_version": node.code_version or "",
                    "fields": fields_list,
                    "dependencies": [dep.to_string() for dep in node.dependencies],
                }
            )

        changed_nodes_list = []
        for node in self.changed_nodes:
            added_fields_list = []
            for field in node.added_fields:
                added_fields_list.append(
                    {
                        "key": field.field_key.to_string(),
                        "version": field.new_version or "",
                        "code_version": field.new_code_version or "",
                    }
                )

            removed_fields_list = []
            for field in node.removed_fields:
                removed_fields_list.append(
                    {
                        "key": field.field_key.to_string(),
                        "version": field.old_version or "",
                        "code_version": field.old_code_version or "",
                    }
                )

            changed_fields_list = []
            for field in node.changed_fields:
                changed_fields_list.append(
                    {
                        "key": field.field_key.to_string(),
                        "old_version": field.old_version or "",
                        "new_version": field.new_version or "",
                        "old_code_version": field.old_code_version or "",
                        "new_code_version": field.new_code_version or "",
                    }
                )

            changed_nodes_list.append(
                {
                    "key": node.feature_key.to_string(),
                    "old_version": node.old_version or "",
                    "new_version": node.new_version or "",
                    "old_code_version": node.old_code_version or "",
                    "new_code_version": node.new_code_version or "",
                    "added_fields": added_fields_list,
                    "removed_fields": removed_fields_list,
                    "changed_fields": changed_fields_list,
                }
            )

        return {
            "added_nodes": added_nodes_list,
            "removed_nodes": removed_nodes_list,
            "changed_nodes": changed_nodes_list,
        }

    @classmethod
    def from_struct(
        cls,
        struct_data: dict[str, Any],
        from_snapshot_version: str,
        to_snapshot_version: str,
    ) -> "GraphDiff":
        """Deserialize from struct.

        Args:
            struct_data: Dict with structure from to_struct()
            from_snapshot_version: Source snapshot version
            to_snapshot_version: Target snapshot version

        Returns:
            GraphDiff instance
        """
        added_nodes = []
        for node_data in struct_data.get("added_nodes", []):
            fields = []
            for field_data in node_data.get("fields", []):
                fields.append(
                    {
                        "key": field_data["key"],
                        "version": field_data["version"]
                        if field_data["version"]
                        else None,
                        "code_version": _normalize_code_version(
                            field_data["code_version"]
                        ),
                    }
                )

            added_nodes.append(
                AddedNode(
                    feature_key=FeatureKey(node_data["key"].split("/")),
                    version=node_data["version"],
                    code_version=_normalize_code_version(node_data["code_version"]),
                    fields=fields,
                    dependencies=[
                        FeatureKey(dep.split("/"))
                        for dep in node_data.get("dependencies", [])
                    ],
                )
            )

        removed_nodes = []
        for node_data in struct_data.get("removed_nodes", []):
            fields = []
            for field_data in node_data.get("fields", []):
                fields.append(
                    {
                        "key": field_data["key"],
                        "version": field_data["version"]
                        if field_data["version"]
                        else None,
                        "code_version": _normalize_code_version(
                            field_data["code_version"]
                        ),
                    }
                )

            removed_nodes.append(
                RemovedNode(
                    feature_key=FeatureKey(node_data["key"].split("/")),
                    version=node_data["version"],
                    code_version=_normalize_code_version(node_data["code_version"]),
                    fields=fields,
                    dependencies=[
                        FeatureKey(dep.split("/"))
                        for dep in node_data.get("dependencies", [])
                    ],
                )
            )

        changed_nodes = []
        for node_data in struct_data.get("changed_nodes", []):
            added_fields = []
            for field_data in node_data.get("added_fields", []):
                added_fields.append(
                    FieldChange(
                        field_key=FieldKey(field_data["key"].split("/")),
                        old_version=None,
                        new_version=field_data["version"]
                        if field_data["version"]
                        else None,
                        old_code_version=None,
                        new_code_version=_normalize_code_version(
                            field_data["code_version"]
                        ),
                    )
                )

            removed_fields = []
            for field_data in node_data.get("removed_fields", []):
                removed_fields.append(
                    FieldChange(
                        field_key=FieldKey(field_data["key"].split("/")),
                        old_version=field_data["version"]
                        if field_data["version"]
                        else None,
                        new_version=None,
                        old_code_version=_normalize_code_version(
                            field_data["code_version"]
                        ),
                        new_code_version=None,
                    )
                )

            changed_fields = []
            for field_data in node_data.get("changed_fields", []):
                changed_fields.append(
                    FieldChange(
                        field_key=FieldKey(field_data["key"].split("/")),
                        old_version=field_data["old_version"]
                        if field_data["old_version"]
                        else None,
                        new_version=field_data["new_version"]
                        if field_data["new_version"]
                        else None,
                        old_code_version=_normalize_code_version(
                            field_data["old_code_version"]
                        ),
                        new_code_version=_normalize_code_version(
                            field_data["new_code_version"]
                        ),
                    )
                )

            changed_nodes.append(
                NodeChange(
                    feature_key=FeatureKey(node_data["key"].split("/")),
                    old_version=node_data["old_version"]
                    if node_data["old_version"]
                    else None,
                    new_version=node_data["new_version"]
                    if node_data["new_version"]
                    else None,
                    old_code_version=_normalize_code_version(
                        node_data["old_code_version"]
                    ),
                    new_code_version=_normalize_code_version(
                        node_data["new_code_version"]
                    ),
                    added_fields=added_fields,
                    removed_fields=removed_fields,
                    changed_fields=changed_fields,
                )
            )

        return cls(
            from_snapshot_version=from_snapshot_version,
            to_snapshot_version=to_snapshot_version,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            changed_nodes=changed_nodes,
        )
