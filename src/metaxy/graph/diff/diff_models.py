"""Graph diff models for migration system.

Provides GraphDiff with struct serialization for storage in migration tables.
"""

from typing import Any

from pydantic import Field

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.utils.constants import DEFAULT_CODE_VERSION
from metaxy.utils.exceptions import MetaxyEmptyCodeVersionError


def _field_to_struct(field: dict[str, Any]) -> dict[str, Any]:
    """Convert a field dict to struct format."""
    key = field["key"]
    return {
        "key": key if isinstance(key, str) else key.to_string(),
        "version": field.get("version", ""),
        "code_version": field["code_version"],
    }


def _validate_code_version(code_version: str | None, context: str) -> None:
    """Validate that code_version is not empty."""
    if not code_version:
        raise MetaxyEmptyCodeVersionError(f"{context} has empty code_version.")


def _is_invalid_code_version(value: str | None) -> bool:
    """Check if a code_version value is invalid (None, empty, or default)."""
    return value in (None, "", DEFAULT_CODE_VERSION)


def _parse_version(value: str | None) -> str | None:
    """Parse a version string, converting empty strings to None."""
    return value if value else None


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

    def _added_or_removed_node_to_struct(
        self, node: "AddedNode | RemovedNode"
    ) -> dict[str, Any]:
        """Convert an AddedNode or RemovedNode to struct format."""
        _validate_code_version(
            node.code_version, f"Node {node.feature_key.to_string()}"
        )
        return {
            "key": node.feature_key.to_string(),
            "version": node.version,
            "code_version": node.code_version,
            "fields": [_field_to_struct(f) for f in node.fields],
            "dependencies": [dep.to_string() for dep in node.dependencies],
        }

    def _changed_node_to_struct(self, node: "NodeChange") -> dict[str, Any]:
        """Convert a NodeChange to struct format."""
        node_key = node.feature_key.to_string()

        added_fields_list = self._added_fields_to_struct(node.added_fields, node_key)
        removed_fields_list = self._removed_fields_to_struct(
            node.removed_fields, node_key
        )
        changed_fields_list = self._changed_fields_to_struct(
            node.changed_fields, node_key
        )

        if not (node.old_code_version and node.new_code_version):
            raise MetaxyEmptyCodeVersionError(
                f"Node {node_key} has empty old/new code_version."
            )

        return {
            "key": node_key,
            "old_version": node.old_version or "",
            "new_version": node.new_version or "",
            "old_code_version": node.old_code_version,
            "new_code_version": node.new_code_version,
            "added_fields": added_fields_list,
            "removed_fields": removed_fields_list,
            "changed_fields": changed_fields_list,
        }

    def _added_fields_to_struct(
        self, fields: list["FieldChange"], node_key: str
    ) -> list[dict[str, Any]]:
        """Convert added fields to struct format."""
        result = []
        for field in fields:
            _validate_code_version(
                field.new_code_version,
                f"Node {node_key} field {field.field_key.to_string()}",
            )
            result.append(
                {
                    "key": field.field_key.to_string(),
                    "version": field.new_version or "",
                    "code_version": field.new_code_version,
                }
            )
        return result

    def _removed_fields_to_struct(
        self, fields: list["FieldChange"], node_key: str
    ) -> list[dict[str, Any]]:
        """Convert removed fields to struct format."""
        result = []
        for field in fields:
            _validate_code_version(field.old_code_version, f"Node {node_key}")
            result.append(
                {
                    "key": field.field_key.to_string(),
                    "version": field.old_version or "",
                    "code_version": field.old_code_version,
                }
            )
        return result

    def _changed_fields_to_struct(
        self, fields: list["FieldChange"], node_key: str
    ) -> list[dict[str, Any]]:
        """Convert changed fields to struct format."""
        result = []
        for field in fields:
            if not (field.old_code_version and field.new_code_version):
                raise MetaxyEmptyCodeVersionError(
                    f"Node {node_key} has empty code_version."
                )
            result.append(
                {
                    "key": field.field_key.to_string(),
                    "old_version": field.old_version or "",
                    "new_version": field.new_version or "",
                    "old_code_version": field.old_code_version,
                    "new_code_version": field.new_code_version,
                }
            )
        return result

    def to_struct(self) -> dict[str, Any]:
        """Serialize to struct (native Python types for storage).

        Returns:
            Dict with structure compatible with Polars struct type
        """
        return {
            "added_nodes": [
                self._added_or_removed_node_to_struct(n) for n in self.added_nodes
            ],
            "removed_nodes": [
                self._added_or_removed_node_to_struct(n) for n in self.removed_nodes
            ],
            "changed_nodes": [
                self._changed_node_to_struct(n) for n in self.changed_nodes
            ],
        }

    @classmethod
    def _field_from_struct(cls, field_data: dict[str, Any]) -> dict[str, Any]:
        """Parse a field dict from struct format."""
        return {
            "key": field_data["key"],
            "version": _parse_version(field_data["version"]),
            "code_version": field_data["code_version"],
        }

    @classmethod
    def _validate_node_code_version(cls, node_data: dict[str, Any]) -> None:
        """Validate that a node has a valid code_version."""
        if _is_invalid_code_version(node_data.get("code_version")):
            raise MetaxyEmptyCodeVersionError(
                f"Node {node_data['key']} has empty code_version."
            )

    @classmethod
    def _added_node_from_struct(cls, node_data: dict[str, Any]) -> "AddedNode":
        """Parse an AddedNode from struct format."""
        cls._validate_node_code_version(node_data)
        return AddedNode(
            feature_key=FeatureKey(node_data["key"].split("/")),
            version=node_data["version"],
            code_version=node_data["code_version"],
            fields=[cls._field_from_struct(f) for f in node_data.get("fields", [])],
            dependencies=[
                FeatureKey(dep.split("/")) for dep in node_data.get("dependencies", [])
            ],
        )

    @classmethod
    def _removed_node_from_struct(cls, node_data: dict[str, Any]) -> "RemovedNode":
        """Parse a RemovedNode from struct format."""
        cls._validate_node_code_version(node_data)
        return RemovedNode(
            feature_key=FeatureKey(node_data["key"].split("/")),
            version=node_data["version"],
            code_version=node_data["code_version"],
            fields=[cls._field_from_struct(f) for f in node_data.get("fields", [])],
            dependencies=[
                FeatureKey(dep.split("/")) for dep in node_data.get("dependencies", [])
            ],
        )

    @classmethod
    def _added_field_from_struct(
        cls, field_data: dict[str, Any], node_key: str
    ) -> "FieldChange":
        """Parse an added field FieldChange from struct format."""
        if _is_invalid_code_version(field_data.get("code_version")):
            raise MetaxyEmptyCodeVersionError(
                f"Field {field_data['key']} in feature {node_key} has empty code_version."
            )
        return FieldChange(
            field_key=FieldKey(field_data["key"].split("/")),
            old_version=None,
            new_version=_parse_version(field_data["version"]),
            old_code_version=None,
            new_code_version=field_data["code_version"],
        )

    @classmethod
    def _removed_field_from_struct(
        cls, field_data: dict[str, Any], node_key: str
    ) -> "FieldChange":
        """Parse a removed field FieldChange from struct format."""
        if _is_invalid_code_version(field_data.get("code_version")):
            raise MetaxyEmptyCodeVersionError(
                f"Field {field_data['key']} in feature {node_key} has empty code_version."
            )
        return FieldChange(
            field_key=FieldKey(field_data["key"].split("/")),
            old_version=_parse_version(field_data["version"]),
            new_version=None,
            old_code_version=field_data["code_version"],
            new_code_version=None,
        )

    @classmethod
    def _changed_field_from_struct(
        cls, field_data: dict[str, Any], node_key: str
    ) -> "FieldChange":
        """Parse a changed field FieldChange from struct format."""
        if _is_invalid_code_version(
            field_data.get("old_code_version")
        ) or _is_invalid_code_version(field_data.get("new_code_version")):
            raise MetaxyEmptyCodeVersionError(
                f"Field {field_data['key']} in feature {node_key} has empty code_version."
            )
        return FieldChange(
            field_key=FieldKey(field_data["key"].split("/")),
            old_version=_parse_version(field_data["old_version"]),
            new_version=_parse_version(field_data["new_version"]),
            old_code_version=field_data["old_code_version"],
            new_code_version=field_data["new_code_version"],
        )

    @classmethod
    def _changed_node_from_struct(cls, node_data: dict[str, Any]) -> "NodeChange":
        """Parse a NodeChange from struct format."""
        node_key = node_data["key"]

        added_fields = [
            cls._added_field_from_struct(f, node_key)
            for f in node_data.get("added_fields", [])
        ]
        removed_fields = [
            cls._removed_field_from_struct(f, node_key)
            for f in node_data.get("removed_fields", [])
        ]
        changed_fields = [
            cls._changed_field_from_struct(f, node_key)
            for f in node_data.get("changed_fields", [])
        ]

        if _is_invalid_code_version(
            node_data.get("old_code_version")
        ) or _is_invalid_code_version(node_data.get("new_code_version")):
            raise MetaxyEmptyCodeVersionError(
                f"Node {node_key} has empty old/new code_version."
            )

        return NodeChange(
            feature_key=FeatureKey(node_key.split("/")),
            old_version=_parse_version(node_data["old_version"]),
            new_version=_parse_version(node_data["new_version"]),
            old_code_version=node_data["old_code_version"],
            new_code_version=node_data["new_code_version"],
            added_fields=added_fields,
            removed_fields=removed_fields,
            changed_fields=changed_fields,
        )

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
        return cls(
            from_snapshot_version=from_snapshot_version,
            to_snapshot_version=to_snapshot_version,
            added_nodes=[
                cls._added_node_from_struct(n)
                for n in struct_data.get("added_nodes", [])
            ],
            removed_nodes=[
                cls._removed_node_from_struct(n)
                for n in struct_data.get("removed_nodes", [])
            ],
            changed_nodes=[
                cls._changed_node_from_struct(n)
                for n in struct_data.get("changed_nodes", [])
            ],
        )
