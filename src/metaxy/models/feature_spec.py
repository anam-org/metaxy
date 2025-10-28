import hashlib
import json
from collections.abc import Mapping
from functools import cached_property

import pydantic

from metaxy.models.constants import ALL_SYSTEM_COLUMNS
from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureKey, FieldKey


class FeatureDep(pydantic.BaseModel):
    """Feature dependency specification with optional column selection and renaming.

    Attributes:
        key: The feature key to depend on
        columns: Optional tuple of column names to select from upstream feature.
            - None (default): Keep all columns from upstream
            - Empty tuple (): Keep only system columns (sample_uid, data_version, etc.)
            - Tuple of names: Keep only specified columns (plus system columns)
        rename: Optional mapping of old column names to new names.
            Applied after column selection.

    Examples:
        >>> # Keep all columns (default behavior)
        >>> FeatureDep(key=FeatureKey(["upstream"]))

        >>> # Keep only specific columns
        >>> FeatureDep(
        ...     key=FeatureKey(["upstream"]),
        ...     columns=("col1", "col2")
        ... )

        >>> # Rename columns to avoid conflicts
        >>> FeatureDep(
        ...     key=FeatureKey(["upstream"]),
        ...     rename={"old_name": "new_name"}
        ... )

        >>> # Select and rename
        >>> FeatureDep(
        ...     key=FeatureKey(["upstream"]),
        ...     columns=("col1", "col2"),
        ...     rename={"col1": "upstream_col1"}
        ... )
    """

    key: FeatureKey
    columns: tuple[str, ...] | None = (
        None  # None = all columns, () = only system columns
    )
    rename: dict[str, str] | None = None  # Column renaming mapping

    @pydantic.model_validator(mode="after")
    def validate_column_operations(self) -> "FeatureDep":
        """Validate column selection and renaming operations."""
        if self.rename is not None:
            for new_name in self.rename.values():
                if new_name in ALL_SYSTEM_COLUMNS:
                    raise ValueError(
                        f"Cannot rename column to system column name: {new_name}. "
                        f"System columns: {ALL_SYSTEM_COLUMNS}"
                    )
        return self

    def table_name(self) -> str:
        """Get SQL-like table name for this feature spec."""
        return self.key.table_name


class FeatureSpec(pydantic.BaseModel):
    key: FeatureKey
    deps: list[FeatureDep] | None
    fields: list[FieldSpec] = pydantic.Field(
        default_factory=lambda: [
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=SpecialFieldDep.ALL,
            )
        ]
    )
    code_version: int = 1
    id_columns: list[str] = pydantic.Field(default_factory=lambda: ["sample_uid"])

    @cached_property
    def fields_by_key(self) -> Mapping[FieldKey, FieldSpec]:
        return {c.key: c for c in self.fields}

    def table_name(self) -> str:
        """Get SQL-like table name for this feature spec."""
        return self.key.table_name

    @pydantic.model_validator(mode="after")
    def validate_unique_field_keys(self) -> "FeatureSpec":
        """Validate that all fields have unique keys."""
        seen_keys: set[tuple[str, ...]] = set()
        for field in self.fields:
            # Convert to tuple for hashability in case it's a plain list
            key_tuple = tuple(field.key)
            if key_tuple in seen_keys:
                raise ValueError(
                    f"Duplicate field key found: {field.key}. "
                    f"All fields must have unique keys."
                )
            seen_keys.add(key_tuple)
        return self

    @pydantic.model_validator(mode="after")
    def validate_id_columns(self) -> "FeatureSpec":
        """Validate that id_columns is non-empty if specified."""
        if self.id_columns is not None and len(self.id_columns) == 0:
            raise ValueError(
                "id_columns must be non-empty if specified. Use None for default."
            )
        return self

    @property
    def feature_spec_version(self) -> str:
        """Compute SHA256 hash of the complete feature specification.

        This property provides a deterministic hash of ALL specification properties,
        including key, deps, fields, code_version, and any metadata/tags.
        Used for audit trail and tracking specification changes.

        Unlike feature_version which only hashes computational properties
        (for migration triggering), feature_spec_version captures the entire specification
        for complete reproducibility and audit purposes.

        Returns:
            SHA256 hex digest of the specification

        Example:
            >>> spec = FeatureSpec(
            ...     key=FeatureKey(["my", "feature"]),
            ...     deps=None,
            ...     fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ...     code_version=1
            ... )
            >>> spec.feature_spec_version
            'abc123...'  # 64-character hex string
        """
        # Use model_dump with mode="json" for deterministic serialization
        # This ensures all types (like FeatureKey) are properly serialized
        spec_dict = self.model_dump(mode="json")

        # Sort keys to ensure deterministic ordering
        spec_json = json.dumps(spec_dict, sort_keys=True)

        # Compute SHA256 hash
        hasher = hashlib.sha256()
        hasher.update(spec_json.encode("utf-8"))

        return hasher.hexdigest()
