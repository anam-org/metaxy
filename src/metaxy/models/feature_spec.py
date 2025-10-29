import hashlib
import json
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any

import pydantic
from pydantic import field_validator

from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureKey, FieldKey

if TYPE_CHECKING:
    from metaxy.models.feature import Feature  # noqa: F401


class FeatureDep(pydantic.BaseModel):
    """Feature dependency specification with optional column selection and renaming.

    Attributes:
        key: The feature key to depend on. Accepts:
            - FeatureKey: Direct key object
            - Feature class: Extracts key from Feature.spec.key
            - FeatureSpec: Extracts key from spec.key
            - str: Converted to FeatureKey([str])
            - list[str]: Converted to FeatureKey(list)
        columns: Optional tuple of column names to select from upstream feature.
            - None (default): Keep all columns from upstream
            - Empty tuple (): Keep only system columns (sample_uid, data_version, etc.)
            - Tuple of names: Keep only specified columns (plus system columns)
        rename: Optional mapping of old column names to new names.
            Applied after column selection.

    Examples:
        >>> # From Feature class (most ergonomic)
        >>> FeatureDep(key=UpstreamFeature)

        >>> # From string (for single-part keys)
        >>> FeatureDep(key="upstream")

        >>> # From list (for multi-part keys)
        >>> FeatureDep(key=["namespace", "upstream"])

        >>> # Traditional explicit key (still supported)
        >>> FeatureDep(key=FeatureKey(["upstream"]))

        >>> # With column selection
        >>> FeatureDep(
        ...     key=UpstreamFeature,
        ...     columns=("col1", "col2")
        ... )

        >>> # With renaming
        >>> FeatureDep(
        ...     key=UpstreamFeature,
        ...     rename={"old_name": "new_name"}
        ... )
    """

    key: FeatureKey
    columns: tuple[str, ...] | None = (
        None  # None = all columns, () = only system columns
    )
    rename: dict[str, str] | None = None  # Column renaming mapping

    @field_validator("key", mode="before")
    @classmethod
    def _coerce_key(cls, value: Any) -> FeatureKey:
        """Convert Feature class, FeatureSpec, str, or list[str] to FeatureKey.

        This allows passing Feature classes directly instead of extracting keys manually.
        """
        # Already a FeatureKey
        if isinstance(value, FeatureKey):
            return value

        # Feature class - extract spec.key
        if hasattr(value, "spec") and hasattr(value.spec, "key"):
            return value.spec.key

        # FeatureSpec object - extract key
        if hasattr(value, "key") and isinstance(getattr(value, "key"), FeatureKey):
            return value.key

        # str or list[str] - will be handled by FeatureKey's own validator
        return value

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

    @field_validator("deps", mode="before")
    @classmethod
    def _coerce_deps(cls, value: Any) -> list[FeatureDep] | None:
        """Convert Feature classes and FeatureSpec objects in deps list to FeatureDep objects.

        Accepts:
        - FeatureDep objects (returned as-is)
        - Feature classes: Extracts key from Feature.spec.key
        - FeatureSpec objects: Extracts key from spec.key
        - Dicts (from deserialization): Unpacks to FeatureDep
        - str, list[str], FeatureKey: Passed to FeatureDep(key=...)

        This allows ergonomic syntax:
        - deps=[UpstreamFeature] instead of deps=[FeatureDep(key=UpstreamFeature.spec.key)]
        - deps=["upstream"] instead of deps=[FeatureDep(key=FeatureKey(["upstream"]))]
        """
        if value is None:
            return None

        if not isinstance(value, list):
            return value

        coerced = []
        for item in value:
            # Already a FeatureDep
            if isinstance(item, FeatureDep):
                coerced.append(item)
            # Feature class - extract spec.key
            elif hasattr(item, "spec") and hasattr(item.spec, "key"):
                coerced.append(FeatureDep(key=item.spec.key))
            # FeatureSpec object - extract key (check for FeatureKey to distinguish from dicts)
            elif hasattr(item, "key") and isinstance(getattr(item, "key"), FeatureKey):
                coerced.append(FeatureDep(key=item.key))
            # Dict from deserialization - unpack it (FeatureDep's own validator will handle key coercion)
            elif isinstance(item, dict):
                coerced.append(FeatureDep(**item))
            # Other types (str, list, FeatureKey) - let FeatureDep's validator handle it
            else:
                coerced.append(FeatureDep(key=item))

        return coerced

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
