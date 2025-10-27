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
