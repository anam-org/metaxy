from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence, Set
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, overload

import narwhals as nw
import pydantic
from pydantic import BeforeValidator
from typing_extensions import Self

from metaxy._decorators import public
from metaxy._hashing import truncate_hash
from metaxy.models.bases import FrozenBaseModel
from metaxy.models.field import CoersibleToFieldSpecsTypeAdapter, FieldSpec
from metaxy.models.fields_mapping import FieldsMapping
from metaxy.models.filter_expression import parse_filter_string
from metaxy.models.lineage import LineageRelationship
from metaxy.models.types import (
    CoercibleToFeatureKey,
    FeatureKey,
    FeatureKeyAdapter,
    FieldKey,
    ValidatedFeatureKey,
)

if TYPE_CHECKING:
    # yes, these are circular imports, the TYPE_CHECKING block hides them at runtime.
    from metaxy.models.feature import BaseFeature


@public
class FeatureDep(pydantic.BaseModel):
    """Feature dependency specification with optional column selection, renaming, and lineage.

    Attributes:
        feature: The feature key to depend on. Accepts string ("a/b/c"), list (["a", "b", "c"]),
            FeatureKey instance, or BaseFeature class.
        select: Optional sequence of column names to select from the upstream feature.
            By default, all columns are selected. System columns are always selected.
            Uses post-rename names when `rename` is also specified.
        rename: Optional mapping of old column names to new names.
            Applied before column selection.
        fields_mapping: Optional field mapping configuration for automatic field dependency resolution.
            When provided, fields without explicit deps will automatically map to matching upstream fields.
            Defaults to `[FieldsMapping.default][metaxy.models.fields_mapping.DefaultFieldsMapping]`.
        filters: Optional SQL-like filter strings applied to this dependency. Automatically parsed into
            Narwhals expressions (accessible via the `filters` property). Filters are automatically
            applied by FeatureDepTransformer after renames during all FeatureDep operations (including
            resolve_update and version computation).
        lineage: The lineage relationship between this upstream dependency and the downstream feature.

            - `LineageRelationship.identity()` (default): 1:1 relationship, same cardinality

            - `LineageRelationship.aggregation(on=...)`: N:1, multiple upstream rows aggregate to one downstream

            - `LineageRelationship.expansion(on=...)`: 1:N, one upstream row expands to multiple downstream rows

        optional: Whether individual samples of the downstream feature can be computed without
            the corresponding samples of the upstream feature. If upstream samples are missing,
            they are going to be represented as NULL values in the joined upstream metadata.
            Defaults to False (required dependency).

    Example: Basic Usage
        ```py
        # Keep all columns with default field mapping (1:1 lineage)
        mx.FeatureDep(feature="upstream")

        # Keep only specific columns
        mx.FeatureDep(feature="upstream/feature", select=("col1", "col2"))

        # Rename columns to avoid conflicts
        mx.FeatureDep(feature="upstream/feature", rename={"old_name": "new_name"})

        # Combined rename + select: select uses post-rename names
        mx.FeatureDep(
            feature="upstream/feature",
            rename={"old_name": "new_name"},
            select=("new_name", "other_col"),
        )

        # SQL filters
        mx.FeatureDep(feature="upstream", filters=["age >= 25", "status = 'active'"])

        # Optional dependency (left join - samples preserved even if no match)
        mx.FeatureDep(feature="enrichment/data", optional=True)
        ```

    Example: Lineage Relationships
        ```py
        from metaxy.models.lineage import LineageRelationship

        # Aggregation: many sensor readings aggregate to one hourly stat
        mx.FeatureDep(feature="sensor_readings", lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]))

        # Expansion: one video expands to many frames
        mx.FeatureDep(feature="video", lineage=LineageRelationship.expansion(on=["video_id"]))

        # Mixed lineage: aggregate from one parent, identity from another
        # In FeatureSpec:
        deps = [
            mx.FeatureDep(feature="readings", lineage=LineageRelationship.aggregation(on=["sensor_id"])),
            mx.FeatureDep(feature="sensor_info", lineage=LineageRelationship.identity()),
        ]
        ```
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    feature: ValidatedFeatureKey
    select: tuple[str, ...] | None = None  # None = all columns, () = only system columns
    rename: dict[str, str] | None = None  # Column renaming mapping
    fields_mapping: FieldsMapping = pydantic.Field(default_factory=FieldsMapping.default)
    sql_filters: tuple[str, ...] | None = pydantic.Field(
        default=None,
        description="SQL-like filter strings applied to this dependency.",
        validation_alias=pydantic.AliasChoices("filters", "sql_filters"),
        serialization_alias="filters",
    )
    lineage: LineageRelationship = pydantic.Field(
        default_factory=LineageRelationship.identity,
        description="Lineage relationship between this upstream dependency and the downstream feature.",
    )
    optional: bool = pydantic.Field(
        default=False,
        description="Whether individual samples of the downstream feature can be computed without "
        "the corresponding samples of the upstream feature. If upstream samples are missing, "
        "they are going to be represented as NULL values in the joined upstream metadata.",
    )

    @pydantic.model_validator(mode="after")
    def validate_select_uses_post_rename_names(self) -> Self:
        if self.select and self.rename:
            renamed_away = set(self.rename.keys()) - set(self.rename.values())
            bad = renamed_away & set(self.select)
            if bad:
                raise ValueError(
                    f"select contains pre-rename column name(s) {sorted(bad)}. "
                    f"Use post-rename names in select (rename is applied first)."
                )
        return self

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            feature: str | Sequence[str] | FeatureKey | type[BaseFeature],
            select: tuple[str, ...] | None = None,
            rename: dict[str, str] | None = None,
            fields_mapping: FieldsMapping | None = None,
            filters: Sequence[str] | None = None,
            lineage: LineageRelationship | None = None,
            optional: bool = False,
        ) -> None: ...

    @cached_property
    def filters(self) -> tuple[nw.Expr, ...]:
        """Parse sql_filters into Narwhals expressions."""
        if self.sql_filters is None:
            return ()
        return tuple(parse_filter_string(filter_str) for filter_str in self.sql_filters)

    def table_name(self) -> str:
        """Get SQL-like table name for this feature spec."""
        return self.feature.table_name


IDColumns: TypeAlias = Sequence[str]  # non-bound, should be used for feature specs with arbitrary id columns

CoercibleToFeatureDep: TypeAlias = FeatureDep | type["BaseFeature"] | str | Sequence[str] | FeatureKey
UniqueKeep: TypeAlias = Literal["any", "latest"]


def _validate_column_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate column names and preserve the user-provided order."""
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence of column names, not a bare string")
    if isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a sequence of column names, not a mapping")
    if isinstance(value, Set):
        raise ValueError(f"{field_name} must be an ordered sequence of column names, not a set")
    try:
        return tuple(dict.fromkeys(value))
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of column names") from exc


def _validate_id_columns(value: Any) -> tuple[str, ...]:
    return _validate_column_sequence(value, "id_columns")


def _validate_unique_subset(value: Any) -> tuple[str, ...]:
    return _validate_column_sequence(value, "unique.subset")


def _validate_unique_order_by(value: Any) -> tuple[str, ...]:
    return _validate_column_sequence(value, "unique.order_by")


def _validate_deps(value: Any) -> list[FeatureDep]:
    """Coerce deps list, converting Feature classes to FeatureDep instances."""
    # Import here to avoid circular dependency at module level
    from metaxy.models.feature import BaseFeature

    if not isinstance(value, list):
        value = list(value) if hasattr(value, "__iter__") else [value]

    result = []
    for item in value:
        if isinstance(item, FeatureDep):
            # Already a FeatureDep, keep as-is
            result.append(item)
        elif isinstance(item, dict):
            # It's a dict (from deserialization), let Pydantic construct FeatureDep from it
            result.append(FeatureDep.model_validate(item))
        elif isinstance(item, type) and issubclass(item, BaseFeature):
            # It's a Feature class, convert to FeatureDep
            result.append(FeatureDep(feature=item))
        else:
            # Try to construct FeatureDep from the item (handles FeatureSpec, etc.)
            result.append(FeatureDep(feature=item))

    return result


@public
class Unique(FrozenBaseModel):
    """Read-time uniqueness settings for a feature."""

    subset: Annotated[tuple[str, ...], BeforeValidator(_validate_unique_subset)] = pydantic.Field(
        ...,
        min_length=1,
        description=(
            "Columns used to determine uniqueness. Records with identical values in these columns "
            "are considered duplicates. Repeated column names are ignored after their first occurrence."
        ),
    )
    keep: UniqueKeep = pydantic.Field(
        default="any",
        description=(
            "Strategy for choosing which row to keep among duplicates. "
            '"any" picks the row with the lexicographically greatest feature ID. '
            '"latest" keeps the row with the lexicographically greatest order_by values.'
        ),
    )
    order_by: Annotated[tuple[str, ...], BeforeValidator(_validate_unique_order_by)] | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            'Columns defining the logical order of duplicates when keep="latest". '
            "The lexicographically greatest values win, with feature ID columns as final logical tie-breakers. "
            "NULL sorts before every non-NULL value. Physical metadata breaks remaining ties; rejecting "
            "conflicting ties is the writer's responsibility. Repeated column names are ignored after "
            "their first occurrence."
        ),
    )

    @pydantic.model_validator(mode="after")
    def validate_latest_has_order_by(self) -> Self:
        if self.keep == "latest" and self.order_by is None:
            raise ValueError('unique.order_by is required when unique.keep="latest"')
        if self.keep == "any" and self.order_by is not None:
            raise ValueError('unique.order_by is only supported when unique.keep="latest"')
        return self

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            subset: Sequence[str],
            keep: UniqueKeep = "any",
            order_by: Sequence[str] | None = None,
        ) -> None: ...

    @pydantic.field_serializer("subset", "order_by")
    @staticmethod
    def _serialize_columns(value: tuple[str, ...] | None) -> list[str] | None:
        return list(value) if value is not None else None


@public
class FeatureSpec(FrozenBaseModel):
    key: Annotated[FeatureKey, BeforeValidator(FeatureKeyAdapter.validate_python)]
    id_columns: Annotated[tuple[str, ...], BeforeValidator(_validate_id_columns)] = pydantic.Field(
        ...,
        min_length=1,
        description="Columns that uniquely identify a sample in this feature.",
    )
    deps: Annotated[list[FeatureDep], BeforeValidator(_validate_deps)] = pydantic.Field(default_factory=list)
    fields: Annotated[
        list[FieldSpec],
        BeforeValidator(CoersibleToFieldSpecsTypeAdapter.validate_python),
    ] = pydantic.Field(
        default_factory=lambda: [
            FieldSpec(
                key=FieldKey(["default"]),
            )
        ],
    )
    metadata: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description="Metadata attached to this feature.",
    )
    description: str | None = pydantic.Field(
        default=None,
        description="Human-readable description of this feature.",
    )
    unique: Unique | None = pydantic.Field(
        default=None,
        description=(
            "Read-time uniqueness settings applied within the selected feature-version candidate set, "
            "after read-mode history and soft-deletion resolution and before user filters."
        ),
    )

    if TYPE_CHECKING:
        # Overload for common case: list of FeatureDep instances
        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns,
            deps: list[FeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: dict[str, Any] | None = None,
            description: str | None = None,
            unique: Unique | Mapping[str, Any] | None = None,
        ) -> None: ...

        # Overload for flexible case: list of coercible types
        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns,
            deps: list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: dict[str, Any] | None = None,
            description: str | None = None,
            unique: Unique | Mapping[str, Any] | None = None,
        ) -> None: ...

        # Implementation signature
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns,
            deps: list[FeatureDep] | list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: dict[str, Any] | None = None,
            description: str | None = None,
            unique: Unique | Mapping[str, Any] | None = None,
        ) -> None: ...

    @cached_property
    def deps_by_key(self) -> Mapping[FeatureKey, FeatureDep]:
        """Get dependencies indexed by their feature key."""
        return {dep.feature: dep for dep in self.deps}

    @cached_property
    def fields_by_key(self) -> Mapping[FieldKey, FieldSpec]:
        return {c.key: c for c in self.fields}

    @cached_property
    def code_version(self) -> str:
        """Hash of this feature's field code_versions only (no dependencies)."""
        hasher = hashlib.sha256()

        # Sort fields by key for deterministic ordering
        sorted_fields = sorted(self.fields, key=lambda field: field.key.to_string())

        for field in sorted_fields:
            hasher.update(field.key.to_string().encode("utf-8"))
            hasher.update(str(field.code_version).encode("utf-8"))

        return truncate_hash(hasher.hexdigest())

    def table_name(self) -> str:
        """Get SQL-like table name for this feature spec."""
        return self.key.table_name

    @pydantic.model_validator(mode="after")
    def validate_unique_field_keys(self) -> Self:
        """Validate that all fields have unique keys."""
        seen_keys: set[tuple[str, ...]] = set()
        for field in self.fields:
            # Convert to tuple for hashability in case it's a plain list
            key_tuple = tuple(field.key)
            if key_tuple in seen_keys:
                raise ValueError(f"Duplicate field key found: {field.key}. All fields must have unique keys.")
            seen_keys.add(key_tuple)
        return self

    @property
    def feature_spec_version(self) -> str:
        """Compute SHA256 hash of the complete feature specification.

        This property provides a deterministic hash of ALL specification properties,
        including key, deps, fields, and any metadata/tags.
        Used for audit trail and tracking specification changes.

        Unlike feature_version which only hashes computational properties
        (for migration triggering), feature_spec_version captures the entire specification
        for complete reproducibility and audit purposes.

        Returns:
            SHA256 hex digest of the specification

        Example:
            ```py
            spec = mx.FeatureSpec(
                key=mx.FeatureKey(["my", "feature"]),
                id_columns=["id"],
            )
            spec.feature_spec_version
            # 'abc123...'  # 64-character hex string
            ```
        """

        # Use model_dump with mode="json" for deterministic serialization
        # This ensures all types (like FeatureKey) are properly serialized
        spec_dict = self.model_dump(mode="json")
        if self.unique is None:
            spec_dict.pop("unique", None)

        # Sort keys to ensure deterministic ordering
        spec_json = json.dumps(spec_dict, sort_keys=True)

        # Compute SHA256 hash
        hasher = hashlib.sha256()
        hasher.update(spec_json.encode("utf-8"))

        return truncate_hash(hasher.hexdigest())


FeatureSpecWithIDColumns: TypeAlias = FeatureSpec

CoercibleToFieldSpec: TypeAlias = str | FieldSpec
