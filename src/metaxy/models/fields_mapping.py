"""Field mapping system for automatic field dependency resolution.

This module provides a flexible system for defining how fields map to upstream
dependencies, supporting both automatic mapping patterns and explicit configurations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_serializer
from pydantic import Field as PydanticField
from typing_extensions import Self

from metaxy._decorators import public
from metaxy.models.types import (
    CoercibleToFieldKey,
    FeatureKey,
    FieldKey,
    ValidatedFieldKeyAdapter,
)

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec


@public
class FieldsMappingType(str, Enum):
    """Type of fields mapping between a field key and the upstream field keys."""

    DEFAULT = "default"
    SPECIFIC = "specific"
    ALL = "all"
    NONE = "none"


class FieldsMappingResolutionContext(BaseModel):
    """Context for resolving field mappings.

    This contains all the information needed to resolve field dependencies
    including the upstream feature being mapped against.
    """

    model_config = ConfigDict(frozen=True)

    field_key: FieldKey
    """The downstream field key being resolved."""

    upstream_feature: FeatureSpec
    """The upstream feature spec being resolved against."""

    @property
    def upstream_feature_key(self) -> FeatureKey:
        """Get the upstream feature key."""
        return self.upstream_feature.key

    @property
    def upstream_feature_fields(self) -> set[FieldKey]:
        """Get the set of field keys from the upstream feature."""
        return {field.key for field in self.upstream_feature.fields}


class BaseFieldsMapping(BaseModel, ABC):
    """Base class for field mapping configurations.

    Field mappings define how a field automatically resolves its dependencies
    based on upstream feature fields.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def resolve_field_deps(
        self,
        context: FieldsMappingResolutionContext,
    ) -> set[FieldKey]:
        """Resolve automatic field mapping to explicit FieldDep list.

        This method should be overridden by concrete implementations.

        Arguments:
            context: The resolution context containing field key and upstream feature.

        Returns:
            Set of [FieldKey][metaxy.models.types.FieldKey] instances for matching fields
        """
        raise NotImplementedError


@public
class SpecificFieldsMapping(BaseFieldsMapping):
    """Field mapping that explicitly depends on specific upstream fields."""

    type: Literal[FieldsMappingType.SPECIFIC] = FieldsMappingType.SPECIFIC
    mapping: dict[FieldKey, set[FieldKey]]

    @field_serializer("mapping", when_used="json")
    def _serialize_mapping(self, value: dict[FieldKey, set[FieldKey]]) -> dict[str, list[list[str]]]:
        """Serialize mapping with FieldKey dict keys converted to strings for JSON.

        JSON dict keys must be strings, so we convert FieldKey objects to their
        string representation (e.g., "faces" or "audio/french").
        Field keys are sorted for deterministic serialization.
        """
        return {
            key.to_string(): [list(field_key.parts) for field_key in sorted(field_keys)]
            for key, field_keys in value.items()
        }

    def resolve_field_deps(
        self,
        context: FieldsMappingResolutionContext,
    ) -> set[FieldKey]:
        desired_upstream_fields = self.mapping.get(context.field_key, set())
        return desired_upstream_fields & context.upstream_feature_fields


@public
class AllFieldsMapping(BaseFieldsMapping):
    """Field mapping that explicitly depends on all upstream fields."""

    type: Literal[FieldsMappingType.ALL] = FieldsMappingType.ALL

    def resolve_field_deps(
        self,
        context: FieldsMappingResolutionContext,
    ) -> set[FieldKey]:
        return context.upstream_feature_fields


@public
class NoneFieldsMapping(BaseFieldsMapping):
    """Field mapping that never matches any upstream fields."""

    type: Literal[FieldsMappingType.NONE] = FieldsMappingType.NONE

    def resolve_field_deps(
        self,
        context: FieldsMappingResolutionContext,
    ) -> set[FieldKey]:
        return set()


@public
class DefaultFieldsMapping(BaseFieldsMapping):
    """Default automatic field mapping configuration.

    When used, automatically maps fields to matching upstream fields based on field keys.

    Attributes:
        match_suffix: If True, allows suffix matching (e.g., "french" matches "audio/french")
        exclude_fields: List of field keys to exclude from auto-mapping
    """

    type: Literal[FieldsMappingType.DEFAULT] = FieldsMappingType.DEFAULT
    match_suffix: bool = False
    exclude_fields: list[FieldKey] = PydanticField(default_factory=list)

    def resolve_field_deps(
        self,
        context: FieldsMappingResolutionContext,
    ) -> set[FieldKey]:
        res = set()

        for upstream_field_key in context.upstream_feature_fields:
            # Skip excluded fields
            if upstream_field_key in self.exclude_fields:
                continue

            # Check for exact match
            if upstream_field_key == context.field_key:
                res.add(upstream_field_key)
            # Check for suffix match if enabled
            elif self.match_suffix and self._is_suffix_match(context.field_key, upstream_field_key):
                res.add(upstream_field_key)

        # If no fields matched, return ALL fields from this upstream feature
        # (excluding any explicitly excluded fields)
        if not res:
            for upstream_field_key in context.upstream_feature_fields:
                if upstream_field_key not in self.exclude_fields:
                    res.add(upstream_field_key)

        return res

    def _is_suffix_match(self, field_key: FieldKey, upstream_field_key: FieldKey) -> bool:
        """Check if field_key is a suffix of upstream_field_key.

        For hierarchical keys like "audio/french", this checks if "french"
        matches the suffix.

        Args:
            field_key: The field key for which to resolve dependencies.
            upstream_fields_by_feature_key: Mapping of upstream feature keys to their fields.

        Returns:
            True if field_key is a suffix of upstream_field_key
        """
        # For single-part keys, check if it's the last part of a multi-part key
        if len(field_key.parts) == 1 and len(upstream_field_key.parts) > 1:
            return field_key.parts[0] == upstream_field_key.parts[-1]

        # For multi-part keys, check if all parts match as suffix
        if len(field_key.parts) <= len(upstream_field_key.parts):
            return upstream_field_key.parts[-len(field_key.parts) :] == field_key.parts

        return False


@public
class FieldsMapping(BaseModel):
    """Base class for field mapping configurations.

    Field mappings define how a field automatically resolves its dependencies
    based on upstream feature fields. This is separate from explicit field
    dependencies which are defined directly.
    """

    model_config = ConfigDict(frozen=True)
    # mapping: BaseFieldsMapping
    mapping: AllFieldsMapping | SpecificFieldsMapping | NoneFieldsMapping | DefaultFieldsMapping = PydanticField(
        ..., discriminator="type"
    )

    def resolve_field_deps(
        self,
        context: FieldsMappingResolutionContext,
    ) -> set[FieldKey]:
        """Resolve field dependencies based on upstream feature fields.

        Invokes the provided mapping to resolve dependencies.

        Args:
            context: The resolution context containing field key and upstream feature.

        Returns:
            Set of [FieldKey][metaxy.models.types.FieldKey] instances for matching fields
        """
        return self.mapping.resolve_field_deps(context)

    @classmethod
    def default(
        cls,
        *,
        match_suffix: bool = False,
        exclude_fields: list[FieldKey] | None = None,
    ) -> Self:
        """Create a default field mapping configuration.

        Args:
            match_suffix: If True, allows suffix matching (e.g., "french" matches "audio/french")
            exclude_fields: List of field keys to exclude from auto-mapping

        Returns:
            Configured FieldsMapping instance.
        """
        return cls(
            mapping=DefaultFieldsMapping(
                match_suffix=match_suffix,
                exclude_fields=exclude_fields or [],
            )
        )

    @classmethod
    def specific(cls, mapping: dict[CoercibleToFieldKey, set[CoercibleToFieldKey]]) -> Self:
        """Create a field mapping that maps downstream field keys into specific upstream field keys.

        Args:
            mapping: Mapping of downstream field keys to sets of upstream field keys.
                Keys and values can be strings, sequences, or FieldKey instances.

        Returns:
            Configured FieldsMapping instance.
        """
        # Validate and coerce the mapping keys and values
        validated_mapping: dict[FieldKey, set[FieldKey]] = {}
        for key, value_set in mapping.items():
            validated_key = ValidatedFieldKeyAdapter.validate_python(key)
            validated_values = {ValidatedFieldKeyAdapter.validate_python(v) for v in value_set}
            validated_mapping[validated_key] = validated_values

        return cls(mapping=SpecificFieldsMapping(mapping=validated_mapping))

    @classmethod
    def all(cls) -> Self:
        """Create a field mapping that explicitly depends on all upstream fields.

        Returns:
            Configured FieldsMapping instance.
        """
        return cls(mapping=AllFieldsMapping())

    @classmethod
    def none(cls) -> Self:
        """Create a field mapping that explicitly depends on no upstream fields.

        This is typically useful when explicitly defining [FieldSpec.deps][metaxy.models.field.FieldSpec] instead.

        Returns:
            Configured FieldsMapping instance.
        """
        return cls(mapping=NoneFieldsMapping())


FieldsMappingAdapter = TypeAdapter(AllFieldsMapping | SpecificFieldsMapping | NoneFieldsMapping | DefaultFieldsMapping)
