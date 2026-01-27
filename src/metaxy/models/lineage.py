"""Lineage relationship types for feature dependencies.

This module defines how features relate to their upstream dependencies in terms of
cardinality and transformation patterns. These types make explicit the relationship
between parent and child features, enabling proper provenance aggregation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from typing_extensions import Self

from metaxy._decorators import public


@public
class LineageRelationshipType(str, Enum):
    """Type of lineage relationship between features."""

    IDENTITY = "1:1"
    AGGREGATION = "N:1"
    EXPANSION = "1:N"


class BaseLineageRelationship(BaseModel, ABC):
    """Base class for lineage relationship configurations.

    Lineage relationships define the cardinality and transformation pattern
    between a child feature and its upstream dependencies.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def get_aggregation_columns(
        self,
        target_id_columns: Sequence[str],
    ) -> Sequence[str] | None:
        """Get columns to aggregate on for this relationship type.

        Args:
            target_id_columns: The target feature's ID columns.

        Returns:
            Columns to group by for aggregation, or None if no aggregation needed.
        """
        raise NotImplementedError


@public
class IdentityRelationship(BaseLineageRelationship):
    """One-to-one relationship where each child row maps to exactly one parent row.

    This is the default relationship type. Parent and child features share the same
    ID columns and have the same cardinality.

    Construct this relationship via [`LineageRelationship.identity`][metaxy.models.lineage.LineageRelationship.identity] classmethod.

    Example:
        ```python
        mx.LineageRelationship.identity()
        ```
    """

    type: Literal[LineageRelationshipType.IDENTITY] = LineageRelationshipType.IDENTITY

    def get_aggregation_columns(
        self,
        target_id_columns: Sequence[str],
    ) -> Sequence[str] | None:
        """No aggregation needed for identity relationships."""
        return None


@public
class AggregationRelationship(BaseLineageRelationship):
    """Many-to-one relationship where multiple parent rows aggregate to one child row.

    Parent features have more granular ID columns than the child. The child aggregates
    multiple parent rows by grouping on a subset of the parent's ID columns.

    Construct this relationship via [`LineageRelationship.aggregation`][metaxy.models.lineage.LineageRelationship.aggregation] classmethod.

    Attributes:
        on: Columns to group by for aggregation. These should be a subset of the
            target feature's ID columns. If not specified, uses all target ID columns.

    Example:
        ```python
        mx.LineageRelationship.aggregation(on=["sensor_id", "hour"])
        ```
    """

    type: Literal[LineageRelationshipType.AGGREGATION] = LineageRelationshipType.AGGREGATION
    on: Sequence[str] | None = PydanticField(
        default=None,
        description="Columns to group by for aggregation. Defaults to all target ID columns.",
    )

    def get_aggregation_columns(
        self,
        target_id_columns: Sequence[str],
    ) -> Sequence[str]:
        """Get columns to aggregate on."""
        return self.on if self.on is not None else target_id_columns


@public
class ExpansionRelationship(BaseLineageRelationship):
    """One-to-many relationship where one parent row expands to multiple child rows.

    Child features have more granular ID columns than the parent. Each parent row
    generates multiple child rows with additional ID columns.

    Construct this relationship via [`LineageRelationship.expansion`][metaxy.models.lineage.LineageRelationship.expansion] classmethod.

    Attributes:
        on: Parent ID columns that identify the parent record. Child records with
            the same parent IDs will share the same upstream provenance.
            If not specified, will be inferred from the available columns.
        id_generation_pattern: Optional pattern for generating child IDs.
            Can be "sequential", "hash", or a custom pattern. If not specified,
            the feature's load_input() method is responsible for ID generation.

    Example:
        ```python
        mx.LineageRelationship.expansion(on=["video_id"], id_generation_pattern="sequential")
        ```
    """

    type: Literal[LineageRelationshipType.EXPANSION] = LineageRelationshipType.EXPANSION
    on: Sequence[str] = PydanticField(
        ...,
        description="Parent ID columns for grouping. Child records with same parent IDs share provenance. Required for expansion relationships.",
    )
    id_generation_pattern: str | None = PydanticField(
        default=None,
        description="Pattern for generating child IDs. If None, handled by load_input().",
    )

    def get_aggregation_columns(
        self,
        target_id_columns: Sequence[str],
    ) -> Sequence[str] | None:
        """Get aggregation columns for the joiner phase.

        For expansion relationships, returns None because aggregation
        happens during diff resolution, not during joining. The joiner
        should pass through all parent records without aggregation.

        Args:
            target_id_columns: The target (child) feature's ID columns.

        Returns:
            None - no aggregation during join phase for expansion relationships.
        """
        # Expansion relationships don't aggregate during join phase
        # Aggregation happens later during diff resolution
        return None


# Discriminated union type for all lineage relationships
LineageRelationshipUnion = IdentityRelationship | AggregationRelationship | ExpansionRelationship


@public
class LineageRelationship(BaseModel):
    """Wrapper class for lineage relationship configurations with convenient constructors.

    This provides a cleaner API for creating lineage relationships while maintaining
    type safety through discriminated unions.
    """

    model_config = ConfigDict(frozen=True)

    relationship: LineageRelationshipUnion = PydanticField(..., discriminator="type")

    @classmethod
    def identity(cls) -> Self:
        """Create an identity (1:1) relationship.

        Returns:
            Configured LineageRelationship for 1:1 relationship.

        Example:
            ```python
            mx.LineageRelationship.identity()
            ```
        """
        return cls(relationship=IdentityRelationship())

    @classmethod
    def aggregation(cls, on: Sequence[str] | None = None) -> Self:
        """Create an aggregation (N:1) relationship.

        Args:
            on: Columns to group by for aggregation. If None, uses all target ID columns.

        Returns:
            Configured LineageRelationship for N:1 relationship.

        Example:
            ```py
            mx.LineageRelationship.aggregation(on=["sensor_id", "hour"])
            ```
        """
        return cls(relationship=AggregationRelationship(on=on))

    @classmethod
    def expansion(
        cls,
        on: Sequence[str],
        id_generation_pattern: str | None = None,
    ) -> Self:
        """Create an expansion (1:N) relationship.

        Args:
            on: Parent ID columns that identify the parent record. Child records with
                the same parent IDs will share the same upstream provenance.
                Required - must explicitly specify which columns link parent to child.
            id_generation_pattern: Pattern for generating child IDs.
                Can be "sequential", "hash", or custom. If None, handled by load_input().

        Returns:
            Configured LineageRelationship for 1:N relationship.

        Example:
            ```py
            mx.LineageRelationship.expansion(on=["video_id"], id_generation_pattern="sequential")
            ```
        """
        return cls(relationship=ExpansionRelationship(on=on, id_generation_pattern=id_generation_pattern))

    def get_aggregation_columns(self, target_id_columns: Sequence[str]) -> Sequence[str] | None:
        """Get columns to aggregate on for this relationship.

        Args:
            target_id_columns: The target feature's ID columns.

        Returns:
            Columns to group by for aggregation, or None if no aggregation needed.
        """
        return self.relationship.get_aggregation_columns(target_id_columns)
