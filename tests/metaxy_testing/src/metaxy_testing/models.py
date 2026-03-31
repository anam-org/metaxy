"""Testing models for Metaxy.

This module contains testing-specific implementations of core Metaxy classes
that are designed for testing and examples, not production use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, overload

import pydantic
from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec
from pydantic import BeforeValidator

if TYPE_CHECKING:
    from metaxy.models.feature_spec import (
        CoercibleToFeatureDep,
        FeatureDep,
        IDColumns,
        Unique,
    )
    from metaxy.models.types import CoercibleToFeatureKey
    from pydantic.types import JsonValue


# Type aliases
DefaultFeatureCols: TypeAlias = tuple[Literal["sample_uid"],]
TestingUIDCols: TypeAlias = tuple[str, ...]


def _validate_sample_feature_spec_id_columns(
    value: Any,
) -> tuple[str, ...]:
    """Coerce id_columns to tuple for SampleFeatureSpec."""
    if value is None:
        return ("sample_uid",)
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return (value,)
    return tuple(value)


class SampleFeatureSpec(FeatureSpec):
    """A testing implementation of FeatureSpec that has a `sample_uid` ID column. Has to be moved to tests."""

    id_columns: Annotated[
        tuple[str, ...],
        BeforeValidator(_validate_sample_feature_spec_id_columns),
    ] = pydantic.Field(
        default_factory=lambda: ("sample_uid",),
        min_length=1,
        description="Columns that uniquely identify a row. They will be used by Metaxy in joins.",
    )

    if TYPE_CHECKING:
        # Overload for common case: list of FeatureDep instances
        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns | None = None,
            deps: list[FeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            unique: Unique | Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns | None = None,
            deps: list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            unique: Unique | Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> None: ...

        # Implementation signature
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns | None = None,
            deps: list[FeatureDep] | list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            unique: Unique | Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> None: ...


class SampleFeature(BaseFeature, spec=None):
    """A testing implementation of BaseFeature with a sample_uid field.

    A default specialization of BaseFeature that uses a `sample_uid` ID column.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class
    sample_uid: str | None = None


__all__ = [
    "DefaultFeatureCols",
    "TestingUIDCols",
    "SampleFeatureSpec",
    "SampleFeature",
]
