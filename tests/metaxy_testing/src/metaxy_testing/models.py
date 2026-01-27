"""Testing models for Metaxy.

This module contains testing-specific implementations of core Metaxy classes
that are designed for testing and examples, not production use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, overload

import pydantic
from pydantic import BeforeValidator

from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec

if TYPE_CHECKING:
    from pydantic.types import JsonValue

    from metaxy.models.feature_spec import (
        CoercibleToFeatureDep,
        FeatureDep,
        IDColumns,
    )
    from metaxy.models.types import CoercibleToFeatureKey


# Type aliases
DefaultFeatureCols: TypeAlias = tuple[Literal["sample_uid"],]
TestingUIDCols: TypeAlias = list[str]


def _validate_sample_feature_spec_id_columns(
    value: Any,
) -> list[str]:
    """Coerce id_columns to list for SampleFeatureSpec."""
    if value is None:
        return ["sample_uid"]
    if isinstance(value, list):
        return value
    return list(value)


class SampleFeatureSpec(FeatureSpec):
    """A testing implementation of FeatureSpec that has a `sample_uid` ID column. Has to be moved to tests."""

    id_columns: Annotated[
        pydantic.SkipValidation[list[str]],
        BeforeValidator(_validate_sample_feature_spec_id_columns),
    ] = pydantic.Field(
        default_factory=lambda: ["sample_uid"],
        description="List of columns that uniquely identify a row. They will be used by Metaxy in joins.",
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
