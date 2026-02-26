"""Feature selection for declaring external feature dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any

import pydantic
from pydantic import BeforeValidator

from metaxy._decorators import public
from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import (
    CoercibleToFeatureKey,
    FeatureKey,
    ValidatedFeatureKeySequenceAdapter,
)

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT


def _validate_keys(value: Any) -> Sequence[FeatureKey]:
    return ValidatedFeatureKeySequenceAdapter.validate_python(value)


@public
class FeatureSelection(FrozenBaseModel):
    """Selects a set of features from a metadata store.

    Fields can be combined — both `projects` and `keys` may be set simultaneously
    to select all features from those projects *plus* the individually listed keys.

    Set `all=True` to select every feature in the store.

    Supports set operators `|`, `&`, and `-` which merge the underlying fields.

    Examples:
        >>> mx.FeatureSelection(projects=["my-project"])
        FeatureSelection(projects=['my-project'], keys=None, all=None)

        >>> mx.FeatureSelection(keys=["raw/video", mx.FeatureKey("ml/embeddings")])
        FeatureSelection(projects=None, keys=[raw/video, ml/embeddings], all=None)

        >>> mx.FeatureSelection(all=True)
        FeatureSelection(projects=None, keys=None, all=True)

        >>> mx.FeatureSelection(projects=["a"]) | mx.FeatureSelection(projects=["b"])
        FeatureSelection(projects=['a', 'b'], keys=None, all=None)

        >>> mx.FeatureSelection(keys=["a/b", "c/d"]) & mx.FeatureSelection(keys=["c/d"])
        FeatureSelection(projects=None, keys=[c/d], all=None)

        >>> mx.FeatureSelection(projects=["a", "b", "c"]) - mx.FeatureSelection(projects=["b"])
        FeatureSelection(projects=['a', 'c'], keys=None, all=None)
    """

    projects: Sequence[str] | None = None
    keys: Annotated[Sequence[FeatureKey], BeforeValidator(_validate_keys)] | None = None
    all: bool | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            projects: Sequence[str] | None = None,
            keys: Sequence[CoercibleToFeatureKey] | None = None,
            all: bool | None = None,
        ) -> None: ...

    @pydantic.model_validator(mode="after")
    def _at_least_one_mode(self) -> FeatureSelection:
        if self.all:
            return self
        if not self.projects and not self.keys:
            msg = "At least one of 'projects', 'keys', or 'all' must be specified"
            raise ValueError(msg)
        return self

    def __or__(self, other: FeatureSelection) -> FeatureSelection:
        if self.all or other.all:
            return FeatureSelection(all=True)
        return FeatureSelection(
            projects=_union(self.projects, other.projects),
            keys=_union(self.keys, other.keys),
        )

    def __and__(self, other: FeatureSelection) -> FeatureSelection:
        if self.all:
            return other
        if other.all:
            return self
        return FeatureSelection(
            projects=_intersect(self.projects, other.projects),
            keys=_intersect(self.keys, other.keys),
        )

    def __sub__(self, other: FeatureSelection) -> FeatureSelection:
        if other.all:
            return FeatureSelection(keys=[])
        return FeatureSelection(
            projects=_subtract(self.projects, other.projects),
            keys=_subtract(self.keys, other.keys),
        )


def _union(
    a: Sequence[SupportsRichComparisonT] | None, b: Sequence[SupportsRichComparisonT] | None
) -> Sequence[SupportsRichComparisonT] | None:
    if a is None:
        return b
    if b is None:
        return a
    return sorted(set(a) | set(b))


def _intersect(
    a: Sequence[SupportsRichComparisonT] | None, b: Sequence[SupportsRichComparisonT] | None
) -> Sequence[SupportsRichComparisonT] | None:
    if a is None or b is None:
        return None
    return sorted(set(a) & set(b)) or None


def _subtract(
    a: Sequence[SupportsRichComparisonT] | None, b: Sequence[SupportsRichComparisonT] | None
) -> Sequence[SupportsRichComparisonT] | None:
    if a is None or b is None:
        return a
    return sorted(set(a) - set(b)) or None
