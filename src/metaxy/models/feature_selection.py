"""Feature selection for declaring external feature dependencies."""

from __future__ import annotations

from collections.abc import Sequence

import pydantic

from metaxy._decorators import public
from metaxy.models.bases import FrozenBaseModel


@public
class FeatureSelection(FrozenBaseModel):
    """Selects a set of features from a metadata store.

    Fields can be combined — both `projects` and `keys` may be set simultaneously
    to select all features from those projects *plus* the individually listed keys.

    Set `all=True` to select every feature in the store.

    Supports set operators `|`, `&`, and `-` which merge the underlying fields.
    """

    projects: Sequence[str] | None = None
    keys: Sequence[str] | None = None
    all: bool | None = None

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


def _union(a: Sequence[str] | None, b: Sequence[str] | None) -> Sequence[str] | None:
    if a is None:
        return b
    if b is None:
        return a
    return sorted(set(a) | set(b))


def _intersect(a: Sequence[str] | None, b: Sequence[str] | None) -> Sequence[str] | None:
    if a is None or b is None:
        return None
    return sorted(set(a) & set(b)) or None


def _subtract(a: Sequence[str] | None, b: Sequence[str] | None) -> Sequence[str] | None:
    if a is None or b is None:
        return a
    return sorted(set(a) - set(b)) or None
