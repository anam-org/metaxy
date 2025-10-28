from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import Field as PydanticField
from pydantic import field_validator

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey

if TYPE_CHECKING:
    from metaxy.models.feature import Feature  # noqa: F401


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class FieldDep(FrozenBaseModel):
    """Field-level dependency specification.

    Attributes:
        feature_key: The feature to depend on. Accepts:
            - FeatureKey: Direct key object
            - Feature class: Extracts key from Feature.spec.key
            - FeatureSpec: Extracts key from spec.key
            - str: Converted to FeatureKey([str])
            - list[str]: Converted to FeatureKey(list)
        fields: Which fields from the feature to depend on.
            - SpecialFieldDep.ALL (default): All fields
            - list[FieldKey]: Specific fields

    Examples:
        >>> # Depend on all fields from a feature
        >>> FieldDep(feature_key=UpstreamFeature)

        >>> # Depend on specific fields (using strings)
        >>> FieldDep(
        ...     feature_key="upstream",
        ...     fields=["field1", "field2"]
        ... )

        >>> # Traditional explicit keys (still supported)
        >>> FieldDep(
        ...     feature_key=FeatureKey(["upstream"]),
        ...     fields=[FieldKey(["field1"])]
        ... )
    """

    feature_key: FeatureKey
    fields: list[FieldKey] | SpecialFieldDep = SpecialFieldDep.ALL

    @field_validator("feature_key", mode="before")
    @classmethod
    def _coerce_feature_key(cls, value: Any) -> FeatureKey:
        """Convert Feature class, FeatureSpec, str, or list[str] to FeatureKey."""
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


class FieldSpec(FrozenBaseModel):
    key: FieldKey = PydanticField(default_factory=lambda: FieldKey(["default"]))
    code_version: str = "1"

    # field-level dependencies can be one of the following:
    # - the default SpecialFieldDep.ALL to depend on all upstream features and all their fields
    # - a list of FieldDep to depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: Any) -> FieldKey:
        return _coerce_field_key(value)


def _coerce_feature_key(value: Any) -> FeatureKey:
    if isinstance(value, FeatureKey):
        return value
    if isinstance(value, str):
        return FeatureKey([value])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if not all(isinstance(part, str) for part in parts):
            raise TypeError("Feature key parts must be strings.")
        return FeatureKey(parts)
    raise TypeError("feature_key must be a FeatureKey, string, or sequence of strings.")


def _coerce_field_key(value: Any) -> FieldKey:
    if isinstance(value, FieldKey):
        return value
    if isinstance(value, str):
        return FieldKey([value])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if not all(isinstance(part, str) for part in parts):
            raise TypeError("Field key parts must be strings.")
        return FieldKey(parts)
    raise TypeError("field key must be a FieldKey, string, or sequence of strings.")
