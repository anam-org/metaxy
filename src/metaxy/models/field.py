from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel, TypeAdapter
from pydantic import Field as PydanticField

from metaxy.models.types import (
    CoercibleToFieldKey,
    FeatureKey,
    FeatureKeyAdapter,
    FieldKey,
    FieldKeyAdapter,
)

if TYPE_CHECKING:
    # yes, these are circular imports, the TYPE_CHECKING block hides them at runtime.
    # neither pyright not basedpyright allow ignoring `reportImportCycles` because they think it's a bad practice
    # and it would be very smart to force the user to restructure their project instead
    # context: https://github.com/microsoft/pyright/issues/1825
    # however, considering the recursive nature of graphs, and the syntactic sugar that we want to support,
    # I decided to just put these errors into `.basedpyright/baseline.json` (after ensuring this is the only error produced by basedpyright)
    from metaxy.models.feature import Feature
    from metaxy.models.feature_spec import (
        CoercibleToFeatureKey,
        FeatureDep,
        FeatureSpec,
    )


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class DefaultFieldsMapping(BaseModel):
    """Automatic field mapping configuration.

    When used in place of explicit FieldDep, automatically maps fields
    to matching upstream fields based on field keys.

    Attributes:
        match_suffix: If True, allows suffix matching (e.g., "french" matches "audio/french")
        exclude_features: List of feature keys to exclude from auto-mapping
        exclude_fields: List of field keys to exclude from auto-mapping
        mapping_hook: Optional callable to customize field mapping logic.
                     Receives (field_key, potential_matches) and returns filtered/modified matches.

    Examples:
        >>> # Exact match only (default)
        >>> DefaultFieldsMapping()

        >>> # Enable suffix matching
        >>> DefaultFieldsMapping(match_suffix=True)

        >>> # Exclude specific upstream features
        >>> DefaultFieldsMapping(exclude_features=[FeatureKey(["some", "feature"])])

        >>> # Exclude specific fields from being auto-mapped
        >>> DefaultFieldsMapping(exclude_fields=[FieldKey(["metadata"])])

        >>> # Custom mapping logic
        >>> def my_hook(field_key, matches):
        ...     # Only keep matches from specific features
        ...     return [(fk, fld) for fk, fld in matches if "important" in fk.to_string()]
        >>> DefaultFieldsMapping(mapping_hook=my_hook)
    """

    match_suffix: bool = False
    exclude_features: list[FeatureKey] = PydanticField(default_factory=list)
    exclude_fields: list[FieldKey] = PydanticField(default_factory=list)
    mapping_hook: (
        Callable[
            [FieldKey, list[tuple[FeatureKey, FieldKey]]],
            list[tuple[FeatureKey, FieldKey]],
        ]
        | None
    ) = None

    def resolve_field_deps(
        self,
        field_key: FieldKey,
        feature_deps: list["FeatureDep"] | None,
    ) -> "list[FieldDep] | SpecialFieldDep":
        """Resolve automatic field mapping to explicit FieldDep list.

        Args:
            field_key: The field key to map
            feature_deps: Feature-level dependencies to search for matching fields

        Returns:
            List of FieldDep instances for matching fields, or SpecialFieldDep.ALL
            if no specific mappings are found.

        Raises:
            ValueError: If ambiguous mappings are found or no feature deps exist
        """
        if not feature_deps:
            # When no feature deps exist, fallback to ALL for backward compatibility
            # This allows DefaultFieldsMapping to be the default without breaking root features
            return SpecialFieldDep.ALL

        # Import here to avoid circular dependency at module level
        from metaxy.models.feature import FeatureGraph

        # Get the active feature graph to look up upstream features
        graph = FeatureGraph.get_active()

        # Track all potential matches
        matches: list[tuple[FeatureKey, FieldKey]] = []

        for feature_dep in feature_deps:
            # Skip excluded features
            if feature_dep.feature in self.exclude_features:
                continue

            # Get the upstream feature spec
            try:
                upstream_feature = graph.get_feature_by_key(feature_dep.feature)
                upstream_spec = upstream_feature.spec()
            except KeyError:
                # Upstream feature not registered yet, skip
                continue

            # Check each field in the upstream feature
            for upstream_field_key in upstream_spec.fields_by_key.keys():
                # Skip excluded fields
                if upstream_field_key in self.exclude_fields:
                    continue

                # Check for exact match
                if upstream_field_key == field_key:
                    matches.append((feature_dep.feature, upstream_field_key))
                # Check for suffix match if enabled
                elif self.match_suffix and self._is_suffix_match(
                    field_key, upstream_field_key
                ):
                    matches.append((feature_dep.feature, upstream_field_key))

        # Apply custom mapping hook if provided
        if self.mapping_hook is not None:
            matches = self.mapping_hook(field_key, matches)

        # Handle results
        if len(matches) == 0:
            # No matches found - could mean no matching fields or upstream features not registered yet
            # Return SpecialFieldDep.ALL as fallback (maintains backward compatibility)
            return SpecialFieldDep.ALL

        # Group fields by feature - a field can naturally map to multiple upstream features
        # This is not ambiguous, it's a normal pattern (e.g., combining audio from multiple sources)
        fields_by_feature: dict[FeatureKey, list[FieldKey]] = {}
        for feature_key, field_key in matches:
            if feature_key not in fields_by_feature:
                fields_by_feature[feature_key] = []
            fields_by_feature[feature_key].append(field_key)

        # Create FieldDep instances
        field_deps = []
        for feature_key, field_keys in fields_by_feature.items():
            field_deps.append(
                FieldDep(
                    feature=feature_key,
                    fields=field_keys,
                )
            )

        return field_deps

    def _is_suffix_match(
        self, field_key: FieldKey, upstream_field_key: FieldKey
    ) -> bool:
        """Check if field_key is a suffix of upstream_field_key.

        For hierarchical keys like "audio/french", this checks if "french"
        matches the suffix.

        Args:
            field_key: The field key to match (e.g., FieldKey(["french"]))
            upstream_field_key: The upstream field key (e.g., FieldKey(["audio", "french"]))

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


class FieldDep(BaseModel):
    feature: FeatureKey
    fields: list[FieldKey] | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL

    @overload
    def __init__(
        self,
        feature: str,
        **kwargs: Any,
    ) -> None:
        """Initialize from string feature key."""
        ...

    @overload
    def __init__(
        self,
        feature: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        feature: FeatureKey,
        **kwargs: Any,
    ) -> None:
        """Initialize from FeatureKey instance."""
        ...

    @overload
    def __init__(
        self,
        feature: "CoercibleToFeatureKey",
        **kwargs: Any,
    ) -> None:
        """Initialize from CoercibleToFeatureKey types."""
        ...

    @overload
    def __init__(
        self,
        feature: "FeatureSpec",
        **kwargs: Any,
    ) -> None:
        """Initialize from FeatureSpec instance."""
        ...

    @overload
    def __init__(
        self,
        feature: type["Feature"],
        **kwargs: Any,
    ) -> None:
        """Initialize from Feature instance."""
        ...

    def __init__(
        self,
        feature: "CoercibleToFeatureKey | FeatureSpec | type[Feature]",
        fields: list[CoercibleToFieldKey]
        | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL,
        *args,
        **kwargs,
    ):
        from metaxy.models.feature import Feature
        from metaxy.models.feature_spec import FeatureSpec

        if isinstance(feature, FeatureSpec):
            feature_key = feature.key
        elif isinstance(feature, type) and issubclass(feature, Feature):
            feature_key = feature.spec().key
        else:
            feature_key = FeatureKeyAdapter.validate_python(feature)

        assert isinstance(feature_key, FeatureKey)

        if isinstance(fields, list):
            validated_fields: Any = TypeAdapter(list[FieldKey]).validate_python(fields)
        else:
            validated_fields = fields  # Keep the enum value as-is

        super().__init__(feature=feature_key, fields=validated_fields, *args, **kwargs)


class FieldSpec(BaseModel):
    key: FieldKey = PydanticField(default_factory=lambda: FieldKey(["default"]))
    code_version: int = 1

    # field-level dependencies can be one of the following:
    # - DefaultFieldsMapping for automatic field mapping (default - auto-maps to matching fields, falls back to ALL)
    # - SpecialFieldDep.ALL to explicitly depend on all upstream features and all their fields
    # - a list of FieldDep to depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] | DefaultFieldsMapping = PydanticField(
        default_factory=DefaultFieldsMapping
    )

    @overload
    def __init__(
        self,
        key: str,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] | DefaultFieldsMapping | None = None,
    ) -> None:
        """Initialize from string key."""
        ...

    @overload
    def __init__(
        self,
        key: Sequence[str],
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] | DefaultFieldsMapping | None = None,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        key: FieldKey,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] | DefaultFieldsMapping | None = None,
    ) -> None:
        """Initialize from FieldKey instance."""
        ...

    @overload
    def __init__(
        self,
        key: None,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] | DefaultFieldsMapping | None = None,
    ) -> None:
        """Initialize with None key (uses default)."""
        ...

    def __init__(
        self,
        key: CoercibleToFieldKey | None,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] | DefaultFieldsMapping | None = None,
        *args,
        **kwargs: Any,
    ) -> None:
        if key is None:
            validated_key = FieldKey(["default"])
        else:
            validated_key = FieldKeyAdapter.validate_python(key)

        # If deps is None, use the default factory (DefaultFieldsMapping())
        if deps is None:
            deps = DefaultFieldsMapping()

        super().__init__(
            key=validated_key,
            code_version=code_version,
            deps=deps,
            *args,
            **kwargs,
        )
