from collections.abc import Mapping
from functools import cached_property

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import (
    BaseFeatureSpecWithIDColumns,
    FeatureKey,
)
from metaxy.models.field import (
    FieldDep,
    FieldKey,
    FieldSpec,
    SpecialFieldDep,
)


class FQFieldKey(FrozenBaseModel):
    field: FieldKey
    feature: FeatureKey

    def to_string(self) -> str:
        return f"{self.feature.to_string()}.{self.field.to_string()}"

    def __repr__(self) -> str:
        return self.to_string()

    def __lt__(self, other: "FQFieldKey") -> bool:
        """Enable sorting of FQFieldKey objects."""
        return self.to_string() < other.to_string()

    def __le__(self, other: "FQFieldKey") -> bool:
        """Enable sorting of FQFieldKey objects."""
        return self.to_string() <= other.to_string()

    def __gt__(self, other: "FQFieldKey") -> bool:
        """Enable sorting of FQFieldKey objects."""
        return self.to_string() > other.to_string()

    def __ge__(self, other: "FQFieldKey") -> bool:
        """Enable sorting of FQFieldKey objects."""
        return self.to_string() >= other.to_string()


class FeaturePlan(FrozenBaseModel):
    """Slice of the feature graph that includes a given feature and its parents"""

    feature: BaseFeatureSpecWithIDColumns
    deps: list[BaseFeatureSpecWithIDColumns] | None

    @cached_property
    def parent_features_by_key(
        self,
    ) -> Mapping[FeatureKey, BaseFeatureSpecWithIDColumns]:
        return {feature.key: feature for feature in self.deps or []}

    @cached_property
    def all_parent_fields_by_key(self) -> Mapping[FQFieldKey, FieldSpec]:
        res: dict[FQFieldKey, FieldSpec] = {}

        for feature in self.deps or []:
            for field in feature.fields:
                res[FQFieldKey(field=field.key, feature=feature.key)] = field

        return res

    @cached_property
    def parent_fields_by_key(self) -> Mapping[FQFieldKey, FieldSpec]:
        res: dict[FQFieldKey, FieldSpec] = {}

        for field in self.feature.fields:
            res.update(self.get_parent_fields_for_field(field.key))

        return res

    def get_parent_fields_for_field(
        self, key: FieldKey
    ) -> Mapping[FQFieldKey, FieldSpec]:
        res = {}

        field = self.feature.fields_by_key[key]

        if field.deps == SpecialFieldDep.ALL:
            # we depend on all upstream features and their fields
            for feature in self.deps or []:
                for field in feature.fields:
                    res[FQFieldKey(field=field.key, feature=feature.key)] = field
        elif isinstance(field.deps, list):
            for field_dep in field.deps:
                if field_dep.fields == SpecialFieldDep.ALL:
                    # we depend on all fields of the corresponding upstream feature
                    for parent_field in self.parent_features_by_key[
                        field_dep.feature_key
                    ].fields:
                        res[
                            FQFieldKey(
                                field=parent_field.key,
                                feature=field_dep.feature_key,
                            )
                        ] = parent_field

                elif isinstance(field_dep, FieldDep):
                    #
                    for field_key in field_dep.fields:
                        fq_key = FQFieldKey(
                            field=field_key,
                            feature=field_dep.feature_key,
                        )
                        res[fq_key] = self.all_parent_fields_by_key[fq_key]
                else:
                    raise ValueError(f"Unsupported dependency type: {type(field_dep)}")
        else:
            raise TypeError(f"Unsupported dependencies type: {type(field.deps)}")

        return res

    @cached_property
    def field_dependencies(
        self,
    ) -> Mapping[FieldKey, Mapping[FeatureKey, list[FieldKey]]]:
        """Get dependencies for each field in this feature.

        Returns a mapping from field key to its upstream dependencies.
        Each dependency maps an upstream feature key to a list of field keys
        that this field depends on.

        This is the format needed by DataVersionResolver.

        Returns:
            Mapping of field keys to their dependency specifications.
            Format: {field_key: {upstream_feature_key: [upstream_field_keys]}}
        """
        result: dict[FieldKey, dict[FeatureKey, list[FieldKey]]] = {}

        for field in self.feature.fields:
            field_deps: dict[FeatureKey, list[FieldKey]] = {}

            if field.deps == SpecialFieldDep.ALL:
                # Depend on all upstream features and all their fields
                for upstream_feature in self.deps or []:
                    field_deps[upstream_feature.key] = [
                        c.key for c in upstream_feature.fields
                    ]
            elif isinstance(field.deps, list):
                # Specific dependencies defined
                for field_dep in field.deps:
                    feature_key = field_dep.feature_key

                    if field_dep.fields == SpecialFieldDep.ALL:
                        # All fields from this upstream feature
                        upstream_feature_spec = self.parent_features_by_key[feature_key]
                        field_deps[feature_key] = [
                            c.key for c in upstream_feature_spec.fields
                        ]
                    elif isinstance(field_dep.fields, list):
                        # Specific fields
                        field_deps[feature_key] = field_dep.fields

            result[field.key] = field_deps

        return result
