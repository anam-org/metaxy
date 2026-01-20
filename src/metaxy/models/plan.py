from collections.abc import Mapping
from functools import cached_property

import pydantic

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureDep, FeatureKey, FeatureSpec
from metaxy.models.field import (
    FieldDep,
    FieldKey,
    FieldSpec,
    SpecialFieldDep,
)
from metaxy.models.fields_mapping import FieldsMappingResolutionContext
from metaxy.models.lineage import (
    AggregationRelationship,
    ExpansionRelationship,
    LineageRelationshipType,
)
from metaxy.models.types import CoercibleToFieldKey, ValidatedFieldKeyAdapter

# Rebuild the model now that FeatureSpec is available
FieldsMappingResolutionContext.model_rebuild()


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

    feature: pydantic.SkipValidation[FeatureSpec]
    deps: pydantic.SkipValidation[list[FeatureSpec] | None]
    feature_deps: list[FeatureDep] | None = None  # The actual dependency specifications with field mappings

    @cached_property
    def parent_features_by_key(
        self,
    ) -> Mapping[FeatureKey, FeatureSpec]:
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

    @cached_property
    def parent_fields_by_feature_key(self) -> Mapping[FeatureKey, set[FieldKey]]:
        res: dict[FeatureKey, set[FieldKey]] = {}

        if self.deps:
            for feature in self.deps:
                res[feature.key] = set([f.key for f in feature.fields])

        return res

    def get_parent_fields_for_field(self, key: CoercibleToFieldKey) -> Mapping[FQFieldKey, FieldSpec]:
        """Get parent fields for a given field key.

        Args:
            key: Field key to get parent fields for. Accepts string, sequence, or FieldKey.

        Returns:
            Mapping of fully qualified field keys to their specs.
        """
        # Validate and coerce the key
        validated_key = ValidatedFieldKeyAdapter.validate_python(key)

        res = {}

        field = self.feature.fields_by_key[validated_key]

        # Get resolved dependencies (combining automatic mapping and explicit deps)
        resolved_deps = self._resolve_field_deps(field)

        for field_dep in resolved_deps:
            if field_dep.fields == SpecialFieldDep.ALL:
                # we depend on all fields of the corresponding upstream feature
                for parent_field in self.parent_features_by_key[field_dep.feature].fields:
                    res[
                        FQFieldKey(
                            field=parent_field.key,
                            feature=field_dep.feature,
                        )
                    ] = parent_field

            elif isinstance(field_dep, FieldDep):
                #
                for field_key in field_dep.fields:
                    fq_key = FQFieldKey(
                        field=field_key,
                        feature=field_dep.feature,
                    )
                    res[fq_key] = self.all_parent_fields_by_key[fq_key]
            else:
                raise ValueError(f"Unsupported dependency type: {type(field_dep)}")

        return res

    def _resolve_field_deps(self, field: FieldSpec) -> list[FieldDep]:
        """Resolve field dependencies by combining explicit deps and automatic mapping.

        Apply field mappings from the FeatureDep and add explicit deps.
        """

        if not self.feature_deps:
            return []

        # Check if field has explicit deps
        if field.deps and field.deps != []:  # Check for non-empty list
            if isinstance(field.deps, SpecialFieldDep):
                # If it's SpecialFieldDep.ALL, return ALL for all upstream features
                return [FieldDep(feature=dep.key, fields=SpecialFieldDep.ALL) for dep in (self.deps or [])]
            else:
                # Use only the explicit deps, no automatic mapping
                return field.deps

        # No explicit deps - use automatic mapping
        field_deps = []

        for feature_dep in self.feature_deps:
            # Resolve field mapping for this specific upstream feature
            # Get the upstream feature spec
            upstream_feature = self.parent_features_by_key.get(feature_dep.feature)
            if not upstream_feature:
                continue

            # Create resolution context
            context = FieldsMappingResolutionContext(field_key=field.key, upstream_feature=upstream_feature)

            mapped_deps = feature_dep.fields_mapping.resolve_field_deps(context)

            if mapped_deps:
                # Add a single FieldDep with all mapped fields
                field_deps.append(FieldDep(feature=feature_dep.feature, fields=list(mapped_deps)))
            # Note: If mapped_deps is empty (e.g., feature excluded),
            # we don't add any dependency for this feature

        if field_deps:
            return field_deps
        else:
            raise RuntimeError(
                f"No upstream fields found for field {field} of feature {self.feature}. Please either specify explicit dependencies on it's FieldSpec or ensure that at least one FeatureDep on the FeatureSpec has a valid field mapping."
            )

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

            # Get resolved dependencies (combining automatic mapping and explicit deps)
            resolved_deps = self._resolve_field_deps(field)

            # Specific dependencies defined
            for field_dep in resolved_deps:
                feature_key = field_dep.feature

                if field_dep.fields == SpecialFieldDep.ALL:
                    # All fields from this upstream feature
                    upstream_feature_spec = self.parent_features_by_key[feature_key]
                    field_deps[feature_key] = [c.key for c in upstream_feature_spec.fields]
                elif isinstance(field_dep.fields, list):
                    # Specific fields
                    field_deps[feature_key] = field_dep.fields

            result[field.key] = field_deps

        return result

    @cached_property
    def upstream_id_columns(self) -> list[str]:
        """Union of all upstream ID columns after renames.

        This is the set of columns used to join multiple upstream features.
        Each upstream feature's id_columns are renamed according to FeatureDep.rename
        before being combined.

        Returns:
            List of column names (order not guaranteed).
        """
        if not self.feature_deps or not self.deps:
            return []

        cols: set[str] = set()
        deps_by_key = {dep.key: dep for dep in self.deps}

        for feature_dep in self.feature_deps:
            upstream_spec = deps_by_key.get(feature_dep.feature)
            if upstream_spec is None:
                continue

            renames = feature_dep.rename or {}
            for col in upstream_spec.id_columns:
                renamed_col = renames.get(col, col)
                cols.add(renamed_col)

        return list(cols)

    def get_input_id_columns_for_dep(self, feature_dep: FeatureDep) -> list[str]:
        """Get the input ID columns for a specific dependency after lineage is applied.

        The returned columns represent the logical unit for this dependency based on
        its lineage relationship:

        - Identity (1:1): Same as upstream ID columns (after renames)
        - Aggregation (N:1): Aggregation columns (each group is a unit)
        - Expansion (1:N): Parent columns from ExpansionRelationship.on

        Args:
            feature_dep: The dependency to get input ID columns for.

        Returns:
            List of column names that define a logical input unit for this dependency.
        """
        relationship = feature_dep.lineage.relationship
        relationship_type = relationship.type

        # Get upstream spec for this dependency
        upstream_spec = self.parent_features_by_key.get(feature_dep.feature)
        if upstream_spec is None:
            return []

        # Apply renames to upstream ID columns
        renames = feature_dep.rename or {}
        renamed_upstream_id_cols = [renames.get(col, col) for col in upstream_spec.id_columns]

        if relationship_type == LineageRelationshipType.IDENTITY:
            return renamed_upstream_id_cols

        elif relationship_type == LineageRelationshipType.AGGREGATION:
            assert isinstance(relationship, AggregationRelationship)
            agg_result = relationship.get_aggregation_columns(list(self.feature.id_columns))
            assert agg_result is not None, "Aggregation relationship must have aggregation columns"
            return list(agg_result)

        elif relationship_type == LineageRelationshipType.EXPANSION:
            assert isinstance(relationship, ExpansionRelationship)
            return list(relationship.on)

        else:
            raise ValueError(f"Unknown lineage relationship type: {relationship_type}")

    @cached_property
    def input_id_columns(self) -> list[str]:
        """Columns that uniquely identify an input sample.

        The "input" is the joined upstream metadata after FeatureDep rules and lineage
        transformations are applied. For features with multiple dependencies, this is
        the intersection of input ID columns from all dependencies.

        Returns:
            List of column names that define a logical input unit.
        """
        if not self.feature_deps:
            return []

        # Collect input ID columns from each dependency
        all_input_cols: list[set[str]] = []
        for feature_dep in self.feature_deps:
            cols = self.get_input_id_columns_for_dep(feature_dep)
            if cols:
                all_input_cols.append(set(cols))

        if not all_input_cols:
            return []

        # Return intersection of all input ID columns
        result = all_input_cols[0]
        for cols in all_input_cols[1:]:
            result = result.intersection(cols)

        return list(result)

    @cached_property
    def optional_deps(self) -> list[FeatureDep]:
        """Dependencies marked as optional (use left join)."""
        return [dep for dep in (self.feature_deps or []) if dep.optional]

    @cached_property
    def required_deps(self) -> list[FeatureDep]:
        """Dependencies that are required (use inner join)."""
        return [dep for dep in (self.feature_deps or []) if not dep.optional]

    @pydantic.model_validator(mode="after")
    def _validate(self) -> "FeaturePlan":
        """Validate the column configuration of this plan.

        Checks for issues that would cause runtime errors:
        - Columns renamed to system column names
        - Columns renamed to upstream ID column names
        - Duplicate rename targets within a single dependency
        - Duplicate column names across dependencies (except allowed ID columns)
        """
        from metaxy.versioning.validation import validate_column_configuration

        validate_column_configuration(self)
        return self
