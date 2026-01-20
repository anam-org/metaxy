"""Validation utilities for feature dependency configuration.

Validates column naming, renames, and potential collisions at feature definition time.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from metaxy.models.constants import ALL_SYSTEM_COLUMNS
from metaxy.models.feature_spec import FeatureDep
from metaxy.versioning.feature_dep_transformer import FeatureDepTransformer

if TYPE_CHECKING:
    from metaxy.models.plan import FeaturePlan


def validate_column_configuration(plan: FeaturePlan) -> None:
    """Validate the column configuration of a FeaturePlan.

    Checks for issues that would cause runtime errors:
    - Columns renamed to system column names
    - Columns renamed to upstream ID column names
    - Duplicate rename targets within a single dependency
    - Duplicate column names across dependencies (except allowed ID columns)

    Args:
        plan: The FeaturePlan to validate

    Raises:
        ValueError: If any validation check fails
    """
    if not plan.feature_deps:
        return

    # First, validate each dependency individually for basic rename errors
    _validate_individual_dep_renames(plan)

    # Then validate cross-dependency column collisions
    _validate_no_column_collisions(plan)


def _validate_individual_dep_renames(plan: FeaturePlan) -> None:
    """Validate rename mappings for each dependency individually."""
    for dep in plan.feature_deps or []:
        if not isinstance(dep, FeatureDep):
            continue

        if not dep.rename:
            continue

        upstream_spec = plan.parent_features_by_key.get(dep.feature)
        upstream_id_columns = upstream_spec.id_columns if upstream_spec else []

        for old_name, new_name in dep.rename.items():
            # Check for renaming to system columns
            if new_name in ALL_SYSTEM_COLUMNS:
                raise ValueError(
                    f"Cannot rename column '{old_name}' to system column name '{new_name}' "
                    f"in dependency '{dep.feature.to_string()}'. "
                    f"System columns: {sorted(ALL_SYSTEM_COLUMNS)}"
                )

            # Check against upstream feature's ID columns
            if new_name in upstream_id_columns:
                raise ValueError(
                    f"Cannot rename column '{old_name}' to ID column '{new_name}' "
                    f"from upstream feature '{dep.feature.to_string()}'. "
                    f"ID columns for '{dep.feature.to_string()}': {upstream_id_columns}"
                )

        # Check for duplicate rename targets within this dependency
        renamed_values = list(dep.rename.values())
        if len(renamed_values) != len(set(renamed_values)):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for name in renamed_values:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            raise ValueError(
                f"Duplicate column names after renaming in dependency '{dep.feature.to_string()}': "
                f"{sorted(duplicates)}. Cannot rename multiple columns to the same name within a single dependency."
            )


def _validate_no_column_collisions(plan: FeaturePlan) -> None:
    """Validate that no non-ID columns collide across dependencies."""
    column_counter: Counter[str] = Counter()
    all_id_columns: set[str] = set()

    for dep in plan.feature_deps or []:
        if not isinstance(dep, FeatureDep):
            continue

        transformer = FeatureDepTransformer(dep=dep, plan=plan)
        renamed_cols = transformer.renamed_columns

        if renamed_cols is not None:
            # Filter out metaxy system columns from validation
            # (they're handled separately and allowed to collide)
            user_cols = [c for c in renamed_cols if c not in transformer.renamed_metaxy_cols]
            column_counter.update(user_cols)

        # Allow both upstream ID columns and output ID columns to repeat
        all_id_columns.update(transformer.renamed_id_columns)
        all_id_columns.update(plan.get_input_id_columns_for_dep(dep))

    # Find columns that appear more than once but aren't ID columns
    repeated_columns = [col for col, count in column_counter.items() if count > 1 and col not in all_id_columns]

    if repeated_columns:
        raise ValueError(
            f"Feature '{plan.feature.key.to_string()}' would have duplicate column names after renaming: "
            f"{sorted(repeated_columns)}. Only ID columns ({sorted(all_id_columns)}) are allowed to be repeated. "
            f"Use the 'rename' parameter in FeatureDep to resolve conflicts, "
            "or use 'columns' to select only the columns you need."
        )
