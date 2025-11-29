"""Flat column versioning engine for databases without native struct support.

This module provides a base class for versioning engines that keep struct-like
data as separate columns using the `{struct_name}__{field_name}` convention
instead of building actual struct columns.

Instead of building struct columns like:
    df["metaxy_provenance_by_field"] = Struct(field1="hash1", field2="hash2")

We keep fields as separate columns with a naming convention:
    df["metaxy_provenance_by_field__field1"] = "hash1"
    df["metaxy_provenance_by_field__field2"] = "hash2"

This avoids struct field access operations (struct.field()) which are not
supported by all SQL databases in Ibis (e.g., PostgreSQL).

The database store is responsible for packing/unpacking these flattened columns
into JSON/JSONB columns at database boundaries.
"""

from __future__ import annotations

from abc import ABC

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.plan import FeaturePlan
from metaxy.versioning.engine import FieldAccessor, VersioningEngine
from metaxy.versioning.ibis import BaseIbisVersioningEngine, IbisHashFn
from metaxy.versioning.types import HashAlgorithm


class FlatFieldAccessor(FieldAccessor):
    """Field accessor for representing struct-like data as flattened columns.

    Uses the `{struct_name}__{field_name}` naming convention instead of building
    actual struct columns.
    """

    @classmethod
    def uses_flattened_struct_columns(cls) -> bool:
        return True

    @staticmethod
    def _get_flattened_column_name(struct_name: str, field_name: str) -> str:
        return f"{struct_name}__{field_name}"

    def record_field_versions(
        self,
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Represent field versions as flattened columns following our naming convention."""
        rename_map = {
            source_col: FlatFieldAccessor._get_flattened_column_name(struct_name, field_name)
            for field_name, source_col in field_columns.items()
        }
        return df.rename(rename_map)  # ty: ignore[invalid-argument-type]

    def access_provenance_field(
        self,
        struct_column: str,
        field_name: str,
    ) -> nw.Expr:
        flattened_name = FlatFieldAccessor._get_flattened_column_name(struct_column, field_name)
        return nw.col(flattened_name)


class FlatVersioningEngine(FlatFieldAccessor, VersioningEngine, ABC):
    def __init__(self, plan: FeaturePlan):
        VersioningEngine.__init__(self, plan)


class IbisFlatVersioningEngine(FlatFieldAccessor, BaseIbisVersioningEngine):
    """Versioning engine for Ibis backends without struct support using flattened columns."""

    def __init__(
        self,
        plan: FeaturePlan,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        BaseIbisVersioningEngine.__init__(self, plan, hash_functions)
