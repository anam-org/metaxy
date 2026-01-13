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

from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.plan import FeaturePlan
from metaxy.versioning.engine import VersioningEngine
from metaxy.versioning.ibis import BaseIbisVersioningEngine, IbisHashFn
from metaxy.versioning.types import HashAlgorithm


class FlatVersioningMixin:
    """Helpers for representing struct-like data as flattened columns."""

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
            source_col: FlatVersioningMixin._get_flattened_column_name(
                struct_name, field_name
            )
            for field_name, source_col in field_columns.items()
        }
        return df.rename(rename_map)  # ty: ignore[invalid-argument-type]

    def access_provenance_field(
        self,
        struct_column: str,
        field_name: str,
    ) -> nw.Expr:
        flattened_name = FlatVersioningMixin._get_flattened_column_name(
            struct_column, field_name
        )
        return nw.col(flattened_name)

    def _add_data_version_columns_from_provenance(self, df: FrameT) -> FrameT:
        """Add data_version columns by aliasing provenance columns for flat engines.

        For flat engines, provenance_by_field is stored as flattened columns like:
          metaxy_provenance_by_field__feature__field

        We need to create corresponding data_version_by_field columns:
          metaxy_data_version_by_field__feature__field
        """
        # First, add the sample-level data_version column
        df = df.with_columns(  # ty: ignore[invalid-argument-type]
            nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
        )

        # For flattened columns, create corresponding data_version_by_field columns
        current_columns = df.collect_schema().names()
        prov_prefix = f"{METAXY_PROVENANCE_BY_FIELD}__"
        data_prefix = f"{METAXY_DATA_VERSION_BY_FIELD}__"

        for col in current_columns:
            if col.startswith(prov_prefix):
                field_name = col.split("__", 1)[1]
                target_col = f"{data_prefix}{field_name}"
                if target_col not in current_columns:
                    df = df.with_columns(nw.col(col).alias(target_col))

        return df


class FlatVersioningEngine(FlatVersioningMixin, VersioningEngine, ABC):
    def __init__(self, plan: FeaturePlan):
        VersioningEngine.__init__(self, plan)


class IbisFlatVersioningEngine(FlatVersioningMixin, BaseIbisVersioningEngine):
    """Versioning engine for Ibis backends without struct support using flattened columns."""

    def __init__(
        self,
        plan: FeaturePlan,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        BaseIbisVersioningEngine.__init__(self, plan, hash_functions)
