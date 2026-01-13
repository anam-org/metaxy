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
    """Base class for versioning engines using flattened column representation.

    Combines flat field access methods with versioning engine logic,
    overriding methods that need special handling for flattened columns.
    """

    def __init__(self, plan: FeaturePlan):
        VersioningEngine.__init__(self, plan)

    def _add_data_version_columns_from_provenance(self, df: FrameT) -> FrameT:
        """Add data_version columns by aliasing provenance columns for flat engines.

        For flat engines, provenance_by_field can be:
          1. Flattened columns: metaxy_provenance_by_field__field
          2. Struct column (for Polars frames): metaxy_provenance_by_field

        We create corresponding data_version columns in the same format.
        """
        # First, add the sample-level data_version column
        df = df.with_columns(  # ty: ignore[invalid-argument-type]
            nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
        )

        current_columns = df.collect_schema().names()

        # Check if we have a struct column or flattened columns
        if METAXY_PROVENANCE_BY_FIELD in current_columns:
            # Have struct column - alias it for data_version_by_field
            df = df.with_columns(
                nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD),
            )
        else:
            # Have flattened columns - create corresponding data_version_by_field columns
            prov_prefix = f"{METAXY_PROVENANCE_BY_FIELD}__"
            data_prefix = f"{METAXY_DATA_VERSION_BY_FIELD}__"

            for col in current_columns:
                if col.startswith(prov_prefix):
                    field_name = col.split("__", 1)[1]
                    target_col = f"{data_prefix}{field_name}"
                    if target_col not in current_columns:
                        df = df.with_columns(nw.col(col).alias(target_col))

        return df  # ty: ignore[invalid-return-type]

    def hash_struct_version_column(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
        struct_column: str = METAXY_PROVENANCE_BY_FIELD,
        hash_column: str = METAXY_PROVENANCE,
        field_names: list[str] | None = None,
    ) -> FrameT:
        """Hash struct column, ensuring data version columns exist for flattened engines.

        For flat engines, when hashing METAXY_DATA_VERSION_BY_FIELD, we need to ensure
        the flattened data version columns exist first by aliasing provenance columns.
        """
        # If we're hashing data_version_by_field, ensure the flattened columns exist
        if struct_column == METAXY_DATA_VERSION_BY_FIELD:
            current_columns = df.collect_schema().names()  # ty: ignore[invalid-argument-type]
            data_prefix = f"{METAXY_DATA_VERSION_BY_FIELD}__"
            has_data_version_columns = any(col.startswith(data_prefix) for col in current_columns)

            if not has_data_version_columns:
                # Create data version columns from provenance columns
                df = self._add_data_version_columns_from_provenance(df)  # ty: ignore[invalid-argument-type]

        # Call VersioningEngine's implementation (skip FlatFieldAccessor in MRO)
        return VersioningEngine.hash_struct_version_column(  # ty: ignore[invalid-argument-type]
            self,  # ty: ignore[invalid-argument-type]
            df,  # ty: ignore[invalid-argument-type]
            hash_algorithm,
            struct_column,
            hash_column,
            field_names,
        )


class IbisFlatVersioningEngine(FlatVersioningEngine, BaseIbisVersioningEngine):
    """Versioning engine for Ibis backends without struct support using flattened columns."""

    def __init__(
        self,
        plan: FeaturePlan,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        BaseIbisVersioningEngine.__init__(self, plan, hash_functions)
