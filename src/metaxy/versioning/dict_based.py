"""Dict-based versioning engine for databases without native struct support.

This module provides a base class for versioning engines that work with
dict[str, str] (mapping field names to column names) instead of struct columns.

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
from metaxy.versioning.engine import VersioningEngine
from metaxy.versioning.ibis import IbisHashFn, IbisVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class FlatVersioningEngine:
    """Helpers for representing struct-like data as flattened columns."""

    @staticmethod
    def _get_flattened_column_name(struct_name: str, field_name: str) -> str:
        """Get the flattened column name for a struct field."""
        return f"{struct_name}__{field_name}"

    @staticmethod
    def build_struct_column(
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build a 'virtual struct' by renaming columns to follow naming convention."""
        rename_map = {
            source_col: FlatVersioningEngine._get_flattened_column_name(
                struct_name, field_name
            )
            for field_name, source_col in field_columns.items()
        }
        return df.rename(rename_map)

    def access_provenance_field(
        self,
        struct_column: str,
        field_name: str,
    ) -> nw.Expr:
        """Access a field from a virtual struct column."""
        flattened_name = FlatVersioningEngine._get_flattened_column_name(
            struct_column, field_name
        )
        return nw.col(flattened_name)


class DictBasedVersioningEngine(FlatVersioningEngine, VersioningEngine, ABC):
    """Base class for versioning engines that use dict-based struct representation."""

    def __init__(self, plan: FeaturePlan):
        VersioningEngine.__init__(self, plan)


class IbisDictBasedVersioningEngine(FlatVersioningEngine, IbisVersioningEngine):
    """Dict-based versioning engine for Ibis backends without struct support.

    Combines dict-based struct representation with Ibis hash functions.
    Suitable for PostgreSQL, SQLite, MySQL, and other SQL databases.

    Reuses hashing/aggregation/keep-latest logic from IbisVersioningEngine and
    only overrides struct handling via FlatVersioningEngine.

    The corresponding metadata store must handle JSON packing/unpacking
    at database boundaries.
    """

    def __init__(
        self,
        plan: FeaturePlan,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        """Initialize the Ibis dict-based engine."""
        IbisVersioningEngine.__init__(self, plan, hash_functions)
