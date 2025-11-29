"""Field accessor for struct columns shared by Polars and Ibis engines."""

from __future__ import annotations

from abc import ABC

import narwhals as nw

from metaxy.versioning.engine import FieldAccessor


class StructFieldAccessor(FieldAccessor, ABC):
    """Field accessor using native struct columns.

    Provides the access method for struct fields. Subclasses must implement
    record_field_versions using their backend-specific struct construction.
    """

    def access_provenance_field(
        self,
        struct_column: str,
        field_name: str,
    ) -> nw.Expr:
        """Access a field from a struct column using narwhals struct.field()."""
        return nw.col(struct_column).struct.field(field_name)
