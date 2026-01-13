from __future__ import annotations

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.versioning.struct_adapter import StructFieldAccessor


class ConcreteStructFieldAccessor(StructFieldAccessor):
    """Test implementation of StructFieldAccessor."""

    def record_field_versions(
        self,
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Not needed for this test."""
        raise NotImplementedError("Not tested")


def test_struct_field_accessor() -> None:
    """Test that StructFieldAccessor.access_provenance_field works correctly."""
    accessor = ConcreteStructFieldAccessor()
    expr = accessor.access_provenance_field("col", "field")
    assert isinstance(expr, nw.Expr)
    assert "col" in repr(expr)
