from __future__ import annotations

import narwhals as nw

from metaxy.versioning.struct_adapter import (
    DefaultStructFieldAccessor,
)


def test_default_struct_field_accessor() -> None:
    accessor = DefaultStructFieldAccessor()
    expr = accessor.field_expr("col", "field")
    assert isinstance(expr, nw.Expr)
    assert "col" in repr(expr)
