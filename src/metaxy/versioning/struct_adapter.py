from __future__ import annotations

from typing import Protocol

import narwhals as nw


class StructFieldAccessor(Protocol):
    def field_expr(self, struct_column: str, field_name: str) -> nw.Expr: ...


class DefaultStructFieldAccessor:
    def field_expr(self, struct_column: str, field_name: str) -> nw.Expr:
        return nw.col(struct_column).struct.field(field_name)
