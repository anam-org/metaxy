"""SQL filter string parsing into Narwhals expressions.

This module exposes Pydantic models that represent backend-agnostic filter
expressions and utilities to parse SQL WHERE-like strings into Narwhals
``IntoExpr`` objects. The primary entry point is :func:`parse_filter_string`,
which returns a Narwhals expression that can be fed directly into
``LazyFrame.filter`` (works across all Narwhals backends).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, ClassVar, Literal

import narwhals as nw
from pydantic import Field, field_validator
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from metaxy.models.bases import FrozenBaseModel

LiteralValue = bool | int | float | str | None


class FilterParseError(ValueError):
    """Raised when a filter string cannot be parsed into a supported expression."""


class ComparisonOperator(str, Enum):
    EQ = "="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="

    def apply(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        if self is ComparisonOperator.EQ:
            return left == right
        if self is ComparisonOperator.NEQ:
            return left != right
        if self is ComparisonOperator.GT:
            return left > right
        if self is ComparisonOperator.LT:
            return left < right
        if self is ComparisonOperator.GTE:
            return left >= right
        if self is ComparisonOperator.LTE:
            return left <= right
        raise FilterParseError(f"Unsupported comparison operator: {self.value}")


class LogicalOperator(str, Enum):
    AND = "and"
    OR = "or"

    def apply(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        if self is LogicalOperator.AND:
            return left & right
        if self is LogicalOperator.OR:
            return left | right
        raise FilterParseError(f"Unsupported logical operator: {self.value}")


class FilterOperand(FrozenBaseModel):
    """Base class for value operands inside comparison expressions."""

    model_config: ClassVar[dict[str, Any]] = {
        "extra": "forbid",
        "frozen": True,
    }

    def to_expr(self) -> nw.Expr:
        raise NotImplementedError


class ColumnOperand(FilterOperand):
    type: Literal["column"] = "column"
    name: str

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            msg = "Column name cannot be empty"
            raise FilterParseError(msg)
        return value

    def to_expr(self) -> nw.Expr:
        return nw.col(self.name)


class LiteralOperand(FilterOperand):
    type: Literal["literal"] = "literal"
    value: LiteralValue

    def to_expr(self) -> nw.Expr:
        return nw.lit(self.value)


FilterOperandType = Annotated[
    ColumnOperand | LiteralOperand,
    Field(discriminator="type"),
]


class FilterExpression(FrozenBaseModel):
    """Base class for boolean filter expressions."""

    model_config: ClassVar[dict[str, Any]] = {
        "extra": "forbid",
        "frozen": True,
    }

    def to_expr(self) -> nw.Expr:
        raise NotImplementedError


class ComparisonExpression(FilterExpression):
    type: Literal["comparison"] = "comparison"
    operator: ComparisonOperator
    left: FilterOperandType
    right: FilterOperandType

    def to_expr(self) -> nw.Expr:
        if self.operator in {ComparisonOperator.EQ, ComparisonOperator.NEQ}:
            # Handle NULL-safe comparisons explicitly for backend correctness.
            null_comparison = _maybe_null_comparison(
                self.left, self.right, self.operator
            )
            if null_comparison is not None:
                return null_comparison

        return self.operator.apply(self.left.to_expr(), self.right.to_expr())


class LogicalExpression(FilterExpression):
    type: Literal["logical"] = "logical"
    operator: LogicalOperator
    left: FilterExpressionType
    right: FilterExpressionType

    def to_expr(self) -> nw.Expr:
        return self.operator.apply(self.left.to_expr(), self.right.to_expr())


class NotExpression(FilterExpression):
    type: Literal["not"] = "not"
    operand: FilterExpressionType

    def to_expr(self) -> nw.Expr:
        return ~self.operand.to_expr()


class OperandExpression(FilterExpression):
    type: Literal["operand"] = "operand"
    operand: FilterOperandType

    def to_expr(self) -> nw.Expr:
        return self.operand.to_expr()


FilterExpressionType = Annotated[
    ComparisonExpression | LogicalExpression | NotExpression | OperandExpression,
    Field(discriminator="type"),
]


class NarwhalsFilter(FrozenBaseModel):
    """Pydantic model for serializable Narwhals filter expressions."""

    expression: FilterExpressionType
    source: str | None = None

    def to_expr(self) -> nw.Expr:
        """Convert the stored expression into a Narwhals ``Expr``."""
        return self.expression.to_expr()

    @classmethod
    def from_string(cls, filter_string: str) -> NarwhalsFilter:
        """Parse a SQL WHERE-like string into a ``NarwhalsFilter``."""
        expression = _parse_to_expression(filter_string)
        return cls(expression=expression, source=filter_string)


_COMPARISON_NODE_MAP: dict[type[exp.Expression], ComparisonOperator] = {}

for class_name, operator in (
    ("EQ", ComparisonOperator.EQ),
    ("NEQ", ComparisonOperator.NEQ),
    ("Gt", ComparisonOperator.GT),
    ("GT", ComparisonOperator.GT),
    ("Lt", ComparisonOperator.LT),
    ("LT", ComparisonOperator.LT),
    ("GTE", ComparisonOperator.GTE),
    ("GE", ComparisonOperator.GTE),
    ("GTOrEq", ComparisonOperator.GTE),
    ("LTE", ComparisonOperator.LTE),
    ("LE", ComparisonOperator.LTE),
    ("LTOrEq", ComparisonOperator.LTE),
):
    cls = getattr(exp, class_name, None)
    if isinstance(cls, type):
        _COMPARISON_NODE_MAP[cls] = operator


def parse_filter_string(filter_string: str) -> nw.Expr:
    """Parse a SQL WHERE-like string into a Narwhals expression."""
    return NarwhalsFilter.from_string(filter_string).to_expr()


def _parse_to_expression(filter_string: str) -> FilterExpressionType:
    if not filter_string or not filter_string.strip():
        raise FilterParseError("Filter string cannot be empty.")

    try:
        parsed = parse_one(filter_string)
    except ParseError as exc:
        msg = f"Failed to parse filter string: {exc}"
        raise FilterParseError(msg) from exc

    if parsed is None:
        raise FilterParseError("Failed to parse filter string into an expression.")

    return _convert_expression(parsed)


def _convert_expression(node: exp.Expression) -> FilterExpressionType:
    node = _strip_parens(node)

    if isinstance(node, exp.Not):
        operand = node.this
        if operand is None:
            raise FilterParseError("NOT operator requires an operand.")
        return NotExpression(operand=_convert_expression(operand))

    if isinstance(node, exp.And):
        return LogicalExpression(
            operator=LogicalOperator.AND,
            left=_convert_expression(node.this),
            right=_convert_expression(node.expression),
        )

    if isinstance(node, exp.Or):
        return LogicalExpression(
            operator=LogicalOperator.OR,
            left=_convert_expression(node.this),
            right=_convert_expression(node.expression),
        )

    operator = _COMPARISON_NODE_MAP.get(type(node))
    if operator:
        left = getattr(node, "this", None)
        right = getattr(node, "expression", None)
        if left is None or right is None:
            raise FilterParseError(
                f"Comparison operator {operator.value} requires two operands."
            )
        return ComparisonExpression(
            operator=operator,
            left=_convert_operand(left),
            right=_convert_operand(right),
        )

    if isinstance(
        node,
        (
            exp.Column,
            exp.Identifier,
            exp.Boolean,
            exp.Literal,
            exp.Null,
        ),
    ):
        return OperandExpression(operand=_convert_operand(node))

    raise FilterParseError(f"Unsupported expression: {node.sql()}")


def _strip_parens(node: exp.Expression) -> exp.Expression:
    current = node
    while isinstance(current, exp.Paren) and current.this is not None:
        current = current.this
    return current


def _identifier_part_to_string(part: exp.Expression | str) -> str:
    if isinstance(part, exp.Identifier):
        return part.name
    if isinstance(part, exp.Star):
        return "*"
    if isinstance(part, exp.Expression):
        return part.sql(dialect="")
    return str(part)


def _column_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Column):
        parts = [_identifier_part_to_string(part) for part in node.parts or ()]
        name = ".".join(part for part in parts if part)
    elif isinstance(node, exp.Identifier):
        name = node.name
    else:
        name = node.sql(dialect="")

    name = name.strip()
    if not name:
        raise FilterParseError("Column reference is malformed.")
    return name


def _convert_operand(node: exp.Expression) -> FilterOperandType:
    node = _strip_parens(node)

    if isinstance(node, exp.Column):
        return ColumnOperand(name=_column_name(node))

    if isinstance(node, exp.Identifier):
        return ColumnOperand(name=_column_name(node))

    if isinstance(node, exp.Neg):
        inner = node.this
        if inner is None:
            raise FilterParseError("Unary minus requires an operand.")
        operand = _convert_operand(inner)
        if not isinstance(operand, LiteralOperand):
            raise FilterParseError("Unary minus only supported for numeric literals.")
        value = operand.value
        if isinstance(value, (int, float)):
            return LiteralOperand(value=-value)
        raise FilterParseError("Unary minus only supported for numeric literals.")

    if isinstance(node, exp.Literal):
        return LiteralOperand(value=_literal_to_python(node))

    if isinstance(node, exp.Boolean):
        return LiteralOperand(value=_literal_to_python(node))

    if isinstance(node, exp.Null):
        return LiteralOperand(value=None)

    raise FilterParseError(f"Unsupported operand: {node.sql()}")


def _literal_to_python(node: exp.Expression) -> LiteralValue:
    match node:
        case exp.Null():
            return None
        case exp.Boolean():
            return node.this is True or str(node.this).lower() == "true"
        case exp.Literal():
            literal = node
            if literal.is_string:
                return literal.name
            if literal.is_int:
                return int(literal.this)
            if literal.is_number:
                return float(literal.this)
            return literal.this
        case _:
            raise FilterParseError(f"Unsupported literal: {node.sql()}")
def _maybe_null_comparison(
    left: FilterOperand,
    right: FilterOperand,
    operator: ComparisonOperator,
) -> nw.Expr | None:
    if (
        isinstance(left, LiteralOperand)
        and left.value is None
        and isinstance(right, ColumnOperand)
    ):
        column_expr = right.to_expr()
        if operator is ComparisonOperator.EQ:
            return column_expr.is_null()
        return ~column_expr.is_null()

    if (
        isinstance(right, LiteralOperand)
        and right.value is None
        and isinstance(left, ColumnOperand)
    ):
        column_expr = left.to_expr()
        if operator is ComparisonOperator.EQ:
            return column_expr.is_null()
        return ~column_expr.is_null()

    return None


__all__ = [
    "ComparisonExpression",
    "ComparisonOperator",
    "FilterExpression",
    "FilterOperand",
    "FilterParseError",
    "LiteralOperand",
    "LogicalExpression",
    "LogicalOperator",
    "NarwhalsFilter",
    "NotExpression",
    "OperandExpression",
    "ColumnOperand",
    "parse_filter_string",
]
