"""SQL filter string parsing into Narwhals expressions.

This module exposes utilities to parse SQL WHERE-like strings into Narwhals
``Expr`` objects. The primary entry point is :func:`parse_filter_string`, which
returns a Narwhals expression that can be fed directly into
``LazyFrame.filter`` (works across all Narwhals backends).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple

import narwhals as nw
import sqlglot
from pydantic import field_serializer, model_validator
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


class OperandInfo(NamedTuple):
    expr: nw.Expr
    is_literal: bool
    literal_value: LiteralValue
    is_column: bool


class NarwhalsFilter(FrozenBaseModel):
    """Pydantic model for serializable Narwhals filter expressions."""

    expression: sqlglot.exp.Expression
    source: str | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
        "frozen": True,
    }

    @model_validator(mode="before")
    @classmethod
    def _parse_expression_from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            expression = _parse_to_sqlglot_expression(data)
            return {"expression": expression, "source": data}
        return data

    @field_serializer("expression")
    def _serialize_expression(self, expression: sqlglot.exp.Expression) -> str:
        return expression.sql()

    def to_expr(self) -> nw.Expr:
        """Convert the stored expression into a Narwhals ``Expr``."""
        return _expression_to_narwhals(self.expression)

    @classmethod
    def from_string(cls, filter_string: str) -> NarwhalsFilter:
        """Parse a SQL WHERE-like string into a ``NarwhalsFilter`` instance."""
        return cls.model_validate(filter_string)


def parse_filter_string(filter_string: str) -> nw.Expr:
    """Parse a SQL WHERE-like string into a Narwhals expression."""
    return NarwhalsFilter.from_string(filter_string).to_expr()


_COMPARISON_NODE_MAP: dict[type[sqlglot.exp.Expression], ComparisonOperator] = {}

_COMPARISON_NODE_ALIASES: dict[ComparisonOperator, tuple[str, ...]] = {
    ComparisonOperator.EQ: ("EQ",),
    ComparisonOperator.NEQ: ("NEQ",),
    ComparisonOperator.GT: ("Gt", "GT"),
    ComparisonOperator.LT: ("Lt", "LT"),
    ComparisonOperator.GTE: ("GTE", "GE", "GTOrEq"),
    ComparisonOperator.LTE: ("LTE", "LE", "LTOrEq"),
}

# SQLGlot publishes both legacy CamelCase comparison classes (``Gt``, ``Lt``)
# and uppercase aliases (``GT``, ``LT``). Some dialect-specific rewrites still
# emit older names like ``GTOrEq``. Register every known alias but keep the
# first match so we do not overwrite identical mappings repeatedly.
for operator, class_names in _COMPARISON_NODE_ALIASES.items():
    for class_name in class_names:
        cls = getattr(sqlglot.exp, class_name, None)
        if not isinstance(cls, type) or cls in _COMPARISON_NODE_MAP:
            continue
        _COMPARISON_NODE_MAP[cls] = operator


def _parse_to_sqlglot_expression(filter_string: str) -> sqlglot.exp.Expression:
    if not filter_string or not filter_string.strip():
        raise FilterParseError("Filter string cannot be empty.")

    try:
        parsed = sqlglot.parse_one(filter_string)
    except ParseError as exc:
        msg = f"Failed to parse filter string: {exc}"
        raise FilterParseError(msg) from exc

    if parsed is None:
        raise FilterParseError(
            f"Failed to parse filter string into an expression for {filter_string}"
        )

    return parsed


def _expression_to_narwhals(node: sqlglot.exp.Expression) -> nw.Expr:
    node = _strip_parens(node)

    if isinstance(node, sqlglot.exp.Not):
        operand = node.this
        if operand is None:
            raise FilterParseError("NOT operator requires an operand.")
        return ~_expression_to_narwhals(operand)

    if isinstance(node, sqlglot.exp.And):
        return _expression_to_narwhals(node.this) & _expression_to_narwhals(
            node.expression
        )

    if isinstance(node, sqlglot.exp.Or):
        return _expression_to_narwhals(node.this) | _expression_to_narwhals(
            node.expression
        )

    operator = _COMPARISON_NODE_MAP.get(type(node))
    if operator:
        left = getattr(node, "this", None)
        right = getattr(node, "expression", None)
        if left is None or right is None:
            raise FilterParseError(
                f"Comparison operator {operator.value} requires two operands."
            )
        left_operand = _operand_info(left)
        right_operand = _operand_info(right)
        null_comparison = _maybe_null_comparison(left_operand, right_operand, operator)
        if null_comparison is not None:
            return null_comparison
        return operator.apply(left_operand.expr, right_operand.expr)

    if isinstance(
        node,
        (
            sqlglot.exp.Column,
            sqlglot.exp.Identifier,
            sqlglot.exp.Boolean,
            sqlglot.exp.Literal,
            sqlglot.exp.Null,
            sqlglot.exp.Neg,
        ),
    ):
        return _operand_info(node).expr

    raise FilterParseError(f"Unsupported expression: {node.sql()}")


def _operand_info(node: sqlglot.exp.Expression) -> OperandInfo:
    node = _strip_parens(node)

    if isinstance(node, sqlglot.exp.Column):
        return OperandInfo(
            expr=nw.col(_column_name(node)),
            is_literal=False,
            literal_value=None,
            is_column=True,
        )

    if isinstance(node, sqlglot.exp.Identifier):
        return OperandInfo(
            expr=nw.col(_column_name(node)),
            is_literal=False,
            literal_value=None,
            is_column=True,
        )

    if isinstance(node, sqlglot.exp.Neg):
        inner = node.this
        if inner is None:
            raise FilterParseError("Unary minus requires an operand.")
        operand = _operand_info(inner)
        if not operand.is_literal or not isinstance(
            operand.literal_value, (int, float)
        ):
            raise FilterParseError("Unary minus only supported for numeric literals.")
        value = -operand.literal_value
        return OperandInfo(
            expr=nw.lit(value), is_literal=True, literal_value=value, is_column=False
        )

    if isinstance(node, sqlglot.exp.Literal):
        value = _literal_to_python(node)
        return OperandInfo(
            expr=nw.lit(value), is_literal=True, literal_value=value, is_column=False
        )

    if isinstance(node, sqlglot.exp.Boolean):
        value = _literal_to_python(node)
        return OperandInfo(
            expr=nw.lit(value), is_literal=True, literal_value=value, is_column=False
        )

    if isinstance(node, sqlglot.exp.Null):
        return OperandInfo(
            expr=nw.lit(None), is_literal=True, literal_value=None, is_column=False
        )

    raise FilterParseError(f"Unsupported operand: {node.sql()}")


def _maybe_null_comparison(
    left: OperandInfo,
    right: OperandInfo,
    operator: ComparisonOperator,
) -> nw.Expr | None:
    if left.is_literal and left.literal_value is None and right.is_column:
        column_expr = right.expr
        if operator is ComparisonOperator.EQ:
            return column_expr.is_null()
        if operator is ComparisonOperator.NEQ:
            return ~column_expr.is_null()
        return None

    if right.is_literal and right.literal_value is None and left.is_column:
        column_expr = left.expr
        if operator is ComparisonOperator.EQ:
            return column_expr.is_null()
        if operator is ComparisonOperator.NEQ:
            return ~column_expr.is_null()
        return None

    return None


def _literal_to_python(node: sqlglot.exp.Expression) -> LiteralValue:
    match node:
        case sqlglot.exp.Null():
            return None
        case sqlglot.exp.Boolean():
            return node.this is True or str(node.this).lower() == "true"
        case sqlglot.exp.Literal():
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


def _strip_parens(node: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
    current = node
    while isinstance(current, sqlglot.exp.Paren) and current.this is not None:
        current = current.this
    return current


def _identifier_part_to_string(part: sqlglot.exp.Expression | str) -> str:
    if isinstance(part, sqlglot.exp.Identifier):
        return part.name
    if isinstance(part, sqlglot.exp.Star):
        return "*"
    if isinstance(part, sqlglot.exp.Expression):
        return part.sql(dialect="")
    return str(part)


def _column_name(node: sqlglot.exp.Expression) -> str:
    if isinstance(node, sqlglot.exp.Column):
        parts = [_identifier_part_to_string(part) for part in node.parts or ()]
        name = ".".join(part for part in parts if part)
    elif isinstance(node, sqlglot.exp.Identifier):
        name = node.name
    else:
        name = node.sql(dialect="")

    name = name.strip()
    if not name:
        raise FilterParseError("Column reference is malformed.")
    return name


__all__ = [
    "ComparisonOperator",
    "FilterParseError",
    "NarwhalsFilter",
    "parse_filter_string",
]
