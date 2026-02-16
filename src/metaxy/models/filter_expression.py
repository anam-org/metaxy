from __future__ import annotations

from typing import Any, NamedTuple

import narwhals as nw
import sqlglot
from pydantic import field_serializer, model_validator
from sqlglot import exp
from sqlglot.errors import ParseError

from metaxy._decorators import public
from metaxy.models.bases import FrozenBaseModel

LiteralValue = bool | int | float | str | None


class FilterParseError(ValueError):
    """Raised when a filter string cannot be parsed into a supported expression."""


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


@public
def parse_filter_string(filter_string: str) -> nw.Expr:
    """Parse a SQL WHERE-like string into a Narwhals expression.

    The parser understands SQL `WHERE` clauses composed of comparison operators, logical operators, parentheses,
    dotted identifiers, and literal values (strings, numbers, booleans, ``NULL``).

    This functionality is implemented with [SQLGlot](https://sqlglot.com/).

    Example:
        ```python
        parse_filter_string("NOT (status = 'deleted') AND deleted_at = NULL")
        # Returns: (~(nw.col("status") == "deleted")) & nw.col("deleted_at").is_null()
        ```
    """
    return NarwhalsFilter.model_validate(filter_string).to_expr()


def _parse_to_sqlglot_expression(filter_string: str) -> sqlglot.exp.Expression:
    if not filter_string or not filter_string.strip():
        raise FilterParseError("Filter string cannot be empty.")

    try:
        parsed = sqlglot.parse_one(filter_string)
    except ParseError as exc:
        msg = f"Failed to parse filter string: {exc}"
        raise FilterParseError(msg) from exc

    if parsed is None:
        raise FilterParseError(f"Failed to parse filter string into an expression for {filter_string}")

    return parsed


def _expression_to_narwhals(node: exp.Expression) -> nw.Expr:
    """Convert a SQLGlot expression AST node to a Narwhals expression."""
    node = _strip_parens(node)

    # Logical operators
    if isinstance(node, exp.Not):
        operand = node.this
        if operand is None:
            raise FilterParseError("NOT operator requires an operand.")
        return ~_expression_to_narwhals(operand)

    if isinstance(node, exp.And):
        return _expression_to_narwhals(node.this) & _expression_to_narwhals(node.expression)

    if isinstance(node, exp.Or):
        return _expression_to_narwhals(node.this) | _expression_to_narwhals(node.expression)

    # IS / IS NOT operators
    if isinstance(node, exp.Is):
        left = node.this
        right = node.expression
        if left is None or right is None:
            raise FilterParseError("IS operator requires two operands.")

        left_operand = _operand_info(left)
        right_operand = _operand_info(right)

        null_comparison = _maybe_null_comparison(left_operand, right_operand, node)
        if null_comparison is not None:
            return null_comparison

        result = left_operand.expr == right_operand.expr
        return ~result if _is_is_not_node(node) else result

    # IN / NOT IN operators
    if isinstance(node, exp.In):
        left = node.this
        right = node.expressions
        if left is None or right is None:
            raise FilterParseError("IN operator requires a column and a list of values.")

        column_operand = _operand_info(left)
        if not column_operand.is_column:
            raise FilterParseError("IN operator left-hand side must be a column.")

        # Extract literal values from the list
        values: list[LiteralValue] = []
        for item in right:
            item_info = _operand_info(item)
            if not item_info.is_literal:
                raise FilterParseError("IN operator values must be literals.")
            values.append(item_info.literal_value)

        result = column_operand.expr.is_in(values)
        # Check if this is NOT IN
        if node.args.get("is_not"):
            return ~result
        return result

    # Comparison operators - direct mapping to Narwhals operations
    if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
        left = node.this
        right = node.expression
        if left is None or right is None:
            raise FilterParseError(f"Comparison operator {type(node).__name__} requires two operands.")
        left_operand = _operand_info(left)
        right_operand = _operand_info(right)

        # Handle NULL comparisons with IS NULL / IS NOT NULL
        null_comparison = _maybe_null_comparison(left_operand, right_operand, node)
        if null_comparison is not None:
            return null_comparison

        # Apply the appropriate Narwhals operator
        if isinstance(node, exp.EQ):
            return left_operand.expr == right_operand.expr
        elif isinstance(node, exp.NEQ):
            return left_operand.expr != right_operand.expr
        elif isinstance(node, exp.GT):
            return left_operand.expr > right_operand.expr
        elif isinstance(node, exp.LT):
            return left_operand.expr < right_operand.expr
        elif isinstance(node, exp.GTE):
            return left_operand.expr >= right_operand.expr
        elif isinstance(node, exp.LTE):
            return left_operand.expr <= right_operand.expr

    # Terminal nodes (operands)
    if isinstance(
        node,
        (
            exp.Column,
            exp.Identifier,
            exp.Boolean,
            exp.Literal,
            exp.Null,
            exp.Neg,
        ),
    ):
        return _operand_info(node).expr

    raise FilterParseError(f"Unsupported expression: {node.sql()}")


def _operand_info(node: exp.Expression) -> OperandInfo:
    """Extract operand information from a SQLGlot expression node."""
    node = _strip_parens(node)

    if isinstance(node, exp.Column):
        return OperandInfo(
            expr=nw.col(_column_name(node)),
            is_literal=False,
            literal_value=None,
            is_column=True,
        )

    if isinstance(node, exp.Identifier):
        return OperandInfo(
            expr=nw.col(_column_name(node)),
            is_literal=False,
            literal_value=None,
            is_column=True,
        )

    if isinstance(node, exp.Neg):
        inner = node.this
        if inner is None:
            raise FilterParseError("Unary minus requires an operand.")
        operand = _operand_info(inner)
        if not operand.is_literal or not isinstance(operand.literal_value, (int, float)):
            raise FilterParseError("Unary minus only supported for numeric literals.")
        value = -operand.literal_value
        return OperandInfo(expr=nw.lit(value), is_literal=True, literal_value=value, is_column=False)

    if isinstance(node, exp.Literal):
        value = _literal_to_python(node)
        return OperandInfo(expr=nw.lit(value), is_literal=True, literal_value=value, is_column=False)

    if isinstance(node, exp.Boolean):
        value = _literal_to_python(node)
        return OperandInfo(expr=nw.lit(value), is_literal=True, literal_value=value, is_column=False)

    if isinstance(node, exp.Null):
        return OperandInfo(expr=nw.lit(None), is_literal=True, literal_value=None, is_column=False)

    raise FilterParseError(f"Unsupported operand: {node.sql()}")


def _is_is_not_node(node: exp.Expression) -> bool:
    """Return True when the node represents an IS NOT comparison."""
    return bool(node.args.get("isnot"))


def _maybe_null_comparison(
    left: OperandInfo,
    right: OperandInfo,
    node: exp.Expression,
) -> nw.Expr | None:
    """Handle SQL NULL comparisons, converting to IS NULL / IS NOT NULL."""
    eq_like = isinstance(node, (exp.EQ, exp.NEQ, exp.Is))
    if not eq_like:
        return None

    negate = isinstance(node, exp.NEQ) or _is_is_not_node(node)

    if left.is_literal and left.literal_value is None and right.is_column:
        column_expr = right.expr
        return ~column_expr.is_null() if negate else column_expr.is_null()

    if right.is_literal and right.literal_value is None and left.is_column:
        column_expr = left.expr
        return ~column_expr.is_null() if negate else column_expr.is_null()

    return None


def _literal_to_python(node: exp.Expression) -> LiteralValue:
    """Convert a SQLGlot literal node to a Python value."""
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


def _strip_parens(node: exp.Expression) -> exp.Expression:
    """Remove surrounding parentheses from an expression."""
    current = node
    while isinstance(current, exp.Paren) and current.this is not None:
        current = current.this
    return current


def _identifier_part_to_string(part: exp.Expression | str) -> str:
    """Convert a column identifier part to a string."""
    if isinstance(part, exp.Identifier):
        return part.name
    if isinstance(part, exp.Star):
        return "*"
    if isinstance(part, exp.Expression):
        return part.sql(dialect="")
    return str(part)


def _column_name(node: exp.Expression) -> str:
    """Extract the column name from a Column or Identifier node."""
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


__all__ = [
    "FilterParseError",
    "NarwhalsFilter",
    "parse_filter_string",
]
