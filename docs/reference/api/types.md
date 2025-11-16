# Types

A few types used in Metaxy here and there.

::: metaxy.provenance.types.LazyIncrement

::: metaxy.provenance.types.Increment

::: metaxy.HashAlgorithm
    options:
      show_if_no_docstring: true

::: metaxy.models.types.SnapshotPushResult

::: metaxy.IDColumns

## Narwhals Filter Expressions

Narwhals filter expressions let you serialize and deserialize backend-agnostic predicates.
They are backed by SQLGlot and understand the following subset of SQL `WHERE` syntax:

- Comparisons: `=`, `!=`, `>`, `<`, `>=`, `<=`

- Logical operators: `AND`, `OR`, `NOT`

- Parentheses for grouping

- Column references (identifiers or dotted paths)

- Literals: strings (`'value'`), numbers, booleans (`TRUE`/`FALSE`), and `NULL`

- Implicit boolean columns (e.g., `NOT is_active`)

Convert SQL `WHERE` condition into Narwhals expressions
```py
from metaxy.models.filter_expression import parse_filter_string

# Parse a SQL WHERE clause into a backend-agnostic Narwhals expression
expr = parse_filter_string("(age > 25 OR age < 18) AND status != 'deleted'")

# Apply the parsed expression to any Narwhals LazyFrame
filtered = lazy_frame.filter(expr)
```

---

::: metaxy.models.filter_expression.NarwhalsFilter
::: metaxy.models.filter_expression.parse_filter_string
