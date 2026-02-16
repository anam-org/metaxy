---
title: "Filters"
description: "SQL-like filter expressions."
---

# Specifying Filters As Text

There are a few occasions with Metaxy where users may want to define custom filter expressions via text, mainly being CLI arguments or configuration files.
For this purpose, Metaxy implements [`parse_filter_string`][metaxy.models.filter_expression.parse_filter_string], which converts SQL-like `WHERE` clauses into Narwhals filter expressions.

The following syntax is supported:

- Comparisons: `=`, `!=`, `>`, `<`, `>=`, `<=`

- Logical operators: `AND`, `OR`, `NOT`

- Set membership: `IN`, `NOT IN`

- Null checks: `IS NULL`, `IS NOT NULL`

- Parentheses for grouping

- Column references (identifiers or dotted paths)

- Literals: strings (`'value'`), numbers, booleans (`TRUE`/`FALSE`), and `NULL`

- Implicit boolean columns (e.g., `NOT is_active`)

!!! example

    ```py
    import polars as pl
    import narwhals as nw
    from metaxy.models.filter_expression import parse_filter_string

    # Create a sample Polars DataFrame
    pdf = pl.DataFrame({"age": [10, 20, 30], "status": ["active", "deleted", "active"]})
    df = nw.from_native(pdf)

    # Parse a SQL WHERE clause into a backend-agnostic Narwhals expression
    expr = parse_filter_string("(age > 25 OR age < 18) AND status != 'deleted'")

    result = df.filter(expr)

    assert result["age"].to_list() == [10, 30]
    ```
