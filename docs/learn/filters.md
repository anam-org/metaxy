# Specifying Filters As Text

There are a few occasions with Metaxy where users may want to define custom filter expressions via text, mainly being CLI arguments or configuration files.
For this purpose, Metaxy implements [`parse_filter_string`][metaxy.models.filter_expression.parse_filter_string], which converts SQL-like `WHERE` clauses into Narwhals filter expressions.

The following syntax is supported:

- Comparisons: `=`, `!=`, `>`, `<`, `>=`, `<=`

- Logical operators: `AND`, `OR`, `NOT`

- Parentheses for grouping

- Column references (identifiers or dotted paths)

- Literals: strings (`'value'`), numbers, booleans (`TRUE`/`FALSE`), and `NULL`

- Implicit boolean columns (e.g., `NOT is_active`)

!!! example

    ```py
    from metaxy.models.filter_expression import parse_filter_string

    df = ...  # a Narwhals frame

    # Parse a SQL WHERE clause into a backend-agnostic Narwhals expression
    expr = parse_filter_string("(age > 25 OR age < 18) AND status != 'deleted'")

    df = df.filter(expr)
    ```
