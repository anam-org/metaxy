---
title: "Metadata Queries"
description: "Metadata query syntax."
---

# Metadata Queries

A metadata query selects a feature by key and can specify a version and filters.

```text
<key>[:<version>][?<filter>[&<filter>...]]
```

??? info "Formal Grammar (EBNF)"

    ```text
    query        = key, [":", version], ["?", filters];
    key          = first-part, {"/", part};
    first-part   = (lowercase-letter | "_"), {lowercase-letter | digit | "_" | "-"};
    part         = (lowercase-letter | digit | "_"), {lowercase-letter | digit | "_" | "-"};
    version      = "current" | "latest" | version-hash;
    version-hash = alphanumeric, {alphanumeric};
    filters      = filter, {"&", filter};
    ```

Feature keys follow [`FeatureKey`][metaxy.FeatureKey] rules. Key parts cannot contain double underscores (`__`).

- `current` selects the version defined by the running code. This is the default.

- `latest` selects the latest version recorded in the metadata store.

- An alphanumeric version hash selects that exact version.

Filters use Metaxy's [SQL-like filter syntax](filters.md). Separate multiple filters with `&`; they are combined with `AND`.

## Example

```py
import metaxy as mx

query = mx.MetadataQuery.from_uri("group/my_feature:latest?age>25&status='active'")
```

`:` separates the key from its version, `?` starts the filters, and `&` separates filters. Inside filters, encode literal `?` as `%3F`, `&` as `%26`, and `%` as `%25`.
