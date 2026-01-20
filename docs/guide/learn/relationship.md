---
title: "Lineage Relationships"
description: "Lineage relationships between features."
---

# Lineage Relationships

Metaxy supports a few common mappings from parent to child samples out of the box. These include:

- `1:1` mapping with [`LineageRelationship.identity`][metaxy.models.lineage.LineageRelationship.identity] (the default one)

- `1:N` mapping with [`LineageRelationship.expansion`][metaxy.models.lineage.LineageRelationship.expansion]

- `N:1` mapping with [`LineageRelationship.aggregation`][metaxy.models.lineage.LineageRelationship.aggregation]

!!! tip

    Always use these classmethods to create instances of lineage relationships.
    They use Pydantic's discriminated unions under the hood to ensure correct type construction.

## Examples

- [1:N expansion](../../examples/expansion.md)
- [N:1 aggregation](../../examples/aggregation.md)
