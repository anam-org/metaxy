# Lineage Relationships

Metaxy supports a few common mappings from parent to child samples out of the box. These include:

- `1:1` mapping with [metaxy.LineageRelationshipType.identity] (the default one)
- `1:N` mapping with [metaxy.LineageRelationshipType.expansion]
- `N:1` mapping with [metaxy.LineageRelationshipType.aggregation]

Always use these classmethods to create instances of lineage relationships. Under the hood, they use Pydantic's discriminated union to ensure that the correct type is constructed based on the provided data.

## Examples

- [1:N example](../examples/one-to-many.md) provides a toy pipeline demonstrating how to define and use `1:N` lineage relationships in Metaxy.
