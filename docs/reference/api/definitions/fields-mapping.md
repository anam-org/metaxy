# Fields Mapping

Metaxy provides a few helpers when defining field-level dependencies:

- the default mapping that matches on field names or suffixes with [metaxy.FieldsMapping.identity] (the default one)
- `specific` mapping with [metaxy.FieldsMapping.default]
- `all` mapping with [metaxy.FieldsMapping.all]

Always use these classmethods to create instances of lineage relationships. Under the hood, they use Pydantic's discriminated union to ensure that the correct type is constructed based on the provided data.

---

::: metaxy.models.fields_mapping.FieldsMapping

---

::: metaxy.models.fields_mapping.FieldsMappingType

---

::: metaxy.models.fields_mapping.DefaultFieldsMapping

::: metaxy.models.fields_mapping.SpecificFieldsMapping

::: metaxy.models.fields_mapping.AllFieldsMapping

::: metaxy.models.fields_mapping.NoneFieldsMapping

---
