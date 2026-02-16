---
title: "Fields Mapping API"
description: "API reference for field dependency mappings."
---

# Fields Mapping

Metaxy provides a few helpers for defining field-level lineage:

 - the default mapping that matches on field names or suffixes with [`FieldsMapping.default`][metaxy.models.fields_mapping.FieldsMapping.default]
 - `specific` mapping with [`FieldsMapping.specific`][metaxy.models.fields_mapping.FieldsMapping.specific]
 - `all` mapping with [`FieldsMapping.all`][metaxy.models.fields_mapping.FieldsMapping.all]
 - `none` mapping with [`FieldsMapping.none`][metaxy.models.fields_mapping.FieldsMapping.none]

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
