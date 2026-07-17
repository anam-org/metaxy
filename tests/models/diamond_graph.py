"""Diamond field graphs for version computation tests.

Layers 0..DEPTH each have the fields "a" and "b". Every field of layer i > 0 depends on both fields
of layer i - 1, so the number of paths through the field graph grows exponentially with the depth.
"""

from collections.abc import Sequence

from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.plan import FQFieldKey
from metaxy.models.types import CoercibleToFieldKey, FeatureKey, FieldKey

DEPTH = 6
MIDDLE = DEPTH // 2
LAYER_FIELDS = ("a", "b")


def layer_key(layer: int) -> FeatureKey:
    return FeatureKey(["diamond", f"l{layer}"])


def layer_field_keys(layer: int) -> list[FQFieldKey]:
    return [FQFieldKey(feature=layer_key(layer), field=FieldKey([name])) for name in LAYER_FIELDS]


def layer_spec(layer: int, code_version: str) -> FeatureSpec:
    return FeatureSpec(
        key=layer_key(layer),
        id_columns=["id"],
        deps=[FeatureDep(feature=layer_key(layer - 1))] if layer > 0 else [],
        fields=[FieldSpec(key=name, code_version=code_version, deps=SpecialFieldDep.ALL) for name in LAYER_FIELDS],
    )


def layer_definition(layer: int, code_version: str, *, hash_truncation_length: int = 8) -> FeatureDefinition:
    return FeatureDefinition(
        spec=layer_spec(layer, code_version),
        feature_schema={},
        feature_class_path=f"tests.diamond.L{layer}",
        project="test",
        hash_truncation_length=hash_truncation_length,
    )


def external_layer_definition(
    layer: int,
    code_version: str,
    provenance_by_field: dict[CoercibleToFieldKey, str] | None,
    *,
    hash_truncation_length: int = 8,
) -> FeatureDefinition:
    """External definition of a layer, with an optional provenance override."""
    return FeatureDefinition.external(
        spec=layer_spec(layer, code_version),
        project="external",
        provenance_by_field=provenance_by_field,
        hash_truncation_length=hash_truncation_length,
        source="tests.models.diamond_graph",
    )


def add_diamond(graph: FeatureGraph, code_versions: Sequence[str]) -> FeatureKey:
    """Add one diamond layer per code version (root first) and return the key of the top layer."""
    for layer, code_version in enumerate(code_versions):
        graph.add_feature_definition(layer_definition(layer, code_version))
    return layer_key(len(code_versions) - 1)
