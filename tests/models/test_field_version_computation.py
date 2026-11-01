"""Test field version computation in the feature graph and in graph snapshots."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from metaxy.ext.polars.handlers.delta import DeltaMetadataStore
from metaxy.graph.diff.differ import GraphDiffer
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import CoercibleToFieldKey, FeatureKey, FieldKey, ValidatedFieldKeyAdapter
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from syrupy.assertion import SnapshotAssertion


def test_load_snapshot_data_computes_proper_field_versions(
    graph: FeatureGraph, snapshot: SnapshotAssertion, tmp_path: Path
):
    """Test that field versions are computed correctly from specs.

    This test validates the field version computation logic. Features defined
    in test functions use standalone spec loading since they can't be imported.
    The important validation is that:
    1. Fields are tracked separately in the snapshot data
    2. Field versions are computed correctly from specs
    """
    differ = GraphDiffer()

    # Create parent feature with two fields with different code versions
    class ParentFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent"]),
            fields=[
                FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                FieldSpec(key=FieldKey(["field2"]), code_version="2"),
            ],
        ),
    ):
        pass

    # Compute expected field versions from the active graph
    parent_field1_version = graph.get_field_version(
        FQFieldKey(feature=FeatureKey(["parent"]), field=FieldKey(["field1"]))
    )
    parent_field2_version = graph.get_field_version(
        FQFieldKey(feature=FeatureKey(["parent"]), field=FieldKey(["field2"]))
    )

    # Field versions should be different from each other (different code versions)
    assert parent_field1_version != parent_field2_version

    with DeltaMetadataStore(root_path=tmp_path / "delta_store").open("w") as store:
        # Record snapshot
        result = SystemTableStorage(store).push_graph_snapshot()

        project_version = result.project_version

        _ = result.already_pushed

        # Load snapshot data - will load standalone specs since test features can't be imported
        snapshot_data = differ.load_snapshot_data(store, project_version)

        # Verify structure is correct
        assert "parent" in snapshot_data
        parent_data = snapshot_data["parent"]
        assert "metaxy_feature_version" in parent_data
        assert "fields" in parent_data
        assert "field1" in parent_data["fields"]
        assert "field2" in parent_data["fields"]

        # Verify field versions are computed correctly (captured via snapshot)
        assert parent_data["fields"] == snapshot(name="field_versions")
        assert parent_data["metaxy_feature_version"] == snapshot(name="feature_version")


def test_load_snapshot_data_fallback_when_graph_reconstruction_fails(
    graph: FeatureGraph, snapshot: SnapshotAssertion, tmp_path: Path
):
    """Test field version computation when feature classes cannot be imported."""
    differ = GraphDiffer()

    # Create a feature defined in test scope (will not be importable)
    class TestFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[
                FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                FieldSpec(key=FieldKey(["field2"]), code_version="2"),
            ],
        ),
    ):
        pass

    with DeltaMetadataStore(root_path=tmp_path / "delta_store").open("w") as store:
        # Record snapshot
        result = SystemTableStorage(store).push_graph_snapshot()

        project_version = result.project_version

        _ = result.already_pushed

        # Load snapshot data - should load standalone specs
        snapshot_data = differ.load_snapshot_data(store, project_version)

        # Verify data was loaded
        assert "test/feature" in snapshot_data
        feature_data = snapshot_data["test/feature"]
        assert "metaxy_feature_version" in feature_data
        assert "fields" in feature_data

        # Verify field versions are computed correctly (captured via snapshot)
        assert feature_data["fields"] == snapshot(name="field_versions")
        assert feature_data["metaxy_feature_version"] == snapshot(name="feature_version")


def test_field_key_normalization(graph: FeatureGraph, tmp_path: Path):
    """Test that field keys are normalized to "/" separator format."""
    differ = GraphDiffer()

    class TestFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test"]),
            fields=[
                FieldSpec(key=FieldKey(["nested", "field"]), code_version="1"),
            ],
        ),
    ):
        pass

    with DeltaMetadataStore(root_path=tmp_path / "delta_store").open("w") as store:
        # Record snapshot
        result = SystemTableStorage(store).push_graph_snapshot()

        project_version = result.project_version

        _ = result.already_pushed

        # Load snapshot data (will load standalone spec since feature is in test scope)
        snapshot_data = differ.load_snapshot_data(store, project_version)

        # Field key should be normalized to "/" format, not "__"
        assert "test" in snapshot_data
        feature_data = snapshot_data["test"]
        assert "nested/field" in feature_data["fields"]
        assert "nested__field" not in feature_data["fields"]


# Diamond field graph: layers 0..DEPTH, each with fields "a" and "b".
# Every field of layer i > 0 depends on both fields of layer i - 1.
DEPTH = 6
MIDDLE = DEPTH // 2
LAYER_FIELDS = ("a", "b")
FULL_MIDDLE_OVERRIDE: dict[CoercibleToFieldKey, str] = {"a": "override_a", "b": "override_b"}
PARTIAL_MIDDLE_OVERRIDE: dict[CoercibleToFieldKey, str] = {"a": "override_a"}
CODE_VERSIONS = st.text(alphabet="0123456789abcdef", min_size=1, max_size=4)


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


def layer_definition(layer: int, code_version: str) -> FeatureDefinition:
    return FeatureDefinition(
        spec=layer_spec(layer, code_version),
        feature_schema={},
        feature_class_path=f"tests.diamond.L{layer}",
        project="test",
    )


def add_diamond(graph: FeatureGraph, code_versions: Sequence[str]) -> FeatureKey:
    """Add one diamond layer per code version (root first) and return the key of the top layer."""
    for layer, code_version in enumerate(code_versions):
        graph.add_feature_definition(layer_definition(layer, code_version))
    return layer_key(len(code_versions) - 1)


@st.composite
def diamond_code_version_change(draw: st.DrawFn) -> tuple[list[str], list[str], int]:
    """Draw per-layer code versions before and after a change of one layer's code version."""
    before = draw(st.lists(CODE_VERSIONS, min_size=1, max_size=DEPTH + 1))
    changed_layer = draw(st.integers(min_value=0, max_value=len(before) - 1))
    new_code_version = draw(CODE_VERSIONS.filter(lambda version: version != before[changed_layer]))
    return before, [*before[:changed_layer], new_code_version, *before[changed_layer + 1 :]], changed_layer


@pytest.fixture
def diamond_graph(graph: FeatureGraph) -> FeatureKey:
    """Diamond field graph in the active graph. Returns the key of the top layer."""
    return add_diamond(graph, ["1"] * (DEPTH + 1))


@pytest.fixture
def root_v2_diamond_graph() -> FeatureGraph:
    """Separate diamond field graph whose root layer has code version "2"."""
    graph = FeatureGraph()
    add_diamond(graph, ["2", *["1"] * DEPTH])
    return graph


@pytest.fixture
def middle_override(request: pytest.FixtureRequest) -> dict[CoercibleToFieldKey, str]:
    """Provenance override of the external middle layer."""
    return request.param


@pytest.fixture
def external_middle_diamond_graph(
    graph: FeatureGraph,
    diamond_graph: FeatureKey,
    middle_override: dict[CoercibleToFieldKey, str],
) -> FeatureKey:
    """Diamond field graph whose middle layer is an external feature with a provenance override.

    Returns the key of the top layer.
    """
    graph.remove_feature(layer_key(MIDDLE))
    graph.add_feature_definition(
        FeatureDefinition.external(
            spec=layer_spec(MIDDLE, "1"),
            project="external",
            provenance_by_field=middle_override,
        )
    )
    return diamond_graph


@pytest.fixture
def parent_field_lookups(monkeypatch: pytest.MonkeyPatch) -> list[FQFieldKey]:
    """Record the field of each `FeaturePlan.get_parent_fields_for_field` call.

    Field version computation looks up the parent fields once per computed field,
    so the recorded fields show which field versions were computed.
    """
    lookups: list[FQFieldKey] = []
    original = FeaturePlan.get_parent_fields_for_field

    def spy(plan: FeaturePlan, key: CoercibleToFieldKey) -> Mapping[FQFieldKey, FieldSpec]:
        lookups.append(FQFieldKey(feature=plan.feature.key, field=ValidatedFieldKeyAdapter.validate_python(key)))
        return original(plan, key)

    monkeypatch.setattr(FeaturePlan, "get_parent_fields_for_field", spy)
    return lookups


def test_diamond_field_versions_match_snapshot(
    graph: FeatureGraph, diamond_graph: FeatureKey, snapshot: SnapshotAssertion
) -> None:
    """Field and feature versions of a diamond field graph stay byte-identical."""
    assert graph.get_feature_version_by_field(diamond_graph) == snapshot(name="version_by_field")
    assert graph.get_feature_version(diamond_graph) == snapshot(name="feature_version")
    assert {
        layer_key(layer).to_string(): graph.get_feature_version_by_field(layer_key(layer)) for layer in range(DEPTH + 1)
    } == snapshot(name="version_by_field_per_layer")


def test_get_feature_version_by_field_computes_each_field_once(
    graph: FeatureGraph, diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> None:
    """One call computes each field of the diamond once, also when fields share upstream fields."""
    graph.get_feature_version_by_field(diamond_graph)

    assert len(parent_field_lookups) == 2 * (DEPTH + 1)
    assert set(parent_field_lookups) == {key for layer in range(DEPTH + 1) for key in layer_field_keys(layer)}


def test_get_field_version_computes_each_upstream_field_once(
    graph: FeatureGraph, diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> None:
    """One call computes the requested field and each of its upstream fields once."""
    graph.get_field_version(FQFieldKey(feature=diamond_graph, field=FieldKey(["a"])))

    assert len(parent_field_lookups) == 2 * DEPTH + 1
    assert set(parent_field_lookups) == {
        FQFieldKey(feature=diamond_graph, field=FieldKey(["a"])),
        *(key for layer in range(DEPTH) for key in layer_field_keys(layer)),
    }


def test_field_versions_reflect_definition_changes_between_calls(
    graph: FeatureGraph, diamond_graph: FeatureKey, root_v2_diamond_graph: FeatureGraph
) -> None:
    """Field versions do not keep computed values from an earlier call."""
    version_by_field_before = graph.get_feature_version_by_field(diamond_graph)

    graph.add_feature_definition(layer_definition(0, "2"))

    assert graph.get_feature_version_by_field(diamond_graph) != version_by_field_before
    assert graph.get_feature_version_by_field(diamond_graph) == root_v2_diamond_graph.get_feature_version_by_field(
        diamond_graph
    )
    assert graph.get_feature_version(diamond_graph) == root_v2_diamond_graph.get_feature_version(diamond_graph)


def test_get_field_version_matches_feature_version_by_field(graph: FeatureGraph, diamond_graph: FeatureKey) -> None:
    """A field version computed alone equals the same field in the feature version map."""
    assert {
        key.field.to_string(): graph.get_field_version(key) for key in layer_field_keys(DEPTH)
    } == graph.get_feature_version_by_field(diamond_graph)


@pytest.mark.parametrize(
    "middle_override",
    [FULL_MIDDLE_OVERRIDE, PARTIAL_MIDDLE_OVERRIDE],
    ids=["full", "partial"],
    indirect=True,
)
def test_external_middle_override_appears_in_middle_versions(
    graph: FeatureGraph,
    external_middle_diamond_graph: FeatureKey,
    middle_override: dict[CoercibleToFieldKey, str],
) -> None:
    """Overridden fields of an external feature use the override as their version."""
    assert graph.get_feature_version_by_field(layer_key(MIDDLE)).items() >= middle_override.items()


@pytest.mark.parametrize(
    "middle_override",
    [FULL_MIDDLE_OVERRIDE, PARTIAL_MIDDLE_OVERRIDE],
    ids=["full", "partial"],
    indirect=True,
)
def test_external_middle_override_top_versions_match_snapshot(
    graph: FeatureGraph, external_middle_diamond_graph: FeatureKey, snapshot: SnapshotAssertion
) -> None:
    """Versions downstream of an external provenance override stay byte-identical."""
    assert graph.get_feature_version_by_field(external_middle_diamond_graph) == snapshot(name="version_by_field")
    assert graph.get_feature_version(external_middle_diamond_graph) == snapshot(name="feature_version")


@pytest.mark.parametrize("middle_override", [FULL_MIDDLE_OVERRIDE], ids=["full"], indirect=True)
def test_full_external_middle_override_skips_layers_below(
    graph: FeatureGraph, external_middle_diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> None:
    """A full provenance override stops the computation at the external feature."""
    graph.get_feature_version_by_field(external_middle_diamond_graph)

    assert {lookup.feature for lookup in parent_field_lookups} == {
        layer_key(layer) for layer in range(MIDDLE + 1, DEPTH + 1)
    }
    assert len(parent_field_lookups) == 2 * (DEPTH - MIDDLE)


@pytest.mark.parametrize("middle_override", [FULL_MIDDLE_OVERRIDE], ids=["full"], indirect=True)
def test_full_external_middle_override_hides_changes_below(
    graph: FeatureGraph, external_middle_diamond_graph: FeatureKey
) -> None:
    """Code version changes below a full provenance override do not change downstream versions."""
    version_by_field_before = graph.get_feature_version_by_field(external_middle_diamond_graph)

    graph.add_feature_definition(layer_definition(0, "2"))

    assert graph.get_feature_version_by_field(external_middle_diamond_graph) == version_by_field_before


@pytest.mark.parametrize("middle_override", [PARTIAL_MIDDLE_OVERRIDE], ids=["partial"], indirect=True)
def test_partial_external_middle_override_computes_layers_below(
    graph: FeatureGraph, external_middle_diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> None:
    """Fields without an override are computed from all upstream layers; overridden fields are not."""
    graph.get_feature_version_by_field(external_middle_diamond_graph)

    assert {lookup.feature for lookup in parent_field_lookups} == {layer_key(layer) for layer in range(DEPTH + 1)}
    assert FQFieldKey(feature=layer_key(MIDDLE), field=FieldKey(["a"])) not in parent_field_lookups
    assert len(parent_field_lookups) == 2 * (DEPTH + 1) - 1


@given(change=diamond_code_version_change())
@settings(max_examples=25, deadline=None)
def test_property_code_version_change_affects_only_layers_at_and_above(
    change: tuple[list[str], list[str], int],
) -> None:
    """Changing the code version of one layer changes all field versions at and above it, and no others."""
    code_versions_before, code_versions_after, changed_layer = change
    graph_before = FeatureGraph()
    graph_after = FeatureGraph()
    add_diamond(graph_before, code_versions_before)
    add_diamond(graph_after, code_versions_after)

    for layer in range(len(code_versions_before)):
        version_by_field_before = graph_before.get_feature_version_by_field(layer_key(layer))
        version_by_field_after = graph_after.get_feature_version_by_field(layer_key(layer))
        if layer < changed_layer:
            assert version_by_field_after == version_by_field_before
        else:
            assert all(version_by_field_after[name] != version_by_field_before[name] for name in LAYER_FIELDS)
