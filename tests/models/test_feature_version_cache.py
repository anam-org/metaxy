"""Test the cache of feature versions in the feature graph.

Cached versions are compared with `tests.models.reference_versions`, which computes versions without the cache.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    MultipleResults,
    RuleBasedStateMachine,
    multiple,
    rule,
    run_state_machine_as_test,
)
from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldDep, FieldSpec, SpecialFieldDep
from metaxy.models.plan import FQFieldKey
from metaxy.models.types import CoercibleToFieldKey, FeatureKey, FieldKey
from metaxy.utils.exceptions import MetaxyMissingFeatureDependency

from tests.models.diamond_graph import (
    DEPTH,
    LAYER_FIELDS,
    add_diamond,
    external_layer_definition,
    layer_definition,
    layer_field_keys,
    layer_key,
)
from tests.models.reference_versions import (
    reference_feature_version,
    reference_feature_version_by_field,
    reference_field_version,
)

LAYERS = st.integers(min_value=0, max_value=DEPTH)
CODE_VERSIONS = st.sampled_from(["1", "2", "3"])
TRUNCATION_LENGTHS = st.sampled_from([8, 12, 16])
# No override, or an override for no field, some fields or all fields of a layer.
PROVENANCE_OVERRIDES = st.none() | st.dictionaries(
    keys=st.sampled_from(LAYER_FIELDS), values=st.sampled_from(["override_1", "override_2"])
)


@dataclass(frozen=True)
class Outcome:
    """What a computation produced: a value, or the type and message of the exception that it raised."""

    value: object = None
    error_type: type[Exception] | None = None
    error_message: str | None = None


def outcome_of(compute: Callable[[], object]) -> Outcome:
    """Run a computation and record its value or its exception."""
    # The exception is the observed result of the computation, so it is recorded and not propagated.
    try:
        return Outcome(value=compute())
    except Exception as error:
        return Outcome(error_type=type(error), error_message=str(error))


@dataclass(frozen=True)
class VersionQuery:
    """A version computation of a feature graph, together with the reference computation of the same version."""

    compute: Callable[[FeatureGraph], object]
    compute_reference: Callable[[FeatureGraph], object]

    def checked_outcome(self, graph: FeatureGraph) -> Outcome:
        """Compute the version, assert that the outcome equals the reference outcome, and return the outcome."""
        outcome = outcome_of(lambda: self.compute(graph))
        assert outcome == outcome_of(lambda: self.compute_reference(graph))
        return outcome


def field_version_query(key: FQFieldKey) -> VersionQuery:
    return VersionQuery(
        compute=lambda graph: graph.get_field_version(key),
        compute_reference=lambda graph: reference_field_version(graph, key),
    )


def feature_version_by_field_query(key: FeatureKey) -> VersionQuery:
    """Query of the version by field of a feature. The field order is part of the result."""
    return VersionQuery(
        compute=lambda graph: list(graph.get_feature_version_by_field(key).items()),
        compute_reference=lambda graph: list(reference_feature_version_by_field(graph, key).items()),
    )


def feature_version_query(key: FeatureKey) -> VersionQuery:
    return VersionQuery(
        compute=lambda graph: graph.get_feature_version(key),
        compute_reference=lambda graph: reference_feature_version(graph, key),
    )


def copy_graph(graph: FeatureGraph) -> FeatureGraph:
    """Create a new graph that holds the same definition objects."""
    graph_copy = FeatureGraph()
    for definition in graph.feature_definitions_by_key.values():
        graph_copy.add_feature_definition(definition)
    return graph_copy


def assign_definition(graph: FeatureGraph, definition: FeatureDefinition) -> None:
    """Put a definition into the graph through the public dict, without `add_feature_definition`."""
    graph.feature_definitions_by_key[definition.key] = definition


def add_unrelated_feature(graph: FeatureGraph) -> None:
    graph.add_feature_definition(
        FeatureDefinition(
            spec=FeatureSpec(key=FeatureKey(["unrelated"]), id_columns=["id"]),
            feature_schema={},
            feature_class_path="tests.unrelated.Unrelated",
            project="test",
        )
    )


def add_downstream_feature(graph: FeatureGraph) -> None:
    graph.add_feature_definition(layer_definition(DEPTH + 1, "1"))


def add_same_root_definition(graph: FeatureGraph) -> None:
    graph.add_feature_definition(graph.feature_definitions_by_key[layer_key(0)])


@pytest.fixture
def no_global_config(config: MetaxyConfig) -> Iterator[None]:
    """Unset the global config for the test, and set the test config again afterwards."""
    MetaxyConfig.reset()
    assert not MetaxyConfig.is_set()
    yield
    MetaxyConfig.set(config)


@pytest.fixture
def diamond_field_keys() -> list[FQFieldKey]:
    """All fields of the diamond field graph, sorted."""
    return sorted(key for layer in range(DEPTH + 1) for key in layer_field_keys(layer))


@pytest.fixture
def cached_version_by_field(
    graph: FeatureGraph, diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> dict[str, str]:
    """Version by field of the diamond top, computed once so that the cache holds it.

    The lookups of this computation are removed from `parent_field_lookups`.
    """
    version_by_field = graph.get_feature_version_by_field(diamond_graph)
    parent_field_lookups.clear()
    return version_by_field


@pytest.fixture(params=["cold", "warm"])
def diamond_cache_state(request: pytest.FixtureRequest, graph: FeatureGraph, diamond_graph: FeatureKey) -> str:
    """State of the cache for the diamond: "cold" with no computed versions, or "warm" with the top computed."""
    if request.param == "warm":
        graph.get_feature_version_by_field(diamond_graph)
    return request.param


@pytest.fixture
def partially_overridden_middle(graph: FeatureGraph) -> FeatureKey:
    """External feature with a provenance override for field "a" only. Its own dependency is not in the graph."""
    spec = FeatureSpec(
        key=FeatureKey(["partial", "middle"]),
        id_columns=["id"],
        deps=[FeatureDep(feature=FeatureKey(["partial", "missing"]))],
        fields=["a", "b"],
    )
    graph.add_feature_definition(
        FeatureDefinition.external(
            spec=spec, project="external", provenance_by_field={"a": "override_a"}, source="tests.partial"
        )
    )
    return spec.key


@pytest.fixture
def child_of_overridden_field(graph: FeatureGraph, partially_overridden_middle: FeatureKey) -> FQFieldKey:
    """Field "x" of a feature that depends only on the overridden field of `partially_overridden_middle`."""
    key = FQFieldKey(feature=FeatureKey(["partial", "child"]), field=FieldKey(["x"]))
    graph.add_feature_definition(
        FeatureDefinition(
            spec=FeatureSpec(
                key=key.feature,
                id_columns=["id"],
                deps=[FeatureDep(feature=partially_overridden_middle)],
                fields=[FieldSpec(key=key.field, deps=[FieldDep(feature=partially_overridden_middle, fields=["a"])])],
            ),
            feature_schema={},
            feature_class_path="tests.partial.Child",
            project="test",
        )
    )
    return key


@pytest.fixture
def unsorted_fields_feature(graph: FeatureGraph, diamond_graph: FeatureKey) -> FeatureKey:
    """Feature that depends on the diamond top. Its spec lists its fields in the order "c", "a", "b"."""
    spec = FeatureSpec(
        key=FeatureKey(["unsorted"]),
        id_columns=["id"],
        deps=[FeatureDep(feature=diamond_graph)],
        fields=[FieldSpec(key=name, deps=SpecialFieldDep.ALL) for name in ("c", "a", "b")],
    )
    graph.add_feature_definition(
        FeatureDefinition(spec=spec, feature_schema={}, feature_class_path="tests.unsorted.Unsorted", project="test")
    )
    return spec.key


def test_cache_reuses_version_by_field(
    graph: FeatureGraph,
    diamond_graph: FeatureKey,
    cached_version_by_field: dict[str, str],
    parent_field_lookups: list[FQFieldKey],
) -> None:
    """A repeated call returns the same versions and computes no field version."""
    assert graph.get_feature_version_by_field(diamond_graph) == cached_version_by_field
    assert parent_field_lookups == []
    assert cached_version_by_field == reference_feature_version_by_field(graph, diamond_graph)


@pytest.mark.usefixtures("cached_version_by_field")
def test_cache_serves_feature_version(
    graph: FeatureGraph, diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> None:
    """The feature version is reused together with the versions by field."""
    feature_version = graph.get_feature_version(diamond_graph)

    assert parent_field_lookups == []
    assert feature_version == reference_feature_version(graph, diamond_graph)


def test_get_field_version_uses_cache(
    graph: FeatureGraph,
    diamond_graph: FeatureKey,
    cached_version_by_field: dict[str, str],
    parent_field_lookups: list[FQFieldKey],
) -> None:
    """After the versions of a feature are computed, a version of one of its fields is not computed again."""
    assert (
        graph.get_field_version(FQFieldKey(feature=diamond_graph, field=FieldKey(["a"])))
        == (cached_version_by_field["a"])
    )
    assert parent_field_lookups == []


def test_feature_versions_compute_only_fields_missing_from_cache(
    graph: FeatureGraph, diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> None:
    """Field versions from `get_field_version` are reused, and only the other fields are computed."""
    graph.get_field_version(FQFieldKey(feature=diamond_graph, field=FieldKey(["a"])))
    parent_field_lookups.clear()

    version_by_field = graph.get_feature_version_by_field(diamond_graph)

    assert parent_field_lookups == [FQFieldKey(feature=diamond_graph, field=FieldKey(["b"]))]
    assert version_by_field == reference_feature_version_by_field(graph, diamond_graph)


@pytest.mark.parametrize(
    ("feature_fixture", "spec_field_order"),
    [("diamond_graph", LAYER_FIELDS), ("unsorted_fields_feature", ("c", "a", "b"))],
    ids=["diamond_top", "unsorted_spec"],
)
def test_feature_version_by_field_keeps_spec_field_order(
    graph: FeatureGraph, request: pytest.FixtureRequest, feature_fixture: str, spec_field_order: tuple[str, ...]
) -> None:
    """The versions by field use the field order of the spec, also when field "b" is computed first."""
    feature_key: FeatureKey = request.getfixturevalue(feature_fixture)
    graph.get_field_version(FQFieldKey(feature=feature_key, field=FieldKey(["b"])))

    version_by_field = graph.get_feature_version_by_field(feature_key)

    assert tuple(version_by_field) == spec_field_order
    assert list(version_by_field.items()) == list(reference_feature_version_by_field(graph, feature_key).items())


def test_cache_returns_a_new_dict_on_each_call(
    graph: FeatureGraph, diamond_graph: FeatureKey, cached_version_by_field: dict[str, str]
) -> None:
    """Changes to a returned dict do not change the result of later calls."""
    cached_version_by_field["a"] = "changed_after_computation"
    graph.get_feature_version_by_field(diamond_graph)["b"] = "changed_after_reuse"

    assert graph.get_feature_version_by_field(diamond_graph) == reference_feature_version_by_field(graph, diamond_graph)


@pytest.mark.parametrize("layer", [0, DEPTH], ids=["root", "top"])
@pytest.mark.parametrize(
    "put_definition",
    [FeatureGraph.add_feature_definition, assign_definition],
    ids=["add_feature_definition", "direct_assignment"],
)
def test_new_definition_in_dependency_closure_invalidates_entry(
    graph: FeatureGraph,
    diamond_graph: FeatureKey,
    cached_version_by_field: dict[str, str],
    put_definition: Callable[[FeatureGraph, FeatureDefinition], None],
    layer: int,
) -> None:
    """A new definition of the feature or of an ancestor makes the next call compute versions again."""
    put_definition(graph, layer_definition(layer, "2"))

    assert graph.get_feature_version_by_field(diamond_graph) != cached_version_by_field
    assert graph.get_feature_version_by_field(diamond_graph) == reference_feature_version_by_field(graph, diamond_graph)


def test_restored_definition_object_gives_original_versions(
    graph: FeatureGraph, diamond_graph: FeatureKey, cached_version_by_field: dict[str, str]
) -> None:
    """When an ancestor is replaced and its first definition object is put back, the first versions apply again."""
    original_root = graph.feature_definitions_by_key[layer_key(0)]
    graph.add_feature_definition(layer_definition(0, "2"))
    assert graph.get_feature_version_by_field(diamond_graph) != cached_version_by_field

    assign_definition(graph, original_root)

    assert graph.get_feature_version_by_field(diamond_graph) == cached_version_by_field
    assert cached_version_by_field == reference_feature_version_by_field(graph, diamond_graph)


def test_removed_ancestor_raises_until_it_is_added_again(
    graph: FeatureGraph, diamond_graph: FeatureKey, cached_version_by_field: dict[str, str]
) -> None:
    """A removed ancestor makes calls raise as in the reference computation; the ancestor added again is used."""
    graph.remove_feature(layer_key(0))

    assert (
        feature_version_by_field_query(diamond_graph).checked_outcome(graph).error_type
        is MetaxyMissingFeatureDependency
    )

    graph.add_feature_definition(layer_definition(0, "2"))

    assert graph.get_feature_version_by_field(diamond_graph) != cached_version_by_field
    assert graph.get_feature_version_by_field(diamond_graph) == reference_feature_version_by_field(graph, diamond_graph)


@pytest.mark.parametrize("truncation_length", [12, 16])
def test_entry_is_bound_to_hash_truncation_length(
    graph: FeatureGraph,
    diamond_graph: FeatureKey,
    config: MetaxyConfig,
    cached_version_by_field: dict[str, str],
    truncation_length: int,
) -> None:
    """Versions are not reused across different hash truncation lengths."""
    with config.model_copy(update={"hash_truncation_length": truncation_length}).use():
        assert {len(version) for version in graph.get_feature_version_by_field(diamond_graph).values()} == {
            truncation_length
        }
        assert graph.get_feature_version_by_field(diamond_graph) == reference_feature_version_by_field(
            graph, diamond_graph
        )

    assert graph.get_feature_version_by_field(diamond_graph) == cached_version_by_field
    assert {len(version) for version in cached_version_by_field.values()} == {config.hash_truncation_length}


@pytest.mark.parametrize(
    "change",
    [add_unrelated_feature, add_downstream_feature, add_same_root_definition],
    ids=["unrelated_feature_added", "downstream_feature_added", "same_root_definition_added"],
)
def test_graph_change_outside_dependency_definitions_keeps_entry(
    graph: FeatureGraph,
    diamond_graph: FeatureKey,
    cached_version_by_field: dict[str, str],
    parent_field_lookups: list[FQFieldKey],
    change: Callable[[FeatureGraph], None],
) -> None:
    """An entry depends on the definitions of the feature and its ancestors, not on the whole graph."""
    change(graph)

    assert graph.get_feature_version_by_field(diamond_graph) == cached_version_by_field
    assert parent_field_lookups == []


def test_separate_graphs_do_not_share_entries(
    graph: FeatureGraph,
    diamond_graph: FeatureKey,
    cached_version_by_field: dict[str, str],
    parent_field_lookups: list[FQFieldKey],
    diamond_field_keys: list[FQFieldKey],
) -> None:
    """A graph with the same definition objects computes its own versions."""
    assert copy_graph(graph).get_feature_version_by_field(diamond_graph) == cached_version_by_field
    assert sorted(parent_field_lookups) == diamond_field_keys


@pytest.mark.usefixtures("diamond_cache_state")
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            field_version_query(FQFieldKey(feature=FeatureKey(["missing"]), field=FieldKey(["a"]))),
            id="field_version_of_missing_feature",
        ),
        pytest.param(
            field_version_query(FQFieldKey(feature=layer_key(DEPTH), field=FieldKey(["missing"]))),
            id="field_version_of_missing_field",
        ),
        pytest.param(feature_version_by_field_query(FeatureKey(["missing"])), id="version_by_field_of_missing_feature"),
        pytest.param(feature_version_query(FeatureKey(["missing"])), id="feature_version_of_missing_feature"),
    ],
)
def test_missing_feature_or_field_raises_like_reference(graph: FeatureGraph, query: VersionQuery) -> None:
    """A missing feature or field raises the same exception as the reference computation, on each call."""
    assert query.checked_outcome(graph).error_type is KeyError
    assert query.checked_outcome(graph).error_type is KeyError


@pytest.mark.parametrize(
    "make_query",
    [
        pytest.param(field_version_query, id="field_version"),
        pytest.param(lambda key: feature_version_by_field_query(key.feature), id="feature_version_by_field"),
        pytest.param(lambda key: feature_version_query(key.feature), id="feature_version"),
    ],
)
def test_child_of_partial_override_with_missing_upstream_gets_versions(
    graph: FeatureGraph,
    child_of_overridden_field: FQFieldKey,
    parent_field_lookups: list[FQFieldKey],
    make_query: Callable[[FQFieldKey], VersionQuery],
) -> None:
    """A field downstream of a partial override gets a version, although the external feature misses a dependency.

    The field depends only on the overridden field. The repeated call computes nothing.
    """
    query = make_query(child_of_overridden_field)
    cold_version = query.compute(graph)
    parent_field_lookups.clear()

    warm_version = query.compute(graph)

    assert parent_field_lookups == []
    assert cold_version == warm_version == query.compute_reference(graph)


@pytest.mark.parametrize(
    "make_query",
    [
        pytest.param(feature_version_by_field_query, id="feature_version_by_field"),
        pytest.param(feature_version_query, id="feature_version"),
        pytest.param(
            lambda key: field_version_query(FQFieldKey(feature=key, field=FieldKey(["b"]))),
            id="field_version_without_override",
        ),
    ],
)
def test_partial_override_with_missing_upstream_raises_on_each_call(
    graph: FeatureGraph,
    partially_overridden_middle: FeatureKey,
    child_of_overridden_field: FQFieldKey,
    make_query: Callable[[FeatureKey], VersionQuery],
) -> None:
    """Versions that need the missing dependency raise, also after the versions of the child are cached."""
    query = make_query(partially_overridden_middle)
    assert query.checked_outcome(graph).error_type is MetaxyMissingFeatureDependency

    graph.get_feature_version_by_field(child_of_overridden_field.feature)

    assert query.checked_outcome(graph).error_type is MetaxyMissingFeatureDependency
    assert query.checked_outcome(graph).error_type is MetaxyMissingFeatureDependency


@pytest.mark.usefixtures("no_global_config")
def test_cache_works_without_global_config(
    graph: FeatureGraph, diamond_graph: FeatureKey, parent_field_lookups: list[FQFieldKey]
) -> None:
    """Without a global config, the default config applies, and repeated calls compute nothing."""
    version_by_field = graph.get_feature_version_by_field(diamond_graph)
    feature_version = graph.get_feature_version(diamond_graph)
    parent_field_lookups.clear()

    assert graph.get_feature_version_by_field(diamond_graph) == version_by_field
    assert graph.get_feature_version(diamond_graph) == feature_version
    assert parent_field_lookups == []
    with MetaxyConfig().use():
        assert version_by_field == reference_feature_version_by_field(graph, diamond_graph)
        assert feature_version == reference_feature_version(graph, diamond_graph)


class FeatureVersionCacheMachine(RuleBasedStateMachine):
    """Change a diamond field graph and the hash truncation length in a random order.

    Each version and each error must always equal the outcome of the reference computation.
    """

    displaced_definitions: Bundle[FeatureDefinition] = Bundle("displaced_definitions")

    def __init__(self, config: MetaxyConfig) -> None:
        super().__init__()
        self.graph = FeatureGraph()
        add_diamond(self.graph, ["1"] * (DEPTH + 1))
        self.config = config

    def put_definition(
        self, definition: FeatureDefinition, put: Callable[[FeatureGraph, FeatureDefinition], None]
    ) -> MultipleResults[FeatureDefinition]:
        """Put a definition into the graph and return the definition that it displaces, if there is one."""
        previous = self.graph.feature_definitions_by_key.get(definition.key)
        put(self.graph, definition)
        if previous is None or self.graph.feature_definitions_by_key.get(definition.key) is previous:
            return multiple()
        return multiple(previous)

    @rule(target=displaced_definitions, layer=LAYERS, code_version=CODE_VERSIONS)
    def add_layer_definition(self, layer: int, code_version: str) -> MultipleResults[FeatureDefinition]:
        return self.put_definition(layer_definition(layer, code_version), FeatureGraph.add_feature_definition)

    @rule(target=displaced_definitions, layer=LAYERS, code_version=CODE_VERSIONS)
    def assign_layer_definition(self, layer: int, code_version: str) -> MultipleResults[FeatureDefinition]:
        return self.put_definition(layer_definition(layer, code_version), assign_definition)

    @rule(
        target=displaced_definitions, layer=LAYERS, code_version=CODE_VERSIONS, provenance_by_field=PROVENANCE_OVERRIDES
    )
    def assign_external_layer_definition(
        self, layer: int, code_version: str, provenance_by_field: dict[CoercibleToFieldKey, str] | None
    ) -> MultipleResults[FeatureDefinition]:
        # Direct assignment, because `add_feature_definition` ignores an external definition
        # when the key holds a non-external definition.
        return self.put_definition(
            external_layer_definition(layer, code_version, provenance_by_field), assign_definition
        )

    @rule(target=displaced_definitions, layer=LAYERS)
    def remove_layer(self, layer: int) -> MultipleResults[FeatureDefinition]:
        if layer_key(layer) not in self.graph.feature_definitions_by_key:
            return multiple()
        removed = self.graph.feature_definitions_by_key[layer_key(layer)]
        self.graph.remove_feature(layer_key(layer))
        return multiple(removed)

    @rule(target=displaced_definitions, definition=displaced_definitions)
    def restore_displaced_definition(self, definition: FeatureDefinition) -> MultipleResults[FeatureDefinition]:
        """Put an earlier definition object back, so that its key holds the same object as before."""
        return self.put_definition(definition, assign_definition)

    @rule()
    def add_unrelated(self) -> None:
        add_unrelated_feature(self.graph)

    @rule(truncation_length=TRUNCATION_LENGTHS)
    def set_truncation_length(self, truncation_length: int) -> None:
        self.config = self.config.model_copy(update={"hash_truncation_length": truncation_length})

    def check(self, query: VersionQuery) -> None:
        """Compare the query with the reference under the current config. Only a missing layer may raise."""
        with self.config.use():
            assert query.checked_outcome(self.graph).error_type in (None, KeyError, MetaxyMissingFeatureDependency)

    @rule(layer=LAYERS)
    def feature_versions_match_reference(self, layer: int) -> None:
        self.check(feature_version_by_field_query(layer_key(layer)))
        self.check(feature_version_query(layer_key(layer)))

    @rule(layer=LAYERS, field=st.sampled_from(LAYER_FIELDS))
    def field_version_matches_reference(self, layer: int, field: str) -> None:
        self.check(field_version_query(FQFieldKey(feature=layer_key(layer), field=FieldKey([field]))))


def test_property_cached_versions_match_reference(config: MetaxyConfig) -> None:
    """Random definition changes, removals, restorations and hash truncation length changes keep versions current."""
    run_state_machine_as_test(
        lambda: FeatureVersionCacheMachine(config),
        settings=settings(max_examples=100, stateful_step_count=25, deadline=None),
    )
