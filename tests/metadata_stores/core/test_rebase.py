"""Tests for MetadataStore.rebase() method."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
import pytest
from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    LineageRelationship,
    MetadataStore,
)
from metaxy.config import MetaxyConfig
from metaxy.ext.polars.handlers.delta import DeltaMetadataStore
from metaxy.metadata_store.system.storage import SystemTableStorage
from metaxy.models.feature import FeatureGraph
from metaxy_testing.models import SampleFeature, SampleFeatureSpec

from metaxy_testing import TempFeatureModule, add_metaxy_provenance_column

UPSTREAM_KEY = FeatureKey(["test", "upstream"])
DOWNSTREAM_KEY = FeatureKey(["test", "downstream"])

UPSTREAM_SPEC = SampleFeatureSpec(
    key=UPSTREAM_KEY,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
)


@pytest.fixture(autouse=True)
def setup_default_config() -> Iterator[None]:
    """Set up default MetaxyConfig for all tests."""
    config = MetaxyConfig(project="default", stores={})
    MetaxyConfig.set(config)
    yield
    MetaxyConfig.reset()


@pytest.fixture()
def make_graph() -> Iterator[list[TempFeatureModule]]:
    """Factory fixture that tracks created modules and cleans them up."""
    modules: list[TempFeatureModule] = []

    yield modules

    for module in modules:
        module.cleanup()


def _create_graph(
    modules: list[TempFeatureModule],
    module_name: str,
    downstream_code_version: str = "1",
) -> TempFeatureModule:
    """Create a simple upstream -> downstream feature graph.

    Upstream spec is always the same. Only downstream code_version changes.
    """
    downstream_spec = SampleFeatureSpec(
        key=DOWNSTREAM_KEY,
        deps=[FeatureDep(feature=UPSTREAM_KEY)],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=downstream_code_version,
                deps=[
                    FieldDep(
                        feature=UPSTREAM_KEY,
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    temp_module = TempFeatureModule(module_name)
    temp_module.write_features({"Upstream": UPSTREAM_SPEC, "Downstream": downstream_spec})
    modules.append(temp_module)
    return temp_module


def test_rebase_updates_provenance(store: MetadataStore, make_graph: list[TempFeatureModule]):
    """Happy path: write data with old version, rebase, verify new provenance."""
    temp_module_v1 = _create_graph(make_graph, "test_rebase_happy_v1", downstream_code_version="1")

    with temp_module_v1.graph.use(), store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [{"default": "h1"}] * 3,
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UPSTREAM_KEY)
        store.write(UPSTREAM_KEY, upstream_data)

        increment = store.resolve_update(DOWNSTREAM_KEY)
        store.write(DOWNSTREAM_KEY, increment.new.to_polars())

        old_feature_version = temp_module_v1.graph.get_feature_version(DOWNSTREAM_KEY)

        # Push graph snapshot so rebase can look up the target graph
        storage = SystemTableStorage(store)
        storage.push_graph_snapshot()

    temp_module_v2 = _create_graph(make_graph, "test_rebase_happy_v2", downstream_code_version="2")

    with temp_module_v2.graph.use(), store.open("w"):
        new_feature_version = temp_module_v2.graph.get_feature_version(DOWNSTREAM_KEY)
        assert old_feature_version != new_feature_version

        # Compute what v2 provenance should look like
        expected_version_by_field = temp_module_v2.graph.get_feature_version_by_field(DOWNSTREAM_KEY)

        # Push v2 graph snapshot
        storage = SystemTableStorage(store)
        storage.push_graph_snapshot()

        existing = store.read(
            DOWNSTREAM_KEY,
            with_feature_history=True,
            filters=[nw.col("metaxy_feature_version") == old_feature_version],
        )

        rebased = store.rebase(
            DOWNSTREAM_KEY,
            existing,
            to_feature_version=new_feature_version,
        )

        store.write(DOWNSTREAM_KEY, rebased.to_native(), preserve_feature_version=True)

        result = (
            store.read(
                DOWNSTREAM_KEY,
                with_feature_history=True,
                filters=[nw.col("metaxy_feature_version") == new_feature_version],
            )
            .collect()
            .to_polars()
        )

        assert result.height == 3
        assert result["metaxy_feature_version"].unique().to_list() == [new_feature_version]

        # project_version should match the target graph's project version
        expected_project_version = temp_module_v2.graph.project_version
        assert result["metaxy_project_version"].unique().to_list() == [expected_project_version]

        # provenance_by_field should have entries for all downstream fields
        for row_pbf in result["metaxy_provenance_by_field"].to_list():
            for field_key in expected_version_by_field:
                assert field_key in row_pbf

        # provenance should be populated and consistent across rows with same upstream data
        provenance_values = result["metaxy_provenance"].to_list()
        assert all(p is not None for p in provenance_values)

        # data_version should equal provenance (default, not user-overridden)
        assert result["metaxy_data_version"].to_list() == provenance_values


def test_rebase_recalculates_provenance_by_field(tmp_path: Path, make_graph: list[TempFeatureModule]):
    """Rebase recalculates provenance_by_field from the target graph, not the source.

    v2 adds a second field to downstream, so provenance_by_field must gain
    a new key after rebase.

    Uses DeltaMetadataStore because DuckDB silently drops new struct keys on insert.
    """
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    downstream_spec_v1 = SampleFeatureSpec(
        key=DOWNSTREAM_KEY,
        deps=[FeatureDep(feature=UPSTREAM_KEY)],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[FieldDep(feature=UPSTREAM_KEY, fields=[FieldKey(["default"])])],
            )
        ],
    )

    temp_v1 = TempFeatureModule("test_rebase_pbf_v1")
    temp_v1.write_features({"Upstream": UPSTREAM_SPEC, "Downstream": downstream_spec_v1})
    make_graph.append(temp_v1)

    with temp_v1.graph.use(), store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [{"default": "h1"}] * 2,
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UPSTREAM_KEY)
        store.write(UPSTREAM_KEY, upstream_data)

        increment = store.resolve_update(DOWNSTREAM_KEY)
        store.write(DOWNSTREAM_KEY, increment.new.to_polars())

        old_feature_version = temp_v1.graph.get_feature_version(DOWNSTREAM_KEY)
        old_pbf = increment.new.to_polars()["metaxy_provenance_by_field"].to_list()

        storage = SystemTableStorage(store)
        storage.push_graph_snapshot()

    # v2: add a second field to downstream — provenance_by_field must gain "extra" key
    downstream_spec_v2 = SampleFeatureSpec(
        key=DOWNSTREAM_KEY,
        deps=[FeatureDep(feature=UPSTREAM_KEY)],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[FieldDep(feature=UPSTREAM_KEY, fields=[FieldKey(["default"])])],
            ),
            FieldSpec(
                key=FieldKey(["extra"]),
                code_version="1",
                deps=[FieldDep(feature=UPSTREAM_KEY, fields=[FieldKey(["default"])])],
            ),
        ],
    )

    temp_v2 = TempFeatureModule("test_rebase_pbf_v2")
    temp_v2.write_features({"Upstream": UPSTREAM_SPEC, "Downstream": downstream_spec_v2})
    make_graph.append(temp_v2)

    with temp_v2.graph.use(), store.open("w"):
        new_feature_version = temp_v2.graph.get_feature_version(DOWNSTREAM_KEY)
        assert old_feature_version != new_feature_version

        storage = SystemTableStorage(store)
        storage.push_graph_snapshot()

        existing = store.read(
            DOWNSTREAM_KEY,
            with_feature_history=True,
            filters=[nw.col("metaxy_feature_version") == old_feature_version],
        )

        rebased = store.rebase(DOWNSTREAM_KEY, existing, to_feature_version=new_feature_version)
        store.write(DOWNSTREAM_KEY, rebased.to_native(), preserve_feature_version=True)

        result = (
            store.read(
                DOWNSTREAM_KEY,
                with_feature_history=True,
                filters=[nw.col("metaxy_feature_version") == new_feature_version],
            )
            .collect()
            .to_polars()
        )

        assert result.height == 2

        # provenance_by_field must differ from v1 — it should now have the "extra" key
        new_pbf = result["metaxy_provenance_by_field"].to_list()
        assert new_pbf != old_pbf
        for row_pbf in new_pbf:
            assert "default" in row_pbf
            assert "extra" in row_pbf


def test_rebase_defaults_to_current_version(store: MetadataStore, make_graph: list[TempFeatureModule]):
    """When to_feature_version is omitted, rebase uses the current active graph."""
    temp_module_v1 = _create_graph(make_graph, "test_rebase_default_v1", downstream_code_version="1")

    with temp_module_v1.graph.use(), store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [{"default": "h1"}] * 3,
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UPSTREAM_KEY)
        store.write(UPSTREAM_KEY, upstream_data)

        increment = store.resolve_update(DOWNSTREAM_KEY)
        store.write(DOWNSTREAM_KEY, increment.new.to_polars())

        old_feature_version = temp_module_v1.graph.get_feature_version(DOWNSTREAM_KEY)

    temp_module_v2 = _create_graph(make_graph, "test_rebase_default_v2", downstream_code_version="2")

    with temp_module_v2.graph.use(), store.open("w"):
        new_feature_version = temp_module_v2.graph.get_feature_version(DOWNSTREAM_KEY)
        assert old_feature_version != new_feature_version

        existing = store.read(
            DOWNSTREAM_KEY,
            with_feature_history=True,
            filters=[nw.col("metaxy_feature_version") == old_feature_version],
        )
        # Omit to_feature_version — should use current graph
        rebased = store.rebase(
            DOWNSTREAM_KEY,
            existing,
        )

        collected = rebased.lazy().collect().to_polars()
        assert collected.height == 3
        # Feature version should be set to the current graph's version
        assert collected["metaxy_feature_version"][0] == new_feature_version

        store.write(DOWNSTREAM_KEY, collected, preserve_feature_version=True)

        result = (
            store.read(
                DOWNSTREAM_KEY,
                with_feature_history=True,
                filters=[nw.col("metaxy_feature_version") == new_feature_version],
            )
            .collect()
            .to_polars()
        )
        assert result.height == 3


def test_rebase_unknown_version_raises(store: MetadataStore, make_graph: list[TempFeatureModule]):
    """Passing an unknown to_feature_version raises ValueError."""
    temp_module = _create_graph(make_graph, "test_rebase_unknown", downstream_code_version="1")

    with temp_module.graph.use(), store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"default": "h1"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UPSTREAM_KEY)
        store.write(UPSTREAM_KEY, upstream_data)

        increment = store.resolve_update(DOWNSTREAM_KEY)
        store.write(DOWNSTREAM_KEY, increment.new.to_polars())

        existing = store.read(DOWNSTREAM_KEY)

        with pytest.raises(ValueError, match="not found in system storage"):
            store.rebase(
                DOWNSTREAM_KEY,
                existing,
                to_feature_version="nonexistent_version",
            )


def test_rebase_root_feature_raises(store: MetadataStore, make_graph: list[TempFeatureModule]):
    """Root features raise ValueError."""
    temp_module = TempFeatureModule("test_rebase_root")
    root_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "root"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )
    temp_module.write_features({"Root": root_spec})
    make_graph.append(temp_module)

    with temp_module.graph.use(), store.open("w"):
        empty_df = nw.from_native(pl.DataFrame({"sample_uid": [], "metaxy_provenance_by_field": []}))
        with pytest.raises(ValueError, match="Cannot rebase root feature"):
            store.rebase(
                FeatureKey(["test", "root"]),
                empty_df,
                to_feature_version="new",
            )


def test_rebase_preserves_user_data_version(store: MetadataStore, make_graph: list[TempFeatureModule]):
    """Rebase recalculates default data_version but preserves user-overridden values."""
    temp_module_v1 = _create_graph(make_graph, "test_rebase_dv_v1", downstream_code_version="1")

    with temp_module_v1.graph.use(), store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [{"default": "h1"}] * 3,
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UPSTREAM_KEY)
        store.write(UPSTREAM_KEY, upstream_data)

        increment = store.resolve_update(DOWNSTREAM_KEY)
        store.write(DOWNSTREAM_KEY, increment.new.to_polars())

        v1_data = store.read(DOWNSTREAM_KEY).collect().to_polars()
        old_provenance = v1_data["metaxy_provenance"].to_list()
        old_data_version = v1_data["metaxy_data_version"].to_list()
        assert old_provenance == old_data_version  # default: data_version == provenance

        # Override data_version for row 0 only
        custom_dv = "custom_user_hash"
        custom_dv_by_field = {"default": "custom_field_hash"}
        overridden = v1_data.with_columns(
            pl.when(pl.col("sample_uid") == 1)
            .then(pl.lit(custom_dv))
            .otherwise(pl.col("metaxy_data_version"))
            .alias("metaxy_data_version"),
            pl.when(pl.col("sample_uid") == 1)
            .then(pl.lit(custom_dv_by_field))
            .otherwise(pl.col("metaxy_data_version_by_field"))
            .alias("metaxy_data_version_by_field"),
        )
        store.write(DOWNSTREAM_KEY, overridden)

        old_feature_version = temp_module_v1.graph.get_feature_version(DOWNSTREAM_KEY)

        # Push v1 graph snapshot
        storage = SystemTableStorage(store)
        storage.push_graph_snapshot()

    temp_module_v2 = _create_graph(make_graph, "test_rebase_dv_v2", downstream_code_version="2")

    with temp_module_v2.graph.use(), store.open("w"):
        new_feature_version = temp_module_v2.graph.get_feature_version(DOWNSTREAM_KEY)

        # Push v2 graph snapshot
        storage = SystemTableStorage(store)
        storage.push_graph_snapshot()

        existing = store.read(
            DOWNSTREAM_KEY,
            with_feature_history=True,
            filters=[nw.col("metaxy_feature_version") == old_feature_version],
        )
        rebased = store.rebase(
            DOWNSTREAM_KEY,
            existing,
            to_feature_version=new_feature_version,
        )

        store.write(DOWNSTREAM_KEY, rebased.to_native(), preserve_feature_version=True)

        result = (
            store.read(
                DOWNSTREAM_KEY,
                with_feature_history=True,
                filters=[nw.col("metaxy_feature_version") == new_feature_version],
            )
            .collect()
            .to_polars()
            .sort("sample_uid")
        )

        assert result.height == 3

        # Row 0 (sample_uid=1): user-overridden data_version should be preserved
        assert result["metaxy_data_version"][0] == custom_dv
        assert result["metaxy_data_version_by_field"][0] == custom_dv_by_field

        # Rows 1-2 (sample_uid=2,3): default data_version should match new provenance
        for i in [1, 2]:
            assert result["metaxy_data_version"][i] == result["metaxy_provenance"][i]


def test_write_fills_null_data_version_from_provenance(store: MetadataStore, make_graph: list[TempFeatureModule]):
    """Write fills null data_version_by_field rows from provenance."""
    temp_module = _create_graph(make_graph, "test_write_null_dv", downstream_code_version="1")

    with temp_module.graph.use(), store.open("w"):
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [{"default": "h1"}] * 3,
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UPSTREAM_KEY)
        store.write(UPSTREAM_KEY, upstream_data)

        increment = store.resolve_update(DOWNSTREAM_KEY)
        base_data = increment.new.to_polars()

        custom_dv_by_field = {"default": "custom_hash"}
        df_with_nulls = base_data.with_columns(
            pl.when(pl.col("sample_uid") == 1)
            .then(pl.lit(custom_dv_by_field))
            .otherwise(pl.lit(None))
            .alias("metaxy_data_version_by_field"),
        ).drop("metaxy_data_version")

        store.write(DOWNSTREAM_KEY, df_with_nulls)

        result = store.read(DOWNSTREAM_KEY).collect().to_polars().sort("sample_uid")

        assert result.height == 3

        # Row 0 (sample_uid=1): custom data_version_by_field preserved
        assert result["metaxy_data_version_by_field"][0] == custom_dv_by_field

        # Rows 1-2: null data_version_by_field filled from provenance
        for i in [1, 2]:
            assert result["metaxy_data_version_by_field"][i] == result["metaxy_provenance_by_field"][i]


DOCUMENT_KEY = FeatureKey(["test", "document"])
CHUNK_KEY = FeatureKey(["test", "chunk"])
SUMMARY_KEY = FeatureKey(["test", "summary"])


def _lineage_graph(chunk_code_version: str = "1", summary_code_version: str = "1") -> FeatureGraph:
    """Documents expand into chunks (1:N) and anchors aggregate into summaries (N:1)."""
    graph = FeatureGraph()
    with graph.use():

        class Document(
            SampleFeature,
            spec=SampleFeatureSpec(key=DOCUMENT_KEY, id_columns=("document_id",), fields=["content"]),
        ):
            pass

        class Chunk(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=CHUNK_KEY,
                id_columns=("chunk_id",),
                deps=[FeatureDep(feature=Document, lineage=LineageRelationship.expansion(on=["document_id"]))],
                fields=[FieldSpec(key=FieldKey(["content"]), code_version=chunk_code_version)],
            ),
        ):
            pass

        class Summary(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=SUMMARY_KEY,
                id_columns=("document_id",),
                deps=[FeatureDep(feature=Chunk, lineage=LineageRelationship.aggregation(on=["document_id"]))],
                fields=[FieldSpec(key=FieldKey(["content"]), code_version=summary_code_version)],
            ),
        ):
            pass

    return graph


def _read_version(store: MetadataStore, key: FeatureKey, feature_version: str) -> nw.LazyFrame[Any]:
    return store.read(key, with_feature_history=True, filters=[nw.col("metaxy_feature_version") == feature_version])


@pytest.fixture
def lineage_metadata(store: MetadataStore) -> dict[FeatureKey, str]:
    """Write v1 metadata for documents d1 and d2, their chunks, and their summaries.

    Chunks carry custom data versions. Returns the v1 feature version of each downstream feature.
    """
    graph = _lineage_graph()
    with graph.use(), store.open("w"):
        documents = pl.DataFrame(
            {"document_id": ["d1", "d2"], "metaxy_provenance_by_field": [{"content": "d1"}, {"content": "d2"}]}
        )
        store.write(DOCUMENT_KEY, add_metaxy_provenance_column(documents, DOCUMENT_KEY))

        parents = (
            store.resolve_update(CHUNK_KEY)
            .new.to_polars()
            .select("document_id", "metaxy_provenance", "metaxy_provenance_by_field")
        )
        chunks = pl.DataFrame(
            {"document_id": ["d1", "d1", "d1", "d2"], "chunk_id": ["d1-a", "d1-b", "d1-c", "d2-a"]}
        ).join(parents, on="document_id")
        store.write(
            CHUNK_KEY,
            chunks.with_columns(
                pl.concat_str(pl.lit("content-"), pl.col("chunk_id")).alias("metaxy_data_version"),
                pl.struct(pl.concat_str(pl.lit("content-"), pl.col("chunk_id")).alias("content")).alias(
                    "metaxy_data_version_by_field"
                ),
            ),
        )

        summaries = store.resolve_update(SUMMARY_KEY).new.to_polars()
        store.write(
            SUMMARY_KEY,
            summaries.select("document_id", "metaxy_provenance", "metaxy_provenance_by_field").unique("document_id"),
        )

        return {key: graph.get_feature_version(key) for key in (CHUNK_KEY, SUMMARY_KEY)}


def test_rebase_expansion_keeps_every_child_and_its_data_version(
    store: MetadataStore, lineage_metadata: dict[FeatureKey, str]
) -> None:
    """Every child row survives the rebase, with its own data version and its parent's provenance."""
    graph = _lineage_graph(chunk_code_version="2")
    with graph.use(), store.open("w"):
        existing = _read_version(store, CHUNK_KEY, lineage_metadata[CHUNK_KEY])
        old_provenance = existing.collect().to_polars().sort("chunk_id")["metaxy_provenance"].to_list()
        rebased = store.rebase(CHUNK_KEY, existing).collect().to_polars().sort("chunk_id")
        expected_provenance = {
            row["document_id"]: row["metaxy_provenance"]
            for row in store.resolve_update(CHUNK_KEY, skip_comparison=True).new.to_polars().iter_rows(named=True)
        }

    assert rebased["chunk_id"].to_list() == ["d1-a", "d1-b", "d1-c", "d2-a"]
    assert rebased["metaxy_data_version"].to_list() == ["content-d1-a", "content-d1-b", "content-d1-c", "content-d2-a"]
    assert rebased["metaxy_feature_version"].unique().to_list() == [graph.get_feature_version(CHUNK_KEY)]
    assert rebased["metaxy_provenance"].to_list() == [expected_provenance[d] for d in rebased["document_id"]]
    assert rebased["metaxy_provenance"].to_list() != old_provenance


def test_rebase_aggregation_returns_one_row_per_sample(
    store: MetadataStore, lineage_metadata: dict[FeatureKey, str]
) -> None:
    """The recomputed provenance has one row per group member, but the rebase keeps one row per sample."""
    graph = _lineage_graph(summary_code_version="2")
    with graph.use(), store.open("w"):
        existing = _read_version(store, SUMMARY_KEY, lineage_metadata[SUMMARY_KEY])
        rebased = store.rebase(SUMMARY_KEY, existing).collect().to_polars().sort("document_id")

    assert rebased["document_id"].to_list() == ["d1", "d2"]
    assert rebased["metaxy_feature_version"].unique().to_list() == [graph.get_feature_version(SUMMARY_KEY)]
    assert rebased["metaxy_provenance"].is_not_null().all()
