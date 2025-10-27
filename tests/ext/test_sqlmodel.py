"""Focused tests for SQLModel integration with Metaxy features.

This module tests the SQLModelFeature base class and SQLModelFeatureMeta metaclass
that allow Metaxy Feature classes to also be SQLModel ORM models. Tests focus on:
1. Basic class creation with both Metaxy spec and SQLModel table parameters
2. Metaxy feature operations (registration, versioning, dependencies)
3. SQLModel integration (table creation, field definitions, __tablename__)
4. DuckDB metadata store integration

Note: Type ignores for __tablename__ are due to SQLModel/SQLAlchemy's use of
declarative_attr which the type checker doesn't fully understand.
"""

from __future__ import annotations

from pathlib import Path

import narwhals as nw
import polars as pl
import pytest
from sqlmodel import Field
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)
from metaxy._utils import collect_to_polars
from metaxy.ext.sqlmodel import SQLModelFeature
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.feature import FeatureGraph

# Basic Creation and Registration Tests


def test_basic_sqlmodel_feature_creation(snapshot: SnapshotAssertion) -> None:
    """Test creating a basic SQLModelFeature with spec and table parameters.

    Verifies that:
    - Class can be created with both spec (Metaxy) and table=True (SQLModel)
    - Feature is registered in the active FeatureGraph
    - Feature has expected Metaxy attributes (spec, graph)
    - Feature version is deterministic and correct
    """

    class VideoFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["video"]),
            deps=None,
            fields=[
                FieldSpec(
                    key=FieldKey(["frames"]), code_version=1
                ),  # Logical data component
            ],
        ),
    ):
        __tablename__: str = "video"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        path: str  # Metadata column: where video is stored

    # Check registration in active graph
    graph = FeatureGraph.get_active()
    assert FeatureKey(["video"]) in graph.features_by_key
    assert graph.features_by_key[FeatureKey(["video"])] is VideoFeature

    # Check Metaxy attributes
    assert hasattr(VideoFeature, "spec")
    assert hasattr(VideoFeature, "graph")
    assert VideoFeature.spec.key == FeatureKey(["video"])

    # Check feature version
    version = VideoFeature.feature_version()
    assert len(version) == 64
    assert version == snapshot


def test_sqlmodel_feature_without_spec() -> None:
    """Test SQLModelFeature base class without spec (for abstract bases).

    Verifies that abstract base classes can be created without a spec parameter.
    """

    class AbstractBase(SQLModelFeature):
        """Abstract base class without spec."""

        uid: str = Field(primary_key=True)

    # Should not crash - abstract bases don't need registration
    assert True


def test_sqlmodel_feature_multiple_fields(snapshot: SnapshotAssertion) -> None:
    """Test SQLModelFeature with multiple Metaxy fields.

    Verifies that:
    - Multiple fields are correctly tracked in spec
    - Feature version reflects all fields
    """

    class MultiFieldFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["multi", "field"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version=1),
                FieldSpec(key=FieldKey(["audio"]), code_version=1),
                FieldSpec(key=FieldKey(["subtitles"]), code_version=2),
            ],
        ),
    ):
        __tablename__: str = "multi_field"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        path: str

    # Check all fields tracked
    assert len(MultiFieldFeature.spec.fields) == 3
    field_keys = {f.key.to_string() for f in MultiFieldFeature.spec.fields}
    assert field_keys == {"frames", "audio", "subtitles"}

    # Check feature version
    version = MultiFieldFeature.feature_version()
    assert len(version) == 64
    assert version == snapshot


# SQLModel Table Functionality Tests


def test_sqlmodel_custom_tablename() -> None:
    """Test that __tablename__ can be customized.

    Verifies that SQLModel __tablename__ attribute works as expected.
    """

    class CustomTableFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["custom", "table"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["content"]), code_version=1)],
        ),
    ):
        __tablename__: str = "my_custom_table_name"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        path: str

    assert CustomTableFeature.__tablename__ == "my_custom_table_name"  # pyright: ignore[reportIncompatibleVariableOverride]


def test_automatic_tablename() -> None:
    """Test that __tablename__ is automatically set from feature key.

    Verifies that if __tablename__ is not provided, it's automatically
    generated from the feature key using double underscore convention.
    """

    class AutoTableFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["my", "auto", "feature"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
        ),
    ):
        # No __tablename__ specified - should be auto-generated
        uid: str = Field(primary_key=True)
        metadata_col: str

    # Should automatically be set to "my__auto__feature"
    tablename = str(AutoTableFeature.__tablename__)  # type: ignore[arg-type]
    assert tablename == "my__auto__feature"

    # Should also match the table_name() method
    assert tablename == AutoTableFeature.table_name()


def test_sqlmodel_field_definitions() -> None:
    """Test that SQLModel field definitions work correctly.

    Verifies that:
    - Primary keys can be defined
    - Various field types work (str, int, float, bool)
    - Optional fields work
    - Metaxy fields (logical data) are distinct from SQLModel columns (metadata)
    """

    class AudioFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["audio"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["waveform"]), code_version=1),  # Logical data
                FieldSpec(key=FieldKey(["spectrum"]), code_version=1),  # Logical data
            ],
        ),
    ):
        __tablename__: str = "audio"  # pyright: ignore[reportIncompatibleVariableOverride]
        # SQLModel columns for metadata
        uid: str = Field(primary_key=True)
        path: str  # Where audio file is stored
        sample_rate: int  # Metadata about audio
        duration: float  # Metadata about audio
        format: str  # File format
        optional_artist: str | None = None  # Optional metadata

    # Verify instance can be created
    instance = AudioFeature(
        uid="1",
        path="/audio/track1.wav",
        sample_rate=44100,
        duration=180.5,
        format="wav",
    )

    assert instance.uid == "1"
    assert instance.path == "/audio/track1.wav"
    assert instance.sample_rate == 44100
    assert instance.optional_artist is None


# Metaxy Feature Functionality Tests


def test_feature_version_method(snapshot: SnapshotAssertion) -> None:
    """Test that SQLModelFeature.feature_version() works correctly.

    Verifies that:
    - feature_version() returns a deterministic hash
    - Hash is 64-character hex string
    """

    class VersionedFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["versioned"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
        ),
    ):
        __tablename__: str = "versioned"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        data: str

    version = VersionedFeature.feature_version()

    # Should be deterministic
    assert version == VersionedFeature.feature_version()

    # Should be 64-character hex string
    assert len(version) == 64
    assert all(c in "0123456789abcdef" for c in version)

    # Snapshot
    assert version == snapshot


def test_data_version_method(snapshot: SnapshotAssertion) -> None:
    """Test that SQLModelFeature.data_version() works correctly.

    Verifies that:
    - data_version() returns dict mapping field keys to hashes
    - Each hash is 64 characters
    """

    class DataVersionFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["data", "version"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["processed_data"]), code_version=1),
                FieldSpec(key=FieldKey(["embeddings"]), code_version=2),
            ],
        ),
    ):
        __tablename__: str = "data_version"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        source_path: str  # Metadata column
        timestamp: int  # Metadata column

    data_version = DataVersionFeature.data_version()

    # Should return dict mapping field keys to hashes
    assert isinstance(data_version, dict)
    assert "processed_data" in data_version
    assert "embeddings" in data_version

    # Each hash should be 64 characters
    assert all(len(v) == 64 for v in data_version.values())

    # Snapshot
    assert data_version == snapshot


def test_feature_with_dependencies(snapshot: SnapshotAssertion) -> None:
    """Test SQLModelFeature with dependencies on other features.

    Verifies that:
    - Parent feature is registered
    - Child feature is registered with correct dependency
    - Feature versions differ between parent and child
    """

    class ParentFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["parent"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["parent_data"]), code_version=1)],
        ),
    ):
        __tablename__: str = "parent"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        parent_data: str

    class ChildFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["child"]),
            deps=[FeatureDep(key=FeatureKey(["parent"]))],
            fields=[FieldSpec(key=FieldKey(["child_data"]), code_version=1)],
        ),
    ):
        __tablename__: str = "child"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        child_data: str

    # Check parent registered
    graph = FeatureGraph.get_active()
    assert FeatureKey(["parent"]) in graph.features_by_key

    # Check child registered with dependency
    assert FeatureKey(["child"]) in graph.features_by_key
    assert ChildFeature.spec.deps is not None
    assert len(ChildFeature.spec.deps) == 1
    assert ChildFeature.spec.deps[0].key == FeatureKey(["parent"])

    # Feature versions differ
    parent_version = ParentFeature.feature_version()
    child_version = ChildFeature.feature_version()
    assert parent_version != child_version

    # Snapshot
    assert {"parent": parent_version, "child": child_version} == snapshot


def test_feature_with_field_dependencies(snapshot: SnapshotAssertion) -> None:
    """Test SQLModelFeature with field-level dependencies.

    Verifies that:
    - Field dependencies are correctly tracked
    - Feature version reflects field dependencies
    """

    class UpstreamFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["upstream"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version=1),
                FieldSpec(key=FieldKey(["audio"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "upstream"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        video_path: str  # Metadata column: where video is stored
        created_at: int  # Metadata column: timestamp

    class DownstreamFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[FeatureDep(key=FeatureKey(["upstream"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["processed"]),
                    code_version=1,
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["upstream"]),
                            fields=[
                                FieldKey(["frames"]),
                                FieldKey(["audio"]),
                            ],
                        )
                    ],
                )
            ],
        ),
    ):
        __tablename__: str = "downstream"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        processed: str

    # Check field dependencies tracked
    deps = DownstreamFeature.spec.fields[0].deps
    assert deps is not None
    assert isinstance(deps, list)  # Ensure it's a list, not SpecialFieldDep
    assert len(deps) == 1

    field_dep = deps[0]
    assert field_dep.feature_key == FeatureKey(["upstream"])
    # field_dep.fields can be SpecialFieldDep or list, check it's a list here
    assert isinstance(field_dep.fields, list)
    assert len(field_dep.fields) == 2

    # Feature version reflects dependencies
    version = DownstreamFeature.feature_version()
    assert len(version) == 64
    assert version == snapshot


def test_version_changes_with_code_version(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version changes when code_version changes.

    Verifies that changing code_version results in different feature version.
    """
    # Use separate graphs to avoid key conflicts
    graph_v1 = FeatureGraph()
    graph_v2 = FeatureGraph()

    with graph_v1.use():

        class FeatureV1(
            SQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["versioned", "v1"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            __tablename__: str = "feature_v1"  # pyright: ignore[reportIncompatibleVariableOverride]
            uid: str = Field(primary_key=True)
            data: str

        v1 = FeatureV1.feature_version()

    with graph_v2.use():

        class FeatureV2(
            SQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["versioned", "v2"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=2)],  # Changed!
            ),
        ):
            __tablename__: str = "feature_v2"  # pyright: ignore[reportIncompatibleVariableOverride]
            uid: str = Field(primary_key=True)
            data: str

        v2 = FeatureV2.feature_version()

    # Should be different
    assert v1 != v2

    # Snapshot both
    assert {"v1": v1, "v2": v2} == snapshot


# Graph Context Tests


def test_custom_graph_context() -> None:
    """Test that SQLModelFeature respects custom graph context.

    Verifies that features can be registered in custom graphs separate from default.
    """
    custom_graph = FeatureGraph()

    with custom_graph.use():

        class CustomGraphFeature(
            SQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["custom", "graph"]),
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            __tablename__: str = "custom_graph"  # pyright: ignore[reportIncompatibleVariableOverride]
            uid: str = Field(primary_key=True)

        # Should be in custom graph
        assert FeatureKey(["custom", "graph"]) in custom_graph.features_by_key
        assert CustomGraphFeature.graph is custom_graph

    # Should NOT be in default graph
    default_graph = FeatureGraph.get_active()
    assert FeatureKey(["custom", "graph"]) not in default_graph.features_by_key


def test_graph_snapshot_inclusion(snapshot: SnapshotAssertion) -> None:
    """Test that SQLModelFeature is included in feature graph snapshots.

    Verifies that SQLModelFeature classes are properly serialized in graph snapshots.
    """

    class SnapshotFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["snapshot", "test"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["value"]), code_version=1)],
        ),
    ):
        __tablename__: str = "snapshot_test"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        value: int

    graph = FeatureGraph.get_active()

    # Create snapshot
    snapshot_data = graph.to_snapshot()

    # Check feature in snapshot
    feature_key_str = "snapshot/test"
    assert feature_key_str in snapshot_data

    # Check snapshot structure
    feature_snapshot = snapshot_data[feature_key_str]
    assert "feature_spec" in feature_snapshot
    assert "feature_version" in feature_snapshot
    assert "feature_class_path" in feature_snapshot

    # Check class path
    assert "SnapshotFeature" in feature_snapshot["feature_class_path"]

    # Snapshot version
    assert feature_snapshot["feature_version"] == snapshot


def test_downstream_dependency_tracking() -> None:
    """Test getting downstream features for SQLModelFeature.

    Verifies that graph correctly tracks downstream dependencies.
    """

    class RootFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["root"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
        ),
    ):
        __tablename__: str = "root"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        data: str

    class MiddleFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["middle"]),
            deps=[FeatureDep(key=FeatureKey(["root"]))],
            fields=[FieldSpec(key=FieldKey(["processed"]), code_version=1)],
        ),
    ):
        __tablename__: str = "middle"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        processed: str

    class LeafFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["leaf"]),
            deps=[FeatureDep(key=FeatureKey(["middle"]))],
            fields=[FieldSpec(key=FieldKey(["final"]), code_version=1)],
        ),
    ):
        __tablename__: str = "leaf"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        final: str

    graph = FeatureGraph.get_active()

    # Get downstream of root
    downstream = graph.get_downstream_features([FeatureKey(["root"])])

    # Should include middle and leaf in topological order
    assert len(downstream) == 2
    assert FeatureKey(["middle"]) in downstream
    assert FeatureKey(["leaf"]) in downstream

    # Topological order: middle before leaf
    middle_idx = downstream.index(FeatureKey(["middle"]))
    leaf_idx = downstream.index(FeatureKey(["leaf"]))
    assert middle_idx < leaf_idx


# Error Handling Tests


def test_duplicate_key_raises() -> None:
    """Test that registering features with duplicate keys raises an error.

    Verifies that FeatureGraph prevents duplicate key registration.
    """

    class Feature1(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["duplicate"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
    ):
        __tablename__: str = "feature1"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)

    # Try to create another with same key
    with pytest.raises(
        ValueError, match="Feature with key duplicate already registered"
    ):

        class Feature2(
            SQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["duplicate"]),  # Same key!
                deps=None,
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            __tablename__: str = "feature2"  # pyright: ignore[reportIncompatibleVariableOverride]
            uid: str = Field(primary_key=True)


def test_inheritance_chain() -> None:
    """Test that SQLModelFeature can be subclassed multiple times.

    Verifies that inheritance chain works correctly with abstract bases.
    """

    class BaseFeature(
        SQLModelFeature,
        spec=None,  # Abstract base
    ):
        """Abstract base class."""

        uid: str = Field(primary_key=True)

    class ConcreteFeature(
        BaseFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["concrete"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
        ),
    ):
        __tablename__: str = "concrete"  # pyright: ignore[reportIncompatibleVariableOverride]
        data: str

    # Check concrete feature registered
    graph = FeatureGraph.get_active()
    assert FeatureKey(["concrete"]) in graph.features_by_key

    # Check both SQLModel and Metaxy functionality
    assert hasattr(ConcreteFeature, "feature_version")
    assert hasattr(ConcreteFeature, "__tablename__")


# DuckDB Metadata Store Integration Test


def test_sqlmodel_feature_with_duckdb_store(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test SQLModelFeature with DuckDB metadata store.

    This is a practical integration test showing that SQLModelFeature works
    with Metaxy's metadata storage system. Verifies:
    - SQLModelFeature can be used with DuckDB store
    - Metadata can be written and read
    - Feature versioning works correctly
    - Data versioning is calculated correctly
    """

    class VideoFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["video", "processing"]),
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version=1),
                FieldSpec(key=FieldKey(["duration"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "video_processing"  # pyright: ignore[reportIncompatibleVariableOverride]
        uid: str = Field(primary_key=True)
        path: str
        duration: float

    # Create DuckDB store
    db_path = tmp_path / "test.duckdb"

    with DuckDBMetadataStore(db_path) as store:
        # Create metadata DataFrame with data_version column (required by metadata store)
        metadata_df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "path": ["/videos/v1.mp4", "/videos/v2.mp4", "/videos/v3.mp4"],
                "duration": [120.5, 90.0, 150.3],
                "data_version": [
                    {"frames": "hash1", "duration": "hash1"},
                    {"frames": "hash2", "duration": "hash2"},
                    {"frames": "hash3", "duration": "hash3"},
                ],
            }
        )

        # Write metadata
        store.write_metadata(VideoFeature, nw.from_native(metadata_df))

        # Read metadata back
        result = store.read_metadata(VideoFeature)
        result_df = collect_to_polars(result)

        # Verify data
        assert len(result_df) == 3
        assert set(result_df["sample_uid"].to_list()) == {1, 2, 3}

        # Check that data_version column was added
        assert "data_version" in result_df.columns

        # Check feature_version column
        assert "feature_version" in result_df.columns
        assert all(
            v == VideoFeature.feature_version()
            for v in result_df["feature_version"].to_list()
        )

        # Snapshot result
        assert result_df.to_dicts() == snapshot
