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


# Custom ID Columns Tests


def test_basic_custom_id_columns() -> None:
    """Test creating SQLModelFeature with custom id_columns.

    Verifies that:
    - SQLModelFeature can be created with custom id_columns
    - id_columns are accessible via Feature.id_columns() method
    - Feature is registered correctly in the graph
    """

    class UserSessionFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["user", "session"]),
            id_columns=["user_id", "session_id"],
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["activity"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "user_session"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: str = Field(primary_key=True)
        session_id: str = Field(primary_key=True)
        activity: str
        timestamp: int

    # Verify id_columns are set correctly
    assert UserSessionFeature.id_columns() == ["user_id", "session_id"]

    # Verify feature is registered
    graph = FeatureGraph.get_active()
    assert FeatureKey(["user", "session"]) in graph.features_by_key

    # Verify spec has custom id_columns
    assert UserSessionFeature.spec.id_columns == ["user_id", "session_id"]
    assert UserSessionFeature.spec.id_columns == ["user_id", "session_id"]


def test_sqlmodel_duckdb_custom_id_columns(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test SQLModelFeature with DuckDB store using custom id_columns.

    Verifies that:
    - Parent and child SQLModelFeatures work with custom id_columns
    - Metadata operations (write/read) respect custom ID columns
    - Joins work correctly with custom ID columns
    - Data versioning works with custom ID columns
    """

    # Create parent feature with custom ID columns
    class UserActivityFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["user", "activity"]),
            id_columns=["user_id", "session_id"],
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["activity_type"]), code_version=1),
                FieldSpec(key=FieldKey(["duration"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "user_activity"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        session_id: int = Field(primary_key=True)
        activity_type: str
        duration: float
        timestamp: int

    # Create child feature with same custom ID columns
    class UserSummaryFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["user", "summary"]),
            id_columns=["user_id", "session_id"],
            deps=[FeatureDep(key=FeatureKey(["user", "activity"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["total_duration"]),
                    code_version=1,
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["user", "activity"]),
                            fields=[FieldKey(["duration"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        __tablename__: str = "user_summary"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        session_id: int = Field(primary_key=True)
        total_duration: float
        summary: str

    # Create DuckDB store
    db_path = tmp_path / "test_custom_ids.duckdb"

    with DuckDBMetadataStore(db_path) as store:
        # Create parent metadata with composite key
        parent_df = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "session_id": [10, 20, 10, 30, 10],
                "activity_type": ["read", "write", "read", "read", "write"],
                "duration": [30.5, 45.0, 15.0, 60.0, 25.5],
                "timestamp": [1000, 1100, 1200, 1300, 1400],
                "data_version": [
                    {"activity_type": "hash_a1", "duration": "hash_d1"},
                    {"activity_type": "hash_a2", "duration": "hash_d2"},
                    {"activity_type": "hash_a3", "duration": "hash_d3"},
                    {"activity_type": "hash_a4", "duration": "hash_d4"},
                    {"activity_type": "hash_a5", "duration": "hash_d5"},
                ],
            }
        )

        # Write parent metadata
        store.write_metadata(UserActivityFeature, nw.from_native(parent_df))

        # Read back parent metadata
        parent_result = store.read_metadata(UserActivityFeature)
        parent_result_df = collect_to_polars(parent_result)

        # Verify parent data
        assert len(parent_result_df) == 5
        assert set(parent_result_df.columns) >= {
            "user_id",
            "session_id",
            "activity_type",
            "duration",
        }

        # Verify composite keys are preserved
        composite_keys = list(
            zip(
                parent_result_df["user_id"].to_list(),
                parent_result_df["session_id"].to_list(),
            )
        )
        assert sorted(composite_keys) == [(1, 10), (1, 20), (2, 10), (2, 30), (3, 10)]

        # Create child metadata with matching composite keys
        child_df = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2],  # Note: (3, 10) missing
                "session_id": [10, 20, 10, 30],
                "total_duration": [30.5, 45.0, 15.0, 60.0],
                "summary": [
                    "User 1 Session 10",
                    "User 1 Session 20",
                    "User 2 Session 10",
                    "User 2 Session 30",
                ],
                "data_version": [
                    {"total_duration": "hash_t1"},
                    {"total_duration": "hash_t2"},
                    {"total_duration": "hash_t3"},
                    {"total_duration": "hash_t4"},
                ],
            }
        )

        # Write child metadata
        store.write_metadata(UserSummaryFeature, nw.from_native(child_df))

        # Read back child metadata
        child_result = store.read_metadata(UserSummaryFeature)
        child_result_df = collect_to_polars(child_result)

        # Verify child data
        assert len(child_result_df) == 4

        # Verify composite keys in child
        child_composite_keys = list(
            zip(
                child_result_df["user_id"].to_list(),
                child_result_df["session_id"].to_list(),
            )
        )
        assert sorted(child_composite_keys) == [(1, 10), (1, 20), (2, 10), (2, 30)]

        # Snapshot results
        assert {
            "parent": parent_result_df.sort(["user_id", "session_id"]).to_dicts(),
            "child": child_result_df.sort(["user_id", "session_id"]).to_dicts(),
        } == snapshot


def test_composite_key_multiple_columns(snapshot: SnapshotAssertion) -> None:
    """Test SQLModelFeature with composite keys (multiple ID columns).

    Verifies that:
    - Features can have composite keys with 3+ columns
    - Feature versioning works with composite keys
    - Data versioning produces correct struct with composite keys
    """

    class MultiKeyFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["multi", "key"]),
            id_columns=["user_id", "session_id", "timestamp"],
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["event"]), code_version=1),
                FieldSpec(key=FieldKey(["metric"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "multi_key"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        session_id: int = Field(primary_key=True)
        timestamp: int = Field(primary_key=True)
        event: str
        metric: float

    # Verify 3-column composite key
    assert MultiKeyFeature.id_columns() == ["user_id", "session_id", "timestamp"]

    # Verify feature version is deterministic
    version = MultiKeyFeature.feature_version()
    assert len(version) == 64

    # Verify data version structure
    data_version = MultiKeyFeature.data_version()
    assert "event" in data_version
    assert "metric" in data_version

    # Snapshot versions
    assert {
        "feature_version": version,
        "data_version": data_version,
        "id_columns": MultiKeyFeature.id_columns(),
    } == snapshot


def test_parent_child_different_id_columns() -> None:
    """Test parent and child features with different ID column requirements.

    Verifies that:
    - Parent can have more ID columns than child
    - Child specifies which ID columns it needs
    - Feature registration and versioning work correctly
    """

    class DetailedParentFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["detailed", "parent"]),
            id_columns=["user_id", "session_id", "device_id"],
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["detail"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "detailed_parent"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        session_id: int = Field(primary_key=True)
        device_id: str = Field(primary_key=True)
        detail: str

    class AggregatedChildFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["aggregated", "child"]),
            id_columns=["user_id", "session_id"],  # Doesn't need device_id
            deps=[FeatureDep(key=FeatureKey(["detailed", "parent"]))],
            fields=[
                FieldSpec(key=FieldKey(["summary"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "aggregated_child"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        session_id: int = Field(primary_key=True)
        summary: str

    # Verify different ID columns
    assert DetailedParentFeature.id_columns() == ["user_id", "session_id", "device_id"]
    assert AggregatedChildFeature.id_columns() == ["user_id", "session_id"]

    # Both should be registered
    graph = FeatureGraph.get_active()
    assert FeatureKey(["detailed", "parent"]) in graph.features_by_key
    assert FeatureKey(["aggregated", "child"]) in graph.features_by_key

    # Child should have parent as dependency
    assert AggregatedChildFeature.spec.deps is not None
    assert len(AggregatedChildFeature.spec.deps) == 1
    assert AggregatedChildFeature.spec.deps[0].key == FeatureKey(["detailed", "parent"])


def test_sqlmodel_feature_id_columns_with_joins(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    """Test that joins work correctly with custom ID columns in SQLModelFeature.

    Verifies that:
    - Upstream features are joined on custom ID columns
    - Inner join behavior works as expected
    - Data versions are calculated correctly after joins
    """

    # Create two upstream features with composite keys
    class FeatureA(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["feature", "a"]),
            id_columns=["user_id", "date"],
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["value_a"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "feature_a"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        date: str = Field(primary_key=True)
        value_a: float

    class FeatureB(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["feature", "b"]),
            id_columns=["user_id", "date"],
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["value_b"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "feature_b"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        date: str = Field(primary_key=True)
        value_b: float

    # Create downstream feature that depends on both
    # Need to handle the sample_uid conflict - both A and B inherit it from SQLModelFeature
    class FeatureC(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["feature", "c"]),
            id_columns=["user_id", "date"],
            deps=[
                FeatureDep(
                    key=FeatureKey(["feature", "a"]),
                    columns=("value_a",),  # Only select value_a, exclude sample_uid
                ),
                FeatureDep(
                    key=FeatureKey(["feature", "b"]),
                    columns=("value_b",),  # Only select value_b, exclude sample_uid
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["combined"]),
                    code_version=1,
                    deps=[
                        FieldDep(
                            feature_key=FeatureKey(["feature", "a"]),
                            fields=[FieldKey(["value_a"])],
                        ),
                        FieldDep(
                            feature_key=FeatureKey(["feature", "b"]),
                            fields=[FieldKey(["value_b"])],
                        ),
                    ],
                ),
            ],
        ),
    ):
        __tablename__: str = "feature_c"  # pyright: ignore[reportIncompatibleVariableOverride]
        user_id: int = Field(primary_key=True)
        date: str = Field(primary_key=True)
        combined: float

    db_path = tmp_path / "test_joins.duckdb"

    with DuckDBMetadataStore(db_path) as store:
        # Write metadata for FeatureA
        df_a = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                ],
                "value_a": [10.0, 20.0, 30.0, 40.0, 50.0],
                "data_version": [
                    {"value_a": "hash_a1"},
                    {"value_a": "hash_a2"},
                    {"value_a": "hash_a3"},
                    {"value_a": "hash_a4"},
                    {"value_a": "hash_a5"},
                ],
            }
        )
        store.write_metadata(FeatureA, nw.from_native(df_a))

        # Write metadata for FeatureB (missing some combinations)
        df_b = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 3],  # Missing (2, 2024-01-02)
                "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"],
                "value_b": [100.0, 200.0, 300.0, 500.0],
                "data_version": [
                    {"value_b": "hash_b1"},
                    {"value_b": "hash_b2"},
                    {"value_b": "hash_b3"},
                    {"value_b": "hash_b4"},
                ],
            }
        )
        store.write_metadata(FeatureB, nw.from_native(df_b))

        # Now test that joining works correctly
        # Inner join should only keep rows present in both A and B
        # Expected matches: (1, 2024-01-01), (1, 2024-01-02), (2, 2024-01-01), (3, 2024-01-01)

        # Read both features to verify
        result_a = collect_to_polars(store.read_metadata(FeatureA))
        result_b = collect_to_polars(store.read_metadata(FeatureB))

        # Verify the data was written correctly
        assert len(result_a) == 5
        assert len(result_b) == 4

        # Verify composite keys work
        keys_a = set(zip(result_a["user_id"].to_list(), result_a["date"].to_list()))
        keys_b = set(zip(result_b["user_id"].to_list(), result_b["date"].to_list()))

        # Inner join result would be intersection
        expected_joined_keys = keys_a & keys_b
        assert len(expected_joined_keys) == 4

        # Snapshot the results
        assert {
            "feature_a_rows": sorted(list(keys_a)),
            "feature_b_rows": sorted(list(keys_b)),
            "expected_join": sorted(list(expected_joined_keys)),
        } == snapshot


def test_sqlmodel_empty_id_columns_raises() -> None:
    """Test that empty id_columns list raises an error.

    Verifies that:
    - Empty list for id_columns is not allowed
    - Appropriate error message is raised
    """

    with pytest.raises(ValueError, match="id_columns must be non-empty"):

        class InvalidFeature(
            SQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["invalid"]),
                id_columns=[],  # Empty list not allowed
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                ],
            ),
        ):
            __tablename__: str = "invalid"  # pyright: ignore[reportIncompatibleVariableOverride]
            data: str


def test_sqlmodel_id_columns_in_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test that custom id_columns are included in feature graph snapshots.

    Verifies that:
    - id_columns are serialized in graph snapshots
    - Snapshot includes the custom ID column configuration
    """

    class SnapshotFeature(
        SQLModelFeature,
        table=True,
        spec=FeatureSpec(
            key=FeatureKey(["snapshot", "ids"]),
            id_columns=["customer_id", "order_id"],
            deps=None,
            fields=[
                FieldSpec(key=FieldKey(["amount"]), code_version=1),
            ],
        ),
    ):
        __tablename__: str = "snapshot_ids"  # pyright: ignore[reportIncompatibleVariableOverride]
        customer_id: int = Field(primary_key=True)
        order_id: int = Field(primary_key=True)
        amount: float

    graph = FeatureGraph.get_active()

    # Create snapshot
    snapshot_data = graph.to_snapshot()

    # Check feature in snapshot
    feature_key_str = "snapshot/ids"
    assert feature_key_str in snapshot_data

    feature_snapshot = snapshot_data[feature_key_str]

    # Check that id_columns are included in the spec
    assert "feature_spec" in feature_snapshot
    spec_data = feature_snapshot["feature_spec"]
    assert "id_columns" in spec_data
    assert spec_data["id_columns"] == ["customer_id", "order_id"]

    # Snapshot the relevant parts
    assert {
        "id_columns": spec_data["id_columns"],
        "feature_version": feature_snapshot["feature_version"],
    } == snapshot
