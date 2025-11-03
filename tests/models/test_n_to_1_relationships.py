"""Comprehensive tests for N->1 (many-to-one) relationships."""

import narwhals as nw
import polars as pl

from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import BaseFeatureSpec, FeatureDep, FieldSpec
from metaxy.models.lineage import LineageRelationship


class TestNToOneRelationships:
    """Test suite for N->1 relationship functionality."""

    def test_basic_n_to_1_aggregation(self, graph):
        """Test that N->1 correctly aggregates multiple parents to single child."""

        # Define sensor readings (many records per hour)
        class SensorReadings(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/readings",
                id_columns=["sensor_id", "hour", "minute"],
            ),
        ):
            pass

        # Define hourly stats (one record per hour) with explicit Aggregation lineage
        class HourlyStats(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/hourly_stats",
                id_columns=["sensor_id", "hour"],
                lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]),
                deps=[
                    FeatureDep(
                        feature="sensor/readings",
                        # No rename needed - sensor_id and hour columns match
                        # minute is not in child ID columns - causes N->1
                    )
                ],
            ),
        ):
            pass

        # Create test data - 3 readings per hour
        readings_data = pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1", "s2", "s2"],
                "hour": [10, 10, 10, 10, 10],
                "minute": [0, 15, 30, 0, 45],
                "temperature": [20.1, 20.5, 20.8, 19.5, 19.9],
                "provenance_by_field": [
                    {"temperature": "hash_s1_00"},
                    {"temperature": "hash_s1_15"},
                    {"temperature": "hash_s1_30"},
                    {"temperature": "hash_s2_00"},
                    {"temperature": "hash_s2_45"},
                ],
            }
        )

        # Test joining
        joiner = NarwhalsJoiner()
        readings_ref = nw.from_native(readings_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"sensor/readings": readings_ref},
            feature_spec=HourlyStats.spec(),
            feature_plan=graph.get_feature_plan(HourlyStats.spec().key),
            # No rename needed - sensor_id and hour column names are the same
        )

        # Verify result
        result_df = result.collect().to_native()

        # Should aggregate 5 readings -> 2 hourly stats (s1/10, s2/10)
        assert result_df.shape[0] == 2

        # Check columns
        assert "sensor_id" in result_df.columns
        assert "hour" in result_df.columns
        assert "temperature" in result_df.columns

        # Verify aggregation happened
        s1_row = result_df.filter(pl.col("sensor_id") == "s1")
        assert len(s1_row) == 1
        assert s1_row["hour"][0] == 10

    def test_n_to_1_with_multiple_fields(self, graph):
        """Test N->1 with multiple fields in provenance struct."""

        class DetailedReadings(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/detailed",
                id_columns=["sensor_id", "timestamp"],
                fields=[
                    FieldSpec(key="temperature", code_version="1"),
                    FieldSpec(key="humidity", code_version="1"),
                    FieldSpec(key="pressure", code_version="1"),
                ],
            ),
        ):
            pass

        class HourlyAggregates(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/aggregates",
                id_columns=["sensor_id", "hour"],
                lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]),
                fields=[
                    FieldSpec(key="avg_temp", code_version="1"),
                    FieldSpec(key="avg_humidity", code_version="1"),
                ],
                deps=[
                    FeatureDep(
                        feature="sensor/detailed",
                        rename={"timestamp": "hour"},  # timestamp renamed to hour
                    )
                ],
            ),
        ):
            pass

        # Create test data with multiple fields
        detailed_data = pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1"],
                "timestamp": [100, 100, 100],  # Same hour
                "temperature": [20.1, 20.5, 20.8],
                "humidity": [45.0, 46.0, 44.0],
                "pressure": [1013.0, 1013.5, 1014.0],
                "provenance_by_field": [
                    {"temperature": "t1", "humidity": "h1", "pressure": "p1"},
                    {"temperature": "t2", "humidity": "h2", "pressure": "p2"},
                    {"temperature": "t3", "humidity": "h3", "pressure": "p3"},
                ],
            }
        )

        # Test joining
        joiner = NarwhalsJoiner()
        detailed_ref = nw.from_native(detailed_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"sensor/detailed": detailed_ref},
            feature_spec=HourlyAggregates.spec(),
            feature_plan=graph.get_feature_plan(HourlyAggregates.spec().key),
            upstream_renames={"sensor/detailed": {"timestamp": "hour"}},
        )

        result_df = result.collect().to_native()

        # Should aggregate 3 readings -> 1 aggregate
        assert result_df.shape[0] == 1
        assert result_df["sensor_id"][0] == "s1"
        assert result_df["hour"][0] == 100

    def test_n_to_1_with_metadata_store(self, graph):
        """Test N->1 through metadata store resolve_update."""

        class Events(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user/events",
                id_columns=["user_id", "event_id"],
                fields=[FieldSpec(key="event_type", code_version="1")],
            ),
        ):
            pass

        class Sessions(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user/sessions",
                id_columns=["user_id", "session_id"],
                lineage=LineageRelationship.aggregation(on=["user_id", "session_id"]),
                fields=[FieldSpec(key="summary", code_version="1")],
                deps=[
                    FeatureDep(
                        feature="user/events",
                        rename={
                            "event_id": "session_id"
                        },  # event_id renamed to session_id
                    )
                ],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            # Write events metadata (many)
            events_metadata = pl.DataFrame(
                {
                    "user_id": ["u1"] * 4 + ["u2"] * 4,
                    "event_id": ["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4"],
                    "event_type": [
                        "click",
                        "scroll",
                        "click",
                        "submit",
                        "view",
                        "exit",
                        "login",
                        "browse",
                    ],
                    "provenance_by_field": [
                        {"event_type": f"hash_{i}"} for i in range(8)
                    ],
                }
            )
            store.write_metadata(Events, nw.from_native(events_metadata))

            # Resolve update for Sessions
            diff = store.resolve_update(Sessions)

            # Check aggregation: 8 events -> fewer sessions
            assert diff.added is not None
            added_df = diff.added.to_polars()

            # We expect 3 unique sessions (e1, e2, e3, e4 but deduplicated by user/session)
            # Actually we expect 4 unique (user_id, session_id) pairs
            assert added_df.shape[0] == 4  # u1/e1, u1/e2, u2/e3, u2/e4

    def test_extreme_n_to_1_ratio(self, graph):
        """Test N->1 with extreme ratio (100:1)."""

        class ManyLogs(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="system/logs",
                id_columns=["request_id", "log_seq"],
            ),
        ):
            pass

        class RequestSummary(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="system/requests",
                id_columns=["request_id"],
                lineage=LineageRelationship.aggregation(on=["request_id"]),
                deps=[
                    FeatureDep(
                        feature="system/logs",
                        # No rename needed - request_id column names match
                        # log_seq not mapped - causes N->1
                    )
                ],
            ),
        ):
            pass

        # Create 100 logs per request
        logs_data = pl.DataFrame(
            {
                "request_id": ["r1"] * 50 + ["r2"] * 50,
                "log_seq": list(range(50)) + list(range(50)),
                "message": [f"log_{i}" for i in range(100)],
                "provenance_by_field": [{"default": f"hash_{i}"} for i in range(100)],
            }
        )

        joiner = NarwhalsJoiner()
        logs_ref = nw.from_native(logs_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"system/logs": logs_ref},
            feature_spec=RequestSummary.spec(),
            feature_plan=graph.get_feature_plan(RequestSummary.spec().key),
            # No rename needed - request_id column names match
        )

        result_df = result.collect().to_native()

        # Should aggregate 100 logs -> 2 requests
        assert result_df.shape[0] == 2

        r1_row = result_df.filter(pl.col("request_id") == "r1")
        assert len(r1_row) == 1

    def test_n_to_1_mixed_with_one_to_one(self, graph):
        """Test feature with both N->1 and 1->1 dependencies."""

        class Metadata(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="metadata",
                id_columns=["entity_id"],
            ),
        ):
            pass

        class Details(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="details",
                id_columns=["entity_id", "detail_seq"],
            ),
        ):
            pass

        class Combined(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="combined",
                id_columns=["entity_id"],
                lineage=LineageRelationship.aggregation(on=["entity_id"]),
                deps=[
                    FeatureDep(
                        feature="metadata",
                        # 1->1: entity_id maps directly
                    ),
                    FeatureDep(
                        feature="details",
                        # No rename needed - entity_id column names match
                        # detail_seq not mapped - N->1
                    ),
                ],
            ),
        ):
            pass

        metadata_data = pl.DataFrame(
            {
                "entity_id": ["e1", "e2"],
                "meta_value": ["meta1", "meta2"],
                "provenance_by_field": [
                    {"default": "meta_hash1"},
                    {"default": "meta_hash2"},
                ],
            }
        )

        details_data = pl.DataFrame(
            {
                "entity_id": ["e1", "e1", "e1", "e2", "e2"],
                "detail_seq": [0, 1, 2, 0, 1],
                "detail_value": ["d1", "d2", "d3", "d4", "d5"],
                "provenance_by_field": [
                    {"default": f"detail_hash{i}"} for i in range(5)
                ],
            }
        )

        joiner = NarwhalsJoiner()

        result, mapping = joiner.join_upstream(
            upstream_refs={
                "metadata": nw.from_native(metadata_data.lazy()),
                "details": nw.from_native(details_data.lazy()),
            },
            feature_spec=Combined.spec(),
            feature_plan=graph.get_feature_plan(Combined.spec().key),
            # No rename needed - entity_id column names match
        )

        result_df = result.collect().to_native()

        # Should have 2 entities (1->1 with metadata, N->1 with details)
        assert result_df.shape[0] == 2

        # Check both metadata and details columns present
        assert "__upstream_metadata__provenance_by_field" in result_df.columns
        assert "__upstream_details__provenance_by_field" in result_df.columns

    def test_provenance_aggregation_correctly_hashes(self, graph):
        """Test that provenance aggregation correctly hashes all values together."""

        class MultiParent(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="multi/parent",
                id_columns=["group_id", "item_id"],
            ),
        ):
            pass

        class SingleChild(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="single/child",
                id_columns=["group_id"],
                lineage=LineageRelationship.aggregation(on=["group_id"]),
                deps=[
                    FeatureDep(
                        feature="multi/parent",
                        # No rename needed - group_id column names match
                    )
                ],
            ),
        ):
            pass

        parent_data = pl.DataFrame(
            {
                "group_id": ["g1", "g1", "g1"],
                "item_id": [1, 2, 3],
                "value": ["a", "b", "c"],
                "provenance_by_field": [
                    {"value": "hash_a"},
                    {"value": "hash_b"},
                    {"value": "hash_c"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        parent_ref = nw.from_native(parent_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"multi/parent": parent_ref},
            feature_spec=SingleChild.spec(),
            feature_plan=graph.get_feature_plan(SingleChild.spec().key),
            # No rename needed - group_id column names match
        )

        result_df = result.collect().to_native()

        # Should have 1 group
        assert result_df.shape[0] == 1

        # Get the provenance
        provenance_col = "__upstream_multi/parent__provenance_by_field"
        provenance = result_df[provenance_col][0]

        # The provenance should be the AGGREGATED hash of all three values
        # Not just the first one!
        # Extract the dict from the provenance column if it's a Series
        if isinstance(provenance, pl.Series):
            provenance = provenance[0] if len(provenance) > 0 else {}

        # Check that it's NOT just the first value
        # If properly aggregated, it should be a hash of all three values
        assert provenance["value"] != "hash_a", (
            f"Should not be just first value, got: {provenance['value']}"
        )

        # The value should be hashed (longer than original)
        assert len(provenance["value"]) > len("hash_a"), "Value should be a hash"

    def test_n_to_1_deterministic_ordering(self, graph):
        """Test that N->1 aggregation is deterministic."""

        class UnorderedParent(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="unordered/parent",
                id_columns=["key", "subkey"],
            ),
        ):
            pass

        class OrderedChild(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="ordered/child",
                id_columns=["key"],
                lineage=LineageRelationship.aggregation(on=["key"]),
                deps=[
                    FeatureDep(
                        feature="unordered/parent",
                        # No rename needed - key column names match
                    )
                ],
            ),
        ):
            pass

        # Create data in different orders
        data1 = pl.DataFrame(
            {
                "key": ["k1", "k1", "k1"],
                "subkey": [3, 1, 2],  # Out of order
                "value": ["c", "a", "b"],
                "provenance_by_field": [
                    {"default": "h3"},
                    {"default": "h1"},
                    {"default": "h2"},
                ],
            }
        )

        data2 = pl.DataFrame(
            {
                "key": ["k1", "k1", "k1"],
                "subkey": [1, 2, 3],  # In order
                "value": ["a", "b", "c"],
                "provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )

        joiner = NarwhalsJoiner()

        # Join with different orderings
        result1, _ = joiner.join_upstream(
            upstream_refs={"unordered/parent": nw.from_native(data1.lazy())},
            feature_spec=OrderedChild.spec(),
            feature_plan=graph.get_feature_plan(OrderedChild.spec().key),
            # No rename needed - key column names match
        )

        result2, _ = joiner.join_upstream(
            upstream_refs={"unordered/parent": nw.from_native(data2.lazy())},
            feature_spec=OrderedChild.spec(),
            feature_plan=graph.get_feature_plan(OrderedChild.spec().key),
            # No rename needed - key column names match
        )

        df1 = result1.collect().to_native()
        df2 = result2.collect().to_native()

        # Results should be identical regardless of input order
        assert df1.shape == df2.shape
        assert df1["key"][0] == df2["key"][0]

    def test_n_to_1_with_nulls(self, graph):
        """Test N->1 with null values in provenance."""

        class NullableParent(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="nullable/parent",
                id_columns=["group", "item"],
            ),
        ):
            pass

        class NonNullChild(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="nonnull/child",
                id_columns=["group"],
                lineage=LineageRelationship.aggregation(on=["group"]),
                deps=[
                    FeatureDep(
                        feature="nullable/parent",
                        # No rename needed - group column names match
                    )
                ],
            ),
        ):
            pass

        parent_data = pl.DataFrame(
            {
                "group": ["g1", "g1", "g1"],
                "item": [1, 2, 3],
                "value": ["a", None, "c"],
                "provenance_by_field": [
                    {"default": "hash1"},
                    None,  # Null provenance
                    {"default": "hash3"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        parent_ref = nw.from_native(parent_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"nullable/parent": parent_ref},
            feature_spec=NonNullChild.spec(),
            feature_plan=graph.get_feature_plan(NonNullChild.spec().key),
            # No rename needed - group column names match
        )

        result_df = result.collect().to_native()

        # Should handle nulls gracefully
        assert result_df.shape[0] == 1
        assert result_df["group"][0] == "g1"

    def test_n_to_1_cross_product(self, graph):
        """Test N->1 with cross product from multiple parents."""

        class LocationReadings(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="location/readings",
                id_columns=["location_id", "sensor_id"],
            ),
        ):
            pass

        class SensorTypes(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/types",
                id_columns=["sensor_id", "type_id"],
            ),
        ):
            pass

        class LocationSummary(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="location/summary",
                id_columns=["location_id"],
                lineage=LineageRelationship.aggregation(on=["location_id"]),
                deps=[
                    FeatureDep(
                        feature="location/readings",
                        # No rename needed - location_id column names match
                    ),
                    FeatureDep(
                        feature="sensor/types",
                        # No mapping - will cause cross product per location
                    ),
                ],
            ),
        ):
            pass

        readings_data = pl.DataFrame(
            {
                "location_id": ["l1", "l1", "l2", "l2"],
                "sensor_id": ["s1", "s2", "s3", "s4"],
                "reading": [10.0, 20.0, 30.0, 40.0],
                "provenance_by_field": [{"default": f"reading_{i}"} for i in range(4)],
            }
        )

        types_data = pl.DataFrame(
            {
                "sensor_id": ["s1", "s2"],
                "type_id": ["temperature", "humidity"],
                "provenance_by_field": [
                    {"default": "type_1"},
                    {"default": "type_2"},
                ],
            }
        )

        joiner = NarwhalsJoiner()

        result, _ = joiner.join_upstream(
            upstream_refs={
                "location/readings": nw.from_native(readings_data.lazy()),
                "sensor/types": nw.from_native(types_data.lazy()),
            },
            feature_spec=LocationSummary.spec(),
            feature_plan=graph.get_feature_plan(LocationSummary.spec().key),
            # No rename needed for location_id
            upstream_renames={
                "sensor/types": {
                    "sensor_id": "type_sensor_id"
                }  # Rename sensor_id to avoid conflict
            },
        )

        result_df = result.collect().to_native()

        # Cross product: 2 locations × (2 readings × 2 types) = 2 locations
        # After grouping by location_id
        assert result_df.shape[0] == 2

    def test_expansion_relationship_no_aggregation(self, graph):
        """Test that Expansion relationship does NOT aggregate (1:N handled in load_input)."""

        class Video(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="video/source",
                id_columns=["video_id"],
            ),
        ):
            pass

        class VideoFrames(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="video/frames",
                id_columns=["video_id", "frame_id"],
                lineage=LineageRelationship.expansion(
                    on=["video_id"],  # Parent ID columns
                    id_generation_pattern="sequential",
                ),
                deps=[
                    FeatureDep(
                        feature="video/source",
                        # No id_columns_mapping needed - names match and 'on' specifies parent columns
                    )
                ],
            ),
        ):
            pass

        video_data = pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "duration": [30.0, 45.0],
                "provenance_by_field": [
                    {"duration": "hash_v1"},
                    {"duration": "hash_v2"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        video_ref = nw.from_native(video_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"video/source": video_ref},
            feature_spec=VideoFrames.spec(),
            feature_plan=graph.get_feature_plan(VideoFrames.spec().key),
            # No rename needed - handled by ExpansionRelationship.on
        )

        result_df = result.collect().to_native()

        # Should have same number of rows as source (no aggregation)
        # The actual expansion to multiple frames would happen in load_input()
        assert result_df.shape[0] == 2
        assert set(result_df["video_id"]) == {"v1", "v2"}

        # Check that duration column is preserved
        assert "duration" in result_df.columns

    def test_identity_relationship_no_aggregation(self, graph):
        """Test that Identity relationship does NOT aggregate."""

        class UserProfile(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user/profile",
                id_columns=["user_id"],
            ),
        ):
            pass

        class UserExtended(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user/extended",
                id_columns=["user_id"],
                lineage=LineageRelationship.identity(),  # Explicit 1:1
                deps=[
                    FeatureDep(
                        feature="user/profile",
                    )
                ],
            ),
        ):
            pass

        profile_data = pl.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "name": ["Alice", "Bob", "Charlie"],
                "provenance_by_field": [
                    {"name": "hash_alice"},
                    {"name": "hash_bob"},
                    {"name": "hash_charlie"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        profile_ref = nw.from_native(profile_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"user/profile": profile_ref},
            feature_spec=UserExtended.spec(),
            feature_plan=graph.get_feature_plan(UserExtended.spec().key),
        )

        result_df = result.collect().to_native()

        # Should have same number of rows (no aggregation)
        assert result_df.shape[0] == 3
        assert set(result_df["user_id"]) == {"u1", "u2", "u3"}

        # Provenance should be preserved as-is (no hashing/aggregation)
        provenance_col = "__upstream_user/profile__provenance_by_field"
        for i, expected_hash in enumerate(["hash_alice", "hash_bob", "hash_charlie"]):
            provenance = result_df[provenance_col][i]
            if isinstance(provenance, pl.Series):
                provenance = provenance[0] if len(provenance) > 0 else {}
            assert provenance["name"] == expected_hash, (
                "Identity should preserve exact provenance"
            )

    def test_manual_provenance_aggregation_logic(self, graph):
        """Test the _hash_struct_list helper method directly."""

        joiner = NarwhalsJoiner()

        # Test with single struct
        single = joiner._hash_struct_list([{"field1": "value1"}])
        assert single == {"field1": "value1"}

        # Test with multiple structs
        multiple = joiner._hash_struct_list(
            [
                {"field1": "value1", "field2": "a"},
                {"field1": "value2", "field2": "b"},
                {"field1": "value3", "field2": "c"},
            ]
        )

        # Should hash each field separately
        assert "field1" in multiple
        assert "field2" in multiple

        # Values should be hashed, not just first
        assert multiple["field1"] != "value1"
        assert multiple["field2"] != "a"

        # Test with empty list
        empty = joiner._hash_struct_list([])
        assert empty == {}

        # Test with None values
        with_none = joiner._hash_struct_list([None, {"field1": "value1"}])
        assert with_none == {"field1": "value1"}
