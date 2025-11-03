"""Test lineage relationships and their effect on provenance calculation."""

import narwhals as nw
import polars as pl

from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import BaseFeatureSpec, FeatureDep, FieldSpec
from metaxy.models.lineage import LineageRelationship


class TestLineageProvenanceHandling:
    """Test that lineage relationships correctly handle provenance calculation."""

    def test_aggregation_runs_in_db_and_produces_flat_table(self, graph):
        """Test that AggregationRelationship aggregates parent metadata in DB."""

        # Parent feature with multiple readings per hour
        class SensorReadings(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/readings",
                id_columns=["sensor_id", "hour", "minute"],
                fields=[
                    FieldSpec(key="temperature", code_version="1"),
                    FieldSpec(key="humidity", code_version="1"),
                ],
            ),
        ):
            pass

        # Child feature aggregating to hourly
        class HourlyStats(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/hourly",
                id_columns=["sensor_id", "hour"],
                lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]),
                fields=[
                    FieldSpec(key="avg_temp", code_version="1"),
                    FieldSpec(key="avg_humidity", code_version="1"),
                ],
                deps=[
                    FeatureDep(
                        feature="sensor/readings",
                        # No mapping needed - names match
                    )
                ],
            ),
        ):
            pass

        # Create parent data with multiple readings per hour
        readings_data = pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1", "s2", "s2"],
                "hour": [10, 10, 10, 10, 10],
                "minute": [0, 15, 30, 0, 45],
                "temperature": [20.1, 20.5, 20.8, 19.5, 19.9],
                "humidity": [45.0, 46.0, 44.0, 50.0, 51.0],
                "provenance_by_field": [
                    {"temperature": "t1_00", "humidity": "h1_00"},
                    {"temperature": "t1_15", "humidity": "h1_15"},
                    {"temperature": "t1_30", "humidity": "h1_30"},
                    {"temperature": "t2_00", "humidity": "h2_00"},
                    {"temperature": "t2_45", "humidity": "h2_45"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        readings_ref = nw.from_native(readings_data.lazy())  # Lazy = runs in DB

        result, mapping = joiner.join_upstream(
            upstream_refs={"sensor/readings": readings_ref},
            feature_spec=HourlyStats.spec(),
            feature_plan=graph.get_feature_plan(HourlyStats.spec().key),
        )

        # Result should be aggregated (flat table)
        result_df = result.collect().to_native()

        # Should have 2 rows (one per sensor/hour combination)
        assert result_df.shape[0] == 2

        # Check that provenance was aggregated (hashed together)
        provenance_col = "__upstream_sensor/readings__provenance_by_field"

        # s1's provenance should be hash of 3 readings
        s1_row = result_df.filter(pl.col("sensor_id") == "s1")
        s1_provenance = s1_row[provenance_col][0]

        # Should NOT be any of the original values (it's hashed)
        assert s1_provenance["temperature"] not in ["t1_00", "t1_15", "t1_30"]
        assert s1_provenance["humidity"] not in ["h1_00", "h1_15", "h1_30"]

        # Should be a hash (16 chars for xxhash64)
        assert len(s1_provenance["temperature"]) == 16
        assert len(s1_provenance["humidity"]) == 16

        # s2's provenance should be different (different inputs)
        s2_row = result_df.filter(pl.col("sensor_id") == "s2")
        s2_provenance = s2_row[provenance_col][0]

        assert s2_provenance["temperature"] != s1_provenance["temperature"]
        assert s2_provenance["humidity"] != s1_provenance["humidity"]

        # Result is flat normalized table ready for DiffResolver
        assert "sensor_id" in result_df.columns
        assert "hour" in result_df.columns
        assert provenance_col in result_df.columns

    def test_expansion_shares_parent_provenance(self, graph):
        """Test that ExpansionRelationship shares parent provenance across children."""

        # Parent feature
        class Document(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="doc/source",
                id_columns=["doc_id"],
                fields=[
                    FieldSpec(key="content", code_version="1"),
                ],
            ),
        ):
            pass

        # Child feature that expands to chunks
        class DocumentChunks(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="doc/chunks",
                id_columns=["doc_id", "chunk_id"],
                lineage=LineageRelationship.expansion(
                    on=["doc_id"],  # Parent columns for grouping
                    id_generation_pattern="sequential",
                ),
                fields=[
                    FieldSpec(key="chunk_text", code_version="1"),
                ],
                deps=[
                    FeatureDep(
                        feature="doc/source",
                        # No mapping needed - doc_id matches
                    )
                ],
            ),
        ):
            pass

        # Create parent data
        doc_data = pl.DataFrame(
            {
                "doc_id": ["d1", "d2"],
                "content": ["Document 1 content", "Document 2 content"],
                "provenance_by_field": [
                    {"content": "hash_d1_content"},
                    {"content": "hash_d2_content"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        doc_ref = nw.from_native(doc_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"doc/source": doc_ref},
            feature_spec=DocumentChunks.spec(),
            feature_plan=graph.get_feature_plan(DocumentChunks.spec().key),
        )

        result_df = result.collect().to_native()

        # Should NOT aggregate (same number of rows as parent)
        # Actual expansion happens in load_input()
        assert result_df.shape[0] == 2

        # Check that parent provenance is preserved as-is
        provenance_col = "__upstream_doc/source__provenance_by_field"

        d1_row = result_df.filter(pl.col("doc_id") == "d1")
        d1_provenance = d1_row[provenance_col][0]

        # Should be exact parent provenance (not hashed/aggregated)
        assert d1_provenance["content"] == "hash_d1_content"

        d2_row = result_df.filter(pl.col("doc_id") == "d2")
        d2_provenance = d2_row[provenance_col][0]
        assert d2_provenance["content"] == "hash_d2_content"

        # Result is flat table ready for expansion in load_input
        assert "doc_id" in result_df.columns
        assert provenance_col in result_df.columns
        # chunk_id not present yet (generated in load_input)
        assert "chunk_id" not in result_df.columns

    def test_identity_preserves_exact_provenance(self, graph):
        """Test that IdentityRelationship preserves exact provenance."""

        class UserProfile(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user/profile",
                id_columns=["user_id"],
                fields=[
                    FieldSpec(key="name", code_version="1"),
                    FieldSpec(key="email", code_version="1"),
                ],
            ),
        ):
            pass

        class UserEnriched(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user/enriched",
                id_columns=["user_id"],
                lineage=LineageRelationship.identity(),  # Explicit 1:1
                fields=[
                    FieldSpec(key="full_profile", code_version="1"),
                ],
                deps=[FeatureDep(feature="user/profile")],
            ),
        ):
            pass

        profile_data = pl.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "name": ["Alice", "Bob", "Charlie"],
                "email": [
                    "alice@example.com",
                    "bob@example.com",
                    "charlie@example.com",
                ],
                "provenance_by_field": [
                    {"name": "hash_alice", "email": "hash_alice_email"},
                    {"name": "hash_bob", "email": "hash_bob_email"},
                    {"name": "hash_charlie", "email": "hash_charlie_email"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        profile_ref = nw.from_native(profile_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"user/profile": profile_ref},
            feature_spec=UserEnriched.spec(),
            feature_plan=graph.get_feature_plan(UserEnriched.spec().key),
        )

        result_df = result.collect().to_native()

        # Should have same number of rows (no aggregation)
        assert result_df.shape[0] == 3

        # Check that provenance is preserved exactly
        provenance_col = "__upstream_user/profile__provenance_by_field"

        for i, user_id in enumerate(["u1", "u2", "u3"]):
            row = result_df.filter(pl.col("user_id") == user_id)
            provenance = row[provenance_col][0]

            # Should be exact values from input
            if user_id == "u1":
                assert provenance["name"] == "hash_alice"
                assert provenance["email"] == "hash_alice_email"
            elif user_id == "u2":
                assert provenance["name"] == "hash_bob"
                assert provenance["email"] == "hash_bob_email"
            else:
                assert provenance["name"] == "hash_charlie"
                assert provenance["email"] == "hash_charlie_email"

        # Result is flat normalized table
        assert "user_id" in result_df.columns
        assert "name" in result_df.columns
        assert "email" in result_df.columns
        assert provenance_col in result_df.columns

    def test_aggregation_with_lazy_frame_runs_in_db(self, graph):
        """Verify that aggregation with LazyFrame actually runs in DB (not materialized early)."""

        class Events(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="events",
                id_columns=["session_id", "event_id"],
                fields=[FieldSpec(key="action", code_version="1")],
            ),
        ):
            pass

        class Sessions(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sessions",
                id_columns=["session_id"],
                lineage=LineageRelationship.aggregation(on=["session_id"]),
                fields=[FieldSpec(key="summary", code_version="1")],
                deps=[FeatureDep(feature="events")],
            ),
        ):
            pass

        # Large dataset that would be expensive to materialize
        events_data = pl.DataFrame(
            {
                "session_id": ["s1"] * 100 + ["s2"] * 100,
                "event_id": list(range(100)) * 2,
                "action": ["click"] * 200,
                "provenance_by_field": [{"action": f"hash_{i}"} for i in range(200)],
            }
        )

        joiner = NarwhalsJoiner()
        # Using lazy frame - should stay lazy through aggregation
        events_ref = nw.from_native(events_data.lazy())

        # This should build the query plan but NOT execute yet
        result, mapping = joiner.join_upstream(
            upstream_refs={"events": events_ref},
            feature_spec=Sessions.spec(),
            feature_plan=graph.get_feature_plan(Sessions.spec().key),
        )

        # Result should still be lazy (aggregation happens in DB when collected)
        assert hasattr(result.to_native(), "_ldf"), "Result should be lazy"

        # Only when we collect does it execute
        result_df = result.collect().to_native()

        # Should be aggregated to 2 sessions
        assert result_df.shape[0] == 2

        # Verify provenance was aggregated
        provenance_col = "__upstream_events__provenance_by_field"
        assert provenance_col in result_df.columns

        # Each session should have aggregated provenance
        s1_provenance = result_df.filter(pl.col("session_id") == "s1")[provenance_col][
            0
        ]
        assert len(s1_provenance["action"]) == 16  # Hash length
