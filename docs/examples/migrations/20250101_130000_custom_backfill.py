"""Example 2: PythonMigration with S3 backfill logic.

This example demonstrates how to implement a completely custom migration
that loads data from an external source (S3), performs transformations,
and writes metadata to the store.

Use Case:
---------
You need to backfill metadata from an external data source (S3, database,
API, etc.) that contains information not available in your current feature
pipeline. This is common when:
- Migrating from a legacy system
- Importing historical data
- Integrating with external services
- Bulk loading from data lakes

PythonMigration provides complete control over:
- Loading data from any source
- Custom transformations and filtering
- Joining with Metaxy's calculated field_provenance
- Writing results to the store

Migration Workflow:
-------------------
1. Load external metadata (from S3, database, API, etc.)
2. Filter and transform as needed
3. Get field_provenance from Metaxy using store.resolve_update()
4. Join external metadata with field_provenance
5. Write to store using store.write_metadata()

Key Differences from DiffMigration:
-----------------------------------
- Not tied to snapshot versions (can operate within a single snapshot)
- No automatic affected feature computation
- User implements full execute() logic
- Complete control over data flow

Example Usage:
--------------
Place this file in .metaxy/migrations/ directory and run:
    metaxy migrations apply

The migration will load videos from S3, filter by size, join with field
provenance, and write to the metadata store.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import narwhals as nw

from metaxy.migrations.models import MigrationResult, PythonMigration
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


class S3VideoBackfillMigration(PythonMigration):
    """Backfill video metadata from S3 bucket.

    This migration loads video files from an S3 bucket, filters by size,
    and writes metadata to the store. It demonstrates:
    - Loading from external source (S3)
    - Custom filtering logic
    - Joining with Metaxy field_provenance
    - Error handling and progress tracking
    """

    # Required: Unique migration ID
    migration_id: str = "20250101_130000_custom_backfill"

    # Required: Timestamp
    created_at: datetime = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

    # Custom fields for this migration
    s3_bucket: str = "prod-videos"
    s3_prefix: str = "processed/"
    min_size_mb: int = 10  # Only process videos >= 10 MB

    # Feature to backfill (as list of path components)
    target_feature_key: list[str] = ["video", "files"]

    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
        """Return list of affected feature keys.

        Args:
            store: Metadata store (not used in this example)
            project: Project name (not used in this example)

        Returns:
            List containing only the target feature key
        """
        return ["/".join(self.target_feature_key)]

    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Execute S3 backfill.

        Process:
        1. Load video files from S3
        2. Filter by size
        3. Get field_provenance from Metaxy
        4. Join and write to store

        Args:
            store: Metadata store to write to
            project: Project name for event tracking
            dry_run: If True, validate and return count without writing

        Returns:
            MigrationResult with execution details

        Raises:
            Exception: If backfill fails
        """
        start_time = datetime.now(timezone.utc)
        feature_key_str = "/".join(self.target_feature_key)
        errors = {}

        try:
            # Step 1: Load from S3
            # Note: In a real migration, you'd use boto3 to connect to S3
            # For this example, we show the structure
            try:
                import boto3  # pyright: ignore[reportMissingImports]
                import polars as pl

                s3 = boto3.client("s3")
                response = s3.list_objects_v2(
                    Bucket=self.s3_bucket, Prefix=self.s3_prefix
                )

                if "Contents" not in response:
                    # No objects found - return success with 0 rows
                    return MigrationResult(
                        migration_id=self.migration_id,
                        status="completed",
                        features_completed=1,
                        features_failed=0,
                        features_skipped=0,
                        affected_features=[feature_key_str],
                        errors={},
                        rows_affected=0,
                        duration_seconds=(
                            datetime.now(timezone.utc) - start_time
                        ).total_seconds(),
                        timestamp=start_time,
                    )

                # Create DataFrame from S3 objects
                external_data = []
                for obj in response["Contents"]:
                    # Use S3 key as sample_uid
                    # Extract video_id from path (assumes format: prefix/video_id.mp4)
                    key = obj["Key"]
                    video_id = key.replace(self.s3_prefix, "").replace(".mp4", "")

                    external_data.append(
                        {
                            "sample_uid": video_id,
                            "path": f"s3://{self.s3_bucket}/{key}",
                            "size_bytes": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                        }
                    )

                external_df = pl.DataFrame(external_data)

            except ImportError:
                # boto3 not available - for demo purposes, create sample data
                import polars as pl

                external_df = pl.DataFrame(
                    {
                        "sample_uid": ["video_001", "video_002", "video_003"],
                        "path": [
                            f"s3://{self.s3_bucket}/{self.s3_prefix}video_001.mp4",
                            f"s3://{self.s3_bucket}/{self.s3_prefix}video_002.mp4",
                            f"s3://{self.s3_bucket}/{self.s3_prefix}video_003.mp4",
                        ],
                        "size_bytes": [
                            15_000_000,
                            8_000_000,
                            20_000_000,
                        ],  # 15MB, 8MB, 20MB
                        "last_modified": [
                            "2025-01-01T00:00:00",
                            "2025-01-01T01:00:00",
                            "2025-01-01T02:00:00",
                        ],
                    }
                )

            # Step 2: Filter by size (only videos >= min_size_mb)
            min_size_bytes = self.min_size_mb * 1024 * 1024
            filtered_df = external_df.filter(
                pl.col("size_bytes") >= min_size_bytes
            ).select(["sample_uid", "path", "size_bytes", "last_modified"])

            if dry_run:
                return MigrationResult(
                    migration_id=self.migration_id,
                    status="skipped",
                    features_completed=1,
                    features_failed=0,
                    features_skipped=0,
                    affected_features=[feature_key_str],
                    errors={},
                    rows_affected=len(filtered_df),
                    duration_seconds=(
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                    timestamp=start_time,
                )

            # Step 3: Get field_provenance from Metaxy
            # First, get the feature class from the graph
            graph = FeatureGraph.get_active()
            feature_key_obj = FeatureKey(self.target_feature_key)
            feature_cls = graph.features_by_key[feature_key_obj]

            # Call resolve_update with sample_uid list
            # This will calculate field_provenance based on upstream dependencies
            samples_df = filtered_df.select(["sample_uid"])
            diff_result = store.resolve_update(
                feature_cls, samples=nw.from_native(samples_df)
            )

            # Step 4: Join external metadata with field_provenance
            # Use 'added' since these are new samples
            if len(diff_result.added) > 0:
                # Convert both to Polars for joining
                added_df = nw.from_native(diff_result.added.to_native()).to_polars()
                provenance_df = added_df.select(
                    ["sample_uid", "metaxy_provenance_by_field"]
                )

                # Join to get final DataFrame
                df_to_write = filtered_df.join(
                    provenance_df, on="sample_uid", how="inner"
                )

                # Step 5: Write to store
                df_to_write_nw = nw.from_native(df_to_write)
                store.write_metadata(feature_cls, df_to_write_nw)

                rows_affected = len(df_to_write)
            else:
                rows_affected = 0

            # Success
            return MigrationResult(
                migration_id=self.migration_id,
                status="completed",
                features_completed=1,
                features_failed=0,
                features_skipped=0,
                affected_features=[feature_key_str],
                errors={},
                rows_affected=rows_affected,
                duration_seconds=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
                timestamp=start_time,
            )

        except Exception as e:
            # Handle errors gracefully
            error_msg = str(e)
            errors[feature_key_str] = error_msg

            return MigrationResult(
                migration_id=self.migration_id,
                status="failed",
                features_completed=0,
                features_failed=1,
                features_skipped=0,
                affected_features=[],
                errors=errors,
                rows_affected=0,
                duration_seconds=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
                timestamp=start_time,
            )


# Key Takeaways:
# --------------
# 1. PythonMigration gives you full control over the migration process
# 2. You can load data from any external source (S3, database, API, etc.)
# 3. Use store.resolve_update() to get Metaxy's field_provenance
# 4. Join external data with field_provenance before writing
# 5. Handle errors and return appropriate MigrationResult
# 6. Support dry_run for validation without side effects

# Alternative Approaches:
# -----------------------
# Instead of PythonMigration, you could subclass MetadataBackfill (see Example 5)
# which provides a more structured interface specifically for backfill operations.
