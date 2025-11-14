"""Example 5: MetadataBackfill operation subclass.

This example demonstrates how to create a custom MetadataBackfill operation
that can be used in DiffMigration or as a standalone migration.

MetadataBackfill vs PythonMigration:
-------------------------------------
- MetadataBackfill: Structured interface for backfill operations
  - Designed to be reusable across migrations
  - Can be composed with other operations in DiffMigration
  - Focused specifically on backfilling metadata
  - Follows operation contract (execute returns row count)

- PythonMigration: Complete control over entire migration
  - Full flexibility for any migration logic
  - Not reusable as an operation
  - Can't be composed with other operations
  - Better for one-off complex migrations

Use Case:
---------
Create reusable backfill operations that:
- Can be used in multiple migrations
- Load from specific external sources (S3, databases, APIs)
- Have well-defined interfaces and validation
- Can be composed with DataVersionReconciliation
- Are testable in isolation

When to Use MetadataBackfill:
------------------------------
- Reusable backfill logic for a specific data source
- Want to compose with other operations in DiffMigration
- Need operation-level abstraction
- Building a library of backfill operations

When to Use PythonMigration Instead:
-------------------------------------
- One-off migration with unique logic
- Complex multi-step migration workflow
- Don't need operation composability
- Full control over migration lifecycle

Example Usage:
--------------
Option 1: Use in DiffMigration YAML:
    ops:
      - type: myproject.migrations.S3VideoBackfillOperation
        id: backfill_videos
        feature_key: ["video", "files"]
        s3_bucket: prod-videos
        s3_prefix: processed/
        reason: Initial backfill from production

Option 2: Use in DiffMigration Python:
    class MyMigration(DiffMigration):
        ops = [
            {
                "type": "myproject.migrations.S3VideoBackfillOperation",
                "id": "backfill_videos",
                "feature_key": ["video", "files"],
                "s3_bucket": "prod-videos",
                "s3_prefix": "processed/",
                "reason": "Initial backfill"
            }
        ]
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import field_validator

from metaxy.migrations.ops import MetadataBackfill

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


class S3VideoBackfillOperation(MetadataBackfill):
    """Backfill video metadata from S3.

    This operation loads video files from an S3 bucket, filters by criteria,
    joins with Metaxy field_provenance, and writes to the store.

    Designed to be reusable across multiple migrations and composable with
    other operations in DiffMigration.

    Configuration:
    --------------
    - s3_bucket: S3 bucket name
    - s3_prefix: Prefix path within bucket
    - min_size_mb: Minimum file size (MB) to include
    - max_size_mb: Maximum file size (MB) to include
    - file_extensions: List of allowed file extensions
    """

    # S3 configuration
    s3_bucket: str
    s3_prefix: str = ""

    # Filtering criteria
    min_size_mb: int = 1  # Minimum file size
    max_size_mb: int = 1000  # Maximum file size (1GB)
    file_extensions: list[str] = [".mp4", ".mov", ".avi"]

    # Processing options
    batch_size: int = 1000  # Process in batches
    include_metadata: bool = True  # Include S3 object metadata

    @field_validator("s3_bucket")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate S3 bucket name format.

        Args:
            v: Bucket name

        Returns:
            Validated bucket name

        Raises:
            ValueError: If bucket name is invalid
        """
        if not v:
            raise ValueError("s3_bucket cannot be empty")

        # Basic S3 bucket name validation
        if not 3 <= len(v) <= 63:
            raise ValueError(
                f"s3_bucket name must be 3-63 characters, got {len(v)}: {v}"
            )

        # Check for valid characters (simplified)
        import re

        if not re.match(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$", v):
            raise ValueError(
                f"s3_bucket name contains invalid characters: {v}. "
                "Must start/end with letter or number, contain only lowercase "
                "letters, numbers, hyphens, and periods."
            )

        return v

    @field_validator("min_size_mb", "max_size_mb")
    @classmethod
    def validate_size_limits(cls, v: int) -> int:
        """Validate file size limits.

        Args:
            v: Size limit in MB

        Returns:
            Validated size limit

        Raises:
            ValueError: If size limit is invalid
        """
        if v < 0:
            raise ValueError(f"Size limit must be non-negative, got: {v}")

        if v > 10_000:  # 10 GB
            raise ValueError(
                f"Size limit too large (max 10,000 MB), got: {v}. "
                "Consider using multiple backfill operations for very large files."
            )

        return v

    def execute_for_feature(
        self,
        store: "MetadataStore",
        feature_key: str,
        *,
        snapshot_version: str,
        from_snapshot_version: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Execute backfill for a single feature.

        Args:
            store: Metadata store
            feature_key: Feature key string
            snapshot_version: Target snapshot version
            from_snapshot_version: Source snapshot (optional)
            dry_run: If True, don't write to store

        Returns:
            Number of rows affected
        """
        # Example stub implementation
        # In a real implementation, you would load from S3, filter, join, and write
        return 0

    def execute(
        self,
        store: "MetadataStore",
        *,
        from_snapshot_version: str,
        to_snapshot_version: str,
        dry_run: bool = False,
        **kwargs,
    ) -> int:
        """Execute S3 video backfill.

        Process:
        1. List objects in S3 bucket with prefix
        2. Filter by size and extension
        3. Create DataFrame with video metadata
        4. Get field_provenance from Metaxy
        5. Join and write to store

        Args:
            store: Metadata store to write to
            from_snapshot_version: Source snapshot (not used for backfill)
            to_snapshot_version: Target snapshot (not used for backfill)
            dry_run: If True, return count without writing

        Returns:
            Number of rows written (or would be written if dry_run)

        Raises:
            Exception: If backfill fails
        """
        # Step 1: List S3 objects
        video_objects = self._list_s3_objects()

        if len(video_objects) == 0:
            return 0

        # Step 2: Create DataFrame
        external_df = self._create_video_dataframe(video_objects)

        if dry_run:
            return len(external_df)

        # Step 3: Get field_provenance from Metaxy
        # Note: In the real implementation, feature_key would be passed as a parameter
        # graph = FeatureGraph.get_active()
        # feature_key_obj = FeatureKey(feature_key)  # feature_key from execute_for_feature
        # feature_cls = graph.features_by_key[feature_key_obj]

        # For this example, just return 0
        return 0

        # Remaining code is for illustration only:
        """
        graph = FeatureGraph.get_active()
        feature_key_obj = FeatureKey(feature_key)
        feature_cls = graph.features_by_key[feature_key_obj]

        # Get samples list
        samples_df = external_df.select(["sample_uid"])
        diff_result = store.resolve_update(
            feature_cls, samples=nw.from_native(samples_df)
        )

        # Step 4: Join with field_provenance
        if len(diff_result.added) > 0:
            # Convert to Polars for joining
            added_df = nw.from_native(diff_result.added.to_native()).to_polars()
            provenance_df = added_df.select(
                ["sample_uid", "metaxy_provenance_by_field"]
            )

            # Join external data with provenance
            import polars as pl

            if not isinstance(external_df, pl.DataFrame):
                external_df = nw.from_native(external_df).to_polars()

            df_to_write = external_df.join(provenance_df, on="sample_uid", how="inner")

            # Step 5: Write to store
            df_to_write_nw = nw.from_native(df_to_write)
            store.write_metadata(feature_cls, df_to_write_nw)

            return len(df_to_write)

        return 0
        """

    def _list_s3_objects(self) -> list[dict[str, Any]]:
        """List objects in S3 bucket matching criteria.

        Returns:
            List of object metadata dicts with keys:
            - key: S3 object key
            - size: Size in bytes
            - last_modified: Last modified timestamp
            - etag: ETag for versioning
        """
        try:
            import boto3  # pyright: ignore[reportMissingImports]

            s3 = boto3.client("s3")

            # List objects with pagination
            objects = []
            paginator = s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(
                Bucket=self.s3_bucket, Prefix=self.s3_prefix
            ):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    # Filter by size
                    size_mb = obj["Size"] / (1024 * 1024)
                    if not self.min_size_mb <= size_mb <= self.max_size_mb:
                        continue

                    # Filter by extension
                    key = obj["Key"]
                    if not any(key.endswith(ext) for ext in self.file_extensions):
                        continue

                    objects.append(
                        {
                            "key": key,
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                            "etag": obj.get("ETag", "").strip('"'),
                        }
                    )

            return objects

        except ImportError:
            # boto3 not available - return sample data for demo
            return [
                {
                    "key": f"{self.s3_prefix}video_001.mp4",
                    "size": 15_000_000,
                    "last_modified": "2025-01-01T00:00:00",
                    "etag": "abc123",
                },
                {
                    "key": f"{self.s3_prefix}video_002.mp4",
                    "size": 20_000_000,
                    "last_modified": "2025-01-01T01:00:00",
                    "etag": "def456",
                },
            ]

    def _create_video_dataframe(self, objects: list[dict[str, Any]]):
        """Create DataFrame from S3 objects.

        Args:
            objects: List of object metadata dicts

        Returns:
            Polars DataFrame with video metadata
        """
        import polars as pl

        rows = []
        for obj in objects:
            # Extract video ID from key
            key = obj["key"]
            video_id = key.replace(self.s3_prefix, "").rsplit(".", 1)[0]

            row = {
                "sample_uid": video_id,
                "s3_path": f"s3://{self.s3_bucket}/{key}",
                "size_bytes": obj["size"],
                "last_modified": obj["last_modified"],
            }

            if self.include_metadata:
                row["s3_etag"] = obj["etag"]
                row["s3_bucket"] = self.s3_bucket
                row["s3_key"] = key

            rows.append(row)

        return pl.DataFrame(rows)


# Example 2: Database backfill operation
# ---------------------------------------
class DatabaseTableBackfillOperation(MetadataBackfill):
    """Backfill metadata from database table.

    Generic database backfill operation that can work with any SQL database
    supported by SQLAlchemy.
    """

    # Database configuration
    database_url: str  # SQLAlchemy connection string
    table_name: str  # Source table
    sample_uid_column: str = "id"  # Column to use as sample_uid

    # Optional SQL query for filtering/transformation
    where_clause: str | None = None  # e.g., "created_at > '2025-01-01'"
    select_columns: list[str] | None = None  # Columns to include (None = all)

    # Processing options
    batch_size: int = 1000

    def execute_for_feature(
        self,
        store: "MetadataStore",
        feature_key: str,
        *,
        snapshot_version: str,
        from_snapshot_version: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Execute backfill for a single feature.

        Args:
            store: Metadata store
            feature_key: Feature key string
            snapshot_version: Target snapshot version
            from_snapshot_version: Source snapshot (optional)
            dry_run: If True, don't write to store

        Returns:
            Number of rows affected
        """
        # Example stub implementation
        return 0

    def execute(
        self,
        store: "MetadataStore",
        *,
        from_snapshot_version: str,
        to_snapshot_version: str,
        dry_run: bool = False,
        **kwargs,
    ) -> int:
        """Execute database backfill.

        Args:
            store: Metadata store
            from_snapshot_version: Source snapshot
            to_snapshot_version: Target snapshot
            dry_run: If True, return count without writing

        Returns:
            Number of rows written
        """
        # Implementation would:
        # 1. Connect to database using database_url
        # 2. Query table_name with optional where_clause
        # 3. Select specified columns
        # 4. Rename sample_uid_column to sample_uid
        # 5. Get field_provenance from Metaxy
        # 6. Join and write to store

        # For this example, return 0
        # In a real implementation, you'd use SQLAlchemy or similar
        return 0


# Example 3: API backfill operation
# ----------------------------------
class RestAPIBackfillOperation(MetadataBackfill):
    """Backfill metadata from REST API.

    Generic REST API backfill operation with pagination support.
    """

    # API configuration
    api_url: str  # Base API URL
    endpoint: str  # API endpoint path
    api_key: str | None = None  # API key for authentication

    # Request configuration
    query_params: dict[str, str] = {}  # Query parameters
    headers: dict[str, str] = {}  # HTTP headers

    # Pagination
    pagination_type: Literal["page", "cursor", "offset"] = "page"
    page_size: int = 100
    max_pages: int = 100  # Limit to prevent runaway pagination

    # Response parsing
    response_data_key: str = "data"  # Key containing data in response
    sample_uid_field: str = "id"  # Field to use as sample_uid

    def execute_for_feature(
        self,
        store: "MetadataStore",
        feature_key: str,
        *,
        snapshot_version: str,
        from_snapshot_version: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Execute backfill for a single feature.

        Args:
            store: Metadata store
            feature_key: Feature key string
            snapshot_version: Target snapshot version
            from_snapshot_version: Source snapshot (optional)
            dry_run: If True, don't write to store

        Returns:
            Number of rows affected
        """
        # Example stub implementation
        return 0

    def execute(
        self,
        store: "MetadataStore",
        *,
        from_snapshot_version: str,
        to_snapshot_version: str,
        dry_run: bool = False,
        **kwargs,
    ) -> int:
        """Execute API backfill.

        Args:
            store: Metadata store
            from_snapshot_version: Source snapshot
            to_snapshot_version: Target snapshot
            dry_run: If True, return count without writing

        Returns:
            Number of rows written
        """
        # Implementation would:
        # 1. Make paginated API requests
        # 2. Parse JSON responses
        # 3. Extract data from response_data_key
        # 4. Rename sample_uid_field to sample_uid
        # 5. Get field_provenance from Metaxy
        # 6. Join and write to store

        # For this example, return 0
        return 0


# Key Takeaways:
# --------------
# 1. MetadataBackfill provides structured interface for backfill operations
# 2. Operations are reusable across multiple migrations
# 3. Can be used in DiffMigration YAML or Python definitions
# 4. Supports composition with other operations
# 5. Focused specifically on backfilling metadata from external sources
# 6. Follows operation contract (execute returns row count)
# 7. Use Pydantic validators for configuration validation
# 8. Support dry_run for preview without side effects

# When to Create MetadataBackfill Operations:
# --------------------------------------------
# - Reusable backfill logic for specific data sources
# - Want operation-level abstraction and composability
# - Building a library of backfill operations
# - Need to combine with DataVersionReconciliation
# - Want to use in multiple migrations

# When to Use PythonMigration Instead:
# -------------------------------------
# - One-off migration with unique logic
# - Complex multi-step workflow
# - Don't need operation composability
# - Full control over migration lifecycle
# - Not reusable as an operation

# MetadataBackfill vs PythonMigration:
# -------------------------------------
# MetadataBackfill:
#   + Reusable operation
#   + Composable with other operations
#   + Structured interface
#   + Can be used in YAML
#   - Less flexible (operation contract)
#   - Focused on backfills only

# PythonMigration:
#   + Complete flexibility
#   + Any migration logic
#   + Full lifecycle control
#   - Not reusable as operation
#   - Can't be used in DiffMigration
#   - Must implement full execute()
