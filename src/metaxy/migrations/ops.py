"""Migration operation types."""

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import pydantic

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


class BaseOperation(pydantic.BaseModel, ABC):
    """Base class for all migration operations.

    All operations must have:
    - id: Unique identifier within migration
    - type: Full class path for polymorphic deserialization (must be Literal in subclasses)
    - feature_key: Root feature this operation affects
    - reason: Human-readable explanation

    Subclasses implement execute() to perform the actual migration logic.

    Note: The 'type' field must be defined as a Literal in each subclass
    for Pydantic discriminated unions to work.
    """

    id: str  # Required, user-provided or auto-generated
    # type field must be Literal in subclasses for discriminated unions
    feature_key: list[str]
    reason: str

    @abstractmethod
    def execute(self, store: "MetadataStore", *, dry_run: bool = False) -> int:
        """Execute the operation.

        Args:
            store: Metadata store to operate on
            dry_run: If True, only validate and return count without executing

        Returns:
            Number of rows affected

        Raises:
            Exception: If operation fails
        """
        pass

    def operation_config_hash(self) -> str:
        """Generate hash of operation config (excluding id).

        Used to detect if operation content changed after partial migration.

        Returns:
            16-character hex hash
        """
        content = self.model_dump_json(exclude={"id"}, by_alias=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class DataVersionReconciliation(BaseOperation):
    """Reconcile data versions when feature definition changes.

    This operation:
    1. Finds rows with old feature_version
    2. Recalculates data_versions based on new feature definition
    3. Writes new rows with updated feature_version and data_version
    4. Preserves all user metadata columns (immutable)

    Used when:
    - Container code_version changes
    - Feature dependencies change
    - Any change that affects feature_version hash

    Example YAML:
        - id: "reconcile_video_abc_to_def"
          type: "metaxy.migrations.ops.DataVersionReconciliation"
          feature_key: ["video", "processing"]
          from: "abc12345"
          to: "def67890"
          reason: "Updated frame extraction algorithm"
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: Literal["metaxy.migrations.ops.DataVersionReconciliation"] = (
        "metaxy.migrations.ops.DataVersionReconciliation"
    )
    from_: str = pydantic.Field(alias="from")  # Old feature_version
    to: str  # New feature_version
    reason: str

    def execute(self, store: "MetadataStore", *, dry_run: bool = False) -> int:
        """Execute data version reconciliation.

        Only works for features with upstream dependencies. For root features
        (no upstream), data_versions are user-defined and cannot be automatically
        reconciled - user must re-run their computation pipeline.

        Process:
        1. Verify feature has upstream dependencies
        2. Load existing metadata with old feature_version (from_)
        3. Use resolve_update() to calculate expected data_versions based on current upstream
        4. Join existing user metadata with new data_versions
        5. Write with new feature_version (to)

        Args:
            store: Metadata store
            dry_run: If True, return row count without executing

        Returns:
            Number of rows affected

        Raises:
            ValueError: If feature has no upstream dependencies (root feature)
        """
        import polars as pl

        from metaxy.metadata_store.base import allow_feature_version_override
        from metaxy.metadata_store.exceptions import FeatureNotFoundError
        from metaxy.models.feature import FeatureRegistry
        from metaxy.models.types import FeatureKey

        feature_key = FeatureKey(self.feature_key)
        registry = FeatureRegistry.get_active()
        feature_cls = registry.features_by_key[feature_key]

        # 1. Verify feature has upstream dependencies
        plan = registry.get_feature_plan(feature_key)
        has_upstream = plan.deps is not None and len(plan.deps) > 0

        if not has_upstream:
            raise ValueError(
                f"DataVersionReconciliation cannot be used for root feature {feature_key.to_string()}. "
                f"Root features have user-defined data_versions that cannot be automatically reconciled. "
                f"User must re-run their computation pipeline to generate new data."
            )

        # 2. Load existing metadata with old feature_version
        try:
            existing_metadata = store.read_metadata(
                feature_cls,
                current_only=False,
                filters=pl.col("feature_version") == self.from_,
                allow_fallback=False,
            )
        except FeatureNotFoundError:
            # Feature doesn't exist yet - nothing to migrate
            return 0

        if len(existing_metadata) == 0:
            # Already migrated (idempotent)
            return 0

        if dry_run:
            return len(existing_metadata)

        # 3. Get sample metadata (exclude data_version and feature_version)
        user_columns = [
            c
            for c in existing_metadata.columns
            if c not in ["data_version", "feature_version"]
        ]
        sample_metadata = existing_metadata.select(user_columns)

        # 4. Use resolve_update to calculate data_versions based on current upstream
        diff_result = store.resolve_update(feature_cls, sample_df=sample_metadata)

        # Use 'changed' for reconciliation (data_versions changed due to upstream)
        # Use 'added' for new feature materialization
        if len(diff_result.changed) > 0:
            new_data_versions = diff_result.changed.select(
                ["sample_id", "data_version"]
            )
            df_to_write = sample_metadata.join(
                new_data_versions, on="sample_id", how="inner"
            )
        elif len(diff_result.added) > 0:
            df_to_write = diff_result.added
        else:
            return 0

        # 5. Write with new feature_version (to)
        df_to_write = df_to_write.with_columns(pl.lit(self.to).alias("feature_version"))

        with allow_feature_version_override():
            store.write_metadata(feature_cls, df_to_write)

        return len(df_to_write)


class MetadataBackfill(BaseOperation, ABC):
    """Base class for metadata backfill operations.

    Users subclass this to implement custom backfill logic with complete
    control over the entire process: loading, transforming, joining, filtering,
    and writing metadata.

    The user implements execute() and can:
    - Load metadata from any external source (S3, database, API, etc.)
    - Perform custom transformations and filtering
    - Join with Metaxy's calculated data_versions however they want
    - Write results to the store

    Example Subclass:
        class S3VideoBackfill(MetadataBackfill):
            type: Literal["myproject.migrations.S3VideoBackfill"]
            s3_bucket: str
            s3_prefix: str
            min_size_mb: int = 10

            def execute(self, store, *, dry_run=False):
                import boto3

                # Load from S3
                s3 = boto3.client('s3')
                objects = s3.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=self.s3_prefix
                )

                external_df = pl.DataFrame([
                    {
                        "sample_id": obj['Key'],
                        "path": f"s3://{self.s3_bucket}/{obj['Key']}",
                        "size_bytes": obj['Size']
                    }
                    for obj in objects['Contents']
                ])

                # Filter
                external_df = external_df.filter(
                    pl.col("size_bytes") > self.min_size_mb * 1024 * 1024
                )

                if dry_run:
                    return len(external_df)

                # Get data versions from Metaxy
                feature_cls = registry.features_by_key[FeatureKey(self.feature_key)]
                diff = store.resolve_update(
                    feature_cls,
                    sample_df=external_df.select(["sample_id"])
                )

                # Join external metadata with calculated data_versions
                to_write = external_df.join(diff.added, on="sample_id", how="inner")

                # Write
                store.write_metadata(feature_cls, to_write)
                return len(to_write)

    Example YAML:
        - id: "backfill_videos_from_s3"
          type: "myproject.migrations.S3VideoBackfill"
          feature_key: ["video", "files"]
          s3_bucket: "prod-videos"
          s3_prefix: "processed/"
          min_size_mb: 10
          reason: "Initial backfill from production S3 bucket"
    """

    # No additional required fields - user subclasses add their own

    @abstractmethod
    def execute(self, store: "MetadataStore", *, dry_run: bool = False) -> int:
        """User implements full backfill logic.

        User has complete control over:
        - Loading external metadata (S3, database, API, files, etc.)
        - Transforming and filtering data
        - Joining with Metaxy's data_versions
        - Writing to store

        Args:
            store: Metadata store to write to
            dry_run: If True, validate and return count without writing

        Returns:
            Number of rows written (or would be written if dry_run)

        Raises:
            Exception: If backfill fails (will be recorded in migration progress)
        """
        pass
