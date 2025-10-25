"""Migration operation types."""

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import pydantic

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


class BaseOperation(pydantic.BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
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
    def execute(
        self,
        store: "MetadataStore",
        *,
        from_snapshot_version: str,
        to_snapshot_version: str,
        dry_run: bool = False,
    ) -> int:
        """Execute the operation.

        Args:
            store: Metadata store to operate on
            from_snapshot_version: Source snapshot version (old state)
            to_snapshot_version: Target snapshot version (new state)
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
        return hashlib.sha256(content.encode()).hexdigest()


class DataVersionReconciliation(BaseOperation):
    """Reconcile data versions when feature definition changes BUT computation is unchanged.

    This operation:
    1. Derives old/new feature_versions from migration's from_snapshot_version/to_snapshot_version
    2. Finds rows with old feature_version
    3. Recalculates data_versions based on new feature definition
    4. Writes new rows with updated feature_version and data_version
    5. Preserves all user metadata columns (immutable)

    Use ONLY when code changed but computation results would be identical:
    - Dependency graph refactoring (more precise field dependencies)
    - Field structure changes (renaming, splitting, better schema)
    - Code organization improvements (imports, typing, refactoring)

    Do NOT use when computation actually changed:
    - Different algorithm/model → re-run pipeline instead
    - Bug fixes that affect output → re-run pipeline instead
    - New model version → re-run pipeline instead

    Feature versions are automatically derived from the migration's snapshot versions,
    eliminating redundancy since each snapshot uniquely identifies all feature versions.

    Example YAML:
        - id: "reconcile_stt_transcription"
          type: "metaxy.migrations.ops.DataVersionReconciliation"
          feature_key: ["speech", "transcription"]
          reason: "Fixed dependency: now depends only on audio field instead of entire video. Transcription logic unchanged."
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: Literal["metaxy.migrations.ops.DataVersionReconciliation"] = (
        "metaxy.migrations.ops.DataVersionReconciliation"
    )
    reason: str

    def execute(
        self,
        store: "MetadataStore",
        *,
        from_snapshot_version: str,
        to_snapshot_version: str,
        dry_run: bool = False,
    ) -> int:
        """Execute data version reconciliation.

        Only works for features with upstream dependencies. For root features
        (no upstream), data_versions are user-defined and cannot be automatically
        reconciled - user must re-run their computation pipeline.

        Process:
        1. Verify feature has upstream dependencies
        2. Query old and new feature_versions from snapshot metadata
        3. Load existing metadata with old feature_version
        4. Use resolve_update() to calculate expected data_versions based on current upstream
        5. Join existing user metadata with new data_versions
        6. Write with new feature_version and snapshot_version

        Args:
            store: Metadata store
            from_snapshot_version: Source snapshot version (old state)
            to_snapshot_version: Target snapshot version (new state)
            dry_run: If True, return row count without executing

        Returns:
            Number of rows affected

        Raises:
            ValueError: If feature has no upstream dependencies (root feature)
        """
        import narwhals as nw

        from metaxy.metadata_store.base import (
            FEATURE_VERSIONS_KEY,
            allow_feature_version_override,
        )
        from metaxy.metadata_store.exceptions import FeatureNotFoundError
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        feature_key = FeatureKey(self.feature_key)
        feature_key_str = feature_key.to_string()
        graph = FeatureGraph.get_active()
        feature_cls = graph.features_by_key[feature_key]

        # 1. Verify feature has upstream dependencies
        plan = graph.get_feature_plan(feature_key)
        has_upstream = plan.deps is not None and len(plan.deps) > 0

        if not has_upstream:
            raise ValueError(
                f"DataVersionReconciliation cannot be used for root feature {feature_key_str}. "
                f"Root features have user-defined data_versions that cannot be automatically reconciled. "
                f"User must re-run their computation pipeline to generate new data."
            )

        # 2. Query feature versions from snapshot metadata
        try:
            from_version_data = store.read_metadata(
                FEATURE_VERSIONS_KEY,
                current_only=False,
                allow_fallback=False,
                filters=[
                    (nw.col("snapshot_version") == from_snapshot_version)
                    & (nw.col("feature_key") == feature_key_str)
                ],
            )
        except FeatureNotFoundError:
            from_version_data = None

        try:
            to_version_data = store.read_metadata(
                FEATURE_VERSIONS_KEY,
                current_only=False,
                allow_fallback=False,
                filters=[
                    (nw.col("snapshot_version") == to_snapshot_version)
                    & (nw.col("feature_key") == feature_key_str)
                ],
            )
        except FeatureNotFoundError:
            to_version_data = None

        # Extract feature versions from lazy frames
        from_feature_version: str | None = None
        to_feature_version: str | None = None

        if from_version_data is not None:
            from_version_df = from_version_data.head(1).collect()
            if from_version_df.shape[0] > 0:
                from_feature_version = str(from_version_df["feature_version"][0])
            else:
                from_version_data = None

        if to_version_data is not None:
            to_version_df = to_version_data.head(1).collect()
            if to_version_df.shape[0] > 0:
                to_feature_version = str(to_version_df["feature_version"][0])
            else:
                to_version_data = None

        if from_version_data is None:
            raise ValueError(
                f"Feature {feature_key_str} not found in from_snapshot {from_snapshot_version}"
            )
        if to_version_data is None:
            raise ValueError(
                f"Feature {feature_key_str} not found in to_snapshot {to_snapshot_version}"
            )

        assert from_feature_version is not None
        assert to_feature_version is not None

        # 3. Load existing metadata with old feature_version
        try:
            existing_metadata = store.read_metadata(
                feature_cls,
                current_only=False,
                filters=[nw.col("feature_version") == from_feature_version],
                allow_fallback=False,
            )
        except FeatureNotFoundError:
            # Feature doesn't exist yet - nothing to migrate
            return 0

        # Collect to check existence and get row count
        existing_metadata_df = existing_metadata.collect()
        if existing_metadata_df.shape[0] == 0:
            # Already migrated (idempotent)
            return 0

        if dry_run:
            return existing_metadata_df.shape[0]

        # 4. Get sample metadata (exclude system columns)
        user_columns = [
            c
            for c in existing_metadata_df.columns
            if c not in ["data_version", "feature_version", "snapshot_version"]
        ]
        sample_metadata = existing_metadata_df.select(user_columns)

        # 5. Use resolve_update to calculate data_versions based on current upstream
        # Convert to Polars for the join to avoid cross-backend issues
        sample_metadata_pl = nw.from_native(sample_metadata.to_native()).to_polars()

        diff_result = store.resolve_update(feature_cls, sample_df=sample_metadata_pl)

        # Use 'changed' for reconciliation (data_versions changed due to upstream)
        # Use 'added' for new feature materialization
        # Convert results to Polars for consistent joining
        if len(diff_result.changed) > 0:
            changed_pl = nw.from_native(diff_result.changed.to_native()).to_polars()
            new_data_versions = changed_pl.select(["sample_uid", "data_version"])
            df_to_write = sample_metadata_pl.join(
                new_data_versions, on="sample_uid", how="inner"
            )
        elif len(diff_result.added) > 0:
            df_to_write = nw.from_native(diff_result.added.to_native()).to_polars()
        else:
            return 0

        # 6. Write with new feature_version and snapshot_version
        # Wrap in Narwhals for write_metadata
        df_to_write_nw = nw.from_native(df_to_write)
        df_to_write_nw = df_to_write_nw.with_columns(
            nw.lit(to_feature_version).alias("feature_version"),
            nw.lit(to_snapshot_version).alias("snapshot_version"),
        )

        with allow_feature_version_override():
            store.write_metadata(feature_cls, df_to_write_nw)

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
                        "sample_uid": obj['Key'],
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
                feature_cls = graph.features_by_key[FeatureKey(self.feature_key)]
                diff = store.resolve_update(
                    feature_cls,
                    sample_df=external_df.select(["sample_uid"])
                )

                # Join external metadata with calculated data_versions
                to_write = external_df.join(diff.added, on="sample_uid", how="inner")

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
    def execute(
        self,
        store: "MetadataStore",
        *,
        from_snapshot_version: str,
        to_snapshot_version: str,
        dry_run: bool = False,
        **kwargs,
    ) -> int:
        """User implements full backfill logic.

        User has complete control over:
        - Loading external metadata (S3, database, API, files, etc.)
        - Transforming and filtering data
        - Joining with Metaxy's data_versions
        - Writing to store

        Args:
            store: Metadata store to write to
            from_snapshot_version: Source snapshot version (old state)
            to_snapshot_version: Target snapshot version (new state)
            dry_run: If True, validate and return count without writing

        Returns:
            Number of rows written (or would be written if dry_run)

        Raises:
            Exception: If backfill fails (will be recorded in migration progress)
        """
        pass
