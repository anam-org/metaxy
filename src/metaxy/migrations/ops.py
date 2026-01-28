"""Migration operation types."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


class BaseOperation(BaseSettings, ABC):
    """Base class for all migration operations with environment variable support.

    Operations are instantiated from YAML configs and execute on individual features.
    Subclasses implement execute_for_feature() to perform the actual migration logic.

    Environment variables are automatically read using pydantic_settings. Define config
    fields as regular Pydantic fields and they will be populated from env vars or config dict.

    The 'type' field is automatically computed from the class's module and name.

    Example:
        class PostgreSQLBackfill(BaseOperation):
            postgresql_url: str  # Reads from POSTGRESQL_URL env var or config dict
            batch_size: int = 1000  # Optional with default

            def execute_for_feature(self, store, feature_key, *, snapshot_version, from_snapshot_version=None, dry_run=False):
                # Implementation here
                return 0
    """

    model_config = SettingsConfigDict(
        extra="ignore",  # Ignore extra fields like 'type' and 'features' from YAML
        frozen=True,
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _substitute_env_vars(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Substitute ${VAR} patterns with environment variables.

        Example:
            postgresql_url: "${POSTGRESQL_URL}" -> postgresql_url: "postgresql://..."
        """
        import os
        import re

        def substitute_value(value):
            if isinstance(value, str):
                # Replace ${VAR} with os.environ.get('VAR')
                def replacer(match):
                    var_name = match.group(1)
                    env_value = os.environ.get(var_name)
                    if env_value is None:
                        raise ValueError(f"Environment variable {var_name} is not set")
                    return env_value

                return re.sub(r"\$\{([^}]+)\}", replacer, value)
            return value

        # Create a new dict to avoid mutating the input
        result = {}
        for key, value in data.items():
            result[key] = substitute_value(value)
        return result

    @property
    def type(self) -> str:
        """Return the fully qualified class name for this operation."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    @abstractmethod
    def execute_for_feature(
        self,
        store: "MetadataStore",
        feature_key: str,
        *,
        snapshot_version: str,
        from_snapshot_version: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Execute operation for a single feature.

        Args:
            store: Metadata store to operate on
            feature_key: Feature key string (e.g., "video/scene")
            snapshot_version: Target snapshot version
            from_snapshot_version: Source snapshot version (optional, for cross-snapshot migrations)
            dry_run: If True, only validate and return count without executing

        Returns:
            Number of rows affected

        Raises:
            Exception: If operation fails
        """
        pass


class DataVersionReconciliation(BaseOperation):
    """Reconcile field provenance when feature definition changes BUT computation is unchanged.

    This operation applies to affected features specified in the migration configuration.
    Feature keys are provided in the migration YAML operations list.

    This operation:
    1. For each affected feature, derives old/new feature_versions from snapshots
    2. Finds rows with old feature_version
    3. Recalculates field_provenance based on new feature definition
    4. Writes new rows with updated feature_version and provenance_by_field
    5. Preserves all user metadata columns (immutable)

    Use ONLY when code changed but computation results would be identical:
    - Dependency graph refactoring (more precise field dependencies)
    - Field structure changes (renaming, splitting, better schema)
    - Code organization improvements (imports, typing, refactoring)

    Do NOT use when computation actually changed:
    - Different algorithm/model → re-run pipeline instead
    - Bug fixes that affect output → re-run pipeline instead
    - New model version → re-run pipeline instead

    Feature versions are automatically derived from the migration's snapshot versions.

    Example YAML:
        operations:
          - type: metaxy.migrations.ops.DataVersionReconciliation
            features: ["video/scene", "video/frames"]
    """

    def execute_for_feature(
        self,
        store: "MetadataStore",
        feature_key: str,
        *,
        snapshot_version: str,
        from_snapshot_version: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Execute field provenance reconciliation for a single feature.

        Only works for features with upstream dependencies. For root features
        (no upstream), field_provenance are user-defined and cannot be automatically
        reconciled - user must re-run their computation pipeline.

        Process:
        1. Verify feature has upstream dependencies
        2. Query old and new feature_versions from snapshot metadata
        3. Load existing metadata with old feature_version
        4. Use resolve_update() to calculate expected field_provenance based on current upstream
        5. Join existing user metadata with new field_provenance
        6. Write with new feature_version and snapshot_version

        Args:
            store: Metadata store
            feature_key: Feature key string (e.g., "examples/child")
            snapshot_version: Target snapshot version (new state)
            from_snapshot_version: Source snapshot version (old state, required for this operation)
            dry_run: If True, return row count without executing

        Returns:
            Number of rows affected

        Raises:
            ValueError: If feature has no upstream dependencies (root feature) or from_snapshot_version not provided
        """
        if from_snapshot_version is None:
            raise ValueError(f"DataVersionReconciliation requires from_snapshot_version for feature {feature_key}")

        to_snapshot_version = snapshot_version
        import narwhals as nw

        from metaxy.metadata_store.base import allow_feature_version_override
        from metaxy.metadata_store.exceptions import FeatureNotFoundError
        from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        feature_key_obj = FeatureKey(feature_key.split("/"))
        feature_key_str = feature_key_obj.to_string()
        graph = FeatureGraph.get_active()

        # 1. Verify feature has upstream dependencies
        plan = graph.get_feature_plan(feature_key_obj)
        has_upstream = plan.deps is not None and len(plan.deps) > 0

        if not has_upstream:
            raise ValueError(
                f"DataVersionReconciliation cannot be used for root feature {feature_key_str}. "
                f"Root features have user-defined field_provenance that cannot be automatically reconciled. "
                f"User must re-run their computation pipeline to generate new data."
            )

        # 2. Query feature versions from snapshot metadata
        try:
            from_version_data = store.read_metadata(
                FEATURE_VERSIONS_KEY,
                current_only=False,
                allow_fallback=False,
                filters=[
                    (nw.col("metaxy_snapshot_version") == from_snapshot_version)
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
                    (nw.col("metaxy_snapshot_version") == to_snapshot_version)
                    & (nw.col("feature_key") == feature_key_str)
                ],
            )
        except FeatureNotFoundError:
            to_version_data = None

        # Extract feature versions from lazy frames
        # Since we filter by snapshot_version and feature_key, there should be exactly one row
        from_feature_version: str | None = None
        to_feature_version: str | None = None

        if from_version_data is not None:
            from_version_df = from_version_data.head(1).collect()
            if from_version_df.shape[0] > 0:
                from_feature_version = str(from_version_df["metaxy_feature_version"][0])
            else:
                from_version_data = None

        if to_version_data is not None:
            to_version_df = to_version_data.head(1).collect()
            if to_version_df.shape[0] > 0:
                to_feature_version = str(to_version_df["metaxy_feature_version"][0])
            else:
                to_version_data = None

        if from_version_data is None:
            raise ValueError(f"Feature {feature_key_str} not found in from_snapshot {from_snapshot_version}")
        if to_version_data is None:
            raise ValueError(f"Feature {feature_key_str} not found in to_snapshot {to_snapshot_version}")

        assert from_feature_version is not None
        assert to_feature_version is not None

        # 3. Load existing metadata with old feature_version
        try:
            existing_metadata = store.read_metadata(
                feature_key_obj,
                current_only=False,
                filters=[nw.col("metaxy_feature_version") == from_feature_version],
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
            if c
            not in [
                "metaxy_provenance_by_field",
                "metaxy_feature_version",
                "metaxy_snapshot_version",
            ]
        ]
        sample_metadata = existing_metadata_df.select(user_columns)

        # 5. Use resolve_update to calculate field_provenance based on current upstream
        # Don't pass samples - let resolve_update auto-load upstream and calculate provenance_by_field
        diff_result = store.resolve_update(feature_key_obj)

        # Convert to Polars for the join to avoid cross-backend issues
        sample_metadata_pl = nw.from_native(sample_metadata.to_native()).to_polars()

        # Use 'changed' for reconciliation (field_provenance changed due to upstream)
        # Use 'added' for new feature materialization
        # Convert results to Polars for consistent joining
        if len(diff_result.changed) > 0:
            changed_pl = nw.from_native(diff_result.changed.to_native()).to_polars()
            new_provenance = changed_pl.select(["sample_uid", "metaxy_provenance_by_field"])
            df_to_write = sample_metadata_pl.join(new_provenance, on="sample_uid", how="inner")
        elif len(diff_result.added) > 0:
            df_to_write = nw.from_native(diff_result.added.to_native()).to_polars()
        else:
            return 0

        # 6. Write with new feature_version and snapshot_version
        # Wrap in Narwhals for write_metadata
        df_to_write_nw = nw.from_native(df_to_write)
        df_to_write_nw = df_to_write_nw.with_columns(
            nw.lit(to_feature_version).alias("metaxy_feature_version"),
            nw.lit(to_snapshot_version).alias("metaxy_snapshot_version"),
        )

        with allow_feature_version_override():
            store.write_metadata(feature_key_obj, df_to_write_nw)

        return len(df_to_write)


class MetadataBackfill(BaseOperation, ABC):
    """Base class for metadata backfill operations.

    Users subclass this to implement custom backfill logic with complete
    control over the entire process: loading, transforming, joining, filtering,
    and writing metadata.

    The user implements execute_for_feature() and can:
    - Load metadata from any external source (S3, database, API, etc.)
    - Perform custom transformations and filtering
    - Join with Metaxy's calculated field_provenance however they want
    - Write results to the store

    Example Subclass:
        class S3VideoBackfill(MetadataBackfill):
            s3_bucket: str
            s3_prefix: str
            min_size_mb: int = 10

            def execute_for_feature(
                self,
                store,
                feature_key,
                *,
                snapshot_version,
                from_snapshot_version=None,
                dry_run=False
            ):
                import boto3
                from metaxy.models.feature import FeatureGraph
                from metaxy.models.types import FeatureKey

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

                # Get field provenance from Metaxy
                feature_key_obj = FeatureKey(feature_key.split("/"))

                diff = store.resolve_update(
                    feature_key_obj,
                    samples=external_df.select(["sample_uid"])
                )

                # Join external metadata with calculated field_provenance
                to_write = external_df.join(diff.added, on="sample_uid", how="inner")

                # Write
                store.write_metadata(feature_key_obj, to_write)
                return len(to_write)

    Example YAML:
        operations:
          - type: "myproject.migrations.S3VideoBackfill"
            features: ["video/files"]
            s3_bucket: "prod-videos"
            s3_prefix: "processed/"
            min_size_mb: 10
    """

    # No additional required fields - user subclasses add their own

    @abstractmethod
    def execute_for_feature(
        self,
        store: "MetadataStore",
        feature_key: str,
        *,
        snapshot_version: str,
        from_snapshot_version: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """User implements backfill logic for a single feature.

        User has complete control over:
        - Loading external metadata (S3, database, API, files, etc.)
        - Transforming and filtering data
        - Joining with Metaxy's field_provenance
        - Writing to store

        Args:
            store: Metadata store to write to
            feature_key: Feature key string (e.g., "video/files")
            snapshot_version: Target snapshot version
            from_snapshot_version: Source snapshot version (optional, for cross-snapshot backfills)
            dry_run: If True, validate and return count without writing

        Returns:
            Number of rows written (or would be written if dry_run)

        Raises:
            Exception: If backfill fails (will be recorded in migration progress)
        """
        pass
