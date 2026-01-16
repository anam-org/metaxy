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

    Use ONLY when code changed but computation results would be identical:
    - Dependency graph refactoring (more precise field dependencies)
    - Field structure changes (renaming, splitting, better schema)
    - Code organization improvements (imports, typing, refactoring)

    Do NOT use when computation actually changed:
    - Different algorithm/model → re-run pipeline instead
    - Bug fixes that affect output → re-run pipeline instead
    - New model version → re-run pipeline instead

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
        """Execute field provenance reconciliation for a single feature."""
        if from_snapshot_version is None:
            raise ValueError(
                f"DataVersionReconciliation requires from_snapshot_version for feature {feature_key}"
            )

        import narwhals as nw

        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        feature_key_obj = FeatureKey(feature_key.split("/"))
        feature_key_str = feature_key_obj.to_string()
        graph = FeatureGraph.get_active()
        feature_cls = graph.features_by_key[feature_key_obj]

        self._verify_has_upstream(graph, feature_key_obj, feature_key_str)

        from_feature_version, to_feature_version = self._get_feature_versions(
            store, feature_key_str, from_snapshot_version, snapshot_version
        )

        existing_metadata_df = self._load_existing_metadata(
            store, feature_cls, from_feature_version
        )
        if existing_metadata_df is None or existing_metadata_df.shape[0] == 0:
            return 0

        if dry_run:
            return existing_metadata_df.shape[0]

        df_to_write = self._compute_new_provenance(
            store, feature_cls, existing_metadata_df
        )
        if df_to_write is None:
            return 0

        df_to_write_nw = nw.from_native(df_to_write)
        df_to_write_nw = df_to_write_nw.with_columns(
            nw.lit(to_feature_version).alias("metaxy_feature_version"),
            nw.lit(snapshot_version).alias("metaxy_snapshot_version"),
        )

        self._write_metadata(store, feature_cls, df_to_write_nw)
        return len(df_to_write)

    def _verify_has_upstream(self, graph, feature_key_obj, feature_key_str) -> None:
        """Verify feature has upstream dependencies."""
        plan = graph.get_feature_plan(feature_key_obj)
        has_upstream = plan.deps is not None and len(plan.deps) > 0
        if not has_upstream:
            raise ValueError(
                f"DataVersionReconciliation cannot be used for root feature {feature_key_str}. "
                f"Root features have user-defined field_provenance that cannot be automatically reconciled."
            )

    def _get_feature_versions(
        self,
        store: "MetadataStore",
        feature_key_str: str,
        from_snapshot_version: str,
        to_snapshot_version: str,
    ) -> tuple[str, str]:
        """Query feature versions from snapshot metadata."""
        from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

        from_feature_version = self._query_single_feature_version(
            store, FEATURE_VERSIONS_KEY, feature_key_str, from_snapshot_version
        )
        to_feature_version = self._query_single_feature_version(
            store, FEATURE_VERSIONS_KEY, feature_key_str, to_snapshot_version
        )

        if from_feature_version is None:
            raise ValueError(
                f"Feature {feature_key_str} not found in from_snapshot {from_snapshot_version}"
            )
        if to_feature_version is None:
            raise ValueError(
                f"Feature {feature_key_str} not found in to_snapshot {to_snapshot_version}"
            )

        return from_feature_version, to_feature_version

    def _query_single_feature_version(
        self, store: "MetadataStore", key, feature_key_str: str, snapshot_version: str
    ) -> str | None:
        """Query a single feature version from snapshot metadata."""
        import narwhals as nw

        from metaxy.metadata_store.exceptions import FeatureNotFoundError

        try:
            version_data = store.read_metadata(
                key,
                current_only=False,
                allow_fallback=False,
                filters=[
                    (nw.col("metaxy_snapshot_version") == snapshot_version)
                    & (nw.col("feature_key") == feature_key_str)
                ],
            )
            version_df = version_data.head(1).collect()
            if version_df.shape[0] > 0:
                return str(version_df["metaxy_feature_version"][0])
        except FeatureNotFoundError:
            pass
        return None

    def _load_existing_metadata(
        self, store: "MetadataStore", feature_cls, from_feature_version: str
    ):
        """Load existing metadata with old feature_version."""
        import narwhals as nw

        from metaxy.metadata_store.exceptions import FeatureNotFoundError

        try:
            existing_metadata = store.read_metadata(
                feature_cls,
                current_only=False,
                filters=[nw.col("metaxy_feature_version") == from_feature_version],
                allow_fallback=False,
            )
            return existing_metadata.collect()
        except FeatureNotFoundError:
            return None

    def _compute_new_provenance(
        self, store: "MetadataStore", feature_cls, existing_metadata_df
    ):
        """Compute new provenance using resolve_update."""
        import narwhals as nw

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
        diff_result = store.resolve_update(feature_cls)

        sample_metadata_pl = nw.from_native(sample_metadata.to_native()).to_polars()

        if len(diff_result.changed) > 0:
            changed_pl = nw.from_native(diff_result.changed.to_native()).to_polars()
            new_provenance = changed_pl.select(
                ["sample_uid", "metaxy_provenance_by_field"]
            )
            return sample_metadata_pl.join(new_provenance, on="sample_uid", how="inner")
        elif len(diff_result.added) > 0:
            return nw.from_native(diff_result.added.to_native()).to_polars()
        return None

    def _write_metadata(
        self, store: "MetadataStore", feature_cls, df_to_write_nw
    ) -> None:
        """Write metadata with version override."""
        from metaxy.metadata_store.base import allow_feature_version_override

        with allow_feature_version_override():
            with store.allow_cross_project_writes():
                store.write_metadata(feature_cls, df_to_write_nw)


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
                graph = FeatureGraph.get_active()
                feature_key_obj = FeatureKey(feature_key.split("/"))
                feature_cls = graph.features_by_key[feature_key_obj]

                diff = store.resolve_update(
                    feature_cls,
                    samples=external_df.select(["sample_uid"])
                )

                # Join external metadata with calculated field_provenance
                to_write = external_df.join(diff.added, on="sample_uid", how="inner")

                # Write
                store.write_metadata(feature_cls, to_write)
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
