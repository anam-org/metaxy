"""System table storage layer for metadata store.

Provides type-safe access to system tables using struct-based storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl

from metaxy.metadata_store.exceptions import SystemDataNotFoundError
from metaxy.metadata_store.system import (
    FEATURE_VERSIONS_KEY,
)
from metaxy.metadata_store.system.events import (
    Event,
)
from metaxy.metadata_store.system.keys import EVENTS_KEY
from metaxy.metadata_store.system.models import POLARS_SCHEMAS, FeatureVersionsModel
from metaxy.models.constants import (
    METAXY_DEFINITION_VERSION,
    METAXY_PROJECT_VERSION,
)
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey, PushResult

if TYPE_CHECKING:
    from metaxy.metadata_store import MetadataStore
    from metaxy.models.feature_definition import FeatureDefinition
    from metaxy.models.feature_selection import FeatureSelection


class SystemTableStorage:
    """Storage layer for system tables.

    Provides type-safe access to snapshots and events.
    Uses struct-based storage (not JSON/bytes) for efficient queries.

    Usage:
        ```python
        with SystemTableStorage(store) as storage:
            storage.write_event(event)
        ```
    """

    def __init__(self, store: MetadataStore):
        """Initialize storage layer.

        Args:
            store: Metadata store to use for system tables
        """
        self.store = store

    def write_event(self, event: Event) -> None:
        """Write event to system table using typed event models.

        Args:
            event: A typed event created via Event classmethods

        Note:
            The store must already be open when calling this method.
        """
        record = event.to_polars()
        self.store.write(EVENTS_KEY, record)

    def push_graph_snapshot(
        self,
        *,
        project: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> PushResult:
        """Record features for a project with a graph snapshot version.

        This should be called during CD (Continuous Deployment) to record what
        feature versions are being deployed. Typically invoked via `metaxy push`.

        Records features for the specified project with the same project_version,
        representing a consistent state of the feature graph based on code definitions.

        The project_version is a deterministic hash of all feature_version hashes
        in the graph, making it idempotent - calling multiple times with the
        same feature definitions produces the same project_version.

        This method detects three scenarios:
        1. New snapshot (computational changes): No existing rows with this project_version
        2. Definition changes: Snapshot exists but some features have different definition_version
        3. No changes: Snapshot exists with identical definition_versions for all features

        Args:
            project: Project name to push features for. If None, uses MetaxyConfig.get().project.
                Raises ValueError if neither is set.
            tags: Optional dictionary of custom tags to attach to the snapshot
                     (e.g., git commit SHA).

        Returns:
            PushResult with project version and list of updated features.

        Raises:
            ValueError: If no project is specified and MetaxyConfig.get().project is None.

        !!! note:
            The store must already be open when calling this method.

            This method automatically loads feature definitions from the metadata store
            before creating the snapshot. This ensures that any external feature dependencies
            are resolved with their actual definitions, preventing incorrect version
            calculations from stale external feature definitions.
        """
        from metaxy.config import MetaxyConfig

        tags = tags or {}
        graph = FeatureGraph.get_active()

        if project is None:
            # Try to infer from graph - only works if all non-external features share the same project
            projects_in_graph = {
                defn.project for defn in graph.feature_definitions_by_key.values() if not defn.is_external
            }
            if len(projects_in_graph) == 1:
                project = projects_in_graph.pop()
            elif len(projects_in_graph) == 0:
                raise ValueError("No features in active graph to push.")
            else:
                # Multiple projects in graph - try config
                project = MetaxyConfig.get().project
                if project is None:
                    raise ValueError(
                        f"Project is required for push_graph_snapshot. Graph contains features from "
                        f"multiple projects: {sorted(projects_in_graph)}. "
                        f"Set 'project' in metaxy.toml or pass project argument."
                    )

        # Now generate snapshot for only this project's features
        project_features = graph.to_snapshot(project=project)

        if not project_features:
            raise ValueError(f"No features found for project '{project}' in the active graph.")

        # Compute project-scoped snapshot version (uses feature_definition_version, excludes external features)
        project_version = graph.get_project_version(project)

        # Check if the latest push for this project already matches the current version
        latest_pushed_version = self._read_latest_project_version(project)
        already_pushed = latest_pushed_version == project_version

        # Convert to DataFrame - need to serialize feature_spec dict to JSON string
        # and add metaxy_project_version and recorded_at columns
        import json
        from datetime import datetime, timezone

        current_snapshot = pl.concat(
            [
                FeatureVersionsModel.model_validate(
                    {
                        "feature_key": k,
                        **{
                            field: (json.dumps(val) if field in ("feature_spec", "feature_schema") else val)
                            for field, val in v.items()
                        },
                        METAXY_PROJECT_VERSION: project_version,
                        "recorded_at": datetime.now(timezone.utc),
                        "tags": json.dumps(tags),
                    }
                ).to_polars()
                for k, v in project_features.items()
            ]
        )

        # Initialize to_push
        to_push = current_snapshot  # Will be updated if snapshot already exists

        if already_pushed:
            latest_pushed_snapshot = self._read_latest_snapshot_data(project_version, project)
            # let's identify features that have updated definitions since the last push
            # Join full current snapshot with latest pushed (keeping all columns)
            pushed_with_current = current_snapshot.join(
                latest_pushed_snapshot.select(
                    "feature_key",
                    pl.col(METAXY_DEFINITION_VERSION).alias(f"{METAXY_DEFINITION_VERSION}_pushed"),
                ),
                on=["feature_key"],
                how="left",
            )

            to_push = pl.concat(
                [
                    # these are records that for some reason have not been pushed previously
                    pushed_with_current.filter(pl.col(f"{METAXY_DEFINITION_VERSION}_pushed").is_null()),
                    # these are the records with actual changes
                    pushed_with_current.filter(pl.col(f"{METAXY_DEFINITION_VERSION}_pushed").is_not_null()).filter(
                        pl.col(METAXY_DEFINITION_VERSION) != pl.col(f"{METAXY_DEFINITION_VERSION}_pushed")
                    ),
                ]
            ).drop(f"{METAXY_DEFINITION_VERSION}_pushed")

        if len(to_push) > 0:
            self.store.write(FEATURE_VERSIONS_KEY, to_push)

        # updated_features only populated when updating existing features
        updated_features = to_push["feature_key"].to_list() if already_pushed and len(to_push) > 0 else []

        return PushResult(
            project_version=project_version,
            already_pushed=already_pushed,
            updated_features=updated_features,
        )

    def _read_system_metadata(self, key: FeatureKey) -> nw.LazyFrame[Any]:
        """Read system metadata.

        System tables are handled specially by MetadataStore.read - they don't
        require feature plan resolution when with_feature_history=True.

        Note:
            The store must already be open when calling this method.

        Returns:
            LazyFrame if table exists, empty LazyFrame with correct schema if it doesn't
        """
        try:
            # read handles system tables specially (no feature plan needed)
            return self.store.read(key, with_feature_history=True)
        except SystemDataNotFoundError:
            return nw.from_native(pl.DataFrame(schema=POLARS_SCHEMAS[key])).lazy()

    def _read_latest_snapshot_data(
        self,
        project_version: str,
        project: str,
    ) -> pl.DataFrame:
        """Read the latest snapshot data for a given snapshot version and project.

        The same snapshot version may include multiple features as their non-topological
        metadata such as Pydantic fields or spec.metadata/tags change. This method
        retrieves the latest feature data for each feature pushed to the metadata store.

        Args:
            project_version: The snapshot version to query.
            project: The project to filter by.

        Returns:
            Polars DataFrame (materialized) with the latest data. Empty if table
            doesn't exist or snapshot not found.
        """
        # Read system metadata
        sys_meta = self._read_system_metadata(FEATURE_VERSIONS_KEY)

        # Filter the data
        lazy = sys_meta.filter(
            nw.col(METAXY_PROJECT_VERSION) == project_version,
            nw.col("project") == project,
        )

        # Deduplicate using Polars (collect and use native operations)
        return (
            lazy.collect().to_polars().sort("recorded_at", descending=True).unique(subset=["feature_key"], keep="first")
        )

    def _read_latest_project_version(self, project: str) -> str | None:
        """Read the most recently pushed project_version for a given project.

        Returns None if no push has ever been recorded for this project.
        """
        sys_meta = self._read_system_metadata(FEATURE_VERSIONS_KEY)

        latest = (
            sys_meta.filter(nw.col("project") == project)
            .sort("recorded_at", descending=True)
            .head(1)
            .collect()
            .to_polars()
        )

        if len(latest) == 0:
            return None
        return latest[METAXY_PROJECT_VERSION][0]

    def read_graph_snapshots(self, project: str | None = None) -> pl.DataFrame:
        """Read recorded graph snapshots from the feature_versions system table.

        Args:
            project: Project name to filter by. If None, returns snapshots from all projects.

        Returns a DataFrame with columns:
        - project_version: Unique identifier for each graph snapshot
        - recorded_at: Timestamp when the snapshot was recorded
        - feature_count: Number of features in this snapshot

        Returns:
            Polars DataFrame with snapshot information, sorted by recorded_at descending

        Raises:
            StoreNotOpenError: If store is not open

        Example:
            ```py
            with store:
                storage = SystemTableStorage(store)
                # Get snapshots for a specific project
                snapshots = storage.read_graph_snapshots(project="my_project")
                latest_snapshot = snapshots[METAXY_PROJECT_VERSION][0]
                print(f"Latest snapshot: {latest_snapshot}")

                # Get snapshots across all projects
                all_snapshots = storage.read_graph_snapshots()
            ```
        """
        # Read system metadata
        versions_lazy = self._read_system_metadata(FEATURE_VERSIONS_KEY)
        if versions_lazy is None:
            # No snapshots recorded yet
            return pl.DataFrame(
                schema={
                    METAXY_PROJECT_VERSION: pl.String,
                    "recorded_at": pl.Datetime("us"),
                    "feature_count": pl.UInt32,
                }
            )

        # Build filters based on project parameter
        if project is not None:
            versions_lazy = versions_lazy.filter(nw.col("project") == project)

        # Materialize
        versions_df = versions_lazy.collect().to_polars()

        if versions_df.height == 0:
            # No snapshots recorded yet
            return pl.DataFrame(
                schema={
                    METAXY_PROJECT_VERSION: pl.String,
                    "recorded_at": pl.Datetime("us"),
                    "feature_count": pl.UInt32,
                }
            )

        # Group by project_version and get earliest recorded_at and count
        snapshots = (
            versions_df.group_by(METAXY_PROJECT_VERSION)
            .agg(
                [
                    pl.col("recorded_at").min().alias("recorded_at"),
                    pl.col("feature_key").count().alias("feature_count"),
                ]
            )
            .sort("recorded_at", descending=True)
        )

        return snapshots

    def read_features(
        self,
        *,
        current: bool = True,
        project_version: str | None = None,
        project: str | None = None,
    ) -> pl.DataFrame:
        """Read feature version information from the feature_versions system table.

        Args:
            current: If True, only return features from the current code snapshot.
                     If False, must provide project_version.
            project_version: Specific snapshot version to filter by. Required if current=False.
            project: Project name to filter by.

        Returns:
            Polars DataFrame with columns from FEATURE_VERSIONS_SCHEMA:
            - feature_key: Feature identifier
            - feature_version: Version hash of the feature
            - recorded_at: When this version was recorded
            - feature_spec: JSON serialized feature specification
            - feature_class_path: Python import path to the feature class
            - project_version: Graph snapshot this feature belongs to

        Raises:
            StoreNotOpenError: If store is not open
            ValueError: If current=False but no project_version provided

        Examples:
            ```py
            # Get features from current code
            with store:
                storage = SystemTableStorage(store)
                features = storage.read_features(current=True)
                print(f"Current graph has {len(features)} features")
            ```

            ```py
            # Get features from a specific snapshot
            with store:
                storage = SystemTableStorage(store)
                features = storage.read_features(current=False, project_version="abc123")
                for row in features.iter_rows(named=True):
                    print(f"{row['feature_key']}: {row['metaxy_feature_version']}")
            ```
        """
        if not current and project_version is None:
            raise ValueError("Must provide project_version when current=False")

        if current:
            # Get current snapshot from active graph, using project-scoped version if project specified
            graph = FeatureGraph.get_active()
            if project is not None:
                project_version = graph.get_project_version(project)
            else:
                # Infer project from graph if single project, otherwise use global snapshot
                snapshot_dict = graph.to_snapshot()
                projects_in_graph = {v["project"] for v in snapshot_dict.values()} if snapshot_dict else set()
                if len(projects_in_graph) == 1:
                    project_version = graph.get_project_version(projects_in_graph.pop())
                else:
                    project_version = graph.project_version

        # Read system metadata
        versions_lazy = self._read_system_metadata(FEATURE_VERSIONS_KEY)
        if versions_lazy is None:
            # No features recorded yet
            return pl.DataFrame(schema=POLARS_SCHEMAS[FEATURE_VERSIONS_KEY])

        # Build filters
        filters = [nw.col(METAXY_PROJECT_VERSION) == project_version]
        if project is not None:
            filters.append(nw.col("project") == project)

        for f in filters:
            versions_lazy = versions_lazy.filter(f)

        # Materialize
        versions_df = versions_lazy.collect().to_polars()

        return versions_df

    def load_graph_from_snapshot(
        self,
        project_version: str,
        project: str | None = None,
    ) -> FeatureGraph:
        """Load and reconstruct a FeatureGraph from a stored snapshot.

        This method creates FeatureDefinition objects directly from the stored snapshot
        data without any dynamic imports. The resulting graph contains all feature
        metadata needed for comparisons.

        Args:
            project_version: The snapshot version to load
            project: Optional project name to filter by

        Returns:
            Reconstructed FeatureGraph with FeatureDefinition objects

        Raises:
            ValueError: If no features found for the snapshot version

        Note:
            The store must already be open when calling this method.

        Example:
            ```python
            with store:
                storage = SystemTableStorage(store)
                graph = storage.load_graph_from_snapshot(project_version="abc123", project="my_project")
                print(f"Loaded {len(graph.feature_definitions_by_key)} features")
            ```
        """
        # Read features for this snapshot
        features_df = self.read_features(
            current=False,
            project_version=project_version,
            project=project,
        )

        if features_df.height == 0:
            raise ValueError(
                f"No features recorded for snapshot {project_version}" + (f" in project {project}" if project else "")
            )

        # Create definitions and build graph
        definitions = self._definitions_from_dataframe(features_df)
        graph = FeatureGraph()
        for definition in definitions:
            graph.add_feature_definition(definition)
        return graph

    def _load_feature_definitions_raw(
        self,
        *,
        projects: Sequence[str] | None = None,
        filters: Sequence[nw.Expr] | None = None,
        graph: FeatureGraph | None = None,
    ) -> list[FeatureDefinition]:
        """Load feature definitions from storage into a graph.

        Args:
            projects: Project(s) to load features from. All projects are used by default.
            filters: Narwhals expressions to filter features.
            graph: Target graph to populate. Uses the current graph by default.

        Returns:
            List of `FeatureDefinition` objects that were loaded.
        """
        if graph is None:
            graph = FeatureGraph.get_active()

        features_df = self._read_latest_features_by_project(projects, filters=filters)

        if features_df.height == 0:
            return []

        definitions = self._definitions_from_dataframe(features_df)
        for definition in definitions:
            graph.add_feature_definition(definition, on_conflict="ignore")

        return definitions

    def resolve_selection(self, selection: FeatureSelection) -> list[FeatureDefinition]:
        """Resolve a feature selection to definitions from the store.

        Args:
            selection: Describes which features to load.

        Returns:
            Matching definitions. Only features that exist in the store are
            returned — missing keys are silently omitted.
        """
        if selection.all:
            return self._definitions_from_dataframe(self._read_latest_features_by_project())

        clauses: list[nw.Expr] = []

        if selection.projects:
            clauses.append(nw.col("project").is_in(selection.projects))
        if selection.keys:
            clauses.append(nw.col("feature_key").is_in([k.to_string() for k in selection.keys]))

        if clauses:
            expr = clauses[0]
            for c in clauses[1:]:
                expr = expr | c
            filters = [expr]
        else:
            filters = None

        return self._definitions_from_dataframe(self._read_latest_features_by_project(filters=filters))

    def _definitions_from_dataframe(self, features_df: pl.DataFrame) -> list[FeatureDefinition]:
        """Create FeatureDefinition objects from a features DataFrame.

        Rows that fail to deserialize are warned about and skipped.

        Args:
            features_df: DataFrame with feature_spec, feature_schema, feature_class_path,
                and project columns.

        Returns:
            List of successfully deserialized FeatureDefinition objects.
        """
        import warnings

        from metaxy._warnings import InvalidStoredFeatureWarning
        from metaxy.models.feature_definition import FeatureDefinition

        source = str(self.store)
        definitions: list[FeatureDefinition] = []
        for row in features_df.iter_rows(named=True):
            try:
                definitions.append(
                    FeatureDefinition.from_stored_data(
                        feature_spec=row["feature_spec"],
                        feature_schema=row["feature_schema"],
                        feature_class_path=row["feature_class_path"],
                        project=row["project"],
                        source=source,
                    )
                )
            except Exception as e:
                feature_key = row.get("feature_key", "<unknown>")
                warnings.warn(
                    f"Skipping feature '{feature_key}': failed to load from store: {e}",
                    InvalidStoredFeatureWarning,
                )
        return definitions

    def _read_latest_features_by_project(
        self,
        projects: Sequence[str] | None = None,
        *,
        filters: Sequence[nw.Expr] | None = None,
    ) -> pl.DataFrame:
        """Read the latest version of each feature.

        Returns the most recently recorded version of each feature across all snapshots.
        This ensures that features from earlier snapshots are included even if later
        snapshots don't contain them (e.g., in multi-project entangled setups).

        Args:
            projects: Optional list of projects to filter by. If None, returns
                features from all projects.
            filters: Narwhals expressions to apply after deduplication. This ensures
                we always load the latest version of features regardless of filters.

        Returns:
            DataFrame with the latest version of each feature.
        """
        # Read all feature versions
        versions_lazy = self._read_system_metadata(FEATURE_VERSIONS_KEY)

        # Collect to polars for grouping operations
        versions_df = versions_lazy.collect().to_polars()

        if versions_df.height == 0:
            return versions_df

        # Filter by projects if specified
        if projects is not None:
            versions_df = versions_df.filter(pl.col("project").is_in(projects))

        if versions_df.height == 0:
            return versions_df

        # Get the latest version of each feature by recorded_at (across all snapshots)
        result = versions_df.sort("recorded_at", descending=True).unique(subset=["feature_key"], keep="first")

        # Apply user filters AFTER deduplication to ensure we always get the latest
        # version of each feature before filtering
        if filters:
            result_nw = nw.from_native(result)
            for expr in filters:
                result_nw = result_nw.filter(expr)
            result = result_nw.to_native()

        return result
