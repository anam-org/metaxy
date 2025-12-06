"""Example 3: PythonMigration with dynamically generated operations.

This example demonstrates how to compute the operations list dynamically
based on runtime conditions, external configuration, or complex logic.

Use Case:
---------
You need to apply different operations to different features based on:
- Feature characteristics (root vs derived features)
- External configuration (feature flags, env variables)
- Runtime inspection (feature metadata, dependencies)
- Conditional logic (only certain features need custom handling)

When to Use:
------------
- Complex migrations with conditional operation logic
- Different operation types for different feature subsets
- Operations determined by external configuration
- Dynamic feature selection based on inspection

Key Technique:
--------------
Override `operations()` to compute the operation list dynamically. The method is
evaluated when the migration class is instantiated, allowing you to inspect the
feature graph, read configuration, or apply complex logic to determine operations.

Example Usage:
--------------
Place this file in .metaxy/migrations/ directory and run:
    metaxy migrations apply

The migration will dynamically determine which operations to apply based on
feature characteristics and configuration.
"""

from datetime import datetime, timezone
from typing import Any

from metaxy.migrations.models import PythonMigration
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey


class DynamicOperationsMigration(PythonMigration):
    """Migration with dynamically computed operations.

    This migration demonstrates how to generate the operations list at runtime
    based on feature characteristics, configuration, or other conditions.

    The operations list is computed dynamically by inspecting the feature graph
    and determining which operations are needed for each feature type.
    """

    migration_id: str = "20250101_140000_dynamic_operations"
    created_at: datetime = datetime(2025, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    parent: str = "20250101_130000_custom_backfill"

    from_snapshot_version: str = "def456..."
    to_snapshot_version: str = "ghi789..."

    # Configuration flags (could be loaded from env vars or config file)
    enable_custom_validation: bool = True
    skip_root_features: bool = False

    def build_operations(self) -> list[dict[str, Any]]:
        """Dynamically compute operations based on feature characteristics.

        This method is evaluated when the migration is loaded, allowing
        inspection of the feature graph and conditional operation generation.

        Returns:
            List of operation dictionaries

        Strategy:
        ---------
        1. Get affected features from snapshot diff
        2. Inspect each feature to determine if it's root or derived
        3. Apply DataVersionReconciliation only to derived features
        4. Root features would need custom handling (not shown here)

        In this example, we always return DataVersionReconciliation, but
        in a real migration you might:
        - Check feature types and apply different operations
        - Read external configuration to enable/disable operations
        - Inspect feature metadata to determine operation strategy
        - Use environment variables to control behavior
        """
        # Get active graph for inspection
        _ = FeatureGraph.get_active()

        # Start with default reconciliation operation
        operations = []

        # Example: Add DataVersionReconciliation
        # In a real scenario, you might conditionally add this based on:
        # - Feature inspection (root vs derived)
        # - Configuration flags
        # - External metadata
        operations.append({"type": "metaxy.migrations.ops.DataVersionReconciliation"})

        # Example: Conditionally add custom operations based on config
        if self.enable_custom_validation:
            # Note: This is a hypothetical operation for demonstration
            # You would implement this in your migrations.ops module
            # operations.append({
            #     "type": "myproject.migrations.ops.CustomValidation",
            #     "validation_level": "strict"
            # })
            pass

        return operations

    def get_affected_features(self, store, project: str | None) -> list[str]:  # type: ignore[override]
        """Override to filter affected features based on configuration.

        This demonstrates how you can customize which features are affected
        by the migration based on runtime conditions.

        Args:
            store: Metadata store
            project: Project name

        Returns:
            Filtered list of affected feature keys
        """
        # Get base affected features from parent implementation
        all_affected = super().get_affected_features(store, project)

        # Example: Filter out root features if configured
        if self.skip_root_features:
            graph = FeatureGraph.get_active()
            filtered = []

            for feature_key_str in all_affected:
                feature_key_obj = FeatureKey(feature_key_str.split("/"))
                if feature_key_obj not in graph.features_by_key:
                    continue

                # Check if feature has upstream dependencies
                plan = graph.get_feature_plan(feature_key_obj)
                has_upstream = plan.deps is not None and len(plan.deps) > 0

                if has_upstream:
                    # Only include derived features (not root)
                    filtered.append(feature_key_str)

            return filtered

        return all_affected


# Alternative Approach: Compute operations based on external config
# ------------------------------------------------------------------
class ConfigDrivenMigration(PythonMigration):
    """Migration that reads operations from external configuration.

    This demonstrates loading operation configuration from a file or
    environment variable, useful for:
    - A/B testing different migration strategies
    - Environment-specific migrations (dev, staging, prod)
    - Dynamic rollout of migration operations
    """

    migration_id: str = "20250101_140000_config_driven"
    created_at: datetime = datetime(2025, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    parent: str = "20250101_130000_custom_backfill"

    from_snapshot_version: str = "def456..."
    to_snapshot_version: str = "ghi789..."

    def build_operations(self) -> list[dict[str, Any]]:
        """Load operations from configuration file or environment.

        Returns:
            Operations list from configuration
        """
        import json
        import os
        from pathlib import Path

        # Try to load from environment variable first
        config_json = os.getenv("MIGRATION_OPS_CONFIG")
        if config_json:
            try:
                config = json.loads(config_json)
                return config.get("operations", self._default_ops())
            except json.JSONDecodeError:
                pass

        # Try to load from config file
        config_path = Path(".metaxy/migration_config.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    return config.get("operations", self._default_ops())
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback to default
        return self._default_ops()

    def _default_ops(self) -> list[dict[str, str]]:
        """Default operations if no configuration found.

        Returns:
            Default operations list
        """
        return [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]


# Alternative Approach: Feature-specific operations
# --------------------------------------------------
class FeatureSpecificMigration(PythonMigration):
    """Apply different operations to different features.

    This demonstrates generating operations based on inspecting the
    feature graph and applying different strategies per feature.
    """

    migration_id: str = "20250101_140000_feature_specific"
    created_at: datetime = datetime(2025, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    parent: str = "20250101_130000_custom_backfill"

    from_snapshot_version: str = "def456..."
    to_snapshot_version: str = "ghi789..."

    # Map feature keys to operation types
    feature_operation_map: dict[str, str] = {
        "video/files": "myproject.migrations.ops.VideoBackfill",
        "audio/transcripts": "myproject.migrations.ops.TranscriptBackfill",
        # All others use default reconciliation
    }

    def build_operations(self) -> list[dict[str, Any]]:
        """Generate operations based on feature-specific rules.

        Returns:
            Operations list with feature-specific operations
        """
        # For DiffMigration with DataVersionReconciliation, we can only
        # have one operation that applies to all affected features.
        # For feature-specific operations, you'd need to use PythonMigration
        # and implement your own execute() logic.

        # This example shows the structure for documentation purposes
        return [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]

    # Note: For true feature-specific operations, use PythonMigration instead
    # and implement execute() to iterate through features and apply different
    # operations based on your rules.


# Key Takeaways:
# --------------
# 1. Override operations() (or build_operations()) to compute ops dynamically
# 2. The method runs when the migration class is instantiated, so you can inspect the current graph/config
# 3. You can inspect feature graph, read config files, check env vars
# 4. Override get_affected_features() to filter which features are processed
# 5. For truly feature-specific operations, use PythonMigration (see Example 2)
# 6. Dynamic operations enable A/B testing, conditional rollout, env-specific logic

# When to Use Dynamic Operations:
# --------------------------------
# - Operations depend on runtime conditions
# - Need different strategies for different environments
# - Want to read configuration from external sources
# - Complex conditional logic for operation selection
# - A/B testing migration strategies

# When NOT to Use:
# ----------------
# - Operations are fixed and known at migration creation time
# - Simple reconciliation is sufficient
# - No conditional logic needed
# Use static operations list in these cases (like Example 1)
