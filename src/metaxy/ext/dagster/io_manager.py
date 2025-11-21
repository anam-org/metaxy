"""Dagster IOManager for Metaxy feature metadata."""

from __future__ import annotations

import ast
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import dagster as dg
import narwhals as nw

from metaxy.ext.dagster.resource import MetaxyMetadataStoreResource
from metaxy.versioning.types import Increment

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.types import FeatureKey

logger = logging.getLogger(__name__)


class MetaxyIOManager(dg.ConfigurableIOManager):
    """Dagster IOManager for Metaxy feature metadata.

    This IOManager integrates Metaxy's metadata versioning with Dagster's asset
    materialization system. It handles:

    - Loading incremental updates (added/changed/removed samples) via `load_input`
    - Logging feature materializations via `handle_output`

    The IOManager expects users to call `store.write_metadata()` in their asset
    functions to persist feature metadata. The `handle_output` method is a no-op
    that only logs the materialization.

    Attributes:
        store: MetaxyMetadataStoreResource to use for metadata operations.
        partition_key_column: Optional column name to use for partitioning. When set,
            load_input will filter the Increment to only include data for the current
            partition. Use this with Dagster partitions for event/key parallel processing.
        sample_limit: Optional limit on number of samples to process. Use for subsampling
            during testing or development.
        sample_filter_expr: Optional Narwhals expression (as string) to filter samples.
            Use for subset-based processing.
        target_key_column: Column name to apply explicit key filtering on (e.g. video_id).
            Useful for ad-hoc runs that should only process selected IDs without setting
            up Dagster partitions.
        target_keys: Optional list of keys to filter to. Can be overridden per-asset via
            AssetIn metadata using ``metaxy/target_keys``.

    Execution Patterns:
        1. **Data Parallel (default)**: Process all documents at once
           - No partition_key_column, no filters
           - All samples in added/changed/removed

        2. **Partitioned**: Process 1 category (with many rows) end-to-end through pipeline
           - Set partition_key_column (e.g., "video_id")
           - Use Dagster partitions to process each key separately
           - Each partition gets filtered Increment for that key only

        3. **Subsampled**: Process a filtered subset of documents (event parallel for a key of 1)
           - Set sample_limit to take first N samples
           - Set sample_filter_expr for custom filtering logic

        4. **Ad-hoc key filtering**: Process only specific IDs without Dagster partitions
           - Set target_key_column (e.g., "video_id") and target_keys (["video_1", ...])
           - Or provide ``metaxy/target_keys`` + ``metaxy/target_key_column`` metadata
             on AssetIn to scope a single asset invocation

    Example:
        ```py
        import dagster as dg
        from metaxy import Feature, FeatureSpec
        from metaxy.ext.dagster import (
            MetaxyMetadataStoreResource,
            MetaxyIOManager,
            build_asset_spec_from_feature,
        )

        # Define resources
        store_resource = MetaxyMetadataStoreResource.from_config()

        # Data parallel (default)
        io_manager_data_parallel = MetaxyIOManager.from_store(store_resource)

        # Partitioned execution with partitions
        io_manager_partitioned = MetaxyIOManager.from_store(
            store_resource,
            partition_key_column="video_id"
        )

        # Subsampled for testing
        io_manager_subsampled = MetaxyIOManager.from_store(
            store_resource,
            sample_limit=10
        )

        # Define feature
        class MyFeature(Feature, spec=FeatureSpec(
            key="my/feature",
            id_columns=["sample_uid"],
        )):
            pass

        # Define asset
        @dg.asset(spec=build_asset_spec_from_feature(MyFeature))
        def my_feature_asset(
            context: dg.AssetExecutionContext,
            store: MetaxyMetadataStoreResource,
        ) -> FeatureSpec:
            increment = store.resolve_update(MyFeature)

            # Process increment and write metadata
            store.write_metadata(MyFeature, increment.added)

            return MyFeature.spec()

        # Define definitions
        defs = dg.Definitions(
            assets=[my_feature_asset],
            resources={
                "metaxy_store": store_resource,
                "io_manager": io_manager_data_parallel,
            },
        )
        ```
    """

    store: MetaxyMetadataStoreResource | None = None
    partition_key_column: str | None = None
    sample_limit: int | None = None
    sample_filter_expr: str | None = None
    target_key_column: str | None = None
    target_keys: list[str] | None = None

    @classmethod
    def from_store(
        cls,
        store: MetaxyMetadataStoreResource,
        partition_key_column: str | None = None,
        sample_limit: int | None = None,
        sample_filter_expr: str | None = None,
        target_key_column: str | None = None,
        target_keys: list[str] | None = None,
    ) -> MetaxyIOManager:
        """Create MetaxyIOManager from a MetaxyMetadataStoreResource.

        This is the recommended way to create the IOManager.

        Args:
            store: MetaxyMetadataStoreResource instance
            partition_key_column: Optional column for partition filtering
            sample_limit: Optional sample limit
            sample_filter_expr: Optional filter expression
            target_key_column: Optional target key column
            target_keys: Optional target keys to filter

        Returns:
            MetaxyIOManager configured with the store

        Example:
            ```py
            store_resource = mxd.MetaxyMetadataStoreResource.from_config(store_name="dev")
            io_manager = mxd.MetaxyIOManager.from_store(store_resource)
            ```
        """
        return cls(
            store=store,
            partition_key_column=partition_key_column,
            sample_limit=sample_limit,
            sample_filter_expr=sample_filter_expr,
            target_key_column=target_key_column,
            target_keys=target_keys,
        )

    def handle_output(self, context: dg.OutputContext, obj: FeatureSpec) -> None:  # noqa: ARG002
        """Handle output from a Metaxy feature asset.

        This is a no-op method that only logs the feature materialization.
        Users should call `store.write_metadata()` themselves within their asset
        functions to persist feature metadata.

        Args:
            context: Dagster output context containing asset information (unused)
            obj: FeatureSpec returned by the asset function

        Example:
            ```py
            @dg.asset
            def my_feature_asset(
                context: dg.AssetExecutionContext,
                store: MetaxyMetadataStoreResource,
            ) -> FeatureSpec:
                increment = store.resolve_update(MyFeature)

                # User explicitly writes metadata
                store.write_metadata(MyFeature, increment.added)

                # Return spec for IOManager
                return MyFeature.spec()
            ```
        """
        # Extract feature key for logging
        try:
            # Try to access key attribute if it's a FeatureSpec
            key_attr = getattr(obj, "key", None)
            if key_attr is not None and hasattr(key_attr, "to_string"):
                feature_key_str: str = key_attr.to_string()  # type: ignore[attr-defined]
            else:
                feature_key_str = str(key_attr) if key_attr is not None else "unknown"
        except Exception:
            feature_key_str = "unknown"

        logger.info(
            f"Metaxy feature '{feature_key_str}' materialized. "
            f"Metadata should be written via store.write_metadata() in asset function."
        )

    def load_input(self, context: dg.InputContext) -> Increment:
        """Load incremental update for a Metaxy feature.

        Resolves the feature's incremental update by calling `store.resolve_update()`.
        Returns an Increment containing added, changed, and removed samples.

        The Increment can be filtered/partitioned based on IOManager configuration:
        - If partition_key_column is set, filters to current Dagster partition
        - If sample_limit is set, limits the number of samples
        - If sample_filter_expr is set, applies custom filtering
        - You can override which feature to resolve by setting
          ``metaxy/resolve_feature_key`` in the AssetIn metadata.

        **Important**: This method only works for **downstream features with dependencies**.
        Root features (features without upstream dependencies) require the `samples`
        parameter to be passed to `resolve_update()`, which this IOManager cannot provide.
        For root features, call `store.resolve_update(feature, samples=...)` directly
        in your asset function.

        Args:
            context: Dagster input context containing asset information

        Returns:
            Increment with added, changed, and removed DataFrames (potentially filtered)

        Raises:
            ValueError: If feature class cannot be determined from context,
                or if feature is a root feature (no dependencies),
                or if partition_key_column is set but partition_key is missing,
                or if partition_key_column is not in the data,
                or if store is not configured
            AttributeError: If feature doesn't have a valid spec

        Example:
            ```py
            from metaxy.ext.dagster import build_asset_in_from_feature

            # Data parallel (default)
            @dg.asset(ins=build_asset_in_from_feature(UpstreamFeature))
            def process_upstream(upstream_diff):
                # upstream_diff is an Increment
                print(f"Processing {len(upstream_diff.added)} added samples")
                print(f"Processing {len(upstream_diff.changed)} changed samples")
                print(f"Processing {len(upstream_diff.removed)} removed samples")

            # Event parallel with partitions
            @dg.asset(
                ins=build_asset_in_from_feature(UpstreamFeature),
                partitions_def=dg.DynamicPartitionsDefinition(name="video_ids")
            )
            def process_upstream_partitioned(context, upstream_diff):
                # upstream_diff is filtered to current partition's video_id
                video_id = context.partition_key
                print(f"Processing video {video_id}")
                print(f"Processing {len(upstream_diff.added)} added samples for this video")

            # Override which feature to resolve (useful when depending on one asset
            # for scheduling but wanting another feature's Increment)
            @dg.asset(
                ins={
                    "diff": dg.AssetIn(
                        key=dg.AssetKey(["upstream", "feature"]),
                        metadata={"metaxy/resolve_feature_key": "downstream/feature"},
                    )
                }
            )
            def process_override(diff):
                # diff is the Increment for downstream/feature, not upstream/feature
                ...
            ```
        """
        if self.store is None:
            raise ValueError(
                "MetaxyIOManager.store is None. "
                "Create IOManager using MetaxyIOManager.from_store(store_resource)."
            )

        feature_cls = self._resolve_feature_cls(context)

        # Resolve the incremental update using store as context manager
        logger.info(
            "Loading increment for feature '%s' from Metaxy store",
            feature_cls.spec().key.to_string(),
        )

        # Get the actual MetadataStore from the resource
        with self.store.get_store() as store:
            increment = store.resolve_update(feature_cls, lazy=False)

        logger.info(
            f"Loaded increment: {len(increment.added)} added, "
            f"{len(increment.changed)} changed, {len(increment.removed)} removed"
        )

        # Apply parallelization filters
        increment = self._apply_parallelization_filters(context, increment)

        logger.info(
            f"After filters: {len(increment.added)} added, "
            f"{len(increment.changed)} changed, {len(increment.removed)} removed"
        )

        return increment

    def _resolve_feature_cls(self, context: dg.InputContext) -> type[BaseFeature]:
        """Determine which feature to resolve for an input.

        The default behaviour is to resolve the upstream asset key, but this can
        be overridden by providing ``metaxy/resolve_feature_key`` metadata on the
        AssetIn. This enables assets to depend on one feature for scheduling
        while requesting a different feature's Increment from the IOManager.
        """
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        override_feature_key = self._get_feature_key_override(context)

        if override_feature_key is not None:
            feature_key = override_feature_key
            logger.info(
                "Using metaxy/resolve_feature_key override to load feature '%s'",
                feature_key.to_string(),
            )
        else:
            asset_key = context.asset_key

            if asset_key is None:
                raise ValueError(
                    "Cannot load input: asset_key is None in InputContext. "
                    "Provide metaxy/resolve_feature_key metadata to explicitly "
                    "select a feature to resolve."
                )

            feature_key = FeatureKey(asset_key.path)

        graph = FeatureGraph.get_active()

        try:
            feature_cls: type[BaseFeature] = graph.get_feature_by_key(feature_key)
        except KeyError as e:
            raise ValueError(
                f"Cannot load input: Feature '{feature_key.to_string()}' not found in active graph. "
                f"Available features: {[k.to_string() for k in graph.list_features(only_current_project=False)]}"
            ) from e

        return feature_cls

    def _get_feature_key_override(self, context: dg.InputContext) -> FeatureKey | None:
        """Read an explicit feature override from AssetIn metadata."""
        metadata = getattr(context, "metadata", None)

        if not isinstance(metadata, dict):
            return None

        override = metadata.get("metaxy/resolve_feature_key")
        if override is None:
            return None

        from metaxy.models.types import FeatureKey

        try:
            return FeatureKey(override)
        except Exception as exc:
            raise ValueError(
                "Invalid metaxy/resolve_feature_key metadata. "
                "Provide a valid FeatureKey (string path like 'video/clean' or list of parts)."
            ) from exc

    def _apply_parallelization_filters(
        self, context: dg.InputContext, increment: Increment
    ) -> Increment:
        """Apply parallelization filters to an Increment.

        Args:
            context: Dagster input context (for partition key)
            increment: The Increment to filter

        Returns:
            Filtered Increment

        Raises:
            ValueError: If partition_key_column is set but partition_key is missing,
                or if partition_key_column is not in the data
        """

        # Apply partition filtering if configured
        if self.partition_key_column is not None:
            if not hasattr(context, "partition_key") or context.partition_key is None:
                raise ValueError(
                    f"IOManager configured with partition_key_column='{self.partition_key_column}' "
                    f"but context has no partition_key. Ensure asset is partitioned."
                )

            partition_key = context.partition_key
            logger.info(
                f"Filtering increment to partition '{partition_key}' "
                f"using column '{self.partition_key_column}'"
            )

            # Filter each DataFrame in the increment
            increment = Increment(
                added=self._filter_dataframe_by_partition(
                    increment.added, self.partition_key_column, partition_key
                ),
                changed=self._filter_dataframe_by_partition(
                    increment.changed, self.partition_key_column, partition_key
                ),
                removed=self._filter_dataframe_by_partition(
                    increment.removed, self.partition_key_column, partition_key
                ),
            )

        # Apply custom filter expression if configured
        filter_value = self.sample_filter_expr
        if filter_value is not None:
            logger.info("Applying custom filter expression")

            if isinstance(filter_value, nw.Expr):
                filter_expr = filter_value
            elif isinstance(filter_value, str):
                filter_expr = self._parse_filter_expr(filter_value)
            else:
                raise TypeError(
                    "sample_filter_expr must be a string expression or narwhals Expr"
                )

            increment = Increment(
                added=increment.added.filter(filter_expr),
                changed=increment.changed.filter(filter_expr),
                removed=increment.removed.filter(filter_expr),
            )

        # Apply explicit key filtering when configured either on the IOManager
        # or via AssetIn metadata (for ad-hoc executions).
        key_filter = self._get_target_key_filter(context)
        if key_filter is not None:
            target_column, target_keys = key_filter
            logger.info(
                "Filtering increment to specific keys on column '%s': %s",
                target_column,
                target_keys,
            )
            for name, frame in (
                ("added", increment.added),
                ("changed", increment.changed),
                ("removed", increment.removed),
            ):
                if len(frame) > 0 and target_column not in frame.columns:
                    raise ValueError(
                        f"Target key column '{target_column}' not found in {name} dataframe. "
                        f"Available columns: {frame.columns}"
                    )
            column_expr = nw.col(target_column)
            increment = Increment(
                added=increment.added.filter(column_expr.is_in(target_keys)),
                changed=increment.changed.filter(column_expr.is_in(target_keys)),
                removed=increment.removed.filter(column_expr.is_in(target_keys)),
            )

        # Apply sample limit if configured
        if self.sample_limit is not None:
            logger.info(f"Limiting increment to {self.sample_limit} samples")

            increment = Increment(
                added=increment.added.head(self.sample_limit),
                changed=increment.changed.head(self.sample_limit),
                removed=increment.removed.head(self.sample_limit),
            )

        return increment

    def _get_target_key_filter(
        self, context: dg.InputContext
    ) -> tuple[str, list[str]] | None:
        """Resolve target key filter configuration.

        Priority:
        1. AssetIn metadata (``metaxy/target_keys`` + ``metaxy/target_key_column``)
        2. IOManager config (``target_keys`` + ``target_key_column``)
        """
        metadata = getattr(context, "metadata", None) or {}
        target_keys: Sequence[str] | str | None = None
        target_column: str | None = None

        if isinstance(metadata, dict):
            target_keys = metadata.get("metaxy/target_keys")  # type: ignore[assignment]
            target_column = metadata.get("metaxy/target_key_column")

        if target_keys is None:
            target_keys = self.target_keys
        if target_column is None:
            target_column = self.target_key_column

        if target_keys is None:
            return None
        if target_column is None:
            raise ValueError(
                "Target keys provided but no target_key_column specified. "
                "Set MetaxyIOManager.target_key_column or provide "
                "metaxy/target_key_column in AssetIn metadata."
            )

        normalized_keys = self._normalize_target_keys(target_keys)
        return target_column, normalized_keys

    @staticmethod
    def _normalize_target_keys(keys: Sequence[str] | str) -> list[str]:
        """Normalize target keys to a list of strings."""
        if isinstance(keys, str):
            return [keys]
        if isinstance(keys, Sequence):
            return [str(key) for key in keys]

        raise TypeError(
            "target_keys must be a string or a sequence of strings "
            "(provided via IOManager config or AssetIn metadata)."
        )

    def _filter_dataframe_by_partition(
        self, df: nw.DataFrame[Any], column: str, partition_key: str
    ) -> nw.DataFrame[Any]:
        """Filter a DataFrame to a specific partition value.

        Args:
            df: DataFrame to filter
            column: Column name to filter on
            partition_key: Value to filter for

        Returns:
            Filtered DataFrame

        Raises:
            ValueError: If column is not in the DataFrame
        """
        import narwhals as nw

        # Check if DataFrame is empty - return as-is
        if len(df) == 0:
            return df

        # Check if column exists
        if column not in df.columns:
            raise ValueError(
                f"Partition column '{column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        # Filter to the partition key
        return df.filter(nw.col(column) == partition_key)

    def _parse_filter_expr(self, expr: str) -> nw.Expr:
        """Parse a user-supplied filter expression safely.

        Only allows simple boolean expressions composed of:
        - `nw.col("<column>")`
        - comparison operators (==, !=, <, <=, >, >=)
        - boolean operators (and/or)
        - arithmetic operators on above (e.g., +, -, *, /)
        - numeric/string literals

        Anything else (function calls, attribute access outside nw.col, etc.)
        is rejected to avoid unsafe evaluation.
        """

        def _validate(node: ast.AST) -> None:
            """Recursively validate AST nodes for safety."""
            if isinstance(node, ast.Expression):
                _validate(node.body)
            elif isinstance(node, ast.BoolOp):
                if not isinstance(node.op, (ast.And, ast.Or)):
                    raise ValueError("Only 'and'/'or' boolean operators are allowed")
                for value in node.values:
                    _validate(value)
            elif isinstance(node, ast.BinOp):
                if not isinstance(
                    node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
                ):
                    raise ValueError("Unsupported binary operator in filter expression")
                _validate(node.left)
                _validate(node.right)
            elif isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)):
                    raise ValueError("Unsupported unary operator in filter expression")
                _validate(node.operand)
            elif isinstance(node, ast.Compare):
                allowed_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
                if not all(isinstance(op, allowed_ops) for op in node.ops):
                    raise ValueError(
                        "Unsupported comparison operator in filter expression"
                    )
                _validate(node.left)
                for comparator in node.comparators:
                    _validate(comparator)
            elif isinstance(node, ast.Call):
                # Only allow nw.col("<col>")
                if not isinstance(node.func, ast.Attribute):
                    raise ValueError("Only nw.col(...) calls are allowed")
                if not (
                    isinstance(node.func.value, ast.Name) and node.func.value.id == "nw"
                ):
                    raise ValueError("Only nw.col(...) calls are allowed")
                if node.func.attr != "col":
                    raise ValueError("Only nw.col(...) calls are allowed")
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                    raise ValueError(
                        "nw.col must be called with a single literal column name"
                    )
                if node.keywords:
                    raise ValueError("nw.col does not support keyword arguments here")
            elif isinstance(node, ast.Attribute):
                # Restrict attribute access strictly to nw.col
                if not (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "nw"
                    and node.attr == "col"
                ):
                    raise ValueError(
                        "Attribute access is not allowed in filter expressions"
                    )
            elif isinstance(node, ast.Name):
                if node.id != "nw":
                    raise ValueError("Only 'nw' name is allowed in filter expressions")
            elif isinstance(node, ast.Constant):
                # Literals are fine
                return
            else:
                raise ValueError("Unsupported syntax in filter expression")

        parsed = ast.parse(expr, mode="eval")
        _validate(parsed)

        # Evaluate with a restricted global namespace
        return eval(  # noqa: S307 - restricted eval after validation
            compile(parsed, filename="<filter_expr>", mode="eval"),
            {"nw": nw, "__builtins__": {}},
            {},
        )
