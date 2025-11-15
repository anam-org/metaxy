"""Mixins and base classes for one-to-many feature patterns.

This module provides reusable patterns for implementing one-to-many
relationships in Metaxy features.
"""

from typing import TYPE_CHECKING, Any, ClassVar

import narwhals as nw

from metaxy.utils.one_to_many import expand_to_children

if TYPE_CHECKING:
    from metaxy.data_versioning.joiners import UpstreamJoiner


class OneToManyFeatureMixin:
    """Mixin for features that implement one-to-many expansion.

    This mixin provides a configurable `load_input()` method that handles
    the mechanical aspects of one-to-many expansion. Subclasses configure
    the expansion behavior through class variables.

    Class Variables to Configure:
        expansion_source_key: str - The upstream feature key to expand (e.g., "video/raw")
        expansion_count: int | dict - Number of children per parent (fixed or variable)
        expansion_namespace: str - Namespace for child UID generation
        parent_ref_column: str - Column name for parent reference (e.g., "parent_video_id")
        child_index_column: str - Column name for child index (default: "child_index")

    Example:
        >>> class VideoChunk(Feature, OneToManyFeatureMixin, spec=...):
        ...     expansion_source_key = "video/raw"
        ...     expansion_count = 10  # 10 chunks per video
        ...     expansion_namespace = "chunk"
        ...     parent_ref_column = "parent_video_id"
        ...
        ...     # No need to implement load_input() - mixin handles it!

    Advanced Example with Dynamic Count:
        >>> class SmartChunks(Feature, OneToManyFeatureMixin, spec=...):
        ...     @classmethod
        ...     def get_expansion_count(cls, parent_data):
        ...         # Calculate based on parent data
        ...         counts = {}
        ...         for row in parent_data.iter_rows(named=True):
        ...             duration = row["duration_seconds"]
        ...             counts[row["sample_uid"]] = int(duration / 10)  # 1 chunk per 10s
        ...         return counts
        ...
        ...     expansion_source_key = "video/raw"
        ...     expansion_namespace = "smart_chunk"
        ...     parent_ref_column = "parent_video_id"
    """

    # Configuration - subclasses should set these
    expansion_source_key: ClassVar[str]
    expansion_count: ClassVar[int | dict[Any, int] | None] = None
    expansion_namespace: ClassVar[str] = ""
    parent_ref_column: ClassVar[str | None] = None
    child_index_column: ClassVar[str] = "child_index"

    @classmethod
    def get_expansion_count(
        cls, parent_data: "nw.LazyFrame[Any]"
    ) -> int | dict[Any, int]:
        """Override to compute expansion count dynamically from parent data.

        Args:
            parent_data: LazyFrame containing parent feature data

        Returns:
            Either a fixed count (int) or variable counts per parent (dict)
        """
        if cls.expansion_count is None:
            raise NotImplementedError(
                f"{cls.__name__} must set expansion_count class variable "
                "or override get_expansion_count()"
            )
        return cls.expansion_count

    @classmethod
    def load_input(
        cls,
        joiner: "UpstreamJoiner",
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Perform one-to-many expansion on configured upstream feature.

        This method:
        1. Identifies the expansion source from upstream_refs
        2. Calls get_expansion_count() to determine fan-out
        3. Performs the expansion using expand_to_children()
        4. Joins with other upstreams using the standard joiner

        Can be overridden for more complex scenarios.
        """
        # Check that expansion_source_key is configured
        if not hasattr(cls, "expansion_source_key"):
            raise AttributeError(
                f"{cls.__name__} must set expansion_source_key class variable"
            )

        # Get the source to expand
        if cls.expansion_source_key not in upstream_refs:
            raise ValueError(
                f"Expansion source '{cls.expansion_source_key}' not found in upstream_refs. "
                f"Available: {list(upstream_refs.keys())}"
            )

        source_ref = upstream_refs[cls.expansion_source_key]

        # Get expansion count (may need to collect parent data)
        expansion_count = cls.get_expansion_count(source_ref)

        # Get ID columns from the feature spec
        # Since this mixin is designed to be used with Feature classes,
        # we can safely assume the attributes exist
        id_columns = getattr(cls, "spec").id_columns if hasattr(cls, "spec") else ["sample_uid"]

        # Perform expansion
        expanded = expand_to_children(
            source_ref,
            num_children_per_parent=expansion_count,
            parent_id_column=id_columns[0],  # Use first ID column
            child_id_column=id_columns[0],
            child_index_column=cls.child_index_column,
            parent_ref_column=cls.parent_ref_column,
            namespace=cls.expansion_namespace,
        )

        # Replace the source with expanded version
        expanded_refs = dict(upstream_refs)
        expanded_refs[cls.expansion_source_key] = expanded

        # Use standard joiner with expanded refs
        # Access the feature's spec and graph through proper channels
        feature_spec = getattr(cls, "spec")
        feature_graph = getattr(cls, "graph")
        feature_plan = feature_graph.get_feature_plan(feature_spec.key)

        # Extract column selections and renames if defined
        upstream_columns = getattr(cls, "_get_upstream_columns", lambda: None)()
        upstream_renames = getattr(cls, "_get_upstream_renames", lambda: None)()

        return joiner.join_upstream(
            expanded_refs,
            feature_spec,
            feature_plan,
            upstream_columns,
            upstream_renames,
        )


class ManyToOneAggregationMixin:
    """Mixin for features that aggregate from many children to one parent.

    This is the inverse of OneToManyFeatureMixin, used when you need to
    aggregate expanded features back to their parent level.

    Class Variables to Configure:
        aggregation_source_key: str - The child feature to aggregate
        parent_id_column: str - Column containing parent ID in children
        target_id_column: str - Column name for aggregated result ID
        aggregation_specs: list[tuple] - List of (column, agg_func, alias) tuples

    Example:
        >>> class VideoSummary(Feature, ManyToOneAggregationMixin, spec=...):
        ...     aggregation_source_key = "video/chunks"
        ...     parent_id_column = "parent_video_id"
        ...     target_id_column = "video_id"
        ...     aggregation_specs = [
        ...         ("score", "mean", "avg_score"),
        ...         ("score", "max", "max_score"),
        ...         ("chunk_uid", "count", "num_chunks"),
        ...     ]
    """

    # Configuration
    aggregation_source_key: ClassVar[str]
    parent_id_column: ClassVar[str] = "parent_video_id"
    target_id_column: ClassVar[str] = "video_id"
    aggregation_specs: ClassVar[list[tuple[str, str, str]]]

    @classmethod
    def load_input(
        cls,
        joiner: "UpstreamJoiner",
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Aggregate child features back to parent level.

        Performs group-by aggregation based on configured specs.
        """
        if not hasattr(cls, "aggregation_source_key"):
            raise AttributeError(
                f"{cls.__name__} must set aggregation_source_key class variable"
            )

        if not hasattr(cls, "aggregation_specs"):
            raise AttributeError(
                f"{cls.__name__} must set aggregation_specs class variable"
            )

        # Get source to aggregate
        if cls.aggregation_source_key not in upstream_refs:
            raise ValueError(
                f"Aggregation source '{cls.aggregation_source_key}' not found in upstream_refs. "
                f"Available: {list(upstream_refs.keys())}"
            )

        source_ref = upstream_refs[cls.aggregation_source_key]

        # Build aggregation expressions
        agg_exprs = []
        for col, func, alias in cls.aggregation_specs:
            if func == "mean":
                agg_exprs.append(nw.col(col).mean().alias(alias))
            elif func == "sum":
                agg_exprs.append(nw.col(col).sum().alias(alias))
            elif func == "max":
                agg_exprs.append(nw.col(col).max().alias(alias))
            elif func == "min":
                agg_exprs.append(nw.col(col).min().alias(alias))
            elif func == "count":
                agg_exprs.append(nw.col(col).count().alias(alias))
            elif func == "std":
                agg_exprs.append(nw.col(col).std().alias(alias))
            else:
                raise ValueError(f"Unsupported aggregation function: {func}")

        # Perform aggregation
        aggregated = (
            source_ref.group_by(cls.parent_id_column)
            .agg(agg_exprs)
            .rename({cls.parent_id_column: cls.target_id_column})
        )

        # Add data_version column for compatibility
        # Note: This is a placeholder - in production you'd compute proper data version
        # For now, just add a constant struct column
        # Using a workaround since nw.lit doesn't support dict literals directly
        import polars as pl
        dummy_df = pl.DataFrame([{"dummy": {"default": "aggregated"}}])
        dummy_lazy = nw.from_native(dummy_df.lazy(), eager_only=False)

        # Cross join with single row to add the column
        aggregated = aggregated.join(
            dummy_lazy.select(
                nw.col("dummy").alias(f"__upstream_{cls.aggregation_source_key}__data_version")
            ),
            how="cross",
        )

        # Build column mapping
        column_mapping = {
            cls.aggregation_source_key: f"__upstream_{cls.aggregation_source_key}__data_version",
        }

        return aggregated, column_mapping