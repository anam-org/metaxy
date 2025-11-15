"""Utilities for handling one-to-many relationships in features.

This module provides helper functions for common patterns when implementing
one-to-many relationships, such as video chunking, frame extraction, etc.
"""

import hashlib
from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    from metaxy.data_versioning.joiners import UpstreamJoiner
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


def generate_child_sample_uid(
    parent_uid: int | str, child_index: int, *, namespace: str = ""
) -> int:
    """Generate a deterministic child sample UID from parent UID and index.

    This function creates reproducible child UIDs for one-to-many expansion,
    ensuring that the same parent+index always produces the same child UID.

    Args:
        parent_uid: The parent sample's UID
        child_index: The index of this child (0-based)
        namespace: Optional namespace to avoid collisions between different
                   child types (e.g., "chunk" vs "frame")

    Returns:
        A deterministic integer UID for the child sample

    Example:
        >>> # Generate UIDs for video chunks
        >>> parent_video_uid = 12345
        >>> chunk_uids = [
        ...     generate_child_sample_uid(parent_video_uid, i, namespace="chunk")
        ...     for i in range(10)
        ... ]
    """
    # Create a deterministic hash
    hasher = hashlib.sha256()
    hasher.update(str(parent_uid).encode())
    hasher.update(str(child_index).encode())
    if namespace:
        hasher.update(namespace.encode())

    # Convert to integer (use first 8 bytes for 64-bit int)
    hash_bytes = hasher.digest()[:8]
    # Use absolute value to ensure positive UIDs
    return abs(int.from_bytes(hash_bytes, byteorder='big'))


def expand_to_children(
    parent_df: "nw.LazyFrame[Any]",
    num_children_per_parent: int | dict[Any, int],
    *,
    parent_id_column: str = "sample_uid",
    child_id_column: str = "sample_uid",
    child_index_column: str = "child_index",
    parent_ref_column: str | None = None,
    namespace: str = "",
) -> "nw.LazyFrame[Any]":
    """Expand parent samples to multiple child samples (one-to-many).

    This function takes a DataFrame of parent samples and expands each parent
    into multiple child samples, generating unique child IDs and preserving
    parent data.

    Args:
        parent_df: LazyFrame containing parent samples
        num_children_per_parent: Either a fixed number of children per parent,
                                  or a dict mapping parent_id -> num_children
        parent_id_column: Name of the parent ID column (default: "sample_uid")
        child_id_column: Name for the child ID column (default: "sample_uid")
        child_index_column: Name for the child index column (default: "child_index")
        parent_ref_column: If specified, adds a column with parent ID reference
        namespace: Namespace for child UID generation to avoid collisions

    Returns:
        LazyFrame with expanded child samples, containing:
        - All original parent columns (with parent_id_column renamed to parent_ref_column if specified)
        - child_id_column: Unique ID for each child
        - child_index_column: 0-based index of child within parent
        - parent_ref_column: Reference to parent ID (if specified)

    Example:
        >>> # Fixed number of children per parent
        >>> children = expand_to_children(
        ...     parent_df,
        ...     num_children_per_parent=10,
        ...     parent_ref_column="parent_video_id",
        ...     namespace="chunk"
        ... )

        >>> # Variable number of children per parent
        >>> children_counts = {video1_id: 5, video2_id: 8, video3_id: 3}
        >>> children = expand_to_children(
        ...     parent_df,
        ...     num_children_per_parent=children_counts,
        ...     parent_ref_column="parent_video_id",
        ...     namespace="chunk"
        ... )
    """
    # Convert to native for processing
    parent_native = parent_df.collect().to_native()

    # Get the parent IDs
    if parent_id_column not in parent_native.columns:
        raise ValueError(f"Parent ID column '{parent_id_column}' not found in parent DataFrame")

    # Determine number of children per parent
    if isinstance(num_children_per_parent, int):
        # Fixed number for all parents
        num_children = num_children_per_parent
        children_per_parent = None
    else:
        # Variable number per parent
        children_per_parent = num_children_per_parent
        num_children = None

    # Build expanded data
    expanded_rows = []

    # Process each parent row
    import polars as pl
    if isinstance(parent_native, pl.DataFrame):
        for row in parent_native.iter_rows(named=True):
            parent_id = row[parent_id_column]

            # Determine number of children for this parent
            if children_per_parent is not None:
                n_children = children_per_parent.get(parent_id, 0)
            else:
                n_children = num_children

            # Generate child rows
            for child_idx in range(n_children or 0):
                child_row = dict(row)  # Copy parent data

                # Generate child UID
                child_uid = generate_child_sample_uid(
                    parent_id, child_idx, namespace=namespace
                )

                # Update/add columns
                child_row[child_id_column] = child_uid
                child_row[child_index_column] = child_idx

                # Add parent reference if requested
                if parent_ref_column:
                    child_row[parent_ref_column] = parent_id

                # If parent_ref_column is different from parent_id_column,
                # and child_id_column is the same as parent_id_column,
                # we've already overwritten it with child_uid above

                expanded_rows.append(child_row)

        # Create expanded DataFrame
        if expanded_rows:
            expanded_df = pl.DataFrame(expanded_rows)
        else:
            # Create empty DataFrame with correct schema
            schema = dict(parent_native.schema)
            schema[child_id_column] = pl.Int64()
            schema[child_index_column] = pl.Int64()
            if parent_ref_column:
                schema[parent_ref_column] = parent_native.schema[parent_id_column]
            expanded_df = pl.DataFrame([], schema=schema)

        # Convert back to Narwhals LazyFrame
        return nw.from_native(expanded_df.lazy(), eager_only=False)

    else:
        # Handle other backends (Ibis, etc.) - for now raise NotImplementedError
        raise NotImplementedError(
            f"expand_to_children not yet implemented for backend: {type(parent_native)}"
        )


class OneToManyJoiner:
    """Joiner that handles one-to-many expansion in load_input.

    This joiner extends the standard joining behavior to support one-to-many
    relationships where a parent feature produces multiple child samples.

    Example:
        >>> class VideoChunk(Feature, spec=...):
        ...     @classmethod
        ...     def load_input(cls, joiner, upstream_refs):
        ...         # Use OneToManyJoiner for expansion
        ...         otm_joiner = OneToManyJoiner(
        ...             joiner,
        ...             expansion_config={
        ...                 "video/raw": {
        ...                     "num_children": 10,  # 10 chunks per video
        ...                     "parent_ref_column": "parent_video_id",
        ...                     "namespace": "chunk",
        ...                 }
        ...             }
        ...         )
        ...         return otm_joiner.join_upstream(
        ...             upstream_refs,
        ...             cls.spec,
        ...             cls.graph.get_feature_plan(cls.spec.key),
        ...         )
    """

    def __init__(
        self,
        base_joiner: "UpstreamJoiner",
        expansion_config: dict[str, dict[str, Any]],
    ):
        """Initialize OneToManyJoiner.

        Args:
            base_joiner: The underlying joiner to use for standard joins
            expansion_config: Configuration for one-to-many expansion.
                Keys are upstream feature keys (as strings).
                Values are dicts with:
                - num_children: int or dict[parent_id, int]
                - parent_ref_column: optional column name for parent reference
                - namespace: optional namespace for child UID generation
                - child_index_column: optional name for child index column
        """
        self.base_joiner = base_joiner
        self.expansion_config = expansion_config

    def join_upstream(
        self,
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_columns: dict[str, tuple[str, ...] | None] | None = None,
        upstream_renames: dict[str, dict[str, str] | None] | None = None,
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Join upstream features with one-to-many expansion.

        For features marked for expansion in expansion_config, this performs
        the one-to-many expansion before joining. Other features are joined
        normally.
        """

        # Process expansions
        expanded_refs = {}
        for upstream_key, upstream_ref in upstream_refs.items():
            if upstream_key in self.expansion_config:
                # Perform one-to-many expansion
                config = self.expansion_config[upstream_key]
                expanded = expand_to_children(
                    upstream_ref,
                    num_children_per_parent=config["num_children"],
                    parent_id_column=feature_spec.id_columns[0],  # Use first ID column
                    child_id_column=feature_spec.id_columns[0],
                    child_index_column=config.get("child_index_column", "child_index"),
                    parent_ref_column=config.get("parent_ref_column"),
                    namespace=config.get("namespace", ""),
                )
                expanded_refs[upstream_key] = expanded
            else:
                # Keep as-is
                expanded_refs[upstream_key] = upstream_ref

        # Now use the base joiner with expanded references
        return self.base_joiner.join_upstream(
            expanded_refs,
            feature_spec,
            feature_plan,
            upstream_columns,
            upstream_renames,
        )