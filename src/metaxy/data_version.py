"""Sample-level data version calculation using Polars expressions."""

import polars as pl
import polars_hash as plh


def calculate_sample_data_versions(
    upstream_data_versions: dict[str, pl.DataFrame],
    container_key: str,
    code_version: int,
    container_deps: dict[str, list[str]],
) -> pl.Expr:
    """Calculate sample-level data versions for a container.

    This function computes a Merkle tree hash for each sample by:
    1. Loading relevant upstream data versions for each sample
    2. Hashing them together (Merkle tree approach)
    3. Adding the current container's code version to the hash

    Args:
        upstream_data_versions: Dictionary mapping upstream feature keys to DataFrames.
            Each DataFrame should have a "data_version" column which is a pl.Struct
            with fields corresponding to containers for that feature.
        container_key: The key of the current container we're computing versions for.
        code_version: The code version of the current container.
        container_deps: Dictionary mapping upstream feature keys to lists of container
            keys that this container depends on.

    Returns:
        A Polars expression that computes the data version struct for each sample.
        The resulting struct will have a field for the current container.

    Example:
        >>> # Upstream feature "video" has containers ["frames", "audio"]
        >>> upstream_df = pl.DataFrame({
        ...     "sample_id": [1, 2, 3],
        ...     "data_version": [
        ...         {"frames": "abc123", "audio": "def456"},
        ...         {"frames": "abc124", "audio": "def457"},
        ...         {"frames": "abc125", "audio": "def458"},
        ...     ]
        ... })
        >>> upstream_data_versions = {"video": upstream_df}
        >>> container_deps = {"video": ["frames", "audio"]}
        >>>
        >>> # Calculate data version for our container
        >>> expr = calculate_sample_data_versions(
        ...     upstream_data_versions=upstream_data_versions,
        ...     container_key="processed",
        ...     code_version=1,
        ...     container_deps=container_deps,
        ... )
    """
    # Start with the container key and code version
    components = [
        pl.lit(container_key),
        pl.lit(str(code_version)),
    ]

    # Add upstream data versions in a deterministic order
    for feature_key in sorted(container_deps.keys()):
        container_keys = sorted(container_deps[feature_key])

        for upstream_container_key in container_keys:
            # Extract the specific container's data version from the upstream struct
            # Format: feature_key/container_key -> data_version
            components.append(pl.lit(f"{feature_key}/{upstream_container_key}"))
            components.append(
                pl.col("data_version").struct.field(upstream_container_key)
            )

    # Concatenate all components with a separator and hash
    # Use SHA256 via polars-hash
    data_version_expr = (
        plh.concat_str(*components, separator="|").chash.sha2_256().alias(container_key)
    )

    # Return as a struct with the container key as the field name
    return pl.struct(data_version_expr)


def calculate_feature_data_versions(
    upstream_data_versions: dict[str, pl.DataFrame],
    feature_containers: dict[str, int],
    feature_deps: dict[str, dict[str, list[str]]],
) -> pl.Expr:
    """Calculate sample-level data versions for all containers in a feature.

    Args:
        upstream_data_versions: Dictionary mapping upstream feature keys to DataFrames.
            Each DataFrame should have a "data_version" column which is a pl.Struct.
        feature_containers: Dictionary mapping container keys to their code versions
            for the current feature.
        feature_deps: Dictionary mapping container keys to their dependencies.
            Each dependency is a dict mapping upstream feature keys to lists of
            container keys.

    Returns:
        A Polars expression that computes a struct with all container data versions.

    Example:
        >>> upstream_data_versions = {
        ...     "video": pl.DataFrame({
        ...         "sample_id": [1, 2],
        ...         "data_version": [
        ...             {"frames": "abc", "audio": "def"},
        ...             {"frames": "ghi", "audio": "jkl"},
        ...         ]
        ...     })
        ... }
        >>> feature_containers = {
        ...     "processed": 1,
        ...     "augmented": 2,
        ... }
        >>> feature_deps = {
        ...     "processed": {"video": ["frames"]},
        ...     "augmented": {"video": ["frames", "audio"]},
        ... }
        >>> expr = calculate_feature_data_versions(
        ...     upstream_data_versions=upstream_data_versions,
        ...     feature_containers=feature_containers,
        ...     feature_deps=feature_deps,
        ... )
    """
    # Calculate data version for each container
    container_exprs = {}

    for container_key in sorted(feature_containers.keys()):
        code_version = feature_containers[container_key]
        container_deps = feature_deps.get(container_key, {})

        data_version_expr = calculate_sample_data_versions(
            upstream_data_versions=upstream_data_versions,
            container_key=container_key,
            code_version=code_version,
            container_deps=container_deps,
        )

        # Extract the field from the struct
        container_exprs[container_key] = data_version_expr.struct.field(container_key)

    # Combine all container data versions into a single struct
    return pl.struct(**container_exprs)


def merge_upstream_data_versions(
    sample_df: pl.DataFrame,
    upstream_data_versions: dict[str, pl.DataFrame],
    feature_key_col: str = "sample_id",
) -> pl.DataFrame:
    """Merge upstream data versions into the sample DataFrame.

    This helper function joins all upstream data version DataFrames with the
    current sample DataFrame, creating columns like "video__data_version",
    "audio__data_version", etc.

    Args:
        sample_df: The DataFrame containing samples for the current feature.
        upstream_data_versions: Dictionary mapping upstream feature keys to their
            DataFrames with data versions.
        feature_key_col: The column name to use for joining (default: "sample_id").

    Returns:
        DataFrame with all upstream data versions joined.
    """
    result_df = sample_df

    for feature_key, upstream_df in upstream_data_versions.items():
        # Rename the data_version column to avoid conflicts
        upstream_df_renamed = upstream_df.select(
            [
                pl.col(feature_key_col),
                pl.col("data_version").alias(f"{feature_key}__data_version"),
            ]
        )

        result_df = result_df.join(
            upstream_df_renamed,
            on=feature_key_col,
            how="left",
        )

    return result_df
