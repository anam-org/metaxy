"""Helper functions for Dagster integration with Metaxy features."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import dagster as dg
import narwhals as nw

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.versioning.types import Increment
else:  # pragma: no cover - runtime imports only needed outside type checking
    from metaxy.models.feature import BaseFeature

from metaxy.models.field import FieldDep, SpecialFieldDep

F = TypeVar("F", bound=Callable[..., Any])


def _sanitize_tag_value(value: str) -> str:
    """Ensure tag values comply with Dagster constraints."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if len(safe) <= 63:
        return safe
    # Preserve uniqueness for long keys
    suffix = hashlib.sha1(safe.encode()).hexdigest()[:8]
    prefix_len = max(0, 63 - 1 - len(suffix))
    return f"{safe[:prefix_len]}-{suffix}"


def _is_generic_pydantic_description(description: str) -> bool:
    """Detect and ignore the generic Pydantic BaseModel description."""
    lowered = description.lower()
    return (
        "base class for creating pydantic models" in lowered
        or lowered.strip().startswith("!!! abstract")
        or ("usage documentation" in lowered and "pydantic models" in lowered)
    )


def _infer_model_description(feature: type[BaseFeature]) -> str | None:
    """Derive a human-readable description from the Feature spec or Pydantic model."""
    spec = feature.spec()

    # Prefer explicit description in feature metadata when provided
    meta_description = spec.metadata.get("description")
    if isinstance(meta_description, str) and meta_description.strip():
        return meta_description.strip()

    # Next, use the class docstring if available
    doc_description = inspect.getdoc(feature)
    if doc_description and not _is_generic_pydantic_description(doc_description):
        return doc_description

    # Finally, use Pydantic model schema description if it isn't the generic BaseModel blurb
    schema = feature.model_json_schema()
    model_description = schema.get("description")
    if model_description and not _is_generic_pydantic_description(model_description):
        return model_description

    return None


def _normalize_schema_type(prop: dict[str, Any]) -> str | None:
    """Extract a concise type description from a JSON schema property."""
    prop_type = prop.get("type")
    if prop_type is None and "anyOf" in prop:
        candidates: set[str] = set()
        for candidate in prop["anyOf"]:
            if not isinstance(candidate, dict):
                continue
            candidate_type = candidate.get("type")
            if isinstance(candidate_type, str):
                candidates.add(candidate_type)
            elif isinstance(candidate_type, list):
                candidates.update(str(t) for t in candidate_type if isinstance(t, str))
        if candidates:
            prop_type = " | ".join(sorted(candidates))
    if isinstance(prop_type, list):
        prop_type = " | ".join(map(str, prop_type))
    return prop_type


def _infer_table_schema(feature: type[BaseFeature]) -> list[dict[str, Any]]:
    """Build a lightweight table schema from the Pydantic model JSON schema."""
    schema = feature.model_json_schema()
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    columns: list[dict[str, Any]] = []
    for name, prop in properties.items():
        column: dict[str, Any] = {
            "name": name,
            "nullable": name not in required,
        }
        normalized_type = _normalize_schema_type(prop)
        if normalized_type:
            column["type"] = normalized_type
        if "description" in prop:
            column["description"] = prop["description"]
        columns.append(column)
    return columns


def _infer_column_lineage(spec: FeatureSpec) -> list[dict[str, Any]]:
    """Derive column-level lineage from FieldSpec dependencies."""

    def format_dep(dep: FieldDep) -> dict[str, Any]:
        fields = dep.fields
        formatted_fields: str | list[str]
        if fields is SpecialFieldDep.ALL:
            formatted_fields = "*"
        else:
            formatted_fields = [field.to_string() for field in fields]
        return {
            "feature": dep.feature.to_string(),
            "fields": formatted_fields,
        }

    lineage: list[dict[str, Any]] = []
    for field_spec in spec.fields:
        deps = field_spec.deps
        if not deps:
            continue

        if deps is SpecialFieldDep.ALL:
            depends_on: list[dict[str, Any] | str] = ["*"]
        elif isinstance(deps, list):
            depends_on = [format_dep(dep) for dep in deps]
        else:
            # Defensive fallback for unexpected types
            depends_on = []

        lineage.append(
            {
                "column": field_spec.key.to_string(),
                "depends_on": depends_on,
            }
        )
    return lineage


def _infer_asset_tags(feature: type[BaseFeature]) -> dict[str, str]:
    """Provide default tags for Dagster assets derived from the Metaxy feature."""
    spec = feature.spec()
    tags: dict[str, str] = {
        "metaxy.feature_key": _sanitize_tag_value(spec.key.to_string()),
        "metaxy.project": _sanitize_tag_value(feature.project),
    }

    user_tags = spec.metadata.get("tags")
    if isinstance(user_tags, dict):
        # Stringify values so Dagster tags remain homogeneous
        tags.update({str(k): str(v) for k, v in user_tags.items()})

    return tags


def build_asset_spec_from_feature(feature: type[BaseFeature]) -> dg.AssetSpec:
    """Build a Dagster AssetSpec from a Metaxy Feature class.

    Converts a Metaxy Feature into a Dagster asset specification, including:
    - Asset key derived from the feature key
    - Dependencies from feature dependencies
    - Metadata from the feature spec

    Args:
        feature: Metaxy Feature class to convert

    Returns:
        Dagster AssetSpec configured for the feature

    Example:
        ```py
        from metaxy import Feature, FeatureSpec
        from metaxy.ext.dagster import build_asset_spec_from_feature

        class MyFeature(Feature, spec=FeatureSpec(
            key="my/feature",
            id_columns=["sample_uid"],
        )):
            pass

        spec = build_asset_spec_from_feature(MyFeature)
        # spec.key == AssetKey(["my", "feature"])
        ```
    """
    spec = feature.spec()

    # Convert FeatureKey to Dagster AssetKey
    # FeatureKey parts are already a sequence of strings
    asset_key = dg.AssetKey(list(spec.key))

    # Convert dependencies to Dagster AssetKeys
    deps: set[dg.AssetKey] | None = None
    if spec.deps:
        deps = {dg.AssetKey(list(dep.feature)) for dep in spec.deps}

    # Extract metadata from the feature spec
    metadata: dict[str, Any] = {
        "feature_key": spec.key.to_string(),
        "feature_version": feature.feature_version(),
        "feature_spec_version": feature.feature_spec_version(),
        "id_columns": json.dumps(list(spec.id_columns)),
        "fields": json.dumps([field.key.to_string() for field in spec.fields]),
        "table_schema": _infer_table_schema(feature),
        "column_lineage": _infer_column_lineage(spec),
        "lineage": spec.lineage.model_dump(mode="json"),
    }

    # Add user-defined metadata if present (convert to JSON string)
    if spec.metadata:
        metadata["user_metadata"] = json.dumps(spec.metadata)

    # Prefer a description supplied via the Pydantic model schema, then fall back to the class docstring
    description = _infer_model_description(feature)

    return dg.AssetSpec(
        key=asset_key,
        deps=deps,
        description=description or f"Metaxy feature: {spec.key.to_string()}",
        metadata=metadata,
    )


def build_asset_in_from_feature(
    feature: type[BaseFeature],
    input_name: str = "diff",
    *,
    resolve_feature: type[BaseFeature] | None = None,
) -> dict[str, dg.AssetIn]:
    """Build a Dagster AssetIn mapping for a Metaxy Feature.

    Creates an input definition that receives an Increment from the feature's
    materialization. The Increment contains added, changed, and removed samples.

    Args:
        feature: Metaxy Feature class to create input for
        input_name: Name for the input parameter (default: "diff")
        resolve_feature: Feature whose Increment should be resolved by the
            IOManager. Defaults to the same feature that defines the dependency,
            but can be set when the consuming asset wants a different feature's
            Increment than the one it depends on (e.g., pull the consumer's
            own Increment while depending on an upstream feature for ordering).

    Returns:
        Dictionary mapping input_name to AssetIn configured for Increment type

    Example:
        ```py
        from metaxy import Feature, FeatureSpec
        from metaxy.ext.dagster import build_asset_in_from_feature

        class MyFeature(Feature, spec=FeatureSpec(
            key="my/feature",
            id_columns=["sample_uid"],
        )):
            pass

        ins = build_asset_in_from_feature(MyFeature, input_name="increment")
        # ins == {"increment": AssetIn(key=AssetKey(["my", "feature"]))}

        @dg.asset(ins=ins)
        def process_changes(increment):
            # increment is an Increment with added, changed, removed DataFrames
            print(f"Added: {len(increment.added)}")
            print(f"Changed: {len(increment.changed)}")
            print(f"Removed: {len(increment.removed)}")

        # Override which feature Increment to resolve (while still depending on
        # another asset for scheduling)
        ins = build_asset_in_from_feature(
            UpstreamFeature,
            resolve_feature=MyFeature,  # IOManager resolves MyFeature Increment
        )
        ```
    """
    spec = feature.spec()
    resolve_spec = feature.spec() if resolve_feature is None else resolve_feature.spec()

    # Convert FeatureKey to Dagster AssetKey
    asset_key = dg.AssetKey(list(spec.key))
    metadata: dict[str, str] = {
        "feature_key": spec.key.to_string(),
        "type": "metaxy.Increment",
    }

    if resolve_feature is not None:
        metadata["metaxy/resolve_feature_key"] = resolve_spec.key.to_string()

    return {
        input_name: dg.AssetIn(
            key=asset_key,
            metadata=metadata,
        )
    }


def sampling_config_schema(
    *,
    default_mode: str = "random",
    default_size: int | None = None,
) -> dict[str, Any]:
    """Standard Dagster config schema for sampling within Metaxy assets."""
    sample_size_field: dg.Field = (
        dg.Field(dg.IntSource, is_required=True)
        if default_size is None
        else dg.Field(dg.IntSource, default_value=default_size)
    )
    return {
        "sample_mode": dg.Field(dg.StringSource, default_value=default_mode),
        "sample_size": sample_size_field,
        "sample_keys": dg.Field([dg.StringSource], is_required=False),
    }


def apply_sampling(
    samples: nw.DataFrame[Any],
    *,
    sample_mode: str | None = "random",
    sample_keys: list[str] | None = None,
    sample_size: int | None = None,
    key_column: str = "video_id",
    default_sample_size: int = 2,
    log_fn: Callable[[str], Any] | None = None,
) -> nw.DataFrame[Any]:
    """Apply key-based or random sampling to a Narwhals DataFrame."""
    log = log_fn or (lambda _msg: None)
    mode = (sample_mode or "random").lower()

    if mode == "keys":
        if not sample_keys:
            log("sample_mode=keys but no sample_keys provided; returning empty set.")
            return samples.head(0)
        log(f"Filtering to keys in {key_column}: {sample_keys}")
        return samples.filter(nw.col(key_column).is_in(sample_keys))

    # Default to random sampling
    if sample_size is None or sample_size <= 0:
        sample_size = default_sample_size
    log(f"Randomly sampling {sample_size} rows")

    native = samples.to_native()

    # Prefer native sampling when available; fallback to slicing/head
    if hasattr(native, "sample"):
        return nw.from_native(
            native.sample(n=min(sample_size, len(native)), shuffle=True)
        )
    if hasattr(native, "head"):
        return nw.from_native(native.head(sample_size))
    if hasattr(native, "slice"):  # PyArrow Table
        return nw.from_native(native.slice(0, sample_size))

    # Ultimate fallback: use Narwhals head then convert
    return nw.from_native(samples.head(sample_size).to_native())


def build_partitioned_asset_spec_from_feature(
    feature: type[BaseFeature],
    partitions_def: dg.PartitionsDefinition,
) -> dg.AssetSpec:
    """Build a partitioned Dagster AssetSpec from a Metaxy Feature class.

    This is a convenience function that combines `build_asset_spec_from_feature`
    with Dagster partitioning. Use this when you want to process features in
    an event/key parallel pattern.

    Args:
        feature: Metaxy Feature class to convert
        partitions_def: Dagster partitions definition (e.g., DynamicPartitionsDefinition)

    Returns:
        Dagster AssetSpec configured for the feature with partitions

    Example:
        ```py
        from metaxy import Feature, FeatureSpec
        from metaxy.ext.dagster import build_partitioned_asset_spec_from_feature
        import dagster as dg

        class VideoFeature(Feature, spec=FeatureSpec(
            key="video/feature",
            id_columns=["video_id", "frame_id"],
        )):
            pass

        # Create partitioned spec
        partitions = dg.DynamicPartitionsDefinition(name="video_ids")
        spec = build_partitioned_asset_spec_from_feature(VideoFeature, partitions)

        # Use with asset
        @dg.asset(spec=spec)
        def video_feature_asset(context: dg.AssetExecutionContext):
            video_id = context.partition_key
            # Process only this video_id
            ...
        ```
    """
    # Get base spec
    base_spec = build_asset_spec_from_feature(feature)

    # Return new spec with partitions
    return dg.AssetSpec(
        key=base_spec.key,
        deps=base_spec.deps,
        description=base_spec.description,
        metadata=base_spec.metadata,
        partitions_def=partitions_def,
    )


def filter_increment_by_partition(
    increment: Increment,
    partition_column: str,
    partition_key: str,
) -> Increment:
    """Filter an Increment to a specific partition value.

    This helper function filters all DataFrames in an Increment (added, changed, removed)
    to only include rows where the partition column matches the partition key.

    Args:
        increment: The Increment to filter
        partition_column: Column name to filter on (e.g., "video_id")
        partition_key: Value to filter for (e.g., "video_123")

    Returns:
        Filtered Increment with only data for the specified partition

    Raises:
        ValueError: If partition_column is not in the data

    Example:
        ```py
        from metaxy.ext.dagster import filter_increment_by_partition

        # Filter increment to specific video
        filtered = filter_increment_by_partition(
            increment=increment,
            partition_column="video_id",
            partition_key="video_123"
        )

        # Now filtered.added/changed/removed only have data for video_123
        ```
    """
    from metaxy.versioning.types import Increment

    def filter_df(df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        """Filter a single DataFrame."""
        # Empty DataFrames - return as-is
        if len(df) == 0:
            return df

        # Check column exists
        if partition_column not in df.columns:
            raise ValueError(
                f"Partition column '{partition_column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        # Filter
        return df.filter(nw.col(partition_column) == partition_key)

    return Increment(
        added=filter_df(increment.added),
        changed=filter_df(increment.changed),
        removed=filter_df(increment.removed),
    )


def limit_increment(increment: Increment, limit: int) -> Increment:
    """Limit the number of samples in an Increment.

    This helper function applies a head/limit operation to all DataFrames
    in an Increment. Useful for subsampling during testing or development.

    Args:
        increment: The Increment to limit
        limit: Maximum number of rows to keep in each DataFrame

    Returns:
        Limited Increment with at most `limit` rows in each DataFrame

    Example:
        ```py
        from metaxy.ext.dagster import limit_increment

        # Limit to 10 samples for testing
        limited = limit_increment(increment, limit=10)

        # Now limited.added/changed/removed have at most 10 rows each
        ```
    """
    from metaxy.versioning.types import Increment

    return Increment(
        added=increment.added.head(limit),
        changed=increment.changed.head(limit),
        removed=increment.removed.head(limit),
    )


def apply_increment_filter(
    increment: Increment,
    filter_expr: nw.Expr,
) -> Increment:
    """Apply a custom filter expression to an Increment.

    This helper function applies a Narwhals filter expression to all DataFrames
    in an Increment. Useful for custom filtering logic.

    Args:
        increment: The Increment to filter
        filter_expr: Narwhals expression to filter with

    Returns:
        Filtered Increment

    Example:
        ```py
        import narwhals as nw
        from metaxy.ext.dagster import apply_increment_filter

        # Filter to samples with confidence > 0.9
        filtered = apply_increment_filter(
            increment,
            nw.col("confidence") > 0.9
        )

        # Filter to specific date range
        filtered = apply_increment_filter(
            increment,
            (nw.col("date") >= "2024-01-01") & (nw.col("date") < "2024-02-01")
        )
        ```
    """
    from metaxy.versioning.types import Increment

    return Increment(
        added=increment.added.filter(filter_expr),
        changed=increment.changed.filter(filter_expr),
        removed=increment.removed.filter(filter_expr),
    )


def filter_samples_by_partition(
    samples: nw.DataFrame[Any],
    partition_column: str,
    partition_key: str,
) -> nw.DataFrame[Any]:
    """Filter samples DataFrame to a specific partition key.

    Use this in root feature assets to filter samples when using partitioned processing.
    This is essential for event/key parallel patterns where you want to process one
    entity (e.g., one video) at a time through your entire feature graph.

    For downstream features, use the IOManager's `partition_key_column` parameter
    instead, which automatically filters Increments.

    Args:
        samples: DataFrame with samples to filter
        partition_column: Column to filter on (e.g., "video_id")
        partition_key: Value to filter for (e.g., "video_123")

    Returns:
        Filtered DataFrame containing only rows matching the partition key

    Raises:
        ValueError: If partition_column is not found in samples

    Example:
        ```py
        import dagster as dg
        from metaxy.ext.dagster import filter_samples_by_partition

        @dg.asset(partitions_def=video_partitions)
        def root_feature(
            context: dg.AssetExecutionContext,
            store: mx.MetadataStore,
        ) -> mx.FeatureSpec:
            # Generate all samples
            all_samples = generate_samples()

            # Filter to current partition (one video)
            samples = filter_samples_by_partition(
                all_samples,
                partition_column="video_id",
                partition_key=context.partition_key,
            )

            # Now process only this video
            increment = store.resolve_update(RootFeature, samples=samples)
            store.write_metadata(RootFeature, increment.added)
            return RootFeature.spec()
        ```
    """
    # Empty DataFrames - return as-is
    if len(samples) == 0:
        return samples

    # Check column exists
    if partition_column not in samples.columns:
        raise ValueError(
            f"Partition column '{partition_column}' not found in samples. "
            f"Available columns: {samples.columns}"
        )

    # Filter to the partition key
    return samples.filter(nw.col(partition_column) == partition_key)


def create_video_partitions_def(
    name: str = "videos",
    initial_partition_keys: list[str] | None = None,
) -> dg.DynamicPartitionsDefinition:
    """Create a DynamicPartitionsDefinition for video processing.

    Convenience function to create partitions for event-parallel processing.
    Use the same partitions_def across all assets in your pipeline to enable
    processing one entity (e.g., one video) at a time through the entire graph.

    Args:
        name: Name for the partitions (default: "videos")
        initial_partition_keys: Optional list of initial partition keys.
            Can be empty and populated later dynamically.

    Returns:
        DynamicPartitionsDefinition that can be shared across assets

    Example:
        ```py
        import dagster as dg
        from metaxy.ext.dagster import create_video_partitions_def

        # Define partitions once
        video_partitions = create_video_partitions_def(
            name="videos",
            initial_partition_keys=["video_1", "video_2", "video_3"],
        )

        # Use across all assets for consistent partitioning
        @dg.asset(partitions_def=video_partitions)
        def raw_video(context: dg.AssetExecutionContext, store):
            video_id = context.partition_key
            # Process this one video...

        @dg.asset(partitions_def=video_partitions)
        def clean_video(context: dg.AssetExecutionContext, store):
            video_id = context.partition_key
            # Process this one video...

        # Add new partitions dynamically:
        # context.instance.add_dynamic_partitions("videos", ["video_4", "video_5"])
        ```
    """
    return dg.DynamicPartitionsDefinition(name=name)


def asset(
    *,
    feature: type[BaseFeature],
    diff_input_name: str = "diff",
    # All standard Dagster @asset parameters below
    name: str | None = None,
    key_prefix: str | list[str] | None = None,
    ins: dict[str, dg.AssetIn] | None = None,
    non_argument_deps: set[dg.AssetKey] | set[str] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
    config_schema: Any = None,
    required_resource_keys: set[str] | None = None,
    resource_defs: dict[str, Any] | None = None,
    io_manager_key: str | None = "io_manager",
    io_manager_def: Any = None,
    compute_kind: str | None = None,
    dagster_type: Any = None,
    partitions_def: dg.PartitionsDefinition | None = None,
    op_tags: dict[str, Any] | None = None,
    group_name: str | None = None,
    output_required: bool = True,
    freshness_policy: Any = None,
    auto_materialize_policy: Any = None,
    backfill_policy: Any = None,
    retry_policy: Any = None,
    code_version: str | None = None,
    check_specs: list[Any] | None = None,
    owners: list[str] | None = None,
    tags: dict[str, str] | None = None,
    **kwargs: Any,
) -> Callable[[F], dg.AssetsDefinition]:
    """Decorator to create Dagster assets from Metaxy features with automatic configuration.

    This decorator wraps Dagster's `@asset` decorator and adds Metaxy-specific functionality:
    - Automatically builds AssetSpec from the feature (key, deps, metadata)
    - Automatically injects Increment (diff) input for downstream features
    - Wraps return value handling for FeatureSpec or MaterializeResult

    **All standard Dagster @asset parameters are supported** - they are passed through to
    the underlying Dagster decorator. This includes tags, compute_kind, partitions_def,
    auto_materialize_policy, and all others.

    Metaxy-Specific Args:
        feature: Metaxy Feature class to create an asset for
        diff_input_name: Name for the diff/increment input parameter (default: "diff").
            Only used for downstream features with dependencies.

    Standard Dagster Args:
        All parameters from `dagster.asset` are supported including:

        - name, key_prefix: Override asset naming

        - ins, non_argument_deps: Additional inputs/dependencies

        - metadata, description: Asset documentation

        - io_manager_key, io_manager_def: I/O manager configuration

        - compute_kind: Categorize computation (e.g., "python", "dbt")

        - partitions_def: Partitioning strategy

        - op_tags, tags, owners: Organization and metadata

        - freshness_policy, auto_materialize_policy: Materialization policies

        - backfill_policy, retry_policy: Execution policies

        - And many more...

    Returns:
        Decorated function as a Dagster AssetsDefinition

    Example:
        ```py
        import dagster as dg
        import metaxy as mx
        import metaxy.ext.dagster as mxd
        from metaxy.versioning.types import Increment

        # Basic usage - Metaxy provides key, metadata, deps
        @mxd.asset(feature=CleanVideo)
        def clean_video(context, diff: Increment, store):
            # Process diff (automatically injected for downstream features)
            with store:
                if len(diff.added) > 0:
                    cleaned = process_data(diff.added)
                    store.write_metadata(CleanVideo, cleaned)
            return CleanVideo.spec()

        # With Dagster parameters - full control over execution
        @mxd.asset(
            feature=CleanVideo,
            compute_kind="python",
            tags={"team": "ml-platform", "priority": "high"},
            op_tags={"concurrency_key": "video_processing"},
            retry_policy=dg.RetryPolicy(max_retries=3),
            auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
        )
        def clean_video(context, diff: Increment, store):
            # Process with full Dagster features...
            with store:
                if len(diff.added) > 0:
                    store.write_metadata(CleanVideo, process_data(diff.added))
            return CleanVideo.spec()

        # Override metadata - merges with Metaxy's feature metadata
        @mxd.asset(
            feature=CleanVideo,
            metadata={"sla_minutes": 30, "model_version": "v2"},
            key_prefix="production",  # Optional: prefix asset key
        )
        def clean_video(context, diff: Increment, store):
            # Metadata includes both Metaxy feature metadata AND user metadata
            return CleanVideo.spec()

        # For complete examples, see:
        # - examples/example-integration-dagster/src/example_integration_dagster/minimal.py
        ```
    """
    spec_obj = feature.spec()
    is_downstream = spec_obj.deps is not None and len(spec_obj.deps) > 0

    # Build the asset spec from feature
    if partitions_def is not None:
        asset_spec = build_partitioned_asset_spec_from_feature(feature, partitions_def)
    else:
        asset_spec = build_asset_spec_from_feature(feature)

    # Determine key - use user override if provided, otherwise from feature
    # When name or key_prefix are provided, we need to let Dagster handle it
    # Otherwise, use the key from the feature
    asset_key = asset_spec.key if (name is None and key_prefix is None) else None

    # If key_prefix but no name, we need to handle it specially
    if key_prefix is not None and name is None:
        prefix = key_prefix if isinstance(key_prefix, list) else [key_prefix]
        asset_key = dg.AssetKey([*prefix, *asset_spec.key.path])

    # Merge metadata - user metadata takes precedence
    merged_metadata = {**(asset_spec.metadata or {})}
    if metadata is not None:
        merged_metadata.update(metadata)

    # Merge description - user description takes precedence
    final_description = (
        description if description is not None else asset_spec.description
    )

    # Merge tags - user-provided tags override inferred ones
    inferred_tags = _infer_asset_tags(feature)
    final_tags = {**inferred_tags}
    if tags is not None:
        final_tags.update(tags)

    # For downstream features, build ins
    # Important: When we have ins, we must remove deps from the spec to avoid conflict
    metaxy_ins: dict[str, dg.AssetIn] | None = None
    deps_to_use = asset_spec.deps
    if is_downstream:
        # Use the first dependency as the input source
        first_dep = spec_obj.deps[0]
        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()
        dep_feature = graph.get_feature_by_key(first_dep.feature)
        metaxy_ins = build_asset_in_from_feature(
            dep_feature, input_name=diff_input_name, resolve_feature=feature
        )
        # Remove deps since we're using ins instead (Dagster doesn't allow both)
        deps_to_use = None

    # Merge ins - user-provided ins take precedence
    final_ins = {**(metaxy_ins or {})}
    if ins is not None:
        final_ins.update(ins)
    final_ins = final_ins if final_ins else None

    def decorator(fn: F) -> dg.AssetsDefinition:
        """Inner decorator that wraps the user function."""

        # Wrap the function to handle return value
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> FeatureSpec:
            result = fn(*args, **kwargs)

            # If generator (yields MaterializeResult), consume it
            if inspect.isgenerator(result):
                # Consume the generator
                for _ in result:
                    pass
                # Return the FeatureSpec
                return feature.spec()

            # If returns FeatureSpec, return it
            if result is not None:
                return result

            # Otherwise return the feature spec
            return feature.spec()

        # Explicitly preserve the original function's signature and annotations for Dagster
        # functools.wraps copies metadata but Dagster needs the full signature
        cast(Any, wrapper).__signature__ = inspect.signature(fn)
        cast(Any, wrapper).__annotations__ = (
            fn.__annotations__.copy() if hasattr(fn, "__annotations__") else {}
        )

        # Apply @dg.asset with all parameters
        # Metaxy-computed values are used as defaults, user values override
        dagster_asset: dg.AssetsDefinition = dg.asset(
            name=name,
            key=asset_key,
            key_prefix=key_prefix
            if key_prefix is not None and name is None and asset_key is None
            else None,
            ins=final_ins,
            non_argument_deps=non_argument_deps,
            metadata=merged_metadata,
            description=final_description,
            config_schema=config_schema,
            required_resource_keys=required_resource_keys,
            resource_defs=resource_defs,
            io_manager_key=io_manager_key,
            io_manager_def=io_manager_def,
            compute_kind=compute_kind,
            dagster_type=dagster_type,
            partitions_def=partitions_def,
            op_tags=op_tags,
            group_name=group_name,
            output_required=output_required,
            freshness_policy=freshness_policy,
            auto_materialize_policy=auto_materialize_policy,
            backfill_policy=backfill_policy,
            retry_policy=retry_policy,
            code_version=code_version,
            check_specs=check_specs,
            owners=owners,
            tags=final_tags,
            deps=deps_to_use,
            **kwargs,  # Any future Dagster parameters
        )(wrapper)

        return dagster_asset

    return decorator
