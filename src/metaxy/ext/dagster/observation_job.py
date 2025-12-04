"""Job builder for observing Metaxy feature assets."""

import logging
from collections.abc import Mapping
from typing import Any

import dagster as dg
from dagster import Nothing

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PARTITION_KEY,
)
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.ext.dagster.utils import (
    build_partition_filter,
    compute_row_count,
)
from metaxy.metadata_store.exceptions import FeatureNotFoundError

logger = logging.getLogger(__name__)


def build_metaxy_observation_job(
    asset: dg.AssetSpec | dg.AssetsDefinition,
    *,
    store_resource_key: str = "store",
    tags: dict[str, str] | None = None,
) -> list[dg.JobDefinition]:
    """Build Dagster job(s) that observe Metaxy feature asset(s).

    Creates job(s) that yield `AssetObservation` events for the given asset.
    The job can be run independently from asset materialization, e.g., on a schedule.

    Returns one job per `metaxy/feature` spec found in the asset.

    Jobs are constructed with matching partitions definitions.
    Job names are always derived as `observe_<FeatureKey.table_name()>`.

    Args:
        asset: Asset spec or asset definition to observe. Must have `metaxy/feature`
            metadata on at least one spec.
        store_resource_key: Resource key for the MetadataStore (default: `"store"`).
        tags: Optional tags to apply to the job(s).

    Returns:
        List of Dagster job definitions, one per `metaxy/feature` spec.

    Raises:
        ValueError: If no specs have `metaxy/feature` metadata.

    Example:
        ```python
        import dagster as dg
        import metaxy.ext.dagster as mxd

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "my/feature"})
        def my_asset():
            ...

        # Build the observation job - partitions_def is extracted automatically
        observation_job = mxd.build_metaxy_observation_job(my_asset)

        # Include in your Definitions
        defs = dg.Definitions(
            jobs=[observation_job],
            resources={"store": my_store_resource},
        )
        ```
    """
    # Extract specs and partitions_def from asset
    if isinstance(asset, dg.AssetSpec):
        specs = [asset]
        partitions_def = None
    elif isinstance(asset, dg.AssetsDefinition):
        specs = list(asset.specs)
        partitions_def = asset.partitions_def
    else:
        raise TypeError(
            f"Expected AssetSpec or AssetsDefinition, got {type(asset).__name__}"
        )

    # Filter to specs with metaxy/feature metadata
    metaxy_specs = [
        spec
        for spec in specs
        if spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY) is not None
    ]

    if not metaxy_specs:
        raise ValueError(
            "Asset has no specs with 'metaxy/feature' metadata. "
            "Ensure your asset has metadata={'metaxy/feature': 'feature/key'}."
        )

    # Build jobs for each metaxy spec
    jobs = [
        _build_observation_job_for_spec(
            spec,
            partitions_def=partitions_def,
            store_resource_key=store_resource_key,
            tags=tags,
        )
        for spec in metaxy_specs
    ]

    return jobs


def _build_observation_job_for_spec(
    spec: dg.AssetSpec,
    *,
    partitions_def: dg.PartitionsDefinition | None,
    store_resource_key: str,
    tags: Mapping[str, str] | None,
) -> dg.JobDefinition:
    """Build an observation job for a single asset spec."""
    tags = tags or {}

    feature_key_str = spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]
    feature_key = mx.coerce_to_feature_key(feature_key_str)
    job_name = f"observe_{feature_key.table_name}"

    op = _build_observation_op(spec, store_resource_key=store_resource_key)

    @dg.job(
        name=job_name,
        partitions_def=partitions_def,
        tags={
            **tags,
            "metaxy/feature": feature_key.to_string(),
        },
        description=f"Observe Metaxy feature {feature_key.to_string()} (Dagster asset: `{spec.key.to_user_string()}`)",
    )
    def observation_job() -> None:
        op()

    return observation_job


def _build_observation_op(
    spec: dg.AssetSpec,
    *,
    store_resource_key: str,
) -> dg.OpDefinition:
    """Build an op that observes a single Metaxy feature asset.

    Args:
        spec: The asset spec with `metaxy/feature` metadata.
        store_resource_key: Resource key for the MetadataStore.

    Returns:
        An op definition that yields an AssetObservation.
    """
    feature_key_str = spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]
    feature_key = mx.coerce_to_feature_key(feature_key_str)
    partition_col = spec.metadata.get(DAGSTER_METAXY_PARTITION_KEY)

    # Create a unique op name based on the asset key
    op_name = f"observe_{spec.key.to_python_identifier()}"

    @dg.op(
        name=op_name,
        required_resource_keys={store_resource_key},
        out=dg.Out(Nothing),
    )
    def observe_feature(context: dg.OpExecutionContext) -> None:
        store: mx.MetadataStore | MetaxyStoreFromConfigResource = getattr(
            context.resources, store_resource_key
        )

        # Build partition filter if partitioned
        partition_key = context.partition_key if context.has_partition_key else None
        partition_filters = build_partition_filter(partition_col, partition_key)

        with store:
            try:
                lazy_df = store.read_metadata(feature_key, filters=partition_filters)
            except FeatureNotFoundError:
                context.log.warning(
                    f"Feature {feature_key.to_string()} not found in store, "
                    "returning empty observation"
                )
                context.log_event(
                    dg.AssetObservation(
                        asset_key=spec.key,
                        metadata={"dagster/row_count": 0, "error": "Feature not found"},
                    )
                )
                return

            # Only log runtime metadata (row counts)
            metadata: dict[str, Any] = {}
            partition_row_count = compute_row_count(lazy_df)

            if context.has_partition_key:
                # Read entire feature (no partition filter) for total count
                full_lazy_df = store.read_metadata(feature_key)
                metadata["dagster/row_count"] = compute_row_count(full_lazy_df)
                metadata["dagster/partition_row_count"] = partition_row_count
            else:
                metadata["dagster/row_count"] = partition_row_count

        context.log_event(
            dg.AssetObservation(
                asset_key=spec.key,
                partition=partition_key,
                metadata=metadata,
            )
        )

    return observe_feature
