"""Job builder for observing Metaxy feature assets."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import dagster as dg

import metaxy as mx
from metaxy._decorators import public
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PARTITION_KEY,
    DAGSTER_METAXY_PARTITION_METADATA_KEY,
)
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.ext.dagster.utils import (
    build_feature_event_tags,
    build_metaxy_partition_filter,
    build_partition_filter,
    compute_row_count,
)
from metaxy.metadata_store.exceptions import FeatureNotFoundError

logger = logging.getLogger(__name__)


@public
def build_metaxy_multi_observation_job(
    name: str,
    *,
    asset_selection: dg.AssetSelection | None = None,
    defs: dg.Definitions | None = None,
    assets: Sequence[dg.AssetSpec | dg.AssetsDefinition | dg.SourceAsset] | None = None,
    store_resource_key: str = "store",
    tags: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> dg.JobDefinition:
    """Build a dynamic Dagster job that observes multiple Metaxy feature assets.

    Creates a job that dynamically spawns one op per asset, yielding
    [`AssetObservation`](https://docs.dagster.io/api/python-api/ops#dagster.AssetObservation) events.
    Uses Dagster's dynamic orchestration to process multiple assets in parallel.

    !!! tip
        This is a very powerful way to observe all your Metaxy features at once.
        Use it in combination with a [Dagster schedule](https://docs.dagster.io/concepts/schedules)
        to run it periodically.

    Provide either:
    - `asset_selection` and `defs`: Select assets from a
      [`Definitions`](https://docs.dagster.io/api/python-api/definitions#dagster.Definitions) object

    - `assets`: Direct list of assets to observe

    !!! note
        All selected assets must share the same partitioning (if any).

    Args:
        name: Name for the job.
        asset_selection: An `AssetSelection` specifying which assets to observe.
            Must be used together with `defs`.
        defs: The `Definitions` object to resolve the selection against.
            Must be used together with `asset_selection`.
        assets: Direct sequence of assets to observe. Each item can be an
            `AssetSpec`, `AssetsDefinition`, or `SourceAsset`.
            Cannot be used together with `asset_selection`/`defs`.
        store_resource_key: Resource key for the MetadataStore (default: `"store"`).
        tags: Optional tags to apply to the job.
        **kwargs: Additional keyword arguments passed to the
            [`@job`](https://docs.dagster.io/api/python-api/jobs#dagster.job) decorator.

    Returns:
        A Dagster job definition that observes all matching Metaxy assets.

    Raises:
        ValueError: If no specs have `metaxy/feature` metadata, if assets have
            inconsistent `partitions_def`, or if invalid argument combinations
            are provided.

    Example:
        ```python
        import dagster as dg
        import metaxy.ext.dagster as mxd


        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "my/feature_a"})
        def feature_a(): ...


        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "my/feature_b"})
        def feature_b(): ...


        # Option 1: Using asset_selection + defs
        my_defs = dg.Definitions(assets=[feature_a, feature_b])
        observation_job = mxd.build_metaxy_multi_observation_job(
            name="observe_my_features",
            asset_selection=dg.AssetSelection.kind("metaxy"),
            defs=my_defs,
        )

        # Option 2: Using direct assets list
        observation_job = mxd.build_metaxy_multi_observation_job(
            name="observe_my_features",
            assets=[feature_a, feature_b],
        )
        ```
    """
    tags = tags or {}

    # Validate argument combinations
    has_selection = asset_selection is not None or defs is not None
    has_assets = assets is not None

    if has_selection and has_assets:
        raise ValueError(
            "Cannot provide both 'assets' and 'asset_selection'/'defs'. "
            "Use either asset_selection + defs, or assets alone."
        )

    if not has_selection and not has_assets:
        raise ValueError("Must provide either 'asset_selection' + 'defs', or 'assets'.")

    if has_selection:
        if asset_selection is None:
            raise ValueError("'defs' requires 'asset_selection' to be provided.")
        if defs is None:
            raise ValueError("'asset_selection' requires 'defs' to be provided.")

        # Resolve selection using defs
        all_assets_defs = list(defs.resolve_asset_graph().assets_defs)
        selected_keys = asset_selection.resolve(all_assets_defs)

        # Get specs for selected keys, with partitions_def
        metaxy_specs: list[dg.AssetSpec] = []
        partitions_defs: list[dg.PartitionsDefinition | None] = []

        for asset_def in all_assets_defs:
            for spec in asset_def.specs:
                if spec.key in selected_keys:
                    if spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY) is not None:
                        metaxy_specs.append(spec)
                        partitions_defs.append(asset_def.partitions_def)
    else:
        # Direct assets list
        assert assets is not None
        metaxy_specs = []
        partitions_defs = []

        for asset in assets:
            if isinstance(asset, dg.AssetSpec):
                if asset.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY) is not None:
                    metaxy_specs.append(asset)
                    partitions_defs.append(asset.partitions_def)
            elif isinstance(asset, dg.AssetsDefinition):
                for spec in asset.specs:
                    if spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY) is not None:
                        metaxy_specs.append(spec)
                        partitions_defs.append(asset.partitions_def)
            elif isinstance(asset, dg.SourceAsset):
                # SourceAsset doesn't have metaxy/feature metadata typically
                pass
            else:
                raise TypeError(f"Expected AssetSpec, AssetsDefinition, or SourceAsset, got {type(asset).__name__}")

    if not metaxy_specs:
        raise ValueError(
            "No assets have specs with 'metaxy/feature' metadata. "
            "Ensure your assets have metadata={'metaxy/feature': 'feature/key'}."
        )

    # Validate all specs have the same partitions_def
    first_partitions_def = partitions_defs[0]
    for i, pdef in enumerate(partitions_defs[1:], start=1):
        if pdef != first_partitions_def:
            raise ValueError(
                f"All assets must have the same partitions_def. "
                f"Asset 0 has {first_partitions_def}, but asset {i} has {pdef}."
            )
    partitions_def = first_partitions_def

    # Build feature keys for description (may have duplicates when multiple assets share a feature)
    feature_keys = [
        mx.coerce_to_feature_key(spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]) for spec in metaxy_specs
    ]

    # Build a mapping of asset key -> spec for the dynamic op
    # This ensures each asset gets its own op, even if multiple assets share the same feature
    spec_by_asset_key = {spec.key.to_user_string(): spec for spec in metaxy_specs}
    all_asset_keys = list(spec_by_asset_key.keys())

    # Config class for runtime filtering of assets to observe
    class _ObserveAssetsConfig(dg.Config):
        asset_keys: list[str] = all_asset_keys

    # Op that emits dynamic outputs for each asset, optionally filtered by config
    @dg.op(
        name=f"{name}_fanout",
        out=dg.DynamicOut(str),
        config_schema=_ObserveAssetsConfig.to_config_schema(),
    )
    def fanout_assets(context: dg.OpExecutionContext) -> Any:
        config = _ObserveAssetsConfig.model_validate(context.op_config)

        # Validate that requested asset keys exist
        requested_keys = set(config.asset_keys)
        available_keys = set(spec_by_asset_key.keys())
        invalid_keys = requested_keys - available_keys
        if invalid_keys:
            raise ValueError(
                f"Requested asset keys not found in job: {sorted(invalid_keys)}. "
                f"Available keys: {sorted(available_keys)}"
            )
        asset_keys_to_observe = [k for k in spec_by_asset_key if k in requested_keys]

        for asset_key_str in asset_keys_to_observe:
            # Use asset key (with / replaced by __) as mapping key for Dagster identifiers
            safe_mapping_key = asset_key_str.replace("/", "__")
            yield dg.DynamicOutput(asset_key_str, mapping_key=safe_mapping_key)

    # Build the shared observation op
    observe_op = _build_observation_op_for_specs(
        name=f"{name}_observe",
        spec_by_asset_key=spec_by_asset_key,
        store_resource_key=store_resource_key,
    )

    # Build job metadata with asset references
    job_metadata: dict[str, Any] = {
        "metaxy/features": [fk.to_string() for fk in feature_keys],
    }
    for spec in metaxy_specs:
        job_metadata[f"metaxy/asset/{spec.key.to_user_string()}"] = dg.MetadataValue.asset(spec.key)

    # Build description as markdown list showing both assets and features
    asset_list = "\n".join(
        f"- `{spec.key.to_user_string()}` → `{spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]}`"
        for spec in metaxy_specs
    )
    description = f"Observe {len(metaxy_specs)} Metaxy assets:\n\n{asset_list}"

    @dg.job(
        name=name,
        partitions_def=partitions_def,
        tags=tags,
        description=description,
        metadata=job_metadata,
        **kwargs,
    )
    def observation_job() -> None:
        asset_keys_dynamic = fanout_assets()
        asset_keys_dynamic.map(observe_op)

    return observation_job


@public
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
        def my_asset(): ...


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
        raise TypeError(f"Expected AssetSpec or AssetsDefinition, got {type(asset).__name__}")

    # Filter to specs with metaxy/feature metadata
    metaxy_specs = [spec for spec in specs if spec.metadata.get(DAGSTER_METAXY_FEATURE_METADATA_KEY) is not None]

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
    **kwargs: Any,
) -> dg.JobDefinition:
    """Build an observation job for a single asset spec."""
    tags = tags or {}

    feature_key_str = spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]
    feature_key = mx.coerce_to_feature_key(feature_key_str)
    job_name = f"observe_{feature_key.table_name}"

    # Build the shared observation op with a single spec (keyed by asset key)
    spec_by_asset_key = {spec.key.to_user_string(): spec}
    observe_op = _build_observation_op_for_specs(
        name=f"observe_{spec.key.to_python_identifier()}",
        spec_by_asset_key=spec_by_asset_key,
        store_resource_key=store_resource_key,
    )

    # Create an op that returns the asset key string (needed for graph composition)
    @dg.op(name=f"get_asset_key_{spec.key.to_python_identifier()}")
    def get_asset_key() -> str:
        return spec.key.to_user_string()

    @dg.job(
        name=job_name,
        partitions_def=partitions_def,
        tags={
            **tags,
            "metaxy/feature": feature_key.to_string(),
        },
        description=f"Observe Metaxy feature {feature_key.to_string()} (Dagster asset: `{spec.key.to_user_string()}`)",
        metadata={
            "metaxy/feature": feature_key.to_string(),
            "dagster/asset": dg.MetadataValue.asset(spec.key),
        },
        **kwargs,
    )
    def observation_job() -> None:
        observe_op(get_asset_key())

    return observation_job


def _build_observation_op_for_specs(
    name: str,
    spec_by_asset_key: dict[str, dg.AssetSpec],
    store_resource_key: str,
) -> dg.OpDefinition:
    """Build an op that observes a Metaxy feature asset.

    This op is shared between single-asset and multi-asset observation jobs.
    It takes an asset key string as input and looks up the corresponding spec.

    Args:
        name: Name for the op.
        spec_by_asset_key: Mapping from asset key strings to asset specs.
        store_resource_key: Resource key for the MetadataStore.

    Returns:
        An op definition that yields an AssetObservation.
    """

    @dg.op(
        name=name,
        required_resource_keys={store_resource_key},
        out=dg.Out(dg.Nothing),
    )
    def observe_asset(context: dg.OpExecutionContext, asset_key_str: str) -> None:
        spec = spec_by_asset_key[asset_key_str]
        feature_key_str = spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]
        feature_key = mx.coerce_to_feature_key(feature_key_str)
        partition_col = spec.metadata.get(DAGSTER_METAXY_PARTITION_KEY)
        metaxy_partition = spec.metadata.get(DAGSTER_METAXY_PARTITION_METADATA_KEY)

        store: mx.MetadataStore | MetaxyStoreFromConfigResource = getattr(context.resources, store_resource_key)

        # Build partition filters:
        # 1. Dagster partition filter (for time/date partitions)
        partition_key = context.partition_key if context.has_partition_key else None
        dagster_partition_filters = build_partition_filter(partition_col, partition_key)
        # 2. Metaxy partition filter (for multi-asset logical partitions)
        metaxy_partition_filters = build_metaxy_partition_filter(metaxy_partition)
        # Combine both filter types
        all_filters = dagster_partition_filters + metaxy_partition_filters

        with store:
            try:
                lazy_df = store.read_metadata(feature_key, filters=all_filters)
            except FeatureNotFoundError:
                context.log.warning(
                    f"Feature {feature_key.to_string()} not found in store, returning empty observation"
                )
                context.log_event(
                    dg.AssetObservation(
                        asset_key=spec.key,
                        partition=partition_key,
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
                tags=build_feature_event_tags(feature_key),
            )
        )

    return observe_asset
