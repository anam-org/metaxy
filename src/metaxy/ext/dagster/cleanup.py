"""Dagster ops for metadata deletion."""

import dagster as dg
from pydantic import Field

import metaxy as mx
from metaxy.models.types import CascadeMode, FeatureKey, ValidatedFeatureKeyList


class DeleteMetadataConfig(dg.Config):
    """Configuration for delete op.

    Attributes:
        feature_key: Feature key validated using ValidatedFeatureKey semantics.
        filters: List of SQL WHERE clause filter expressions (e.g., ["status = 'inactive'", "age > 18"]).
            See https://docs.metaxy.org/guide/concepts/filters/ for syntax.
        soft: Whether to use soft deletes or hard deletes.
        cascade: Cascade deletion to dependent/dependency features. Options: none, downstream, upstream, both.
    """

    feature_key: ValidatedFeatureKeyList
    filters: list[str] = Field(
        description="List of SQL WHERE clause filter expressions. See https://docs.metaxy.org/guide/concepts/filters/ for syntax."
    )
    soft: bool = Field(
        default=True,
        description="Whether to use soft deletes or hard deletes.",
    )
    cascade: CascadeMode = Field(
        default=CascadeMode.NONE,
        description="Cascade deletion to dependent/dependency features. Options: none, downstream, upstream, both.",
    )


@dg.op
def delete(
    context: dg.OpExecutionContext,
    config: DeleteMetadataConfig,
    metaxy_store: dg.ResourceParam[mx.MetadataStore],
) -> None:
    """Execute metadata deletion operation.

    Args:
        context: Dagster execution context.
        config: Deletion configuration.
        metaxy_store: Configured Metaxy metadata store resource.

    Example:
        ```python
        import dagster as dg
        from metaxy.ext.dagster import delete
        from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource


        # Define a job with the delete op
        @dg.job(resource_defs={"metaxy_store": MetaxyStoreFromConfigResource(name="dev")})
        def cleanup_job():
            delete()


        # Execute with config to delete inactive customer segments
        cleanup_job.execute_in_process(
            run_config={
                "ops": {
                    "delete": {
                        "config": {
                            "feature_key": ["customer", "segment"],
                            "filters": ["status = 'inactive'"],
                            "soft": True,
                        }
                    }
                }
            }
        )
        ```
    """
    from metaxy.models.filter_expression import parse_filter_string

    store = metaxy_store

    # Convert validated list[str] to FeatureKey
    feature_key = FeatureKey(config.feature_key)

    # Parse filter strings into Narwhals expressions
    filter_exprs = [parse_filter_string(f) for f in config.filters]

    context.log.info(f"Executing {'soft' if config.soft else 'hard'} delete for {feature_key.to_string()}")

    # Determine features to delete (with or without cascading)
    if config.cascade != CascadeMode.NONE:
        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()
        features_to_delete: list[FeatureKey] = graph.get_cascade_deletion_order(feature_key, config.cascade)

        context.log.info(
            f"Cascading {'soft' if config.soft else 'hard'} delete "
            f"for {feature_key.to_string()} in {config.cascade.value} direction"
        )
        context.log.info(
            f"Will delete {len(features_to_delete)} features in order: "
            f"{', '.join(fk.to_string() for fk in features_to_delete)}"
        )
    else:
        features_to_delete = [feature_key]

    with store.open("w"):
        for fk in features_to_delete:
            context.log.info(f"Deleting {fk.to_string()}...")
            store.delete(fk, filters=filter_exprs, soft=config.soft)

    cascade_info = f" ({config.cascade.value} cascade)" if config.cascade != CascadeMode.NONE else ""
    context.log.info(f"Successfully completed delete for {feature_key.to_string()}{cascade_info}")
