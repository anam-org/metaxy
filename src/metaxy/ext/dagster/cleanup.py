"""Dagster ops for metadata deletion."""

import dagster as dg
from pydantic import Field

import metaxy as mx
from metaxy.models.types import FeatureKey, ValidatedFeatureKeyList


class DeleteMetadataConfig(dg.Config):
    """Configuration for delete_metadata op.

    Attributes:
        feature_key: Feature key validated using ValidatedFeatureKey semantics.
        filters: List of SQL WHERE clause filter expressions (e.g., ["status = 'inactive'", "age > 18"]).
            See https://docs.metaxy.org/guide/learn/filters/ for syntax.
        soft: Whether to use soft deletes or hard deletes.
    """

    feature_key: ValidatedFeatureKeyList
    filters: list[str] = Field(
        description="List of SQL WHERE clause filter expressions. See https://docs.metaxy.org/guide/learn/filters/ for syntax."
    )
    soft: bool = Field(
        default=True,
        description="Whether to use soft deletes or hard deletes.",
    )


@dg.op
def delete_metadata(
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
        from metaxy.ext.dagster import delete_metadata
        from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource


        # Define a job with the delete op
        @dg.job(resource_defs={"metaxy_store": MetaxyStoreFromConfigResource(name="dev")})
        def cleanup_job():
            delete_metadata()


        # Execute with config to delete inactive customer segments
        cleanup_job.execute_in_process(
            run_config={
                "ops": {
                    "delete_metadata": {
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

    with store.open("write"):
        store.delete_metadata(feature_key, filters=filter_exprs, soft=config.soft)

    context.log.info(f"Successfully completed delete for {feature_key.to_string()}")
