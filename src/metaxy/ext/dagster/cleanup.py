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
        cascade: Cascade deletion to dependent/dependency features. Options: none, downstream, upstream, both.
    """

    feature_key: ValidatedFeatureKeyList
    filters: list[str] = Field(
        description="List of SQL WHERE clause filter expressions. See https://docs.metaxy.org/guide/learn/filters/ for syntax."
    )
    soft: bool = Field(
        default=True,
        description="Whether to use soft deletes or hard deletes.",
    )
    cascade: str = Field(
        default="none",
        description="Cascade deletion to dependent/dependency features. Options: none, downstream, upstream, both.",
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
        @dg.job(
            resource_defs={"metaxy_store": MetaxyStoreFromConfigResource(name="dev")}
        )
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
                            "cascade": "none",  # or "downstream", "upstream", "both"
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

    # Validate cascade parameter
    valid_cascade_options = ("none", "downstream", "upstream", "both")
    if config.cascade not in valid_cascade_options:
        raise ValueError(f"Invalid cascade option: {config.cascade}. Valid options: {', '.join(valid_cascade_options)}")

    context.log.info(f"Executing {'soft' if config.soft else 'hard'} delete for {feature_key.to_string()}")

    # Determine features to delete (with or without cascading)
    if config.cascade != "none":
        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()
        features_to_delete = graph.get_cascade_features(feature_key, config.cascade)

        context.log.info(
            f"Cascading {'soft' if config.soft else 'hard'} delete "
            f"for {feature_key.to_string()} in {config.cascade} direction"
        )
        context.log.info(
            f"Will delete {len(features_to_delete)} features in order: "
            f"{', '.join(fk.to_string() for fk in features_to_delete)}"
        )
    else:
        features_to_delete = [feature_key]
        context.log.info(f"Executing {'soft' if config.soft else 'hard'} delete for {feature_key.to_string()}")

    # Execute deletions in order
    with store.open("write"):
        for fk in features_to_delete:
            context.log.info(f"Deleting {fk.to_string()}...")
            store.delete_metadata(fk, filters=filter_exprs, soft=config.soft)

    cascade_info = f" ({config.cascade} cascade)" if config.cascade != "none" else ""
    context.log.info(f"Successfully completed delete for {feature_key.to_string()}{cascade_info}")
