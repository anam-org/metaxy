"""Dagster ops for orchestrated metadata deletion and mutation operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import dagster as dg
import narwhals as nw
from pydantic import Field

import metaxy as mx
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.metadata_store.cleanup import DeletionResult, MutationResult
from metaxy.models.types import FeatureKey


class DeleteMetadataConfig(dg.Config):
    """Configuration for delete_metadata op.

    Defines deletion parameters for Dagster config system.
    """

    feature_key: list[str] = Field(
        description="Feature key as list of strings (e.g., ['user', 'profile'])",
    )
    filter_expr: str | None = Field(
        default=None,
        description="Narwhals filter expression as string (e.g., \"nw.col('status') == 'inactive'\")",
    )
    retention_days: int | None = Field(
        default=None,
        description="Delete records older than this many days (alternative to filter_expr)",
    )
    timestamp_column: str = Field(
        default="metaxy_created_at",
        description="Column to use for retention calculation",
    )
    hard: bool = Field(
        default=False,
        description="Use hard delete (physical removal) instead of soft delete",
    )


@dg.op(
    required_resource_keys={"store"},
    tags={"kind": "metaxy", "operation": "delete"},
)
def delete_metadata(
    context,
) -> DeletionResult:
    """Execute metadata deletion operation.

    Performs actual deletion operation, removing data according to the configuration.
    Always logs the operation to the audit trail.

    WARNING: `filter_expr` is evaluated with Python ``eval``. Do not supply
    untrusted input here; prefer ``retention_days`` or vetted templates.

    Args:
        context: Dagster op execution context
        config: Deletion configuration

    Returns:
        DeletionResult with details of what was deleted

    Example:
        ```python
        import dagster as dg
        import metaxy.ext.dagster as mxd

        @dg.job(resource_defs={"store": mxd.MetaxyStoreFromConfigResource(name="default")})
        def cleanup_job():
            mxd.delete_metadata(
                feature_key=["logs"],
                retention_days=90,
                hard=False,  # Soft delete
            )
        ```
    """
    config = context.op_config
    store: mx.MetadataStore | MetaxyStoreFromConfigResource = context.resources.store
    feature_key = FeatureKey(config["feature_key"])

    # Build filter expression
    if config.get("retention_days"):
        cutoff = datetime.now(timezone.utc) - timedelta(days=config["retention_days"])
        timestamp_col = config.get("timestamp_column", "metaxy_created_at")
        filter_expr = nw.col(timestamp_col) < nw.lit(cutoff)
    elif config.get("filter_expr"):
        # Evaluate the filter expression string
        filter_expr = eval(config["filter_expr"])
    else:
        raise ValueError("Must provide either filter_expr or retention_days")

    hard = config.get("hard", False)
    context.log.info(
        f"Executing {'hard' if hard else 'soft'} delete for {feature_key.to_string()}"
    )

    with store.open("write"):
        if hard:
            result = store.delete_metadata(
                feature_key,
                filter=filter_expr,
            )
        else:
            result = store.soft_delete_metadata(
                feature_key,
                filter=filter_expr,
            )

    context.log.info(f"Deletion complete: {result.rows_affected} rows deleted")

    if result.error:
        context.log.error(f"Deletion error: {result.error}")

    # Add metadata for Dagster UI
    context.add_output_metadata(
        {
            "rows_deleted": result.rows_affected,
            "deletion_mode": "hard" if hard else "soft",
            "timestamp": result.timestamp.isoformat(),
            "run_id": context.run_id,
            "error": result.error,
        }
    )

    return result


class MutateMetadataConfig(dg.Config):
    """Configuration for mutate_metadata op.

    Defines mutation parameters for Dagster config system.
    """

    feature_key: list[str] = Field(
        description="Feature key as list of strings (e.g., ['user', 'profile'])",
    )
    filter_expr: str = Field(
        description="Narwhals filter expression as string (e.g., \"nw.col('user_id') == 'user_123'\")",
    )
    updates: dict[str, Any] = Field(
        description="Column -> value mapping for updates (e.g., {'email': '[REDACTED]'})",
    )


@dg.op(
    required_resource_keys={"store"},
    tags={"kind": "metaxy", "operation": "mutate"},
)
def mutate_metadata(
    context,
) -> MutationResult:
    """Execute metadata mutation operation.

    Updates column values for records matching the filter. Useful for GDPR anonymization,
    setting flags, or other in-place updates.

    WARNING: `filter_expr` is evaluated with Python ``eval``. Do not supply
    untrusted input; keep configs trusted or migrate to a structured filter
    representation when available.

    Args:
        context: Dagster op execution context
        config: Mutation configuration

    Returns:
        MutationResult with details of what was mutated

    Example:
        ```python
        import dagster as dg
        import metaxy.ext.dagster as mxd

        @dg.job(resource_defs={"store": mxd.MetaxyStoreFromConfigResource(name="default")})
        def anonymize_job():
            mxd.mutate_metadata(
                feature_key=["user", "profile"],
                filter_expr="nw.col('user_id') == 'user_123'",
                updates={"email": "[REDACTED]", "phone": None},
            )
        ```
    """
    config = context.op_config
    store: mx.MetadataStore | MetaxyStoreFromConfigResource = context.resources.store
    feature_key = FeatureKey(config["feature_key"])

    # Evaluate filter expression
    filter_expr = eval(config["filter_expr"])
    updates = config["updates"]
    context.log.info(
        f"Executing mutation for {feature_key.to_string()}, updates={updates}"
    )

    with store.open("write"):
        result = store.mutate_metadata(
            feature_key,
            filter=filter_expr,
            updates=updates,
        )

    context.log.info(f"Mutation complete: {result.rows_affected} rows updated")

    if result.error:
        context.log.error(f"Mutation error: {result.error}")

    # Add metadata for Dagster UI
    context.add_output_metadata(
        {
            "rows_updated": result.rows_affected,
            "updates": str(updates),
            "timestamp": result.timestamp.isoformat(),
            "run_id": context.run_id,
            "error": result.error,
        }
    )

    return result
