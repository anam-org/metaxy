"""Dagster ops for metadata deletion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dagster as dg
import narwhals as nw
from pydantic import Field

import metaxy as mx
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.models.types import FeatureKey


class DeleteMetadataConfig(dg.Config):
    """Configuration for delete_metadata op."""

    feature_key: list[str] = Field(
        description="Feature key as list of strings (e.g., ['user', 'profile']).",
    )
    filter_expr: str | None = Field(
        default=None,
        description="Narwhals filter expression as string (e.g., \"nw.col('status') == 'inactive'\").",
    )
    retention_days: int | None = Field(
        default=None,
        description="Delete records older than this many days (alternative to filter_expr).",
    )
    timestamp_column: str = Field(
        default="metaxy_created_at",
        description="Column to use for retention calculation.",
    )
    hard: bool = Field(
        default=False,
        description="Use hard delete (physical removal) instead of soft delete.",
    )


@dg.op(
    required_resource_keys={"store"},
    tags={"kind": "metaxy", "operation": "delete"},
)
def delete_metadata(context):
    """Execute metadata deletion operation."""

    config = context.op_config
    store: mx.MetadataStore | MetaxyStoreFromConfigResource = context.resources.store
    feature_key = FeatureKey(config["feature_key"])

    if config.get("retention_days"):
        cutoff = datetime.now(timezone.utc) - timedelta(days=config["retention_days"])
        timestamp_col = config.get("timestamp_column", "metaxy_created_at")
        filter_expr = nw.col(timestamp_col) < nw.lit(cutoff)
    elif config.get("filter_expr"):
        filter_expr = eval(config["filter_expr"])
    else:
        raise ValueError("Must provide either filter_expr or retention_days")

    hard = config.get("hard", False)
    context.log.info(
        f"Executing {'hard' if hard else 'soft'} delete for {feature_key.to_string()}"
    )

    with store.open("write"):
        store.delete_metadata(feature_key, filters=filter_expr, soft=not hard)
