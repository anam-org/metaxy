"""Pydantic models for system tables."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from pydantic import BaseModel, Field

from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY
from metaxy.metadata_store.system.events import EVENTS_SCHEMA
from metaxy.metadata_store.system.keys import EVENTS_KEY
from metaxy.models.constants import (
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_FULL_DEFINITION_VERSION,
    METAXY_SNAPSHOT_VERSION,
)

# Common Polars schemas for system tables
FEATURE_VERSIONS_SCHEMA = {
    "project": pl.String,
    "feature_key": pl.String,
    METAXY_FEATURE_VERSION: pl.String,
    METAXY_FEATURE_SPEC_VERSION: pl.String,  # Hash of complete FeatureSpec (all properties)
    METAXY_FULL_DEFINITION_VERSION: pl.String,  # Hash of feature_spec_version + project (for migration detection)   # TODO: this is probably not needed, we can just use a combination of project and metaxy_feature_version instead
    "recorded_at": pl.Datetime("us"),
    "feature_spec": pl.String,  # Full serialized FeatureSpec
    "feature_class_path": pl.String,
    METAXY_SNAPSHOT_VERSION: pl.String,
}


class FeatureVersionsModel(BaseModel):
    """Pydantic model for feature_versions system table.

    This table records when feature specifications are pushed to production,
    tracking the evolution of feature definitions over time.
    """

    project: str
    feature_key: str
    metaxy_feature_version: str = Field(
        ...,
        description="Hash of versioned feature topology (combined versions of fields on this feature)",
    )
    metaxy_feature_spec_version: str = Field(
        ..., description="Hash of complete FeatureSpec (all properties)"
    )
    metaxy_full_definition_version: str = Field(
        ..., description="Hash of feature_spec_version + project"
    )
    recorded_at: datetime = Field(
        ..., description="Timestamp when feature version was recorded"
    )
    feature_spec: str = Field(
        ..., description="Full serialized FeatureSpec as JSON string"
    )
    feature_class_path: str = Field(
        ..., description="Python import path to Feature class"
    )
    metaxy_snapshot_version: str = Field(
        ..., description="Deterministic hash of entire Metaxy project"
    )

    def to_polars(self) -> pl.DataFrame:
        """Convert this model instance to a single-row Polars DataFrame.

        Returns:
            Polars DataFrame with one row matching FEATURE_VERSIONS_SCHEMA
        """
        # Polars can directly convert Pydantic models to DataFrames
        return pl.DataFrame([self], schema=FEATURE_VERSIONS_SCHEMA)


POLARS_SCHEMAS = {
    FEATURE_VERSIONS_KEY: FEATURE_VERSIONS_SCHEMA,
    EVENTS_KEY: EVENTS_SCHEMA,
}
