"""Pydantic models for system tables."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from pydantic import BaseModel, Field, field_validator

from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY
from metaxy.metadata_store.system.events import EVENTS_SCHEMA
from metaxy.metadata_store.system.keys import EVENTS_KEY
from metaxy.models.constants import (
    METAXY_DEFINITION_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_SNAPSHOT_VERSION,
)

# Common Polars schemas for system tables
# Note: tags field schema is omitted - Polars will infer the Struct schema from data
FEATURE_VERSIONS_SCHEMA = {
    "project": pl.String,
    "feature_key": pl.String,
    METAXY_FEATURE_VERSION: pl.String,
    METAXY_DEFINITION_VERSION: pl.String,  # Hash of feature definition (spec + schema), excludes project
    "recorded_at": pl.Datetime("us"),
    "feature_spec": pl.String,  # Full serialized FeatureSpec
    "feature_schema": pl.String,  # Full Pydantic model schema as JSON
    "feature_class_path": pl.String,
    METAXY_SNAPSHOT_VERSION: pl.String,
    "tags": pl.String,
    "deleted_at": pl.Datetime("us"),  # Timestamp when feature was removed from the project (nullable)
}


METAXY_TAG = "metaxy"
METAXY_VERSION_KEY = "version"


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
    metaxy_definition_version: str = Field(
        ..., description="Hash of feature definition (spec + schema), excludes project"
    )
    recorded_at: datetime = Field(..., description="Timestamp when feature version was recorded")
    feature_spec: str = Field(..., description="Full serialized FeatureSpec as JSON string")
    feature_schema: str = Field(..., description="Full Pydantic model schema as JSON string")
    feature_class_path: str = Field(..., description="Python import path to Feature class")
    metaxy_snapshot_version: str = Field(..., description="Deterministic hash of entire Metaxy project")
    tags: dict[str, str] | str = Field(
        default="{}",
        description="Snapshot tags as JSON string (key-value pairs). The metaxy tag is reserved for internal use.",
        validate_default=True,
    )
    deleted_at: datetime | None = Field(
        default=None, description="Timestamp when the feature has been removed from the project"
    )

    @field_validator("tags", mode="before")
    @classmethod
    def serialize_tags(cls, v: dict[str, str] | str | None) -> str:
        """Convert tags dict to JSON string if needed."""
        import json

        # Parse to dict if string
        tags_dict: dict[str, str]
        if isinstance(v, str):
            try:
                tags_dict = json.loads(v)
            except json.JSONDecodeError:
                tags_dict = {}
        else:
            # Handle None or dict
            tags_dict = v or {}

        # Ensure metaxy.version is set
        from metaxy._version import __version__

        metaxy_tag_value = tags_dict.get(METAXY_TAG, "{}")
        metaxy_tag_dict = json.loads(metaxy_tag_value) if isinstance(metaxy_tag_value, str) else metaxy_tag_value
        if not isinstance(metaxy_tag_dict, dict):
            metaxy_tag_dict = {}
        metaxy_tag_dict[METAXY_VERSION_KEY] = metaxy_tag_dict.get(METAXY_VERSION_KEY, __version__)
        tags_dict[METAXY_TAG] = json.dumps(metaxy_tag_dict)

        return json.dumps(tags_dict)

    def to_polars(self) -> pl.DataFrame:
        """Convert this model instance to a single-row Polars DataFrame.

        Returns:
            Polars DataFrame with one row matching FEATURE_VERSIONS_SCHEMA
        """
        # tags is already a JSON string, no need to serialize
        return pl.DataFrame([self.model_dump()], schema=FEATURE_VERSIONS_SCHEMA)


POLARS_SCHEMAS = {
    FEATURE_VERSIONS_KEY: FEATURE_VERSIONS_SCHEMA,
    EVENTS_KEY: EVENTS_SCHEMA,
}
