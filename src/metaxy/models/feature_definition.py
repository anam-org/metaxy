"""Feature definition model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

from pydantic import Field

from metaxy._hashing import truncate_hash
from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature


class FeatureDefinition(FrozenBaseModel):
    """Complete feature definition wrapping all feature information.

    Attributes:
        spec: The complete feature specification
        feature_schema: Pydantic JSON schema dict for the feature model
        feature_class_path: Python import path (e.g., 'myapp.features.VideoFeature')
        project: The metaxy project this feature belongs to
    """

    spec: FeatureSpec = Field(description="Complete feature specification")
    feature_schema: dict[str, Any] = Field(description="Pydantic JSON schema dict")
    feature_class_path: str = Field(description="Python import path", min_length=2)
    project: str = Field(description="The metaxy project this feature belongs to", min_length=1)

    @classmethod
    def from_feature_class(cls, feature_cls: type[BaseFeature]) -> FeatureDefinition:
        """Create a FeatureDefinition from a Feature class."""
        spec = feature_cls.spec()
        schema = feature_cls.model_json_schema()
        class_path = f"{feature_cls.__module__}.{feature_cls.__name__}"
        project = feature_cls.metaxy_project()

        return cls(
            spec=spec,
            feature_schema=schema,
            feature_class_path=class_path,
            project=project,
        )

    @staticmethod
    def _compute_definition_version(spec: FeatureSpec, schema: dict[str, Any]) -> str:
        """Compute hash of spec + schema (excludes project)."""
        hasher = hashlib.sha256()
        hasher.update(spec.feature_spec_version.encode())
        hasher.update(json.dumps(schema, sort_keys=True).encode())
        return truncate_hash(hasher.hexdigest())

    @cached_property
    def feature_definition_version(self) -> str:
        """Hash of spec + schema (excludes project)."""
        return self._compute_definition_version(self.spec, self.feature_schema)

    @property
    def key(self) -> FeatureKey:
        """Get the feature key from the spec."""
        return self.spec.key

    @property
    def table_name(self) -> str:
        """Get SQL-like table name for this feature."""
        return self.spec.table_name()

    @property
    def id_columns(self) -> tuple[str, ...]:
        """Get ID columns from the spec."""
        return self.spec.id_columns

    @cached_property
    def columns(self) -> Sequence[str]:
        """Get column names from the feature schema."""
        return list(self.feature_schema.get("properties", {}).keys())
