"""Feature definition model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

from pydantic import Field, PrivateAttr

from metaxy._decorators import public
from metaxy._hashing import truncate_hash
from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature


@public
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

    # Runtime-only reference to the feature class. Not serialized. Will be removed in the future once we figure out a good way to extract Pydantic fields from the serialized schema.
    _feature_class: type[BaseFeature] | None = PrivateAttr(default=None)

    @classmethod
    def from_feature_class(cls, feature_cls: type[BaseFeature]) -> FeatureDefinition:
        """Create a FeatureDefinition from a Feature class."""
        spec = feature_cls.spec()
        schema = feature_cls.model_json_schema()
        class_path = f"{feature_cls.__module__}.{feature_cls.__name__}"
        project = feature_cls.metaxy_project()

        definition = cls(
            spec=spec,
            feature_schema=schema,
            feature_class_path=class_path,
            project=project,
        )
        definition._feature_class = feature_cls
        return definition

    def _get_feature_class(self) -> type[BaseFeature]:
        """Return the feature class, using cached reference or importing by path.

        Internal method - will be removed in a future version.
        The Dagster integration currently requires feature classes to be
        importable at runtime (i.e., defined at module level, not inside functions).

        Returns:
            The feature class.

        Raises:
            ImportError: If the class cannot be imported and no cached reference exists.
        """
        # Return cached reference if available
        if self._feature_class is not None:
            return self._feature_class

        import sys

        module_path, class_name = self.feature_class_path.rsplit(".", 1)

        # Try to get from already-loaded modules
        if module_path in sys.modules:
            module = sys.modules[module_path]
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                if isinstance(cls, type):
                    return cls  # type: ignore[return-value]

        # Fall back to importing
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls  # type: ignore[return-value]
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(
                f"Cannot import feature class '{self.feature_class_path}': {e}. "
                f"The Dagster integration currently only supports inferring column schema "
                f"from features that are available as classes at runtime. "
                f"This method will be removed in a future version."
            ) from e

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
