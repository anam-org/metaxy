"""Feature definition model for internal machinery.

This module provides the FeatureDefinition model, which is the central data structure
that all internal Metaxy operations work with. It decouples internal machinery from
Feature classes, enabling multi-project support and external feature loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature


class FeatureDefinition(FrozenBaseModel):
    """Complete feature definition for internal machinery.

    This is the central data structure that all internal Metaxy operations
    work with. It can be constructed from:

    1. A Feature class (local project) via `from_feature_class()`
    2. Stored metadata (external project) via `from_stored_metadata()`

    The FeatureDefinition contains all information needed to work with a feature
    without requiring the Feature class to be importable. This enables:

    - Multi-project workflows where features from different projects coexist
    - External feature loading from MetadataStore
    - Decoupling internal machinery from Feature class imports

    Attributes:
        spec: The complete feature specification (key, deps, fields, etc.)
        feature_schema: Pydantic JSON schema dict for the feature model
        project_name: Project name for metadata isolation
        feature_version: Hash of feature specification (dependencies + fields + code_versions)
        feature_code_version: Hash of this feature's field code_versions only (no dependencies)
        feature_definition_version: Hash of complete definition including Pydantic schema
        feature_class_path: Import path for potential future reconstruction
    """

    spec: FeatureSpec = Field(
        description="Complete feature specification including key, deps, fields, etc."
    )
    feature_schema: dict[str, Any] = Field(
        description="Pydantic JSON schema dict for the feature model"
    )
    project_name: str = Field(description="Project name for metadata isolation")
    feature_version: str = Field(
        description="Hash of feature specification (dependencies + fields + code_versions)"
    )
    feature_code_version: str = Field(
        description="Hash of this feature's field code_versions only (no dependencies)"
    )
    feature_definition_version: str = Field(
        description="Hash of complete definition including Pydantic schema"
    )
    feature_class_path: str = Field(
        description="Import path for potential future reconstruction (e.g., 'myapp.features.VideoFeature')"
    )

    @classmethod
    def from_feature_class(cls, feature_cls: type[BaseFeature]) -> FeatureDefinition:
        """Create a FeatureDefinition from a Feature class.

        This is the primary way to create a FeatureDefinition for features
        defined in the local project.

        Args:
            feature_cls: The Feature class to create a definition from.
                Must be a subclass of BaseFeature with a valid spec.

        Returns:
            A new FeatureDefinition instance containing all feature metadata.

        Example:
            ```python
            class MyFeature(BaseFeature, spec=FeatureSpec(...)):
                value: str

            definition = FeatureDefinition.from_feature_class(MyFeature)
            print(definition.key)  # FeatureKey(['my', 'feature'])
            ```
        """
        return cls(
            spec=feature_cls.spec(),
            feature_schema=feature_cls.model_json_schema(),
            project_name=feature_cls.project,
            feature_version=feature_cls.feature_version(),
            feature_code_version=feature_cls.spec().code_version,
            feature_definition_version=feature_cls.full_definition_version(),
            feature_class_path=f"{feature_cls.__module__}.{feature_cls.__name__}",
        )

    @classmethod
    def from_stored_metadata(cls, data: dict[str, Any]) -> FeatureDefinition:
        """Create a FeatureDefinition from stored metadata.

        This is used to reconstruct FeatureDefinitions from data stored in
        the metadata store, enabling external feature loading without requiring
        the Feature class to be importable.

        Args:
            data: Dictionary containing feature metadata as stored in the
                metadata store. Expected keys:
                - feature_spec: dict (FeatureSpec as JSON)
                - feature_schema: dict (Pydantic JSON schema)
                - project: str (project name)
                - metaxy_feature_version: str (feature version hash)
                - metaxy_feature_spec_version: str (spec version, used for code_version)
                - metaxy_full_definition_version: str (full definition hash)
                - feature_class_path: str (import path)

        Returns:
            A new FeatureDefinition instance reconstructed from stored data.

        Example:
            ```python
            # From metadata store query result
            stored_data = {
                "feature_spec": {...},
                "feature_schema": {...},
                "project": "myproject",
                "metaxy_feature_version": "abc123",
                "metaxy_feature_spec_version": "def456",
                "metaxy_full_definition_version": "ghi789",
                "feature_class_path": "myapp.features.MyFeature",
            }
            definition = FeatureDefinition.from_stored_metadata(stored_data)
            ```
        """
        # Parse FeatureSpec from stored dict
        spec = FeatureSpec.model_validate(data["feature_spec"])

        return cls(
            spec=spec,
            feature_schema=data["feature_schema"],
            project_name=data["project"],
            feature_version=data["metaxy_feature_version"],
            feature_code_version=spec.code_version,
            feature_definition_version=data["metaxy_full_definition_version"],
            feature_class_path=data["feature_class_path"],
        )

    @property
    def key(self) -> FeatureKey:
        """Get the feature key.

        Returns:
            The feature key from the spec.
        """
        return self.spec.key

    @property
    def table_name(self) -> str:
        """Get SQL-like table name for this feature.

        Returns:
            Table name string (e.g., "my_namespace__my_feature")
        """
        return self.spec.table_name()

    @property
    def id_columns(self) -> tuple[str, ...]:
        """Get the ID columns for this feature.

        Returns:
            Tuple of column names that uniquely identify a sample.
        """
        return self.spec.id_columns

    @property
    def columns(self) -> list[str]:
        """Get the list of column names from the Pydantic schema.

        Returns all field names defined on the feature, including both
        user-defined fields and metaxy metadata fields.

        Returns:
            List of column/field names.

        Example:
            ```python
            definition.columns
            # ['sample_id', 'value', 'metaxy_provenance', 'metaxy_feature_version', ...]
            ```
        """
        properties = self.feature_schema.get("properties", {})
        return list(properties.keys())

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary format for storage in MetadataStore.

        This format is used for storing feature definitions in the database
        and can be reconstructed via `from_stored_metadata()`. The keys match
        the column names used in the feature_versions system table.

        Returns:
            Dictionary containing all feature metadata in storage format.
        """
        return {
            "feature_spec": self.spec.model_dump(mode="json"),
            "feature_schema": self.feature_schema,
            "metaxy_feature_version": self.feature_version,
            "metaxy_feature_spec_version": self.spec.feature_spec_version,
            "metaxy_full_definition_version": self.feature_definition_version,
            "feature_class_path": self.feature_class_path,
            "project": self.project_name,
        }
