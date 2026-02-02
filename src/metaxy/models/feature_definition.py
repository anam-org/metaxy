"""Feature definition model."""

from __future__ import annotations

import hashlib
import inspect
import json
import warnings
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, Json, PrivateAttr, TypeAdapter, field_serializer, field_validator

from metaxy._decorators import public
from metaxy._hashing import truncate_hash
from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.types import CoercibleToFieldKey, FeatureKey, ValidatedFieldKeyAdapter

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
    feature_schema: Json[dict[str, Any]] = Field(description="Pydantic JSON schema dict")
    feature_class_path: str | None = Field(default=None, description="Python import path")
    project: str = Field(description="The metaxy project this feature belongs to", min_length=1)

    # Runtime-only reference to the feature class. Not serialized. Will be removed in the future once we figure out a good way to extract Pydantic fields from the serialized schema.
    _feature_class: type[BaseFeature] | None = PrivateAttr(default=None)

    @field_validator("feature_schema", mode="before")
    @staticmethod
    def _convert_dict_to_json(value: dict[str, Any] | str) -> str:
        """Convert dict to JSON string so Json[...] type can parse it."""
        if isinstance(value, dict):
            return TypeAdapter(dict[str, Any]).dump_json(value, exclude_none=True).decode()
        return value

    @field_serializer("feature_schema", when_used="json")
    @staticmethod
    def _serialize_feature_schema(value: dict[str, Any]) -> str:
        """Serialize feature_schema as JSON string, excluding None values."""
        return TypeAdapter(dict[str, Any]).dump_json(value, exclude_none=True).decode()

    _is_external: bool = PrivateAttr(default=False)
    _provenance_by_field: dict[str, str] | None = PrivateAttr(default=None)
    _on_version_mismatch: Literal["warn", "error"] = PrivateAttr(default="warn")
    _source: str | None = PrivateAttr(default=None)

    @classmethod
    def from_feature_class(cls, feature_cls: type[BaseFeature]) -> FeatureDefinition:
        """Create a FeatureDefinition from a Feature class."""
        spec = feature_cls.spec()

        # Inject class docstring as description if not already set
        if spec.description is None and feature_cls.__doc__:
            spec = spec.model_copy(update={"description": feature_cls.__doc__.strip()})

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

    @classmethod
    def from_stored_data(
        cls,
        feature_spec: dict[str, Any] | str,
        feature_schema: dict[str, Any] | str,
        feature_class_path: str,
        project: str,
        source: str | None = None,
    ) -> FeatureDefinition:
        """Create a FeatureDefinition from stored data.

        Handles JSON string or dict inputs for spec and schema fields.

        Args:
            feature_spec: Feature specification as dict or JSON string.
            feature_schema: Pydantic JSON schema as dict or JSON string.
            feature_class_path: Python import path of the feature class.
            project: The metaxy project name.
            source: Human-readable string describing where this definition came from.

        Returns:
            A new FeatureDefinition instance.
        """
        import json

        if isinstance(feature_spec, str):
            feature_spec = json.loads(feature_spec)
        if isinstance(feature_schema, str):
            feature_schema = json.loads(feature_schema)

        spec = FeatureSpec.model_validate(feature_spec)
        definition = cls(
            spec=spec,
            feature_schema=feature_schema,
            feature_class_path=feature_class_path,
            project=project,
        )
        definition._source = source
        return definition

    @staticmethod
    def _capture_call_site() -> str:
        """Capture the call site location for external feature definitions.

        Returns a string like "mymodule.submodule:MyClass.method" or
        "mymodule.submodule:top_level_function" or "mymodule.submodule" for
        module-level calls.
        """
        # Walk up the stack to find the first frame outside this module
        for frame_info in inspect.stack():
            if frame_info.filename != __file__:
                module = inspect.getmodule(frame_info.frame)
                module_name = module.__name__ if module else frame_info.filename
                func_name = frame_info.function

                # Try to get class context if we're in a method
                local_vars = frame_info.frame.f_locals
                if "self" in local_vars:
                    class_name = type(local_vars["self"]).__name__
                    return f"{module_name}:{class_name}.{func_name}"
                elif "cls" in local_vars:
                    cls_obj = local_vars["cls"]
                    if isinstance(cls_obj, type):
                        return f"{module_name}:{cls_obj.__name__}.{func_name}"

                # Module-level or function-level call
                if func_name == "<module>":
                    return module_name
                return f"{module_name}:{func_name}"

        return "?"

    @classmethod
    def external(
        cls,
        *,
        spec: FeatureSpec,
        project: str,
        feature_schema: dict[str, Any] | None = None,
        provenance_by_field: dict[CoercibleToFieldKey, str] | None = None,
        on_version_mismatch: Literal["warn", "error"] = "warn",
        source: str | None = None,
    ) -> FeatureDefinition:
        """Create an external FeatureDefinition without a Feature class.

        External features are definitions loaded from another project or system
        that don't have corresponding Python Feature classes in the current codebase.

        Args:
            spec: The feature specification.
            project: The metaxy project this feature belongs to.
            feature_schema: Pydantic JSON schema dict describing the feature's fields.
                Typically doesn't have to be provided, unless some user code attempts
                to use it before the real feature definition is loaded from the metadata store.
                This argument is experimental and may be changed in the future.
            provenance_by_field: Optional manually-specified field provenance map.
                Use this argument to avoid providing too many upstream external features.
                Make sure to provide the actual values from the real external feature.
            on_version_mismatch: How to handle a version mismatch if the actual feature loaded from the
                metadata store has a different version than the version specified in the corresponding external feature.
            source: Human-readable string describing where this definition came from.
                If not provided, captures the call site location automatically.

        Returns:
            A new FeatureDefinition marked as external.
        """
        normalized_provenance: dict[str, str] | None = None
        if provenance_by_field is not None:
            normalized_provenance = {
                ValidatedFieldKeyAdapter.validate_python(k).to_string(): v for k, v in provenance_by_field.items()
            }

        # Capture call site location if source is not provided
        if source is None:
            source = cls._capture_call_site()

        definition = cls(
            spec=spec,
            feature_schema=feature_schema or {},
            feature_class_path=None,
            project=project,
        )
        definition._is_external = True
        definition._provenance_by_field = normalized_provenance
        definition._on_version_mismatch = on_version_mismatch
        definition._source = source
        return definition

    def _get_feature_class(self) -> type[BaseFeature]:
        """Return the feature class, using cached reference or importing by path.

        The Dagster integration requires feature classes to be importable at runtime
        (i.e., defined at module level, not inside functions).

        Returns:
            The feature class.

        Raises:
            ImportError: If the class cannot be imported and no cached reference exists.
        """
        # Return cached reference if available
        if self._feature_class is not None:
            return self._feature_class

        if self.feature_class_path is None:
            raise ImportError(
                f"Cannot get feature class for external feature '{self.key}'. "
                "External features do not have an associated Python class."
            )

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
                f"The Dagster integration requires feature classes to be importable at runtime."
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

    @property
    def is_external(self) -> bool:
        """Check if this is an external feature definition."""
        return self._is_external

    @property
    def provenance_by_field_override(self) -> dict[str, str]:
        """The manually-specified field provenance map.

        Raises:
            RuntimeError: If no provenance override was set.
        """
        if self._provenance_by_field is None:
            from metaxy.utils.exceptions import MetaxyInvariantViolationError

            raise MetaxyInvariantViolationError(
                f"No provenance override set for feature '{self.key}'. "
                "Check has_provenance_override before accessing this property."
            )
        return self._provenance_by_field

    @property
    def has_provenance_override(self) -> bool:
        """True if this external feature has a provenance override."""
        if not self.is_external:
            from metaxy.utils.exceptions import MetaxyInvariantViolationError

            raise MetaxyInvariantViolationError(
                f"Feature '{self.key}' is not an external feature. "
                "Only external features can have provenance overrides."
            )
        return self._provenance_by_field is not None

    @property
    def on_version_mismatch(self) -> Literal["warn", "error"]:
        """What to do when actual feature version differs from expected."""
        return self._on_version_mismatch

    @property
    def source(self) -> str:
        """Human-readable string describing where this definition came from."""
        return self._source or self.feature_class_path or "?"

    def check_version_mismatch(
        self,
        *,
        expected_version: str,
        actual_version: str,
        expected_version_by_field: dict[str, str],
        actual_version_by_field: dict[str, str],
    ) -> None:
        """Check if the actual feature version matches expected version.

        Called by load_feature_definitions after loading external features from
        the metadata store, comparing provenance-carrying feature versions.

        Args:
            expected_version: The feature version before loading (from graph).
            actual_version: The feature version after loading (from graph).
            expected_version_by_field: Field-level versions before loading.
            actual_version_by_field: Field-level versions after loading.

        Raises:
            ValueError: If versions mismatch and on_version_mismatch is "error".
        """
        if not self.is_external:
            return

        if expected_version == actual_version:
            return

        # Find which fields differ
        mismatched_fields = []
        all_fields = set(expected_version_by_field.keys()) | set(actual_version_by_field.keys())
        for field in sorted(all_fields):
            expected_field_ver = expected_version_by_field.get(field, "<missing>")
            actual_field_ver = actual_version_by_field.get(field, "<missing>")
            if expected_field_ver != actual_field_ver:
                mismatched_fields.append(f"  - {field}: expected '{expected_field_ver}', got '{actual_field_ver}'")

        field_details = "\n".join(mismatched_fields) if mismatched_fields else "  (no field-level details available)"

        message = (
            f"Version mismatch for external feature '{self.key}': "
            f"expected feature version '{expected_version}', got '{actual_version}'.\n"
            f"Field-level mismatches:\n{field_details}\n"
            f"The external feature definition may be out of sync with the metadata store."
        )

        if self._on_version_mismatch == "error":
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)
