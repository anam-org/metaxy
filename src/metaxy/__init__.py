from collections.abc import Sequence
from pathlib import Path

import narwhals as nw

from metaxy._decorators import public
from metaxy._version import __version__
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.entrypoints import (
    load_features,
    load_module_entrypoint,
    load_package_entrypoints,
)
from metaxy.metadata_store import AccessMode, MetadataStore
from metaxy.migrations import (
    BaseOperation,
    DataVersionReconciliation,
    DiffMigration,
    FullGraphMigration,
    MetadataBackfill,
    Migration,
    MigrationExecutor,
    MigrationResult,
    SystemTableStorage,
    detect_diff_migration,
)
from metaxy.models.feature import (
    BaseFeature,
    FeatureGraph,
    current_graph,
    graph,
)
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.feature_spec import (
    FeatureDep,
    FeatureSpec,
    FeatureSpecWithIDColumns,
    IDColumns,
)
from metaxy.models.field import (
    FieldDep,
    FieldSpec,
    SpecialFieldDep,
)
from metaxy.models.fields_mapping import (
    AllFieldsMapping,
    DefaultFieldsMapping,
    FieldsMapping,
    FieldsMappingType,
)
from metaxy.models.lineage import LineageRelationship
from metaxy.models.types import (
    CoercibleToFeatureKey,
    CoercibleToFieldKey,
    FeatureDepMetadata,
    FeatureKey,
    FieldKey,
    ValidatedFeatureKey,
    ValidatedFeatureKeyAdapter,
    ValidatedFeatureKeySequence,
    ValidatedFeatureKeySequenceAdapter,
    ValidatedFieldKey,
    ValidatedFieldKeyAdapter,
    ValidatedFieldKeySequence,
    ValidatedFieldKeySequenceAdapter,
)
from metaxy.utils import BatchedMetadataWriter
from metaxy.utils.exceptions import MetaxyMissingFeatureDependency
from metaxy.versioning.types import HashAlgorithm


@public
def get_feature_by_key(key: CoercibleToFeatureKey) -> FeatureDefinition:
    """Get a FeatureDefinition by its key from the current graph.

    Args:
        key: Feature key to look up (can be FeatureKey, list of strings, slash-separated string, etc.)

    Returns:
        FeatureDefinition for the feature

    Raises:
        KeyError: If no feature with the given key is registered
    """
    return current_graph().get_feature_definition(key)


@public
def coerce_to_feature_key(value: CoercibleToFeatureKey) -> FeatureKey:
    """Coerce a value to a [`FeatureKey`][metaxy.FeatureKey].

    Accepts:

    - slashed `str`: `"a/b/c"`
    - `Sequence[str]`: `["a", "b", "c"]`
    - `FeatureKey`: pass through
    - `type[BaseFeature]`: extracts `.spec().key`

    Args:
        value: Value to coerce to `FeatureKey`

    Returns:
        The coerced `FeatureKey`

    Raises:
        ValidationError: If the value cannot be coerced to a `FeatureKey`
    """
    return ValidatedFeatureKeyAdapter.validate_python(value)


@public
def init_metaxy(
    config: MetaxyConfig | Path | str | None = None,
    search_parents: bool = True,
) -> MetaxyConfig:
    """Main user-facing initialization function for Metaxy. It loads feature definitions and Metaxy configuration.

    Features are [discovered](../../guide/learn/feature-discovery.md) from installed Python packages metadata.

    Args:
        config: Metaxy configuration to use for initialization. Will be auto-discovered if not provided.

            !!! tip
                `METAXY_CONFIG` environment variable can be used to set the config file path.

        search_parents: Whether to search parent directories for configuration files during config auto-discovery.

    Returns:
        The activated Metaxy configuration.
    """
    if isinstance(config, MetaxyConfig):
        MetaxyConfig.set(config)
    else:
        config = MetaxyConfig.load(
            config_file=config,
            search_parents=search_parents,
        )
    load_features(config.entrypoints)
    return config


@public
def load_feature_definitions(
    store: MetadataStore,
    *,
    projects: str | list[str] | None = None,
    filters: Sequence[nw.Expr] | None = None,
) -> list[FeatureDefinition]:
    """Load feature definitions from a metadata store into the active graph.

    Loads [`FeatureDefinition`][metaxy.FeatureDefinition] objects from the metadata store
    without requiring the original Python feature classes to be importable.

    This enables depending on features from external projects or historical snapshots where
    the source code is not available at runtime.

    Args:
        store: Metadata store to load from. Will be opened automatically if not already open.
        projects: Project(s) to load features from. If not provided, loads from all projects.
        filters: Narwhals expressions to filter features. Applied after snapshot selection
            and deduplication, ensuring the latest version of each feature is loaded
            before filtering. Available columns: `project`, `feature_key`,
            `metaxy_feature_version`, `metaxy_definition_version`, `recorded_at`,
            `feature_spec`, `feature_schema`, `feature_class_path`,
            `metaxy_snapshot_version`, `tags`, `deleted_at`.

    Returns:
        List of loaded FeatureDefinition objects. Empty if no features found.

    Note:
        Features loaded this way have their `feature_schema` preserved from when they
        were originally saved, but the Pydantic model class is not available. Operations
        requiring the actual Python class (like schema extraction or model validation)
        will not work.

    Example:
        ```python
        import metaxy as mx
        import narwhals as nw

        # Load all features from latest snapshots into active graph
        definitions = mx.load_feature_definitions(store)

        # Load from specific project into a custom graph
        with mx.FeatureGraph().use():
            definitions = mx.load_feature_definitions(store, projects="external-project")

        # Load specific features by key
        definitions = mx.load_feature_definitions(
            store,
            filters=[nw.col("feature_key").is_in(["my/feature", "other/feature"])],
        )
        ```
    """
    from contextlib import nullcontext

    # Use nullcontext if store is already open, otherwise open it
    cm = nullcontext(store) if store._is_open else store
    with cm:
        storage = SystemTableStorage(store)
        return storage.load_feature_definitions(projects=projects, filters=filters)


__all__ = [
    "BatchedMetadataWriter",
    "BaseFeature",
    "FeatureDefinition",
    "FeatureGraph",
    "graph",
    "FeatureSpec",
    "FeatureDep",
    "FeatureDepMetadata",
    "FeatureSpec",
    "FeatureSpecWithIDColumns",
    "AllFieldsMapping",
    "DefaultFieldsMapping",
    "FieldsMapping",
    "FieldsMappingType",
    "FieldDep",
    "FieldSpec",
    "SpecialFieldDep",
    "FeatureKey",
    "FieldKey",
    "CoercibleToFeatureKey",
    "CoercibleToFieldKey",
    "coerce_to_feature_key",
    "get_feature_by_key",
    "ValidatedFeatureKey",
    "ValidatedFieldKey",
    "ValidatedFeatureKeySequence",
    "ValidatedFieldKeySequence",
    "MetadataStore",
    "load_features",
    "load_module_entrypoint",
    "load_package_entrypoints",
    "Migration",
    "DiffMigration",
    "FullGraphMigration",
    "MigrationResult",
    "MigrationExecutor",
    "SystemTableStorage",
    "BaseOperation",
    "DataVersionReconciliation",
    "MetadataBackfill",
    "detect_diff_migration",
    "MetaxyConfig",
    "StoreConfig",
    "init_metaxy",
    "load_feature_definitions",
    "IDColumns",
    "HashAlgorithm",
    "LineageRelationship",
    "AccessMode",
    "current_graph",
    "MetaxyMissingFeatureDependency",
    "ValidatedFeatureKeyAdapter",
    "ValidatedFieldKeyAdapter",
    "ValidatedFeatureKeySequenceAdapter",
    "ValidatedFieldKeySequenceAdapter",
    "__version__",
]
