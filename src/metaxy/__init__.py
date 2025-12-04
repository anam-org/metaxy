from pathlib import Path

from metaxy._version import __version__
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.entrypoints import (
    load_features,
    load_module_entrypoint,
    load_package_entrypoints,
)
from metaxy.metadata_store import AccessMode, InMemoryMetadataStore, MetadataStore
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
    get_feature_by_key,
    graph,
)
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
from metaxy.versioning.types import HashAlgorithm


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


def init_metaxy(
    config_file: Path | None = None, search_parents: bool = True
) -> MetaxyConfig:
    """Main user-facing initialization function for Metaxy. It loads the configuration and features.

    Features are [discovered](../../guide/learn/feature-discovery.md) from installed Python packages metadata.

    Args:
        config_file (Path | None, optional): Path to the configuration file.

            Will be auto-discovered in current or parent directories if not provided.

            !!! tip
                `METAXY_CONFIG` environment variable can be used to set this parameter

        search_parents (bool, optional): Whether to search parent directories for configuration files. Defaults to True.

    Returns:
        MetaxyConfig: The initialized Metaxy configuration.
    """
    cfg = MetaxyConfig.load(
        config_file=config_file,
        search_parents=search_parents,
    )
    load_features(cfg.entrypoints)
    return cfg


__all__ = [
    "BaseFeature",
    "FeatureGraph",
    "graph",
    "FeatureSpec",
    "get_feature_by_key",
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
    "ValidatedFeatureKey",
    "ValidatedFieldKey",
    "ValidatedFeatureKeySequence",
    "ValidatedFieldKeySequence",
    "MetadataStore",
    "InMemoryMetadataStore",
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
    "IDColumns",
    "HashAlgorithm",
    "LineageRelationship",
    "AccessMode",
    "current_graph",
    "ValidatedFeatureKeyAdapter",
    "ValidatedFieldKeyAdapter",
    "ValidatedFeatureKeySequenceAdapter",
    "ValidatedFieldKeySequenceAdapter",
    "__version__",
]
