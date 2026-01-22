from pathlib import Path

from metaxy._public import public
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
from metaxy.utils import BatchedMetadataWriter
from metaxy.versioning.types import HashAlgorithm


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


__all__ = [
    "BatchedMetadataWriter",
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
