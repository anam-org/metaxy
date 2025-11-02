from pathlib import Path

from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.entrypoints import (
    load_features,
    load_module_entrypoint,
    load_package_entrypoints,
)
from metaxy.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
)
from metaxy.migrations import (
    BaseOperation,
    CustomMigration,
    DataVersionReconciliation,
    DiffMigration,
    FullGraphMigration,
    MetadataBackfill,
    Migration,
    MigrationExecutor,
    MigrationResult,
    SystemTableStorage,
    detect_migration,
)
from metaxy.models.feature import (
    BaseFeature,
    Feature,
    FeatureGraph,
    TestingFeature,
    get_feature_by_key,
    graph,
)
from metaxy.models.feature_spec import (
    BaseFeatureSpec,
    BaseFeatureSpecWithIDColumns,
    FeatureDep,
    FeatureSpec,
    IDColumns,
    IDColumnsT,
    TestingFeatureSpec,
)
from metaxy.models.field import FieldDep, FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureDepMetadata, FeatureKey, FieldKey


def init_metaxy(
    config_file: Path | None = None, search_parents: bool = True
) -> MetaxyConfig:
    """Main user-facing initialization function for Metaxy. It loads the configuration and features.

    Features are [discovered](../../learn/feature-discovery.md) from installed Python packages metadata.

    Args:
        config_file (Path | None, optional): Path to the configuration file. Defaults to None.
        search_parents (bool, optional): Whether to search parent directories for configuration files. Defaults to True.

    Returns:
        MetaxyConfig: The initialized Metaxy configuration.
    """
    cfg = MetaxyConfig.load(
        config_file=config_file,
        search_parents=search_parents,
    )
    load_features(cfg.entrypoints)
    MetaxyConfig.set(cfg)
    return cfg


__all__ = [
    "BaseFeature",
    "FeatureGraph",
    "Feature",
    "TestingFeature",
    "graph",
    "FeatureSpec",
    "TestingFeatureSpec",
    "get_feature_by_key",
    "FeatureDep",
    "FeatureDepMetadata",
    "BaseFeatureSpec",
    "BaseFeatureSpecWithIDColumns",
    "FieldDep",
    "FieldSpec",
    "SpecialFieldDep",
    "FeatureKey",
    "FieldKey",
    "MetadataStore",
    "InMemoryMetadataStore",
    "load_features",
    "load_module_entrypoint",
    "load_package_entrypoints",
    "Migration",
    "DiffMigration",
    "FullGraphMigration",
    "CustomMigration",
    "MigrationResult",
    "MigrationExecutor",
    "SystemTableStorage",
    "BaseOperation",
    "DataVersionReconciliation",
    "MetadataBackfill",
    "detect_migration",
    "MetaxyConfig",
    "StoreConfig",
    "init_metaxy",
    "IDColumns",
    "IDColumnsT",
    "HashAlgorithm",
]
