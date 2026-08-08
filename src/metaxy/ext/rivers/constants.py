"""
Ported 1:1 from metaxy/ext/dagster/constants.py where the concept applies
to Rivers. Dagster-only keys (column schema/lineage) are dropped.
"""

RIVERS_METAXY_FEATURE_METADATA_KEY = "metaxy/feature"
RIVERS_METAXY_PARTITION_KEY = "partition_by"
RIVERS_METAXY_PARTITION_METADATA_KEY = "metaxy/partition"
RIVERS_METAXY_PROJECT_TAG_KEY = "metaxy/project"
RIVERS_METAXY_INFO_METADATA_KEY = "metaxy/info"
