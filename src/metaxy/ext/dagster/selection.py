"""Asset selection helpers for Metaxy assets."""

from __future__ import annotations

import dagster as dg

import metaxy as mx
from metaxy._decorators import public
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PROJECT_TAG_KEY,
)


@public
def select_metaxy_assets(
    *,
    project: str | None = None,
    feature: mx.CoercibleToFeatureKey | None = None,
) -> dg.AssetSelection:
    """Select Metaxy assets by project and/or feature.

    This helper creates an `AssetSelection` that filters assets tagged by `@metaxify`.

    Args:
        project: Filter by project name. If None, uses `MetaxyConfig.get().project`.
        feature: Filter by specific feature key. If provided, further narrows the selection.

    Returns:
        An `AssetSelection` that can be used with `dg.define_asset_job`,
        `dg.materialize`, or `AssetSelection` operations like `|` and `&`.

    Example: Select all Metaxy assets in current project
        ```python
        import metaxy.ext.dagster as mxd

        all_metaxy = mxd.select_metaxy_assets()
        ```

    Example: Select assets for a specific project
        ```python
        prod_assets = mxd.select_metaxy_assets(project="production")
        ```

    Example: Select a specific feature's assets
        ```python
        feature_assets = mxd.select_metaxy_assets(feature="my/feature/key")
        ```

    Example: Use with asset jobs
        ```python
        metaxy_job = dg.define_asset_job(
            name="materialize_metaxy",
            selection=mxd.select_metaxy_assets(),
        )
        ```

    Example: Combine with other selections
        ```python
        # All metaxy assets plus some other assets
        combined = mxd.select_metaxy_assets() | dg.AssetSelection.keys("other_asset")

        # Metaxy assets that are also in a specific group
        filtered = mxd.select_metaxy_assets() & dg.AssetSelection.groups("my_group")
        ```
    """
    resolved_project = project if project is not None else mx.MetaxyConfig.get().project

    selection = dg.AssetSelection.tag(DAGSTER_METAXY_PROJECT_TAG_KEY, resolved_project)

    if feature is not None:
        feature_key = mx.coerce_to_feature_key(feature)
        selection = selection & dg.AssetSelection.tag(DAGSTER_METAXY_FEATURE_METADATA_KEY, str(feature_key))

    return selection
