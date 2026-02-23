import dagster as dg
from typing_extensions import assert_type

import metaxy.ext.dagster as mxd


def test_metaxify_preserves_asset_type_without_parentheses() -> None:
    @mxd.metaxify
    @dg.asset
    def raw_numbers() -> None: ...

    assert_type(raw_numbers, dg.AssetsDefinition)

    @mxd.metaxify
    @dg.asset(deps=[raw_numbers])
    def processed_numbers() -> None: ...

    assert_type(processed_numbers, dg.AssetsDefinition)


def test_metaxify_preserves_asset_type_with_parentheses() -> None:
    @mxd.metaxify()
    @dg.asset
    def source_asset() -> None: ...

    assert_type(source_asset, dg.AssetsDefinition)

    @mxd.metaxify()
    @dg.asset(deps=[source_asset])
    def downstream_asset() -> None: ...

    assert_type(downstream_asset, dg.AssetsDefinition)
