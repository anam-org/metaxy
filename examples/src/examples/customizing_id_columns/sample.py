from typing import NamedTuple

import pydantic

from metaxy import BaseFeature, BaseFeatureSpec, FeatureKey


class ProductColumns(NamedTuple):
    store_id: str
    item_id: str


class ProductFeatureSpec(BaseFeatureSpec[ProductColumns]):
    """A testing concrete implementation of BaseBaseFeatureSpec that has a `sample_uid` ID column."""

    id_columns: ProductColumns = pydantic.Field(
        default=ProductColumns("store_id", "item_id"),
        description="List of columns that uniquely identify a row. They will be used by Metaxy in joins.",
    )


class ProductFeature(
    BaseFeature[ProductColumns],
    spec=ProductFeatureSpec(
        key=FeatureKey(["stats", "by-store", "daily-purchases"]), deps=None
    ),
):
    store_id: str | None = None
    item_id: str | None = None


# Access id_columns as a property, not a method
ProductFeature.spec().id_columns
