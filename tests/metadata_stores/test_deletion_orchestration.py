"""Orchestration-focused deletion tests without graph-wide propagation.

These tests mirror the recommended workflow: orchestrators decide ordering and
invoke per-feature operations explicitly (e.g., via Dagster assets).
"""

from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch

import narwhals as nw
import pytest

from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store import InMemoryMetadataStore


@pytest.fixture
def feature_classes():
    class Root(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["root"]),
            fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
        ),
    ):
        pass

    class Child(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["child"]),
            deps=[],
            fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
        ),
    ):
        pass

    return {"Root": Root, "Child": Child}


def test_delete_does_not_implicitly_propagate(feature_classes):
    """delete_metadata should only touch the requested feature."""
    store = InMemoryMetadataStore()

    called = []

    def mock_delete(feature_key, filter_expr):
        called.append(feature_key)
        return 3

    with patch.object(store, "_delete_metadata_impl", side_effect=mock_delete):
        with store.open("write"):
            result = store.delete_metadata(
                feature_classes["Root"], filter=nw.col("field1") == "x"
            )

    assert called == [FeatureKey(["root"])]  # single feature only
    assert result.rows_affected == 3


def test_asset_style_workflow_calls_delete_once(feature_classes):
    """Simulate the recommended per-feature asset workflow."""
    Update = namedtuple("Update", ["added", "changed", "to_delete"])

    store = MagicMock()
    store.resolve_update.return_value = Update("added_rows", "changed_rows", "to_del")

    def asset_fn(store):
        added, changed, to_delete = store.resolve_update(feature_classes["Root"])
        # materialize(added, changed) would go here
        store.delete_metadata(feature_classes["Root"], filter=to_delete)
        return added, changed, to_delete

    asset_fn(store)

    store.resolve_update.assert_called_once()
    store.delete_metadata.assert_called_once_with(
        feature_classes["Root"], filter="to_del"
    )
