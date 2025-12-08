"""Dagster cleanup op tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import metaxy.ext.dagster as mxd
from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec


@pytest.fixture
def feature_cls():
    class Logs(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["logs"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    return Logs


def test_delete_metadata_op_builds_retention_filter(feature_cls):
    cutoff_days = 30
    store = MagicMock()
    store.delete_metadata.return_value = 2

    from dagster import build_op_context

    ctx = build_op_context(
        resources={"store": store},
        op_config={
            "feature_key": ["logs"],
            "retention_days": cutoff_days,
            "timestamp_column": "metaxy_created_at",
            "hard": True,
        },
    )

    mxd.delete_metadata(ctx)
    store.delete_metadata.assert_called_once()
    args, kwargs = store.delete_metadata.call_args
    filter_expr = kwargs["filters"]
    assert "metaxy_created_at" in str(filter_expr)


def test_soft_delete_metadata_op(feature_cls):
    store = MagicMock()
    store.delete_metadata.return_value = 1

    from dagster import build_op_context

    ctx = build_op_context(
        resources={"store": store},
        op_config={
            "feature_key": ["logs"],
            "filter_expr": "nw.col('level') == 'warn'",
            "hard": False,
        },
    )

    mxd.delete_metadata(ctx)
    store.delete_metadata.assert_called_once()
    args, kwargs = store.delete_metadata.call_args
    assert args[0] == FeatureKey(["logs"])
    assert kwargs["soft"] is True
