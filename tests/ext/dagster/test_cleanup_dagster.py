"""Basic Dagster cleanup/mutation op tests without propagation."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import metaxy.ext.dagster as mxd
from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store.cleanup import DeletionResult, MutationResult


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
    store.delete_metadata.return_value = DeletionResult(
        feature_key=FeatureKey(["logs"]),
        rows_affected=2,
        timestamp=datetime.now(timezone.utc),
        error=None,
    )

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

    result = mxd.delete_metadata(ctx)

    assert result.rows_affected == 2
    store.delete_metadata.assert_called_once()
    args, kwargs = store.delete_metadata.call_args
    # Expect filter expr applied; verify it targets timestamp column
    filter_expr = kwargs["filter"]
    assert "metaxy_created_at" in str(filter_expr)


def test_mutate_metadata_op_passes_updates(feature_cls):
    store = MagicMock()
    store.mutate_metadata.return_value = MutationResult(
        feature_key=FeatureKey(["logs"]),
        rows_affected=1,
        updates={"status": "archived"},
        timestamp=datetime.now(timezone.utc),
        error=None,
    )

    from dagster import build_op_context

    ctx = build_op_context(
        resources={"store": store},
        op_config={
            "feature_key": ["logs"],
            "filter_expr": "nw.col('level') == 'warn'",
            "updates": {"status": "archived"},
        },
    )

    result = mxd.mutate_metadata(ctx)

    assert result.rows_affected == 1
    store.mutate_metadata.assert_called_once()
    args, kwargs = store.mutate_metadata.call_args
    assert args[0] == FeatureKey(["logs"])
    assert kwargs["updates"] == {"status": "archived"}
