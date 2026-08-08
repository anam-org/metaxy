"""
Ported 1:1 from metaxy/ext/dagster/utils.py::build_feature_info_metadata.

Touches only metaxy-core APIs -- nothing orchestrator-specific -- so it
needs no adaptation for Rivers.
"""

from __future__ import annotations

from typing import Any

import metaxy as mx


def build_feature_info_metadata(feature: mx.CoercibleToFeatureKey) -> dict[str, Any]:
    """Build feature info metadata dict.

    NOTE: unlike Dagster (dict[str, Any] metadata), Rivers asset metadata
    is dict[str, str] only. Callers (io_handler.py, metaxify.py) JSON-encode
    this return value before attaching it as asset/output metadata.
    """
    feature_key = mx.coerce_to_feature_key(feature)
    feature_def = mx.get_feature_by_key(feature_key)
    feature_version = mx.current_graph().get_feature_version(feature_key)

    return {
        "feature": {
            "project": feature_def.project,
            "spec": feature_def.spec.model_dump(mode="json"),
            "version": feature_version,
            "type": feature_def.feature_class_path,
        },
        "metaxy": {
            "version": mx.__version__,
            "plugins": mx.MetaxyConfig.get().plugins,
        },
    }
