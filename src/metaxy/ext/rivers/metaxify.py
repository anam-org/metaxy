"""
Adapted from metaxy/ext/dagster/metaxify.py.

NOW FULLY CONFIRMED against python\\rivers\\_core\\assets\\__init__.pyi's
`Asset.__new__` overload for the `@Asset(...)` (parenthesized) form:

    def __new__(cls, *, name: str | None = ..., tags: ..., kinds: ...,
                 group: ..., code_version: str | None = ...,
                 io_handler: IOHandler | str | None = ...,
                 metadata: dict[str, str] | None = ...,
                 partitions_def: ..., deps: list["DepDef"] | None = ...,
                 hooks: ..., automation_condition: ...,
                 backfill_strategy: ..., pool: ..., pool_slots: ...,
                 retry: ..., compute: ...) -> Callable[[Callable], "SingleAsset"]

Two things this resolves vs. the earlier draft:

  1. `name=` IS a real kwarg. This means we can pin the registered asset
     name directly to the Metaxy feature's table_name -- no more in-memory
     feature->asset-name registry, no more import-order dependency for
     deps resolution. `AssetDef.input(upstream_key.table_name)` now works
     deterministically as long as every metaxify-decorated asset for that
     feature also sets `name=` to the same table_name (which it does,
     always, below).

  2. `description=` is CONFIRMED ABSENT from the overload -- removed
     entirely rather than guessed at.

STILL NOT PORTED (no confirmed Rivers equivalent -- Dagster-only concepts,
same as before): column schema injection, column lineage injection,
"kinds"/"tags" auto-injection from the feature graph, key_prefix-style
remapping.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TypeVar

import metaxy as mx
from rivers import Asset, AssetDef

from .constants import RIVERS_METAXY_FEATURE_METADATA_KEY, RIVERS_METAXY_INFO_METADATA_KEY
from .io_handler import MetaxyIOHandler
from .utils import build_feature_info_metadata

_F = TypeVar("_F", bound=Callable[..., Any])


def metaxify(
    feature: mx.CoercibleToFeatureKey,
    *,
    inject_code_version: bool = True,
    io_handler_name: str | None = None,
    **asset_kwargs: Any,
) -> Callable[[_F], Any]:
    """Decorator that wires Metaxy feature info into a Rivers `@Asset`.

    Unlike Dagster's `@metaxify` (which reads the feature key from
    `metadata={"metaxy/feature": ...}` on an already-decorated asset),
    this takes the feature explicitly, since wiring happens at decoration
    time here:

        @metaxify("my/feature")
        def my_asset(context):
            ...

    The asset is always registered under `name=<feature's table_name>`
    (overriding any `name=` you pass in `asset_kwargs`), so that
    dependency resolution between metaxify-decorated assets is
    deterministic. Any other `asset_kwargs` (e.g. `partitions_def=`,
    `retry=`, `pool=`, `tags=`) are passed straight through to `Asset(...)`.
    """

    def decorator(fn: _F) -> Any:
        feature_key = mx.coerce_to_feature_key(feature)
        feature_def = mx.get_feature_by_key(feature_key)
        feature_spec = feature_def.spec

        # --- deps: resolve upstream Metaxy features directly by table_name.
        # Deterministic now that every metaxify-decorated asset is
        # registered under name=<table_name> (see docstring).
        deps: list[Any] = [
            AssetDef.input(mx.coerce_to_feature_key(dep.feature).table_name)
            for dep in feature_spec.deps
        ]
        deps.extend(asset_kwargs.pop("deps", []))

        # --- code_version: append metaxy version, same format as Dagster.
        code_version = asset_kwargs.pop("code_version", None)
        if inject_code_version:
            metaxy_code_version = f"metaxy:{feature_spec.code_version}"
            code_version = (
                f"{code_version},{metaxy_code_version}" if code_version else metaxy_code_version
            )

        # --- metadata: dict[str, str] ONLY (confirmed) -- nested info is
        # JSON-encoded, unlike Dagster's dict[str, Any] metadata.
        metadata: dict[str, str] = {
            **{k: str(v) for k, v in asset_kwargs.pop("metadata", {}).items()},
            RIVERS_METAXY_FEATURE_METADATA_KEY: feature_key.to_string(),
            RIVERS_METAXY_INFO_METADATA_KEY: json.dumps(build_feature_info_metadata(feature_key)),
            "metaxy/feature_code_version": feature_spec.code_version,
        }

        asset_kwargs.pop("name", None)  # always overridden, see docstring

        return Asset(
            name=feature_key.table_name,
            io_handler=MetaxyIOHandler(name=io_handler_name),
            deps=deps,
            code_version=code_version,
            metadata=metadata,
            **asset_kwargs,
        )(fn)

    return decorator
