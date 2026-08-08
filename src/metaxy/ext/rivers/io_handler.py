"""
Ported from metaxy/ext/dagster/io_manager.py (MetaxyIOManager).

CONFIRMED against python\rivers\io_handlers\base.py and
python\rivers\_core\__init__.pyi:

    class BaseIOHandler(BaseSettings, ABC):
        def handle_output(self, context: OutputContext, obj: Any) -> None: ...
        def load_input(self, context: InputContext) -> Any: ...
    # attached as an INSTANCE: @Asset(io_handler=MetaxyIOHandler(name=...))
    # (not a class, unlike our first draft)

    class OutputContext:
        asset_name: str
        asset_metadata: dict[str, str] | None
        partition: PartitionContext | None
        add_output_metadata(dict[str, str|int|float|bool|None|MetadataValue])
        register_data_version(str)

    class InputContext:
        asset_name: str            # the UPSTREAM asset (this context describes it)
        downstream_asset: str
        asset_metadata: dict[str, str] | None   # the upstream asset's metadata
        partition: PartitionContext | None

    class PartitionContext:
        keys: list[PartitionKey]
        key: PartitionKey          # keys[0]

Because BaseIOHandler is itself pydantic BaseSettings with NO setup()/
teardown() hooks (unlike Resource), this handler resolves its own store
directly rather than depending on a separately-injected MetaxyResource --
there's no confirmed mechanism for Rivers to inject a Resource into an
IOHandler instance.

CONFIRMED (partitions/__init__.pyi):
    class PartitionKey.Single(PartitionKey):
        key: list[str]   # NOTE: a list, even for the "Single" variant --
                          # presumably supports multiple keys per step
                          # (e.g. a backfill range materialized in one run).
    class PartitionKey.Multi(PartitionKey):
        keys: dict[str, list[str]]

REMAINING TODO (documented, not silently guessed):
    - Since `Single.key` is a list, `_build_filters()` below filters with
      `.is_in(...)` when there are multiple keys and `==` for exactly one,
      mirroring Dagster's build_partition_filter_from_input_context
      single-vs-multiple-partition-keys branch. `Multi` (dict of dimension
      -> list[str]) is NOT handled yet -- multi-dimensional partitions
      will raise NotImplementedError rather than silently filtering wrong.
    - `metaxy/partition` (the static multi-asset-to-one-feature dict) has
      to be JSON-encoded since Rivers metadata is dict[str, str] only
      (Dagster's metadata dict allowed arbitrary values). Both the
      writer (metaxify / user code setting this metadata) and this reader
      must agree on that encoding -- documented in metaxify.py too.
    - Unlike Dagster's MetaxyIOManager, this handler does not log the
      "metaxy/materialized_in_run" count or any resolved-store/table_name
      input metadata -- Rivers' OutputContext/InputContext do not expose a
      run id or resolved-store info the way Dagster's contexts do (see
      resources.py's KNOWN LIMITATION note re: no run id at setup time).
"""

from __future__ import annotations

import json
from typing import Any

import narwhals as nw

import metaxy as mx
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from rivers import BaseIOHandler
from rivers._core import InputContext, OutputContext

from .constants import (
    RIVERS_METAXY_FEATURE_METADATA_KEY,
    RIVERS_METAXY_INFO_METADATA_KEY,
    RIVERS_METAXY_PARTITION_KEY,
    RIVERS_METAXY_PARTITION_METADATA_KEY,
)
from .utils import build_feature_info_metadata

MetaxyOutput = Any


def _partition_key_values(context: InputContext | OutputContext) -> list[str] | None:
    """Return the list of partition key string(s) for this context, or None.

    Only handles PartitionKey.Single (a plain list[str]). Raises for
    PartitionKey.Multi since there's no confirmed mapping yet from a
    {dimension: [values]} dict onto a single filter column.
    """
    partition = context.partition
    if partition is None:
        return None
    key = partition.key
    if hasattr(key, "keys") and not hasattr(key, "key"):
        # PartitionKey.Multi
        raise NotImplementedError(
            "Multi-dimensional Rivers partitions are not yet supported by "
            "MetaxyIOHandler -- only PartitionKey.Single is handled."
        )
    return list(getattr(key, "key", []))


class MetaxyIOHandler(BaseIOHandler):
    """IO handler that reads/writes data to a Metaxy MetadataStore,
    mirroring metaxy.ext.dagster.MetaxyIOManager.

    Configured directly (it's pydantic BaseSettings, like MetaxyResource):

        @Asset(io_handler=MetaxyIOHandler(name="dev"), ...)

    Expects `"metaxy/feature"` to be set in the asset's metadata, mapping
    the Rivers asset to a Metaxy feature key.
    """

    name: str | None = None

    def _get_store(self) -> mx.MetadataStore:
        return mx.MetaxyConfig.get().get_store(self.name)

    def _feature_key_from_metadata(self, metadata: dict[str, str] | None) -> mx.FeatureKey:
        metadata = metadata or {}
        raw_key = metadata.get(RIVERS_METAXY_FEATURE_METADATA_KEY)
        if raw_key is None:
            raise ValueError(
                f'Missing `"{RIVERS_METAXY_FEATURE_METADATA_KEY}"` key in asset metadata'
            )
        return mx.ValidatedFeatureKeyAdapter.validate_python(raw_key)

    def _build_filters(
        self, metadata: dict[str, str] | None, context: InputContext | OutputContext
    ) -> list[nw.Expr]:
        metadata = metadata or {}
        filters: list[nw.Expr] = []

        partition_col = metadata.get(RIVERS_METAXY_PARTITION_KEY)
        partition_keys = _partition_key_values(context)
        if partition_col and partition_keys:
            if len(partition_keys) == 1:
                filters.append(nw.col(partition_col) == partition_keys[0])
            else:
                filters.append(nw.col(partition_col).is_in(partition_keys))

        # metaxy/partition: JSON-encoded {col: value} dict (see module docstring)
        raw_metaxy_partition = metadata.get(RIVERS_METAXY_PARTITION_METADATA_KEY)
        if raw_metaxy_partition:
            metaxy_partition = json.loads(raw_metaxy_partition)
            for col, value in metaxy_partition.items():
                if isinstance(value, list):
                    filters.append(nw.col(col).is_in(value))
                else:
                    filters.append(nw.col(col) == value)

        return filters

    # -- IO handler interface --------------------------------------------

    def load_input(self, context: InputContext) -> nw.LazyFrame[Any]:
        store = self._get_store()
        with store:
            # InputContext describes the UPSTREAM asset directly -- no
            # separate "upstream_output" indirection needed, unlike Dagster.
            feature_key = self._feature_key_from_metadata(context.asset_metadata)
            filters = self._build_filters(context.asset_metadata, context)

            lazy_frame, _resolved_store = store.read(
                feature=feature_key,
                filters=filters,
                with_store_info=True,
            )
            return lazy_frame

    def handle_output(self, context: OutputContext, obj: MetaxyOutput) -> None:
        store = self._get_store()
        feature_key = self._feature_key_from_metadata(context.asset_metadata)
        feature = mx.get_feature_by_key(feature_key)

        if obj is not None:
            with store.open("w"):
                store.write(feature=feature, df=obj)

        self._log_output_metadata(context, store, feature_key)

    def _log_output_metadata(
        self, context: OutputContext, store: mx.MetadataStore, feature_key: mx.FeatureKey
    ) -> None:
        with store:
            try:
                mx.get_feature_by_key(feature_key)  # raises FeatureNotFoundError if missing
                context.add_output_metadata(
                    {
                        "metaxy/feature": feature_key.to_string(),
                        # NOTE: metadata values here must be JSON-serializable to
                        # a str|int|float|bool|None|MetadataValue -- build_feature_info_metadata
                        # returns a nested dict, so it's JSON-encoded to a string.
                        RIVERS_METAXY_INFO_METADATA_KEY: json.dumps(
                            build_feature_info_metadata(feature_key)
                        ),
                    }
                )
            except FeatureNotFoundError:
                pass