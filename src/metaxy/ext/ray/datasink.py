from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING

from ray.data import Datasink

import metaxy as mx

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import TaskContext
    from ray.data.block import Block


class MetaxyDatasink(Datasink[None]):
    """A Ray Data Datasink for writing to a Metaxy metadata store.

    !!! example

        ```python
        import metaxy as mx
        import ray

        cfg = mx.init_metaxy()
        dataset = ...  # a ray.data.Dataset

        dataset.write_datasink(
            MetaxyDatasink(
                feature="my/feature",
                store=cfg.get_store(),
                config=cfg,
            )
        )
        ```

    !!! note
        In the future this Datasink will support writing multiple features at once.

    Args:
        feature: Feature to write metadata for.
        store: Metadata store to write to.
        config: Metaxy configuration. Will be auto-discovered by the worker if not provided.

            !!! warning
                Ensure the Ray environment is set up properly when not passing `config` explicitly.
                This can be achieved by setting `METAXY_CONFIG` and other `METAXY_` environment variables.
                The best practice is to pass `config` explicitly to avoid surprises.

        allow_cross_project_writes: Whether to allow writing metadata for features from other projects.
    """

    def __init__(
        self,
        feature: mx.CoercibleToFeatureKey,
        store: mx.MetadataStore,
        config: mx.MetaxyConfig | None = None,
        allow_cross_project_writes: bool = False,
    ):
        self.config = mx.init_metaxy(config)

        self.store = store
        self.config = config
        self.allow_cross_project_writes = allow_cross_project_writes

        self._feature_key = mx.coerce_to_feature_key(feature)

    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> None:
        """Write blocks of metadata to the store."""
        # Initialize metaxy on the worker - config and features are needed for write_metadata
        mx.init_metaxy(self.config)

        for i, block in enumerate(blocks):
            try:
                with (
                    self.store.open("write"),
                    self.store.allow_cross_project_writes() if self.allow_cross_project_writes else nullcontext(),
                ):
                    self.store.write_metadata(self._feature_key, block)
            except Exception:
                logger.exception(
                    f"Failed to write metadata for feature {self._feature_key.to_string()} block {i} of task {ctx.task_idx} ({ctx.op_name})"
                )
