from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ray.data import Datasink
from ray.data.block import BlockAccessor
from ray.data.datasource.datasink import WriteResult

import metaxy as mx

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import TaskContext
    from ray.data.block import Block


@dataclass
class _WriteTaskResult:
    """Result of a single write task (internal)."""

    rows_written: int
    rows_failed: int


@dataclass
class MetaxyWriteResult:
    """Result of a MetaxyDatasink write operation."""

    rows_written: int
    rows_failed: int


class MetaxyDatasink(Datasink[_WriteTaskResult]):
    """A Ray Data Datasink for writing to a Metaxy metadata store.

    !!! example

        ```python
        import metaxy as mx
        import ray

        cfg = mx.init_metaxy()
        dataset = ...  # a ray.data.Dataset

        datasink = MetaxyDatasink(
            feature="my/feature",
            store=cfg.get_store(),
            config=cfg,
        )
        dataset.write_datasink(datasink)

        print(f"Wrote {datasink.result.rows_written} rows, {datasink.result.rows_failed} failed")
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

        # Populated after write completes
        self._result: MetaxyWriteResult | None = None

    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> _WriteTaskResult:
        """Write blocks of metadata to the store."""
        # Initialize metaxy on the worker - config and features are needed for write_metadata
        mx.init_metaxy(self.config)

        rows_written = 0
        rows_failed = 0

        for i, block in enumerate(blocks):
            block_accessor = BlockAccessor.for_block(block)
            num_rows = block_accessor.num_rows()

            try:
                with (
                    self.store.open("write"),
                    self.store.allow_cross_project_writes() if self.allow_cross_project_writes else nullcontext(),
                ):
                    self.store.write_metadata(self._feature_key, block)
                rows_written += num_rows
            except Exception:
                logger.exception(
                    f"Failed to write {num_rows} metadata rows for feature {self._feature_key.to_string()} block {i} of task {ctx.task_idx} ({ctx.op_name})"
                )
                rows_failed += num_rows

        return _WriteTaskResult(rows_written=rows_written, rows_failed=rows_failed)

    def on_write_complete(self, write_result: WriteResult[_WriteTaskResult]) -> None:
        """Aggregate write statistics from all tasks."""
        rows_written = 0
        rows_failed = 0

        for task_result in write_result.write_returns:
            rows_written += task_result.rows_written
            rows_failed += task_result.rows_failed

        self._result = MetaxyWriteResult(rows_written=rows_written, rows_failed=rows_failed)

        logger.info(
            f"MetaxyDatasink write complete for {self._feature_key.to_string()}: "
            f"{rows_written} rows written, {rows_failed} rows failed"
        )

    @property
    def result(self) -> MetaxyWriteResult:
        """Result of the write operation.

        Raises:
            RuntimeError: If accessed before the write operation completes.
        """
        if self._result is None:
            raise RuntimeError("Write operation has not completed yet")
        return self._result
