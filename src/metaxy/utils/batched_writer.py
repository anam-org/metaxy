"""Batched metadata writer for high-throughput streaming writes.

This module provides a batched writer that queues data and writes to a
MetadataStore in batches, either when a batch size threshold is reached or
after a time interval. This is useful for streaming scenarios where data
arrives continuously and needs to be written efficiently.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw

from metaxy._decorators import public
from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey, ValidatedFeatureKeyAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.typing import Frame, IntoFrame

    from metaxy.metadata_store import MetadataStore


logger = logging.getLogger(__name__)

# Type for queue items: dict mapping feature keys to frames
QueueItem = dict[FeatureKey, "Frame"]


@public
class BatchedMetadataWriter:
    """Batched metadata writer with background flush thread.

    Queues data and writes to a MetadataStore in batches either when:

    - The batch reaches `flush_batch_size` rows (if set)
    - `flush_interval` seconds have passed since last flush

    The writer runs a background thread that handles flushing, allowing the
    main thread to continue processing data without blocking on writes.

    Example:
        ```py
        import polars as pl

        with mx.BatchedMetadataWriter(store) as writer:
            batch = {
                MyFeature: pl.DataFrame(
                    {
                        "id": ["x"],
                        "metaxy_provenance_by_field": [{"part_1": "h1", "part_2": "h2"}],
                    }
                )
            }
            writer.put(batch)

        with store:
            assert len(store.read_metadata(MyFeature).collect()) == 1
        ```

    ??? example "Manual lifecycle management"
        <!-- skip next -->
        ```py
        writer = mx.BatchedMetadataWriter(store)
        writer.start()
        try:
            for batch_dict in data_stream:
                writer.put(batch_dict)
        finally:
            rows_written = writer.stop()
        ```

    Args:
        store: The MetadataStore to write to. Must be opened before use.
        flush_batch_size: Number of rows to accumulate before flushing.
            If not set, flushes are only triggered by `flush_interval`
            or when stopping the writer.

            !!! note
                Setting this triggers row counting which materializes lazy frames.

        flush_interval: Maximum seconds between flushes. The timer resets after the end of each flush.

    Raises:
        RuntimeError: If the background thread encounters an error during flush.
    """

    def __init__(
        self,
        store: MetadataStore,
        flush_batch_size: int | None = None,
        flush_interval: float = 2.0,
    ) -> None:
        self._store = store
        self._flush_batch_size = flush_batch_size
        self._flush_interval = flush_interval

        self._queue: queue.Queue[QueueItem] = queue.Queue()
        self._should_stop = threading.Event()
        self._stopped = threading.Event()
        self._num_written: dict[FeatureKey, int] = {}
        self._lock = threading.Lock()
        self._error: BaseException | None = None

        self._thread: threading.Thread | None = None
        self._started = False

        # Capture context at construction time to propagate to background thread
        # This is necessary because ContextVars don't propagate to child threads
        self._graph = FeatureGraph.get_active()
        self._config = MetaxyConfig.get()

    def start(self) -> None:
        """Start the background flush thread.

        This method must be called before putting data. When using the writer
        as a context manager, this is called automatically on entry.

        Raises:
            RuntimeError: If the writer has already been started.
        """
        if self._started:
            raise RuntimeError("Writer has already been started")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started = True

    def put(self, batches: Mapping[CoercibleToFeatureKey, IntoFrame]) -> None:
        """Queue batches for writing.

        The batches are accumulated per-feature and written together using
        `[MetadataStore.write_metadata_multi][]`.

        Args:
            batches: Mapping from feature keys to dataframes.
                Dataframes can be of any type supported by [Narwhals](https://narwhals-dev.github.io/narwhals/).

        Raises:
            RuntimeError: If the writer has not been started, has been stopped,
                or encountered an error.
        """
        self._check_can_put()

        # Convert all keys and values
        converted: dict[FeatureKey, Frame] = {}
        for key, batch in batches.items():
            feature_key = ValidatedFeatureKeyAdapter.validate_python(key)
            batch_nw = self._to_narwhals(batch)
            converted[feature_key] = batch_nw

        self._queue.put(converted)

    def _check_can_put(self) -> None:
        """Check if we can accept new data."""
        if not self._started:
            raise RuntimeError("Writer has not been started. Call start() first or use as context manager.")

        if self._should_stop.is_set():
            raise RuntimeError("Cannot put data after writer has been stopped")

        if self._error is not None:
            raise RuntimeError(f"Writer encountered an error: {self._error}") from self._error

    def _to_narwhals(self, batch: IntoFrame) -> Frame:
        """Convert input to Narwhals Frame."""
        if isinstance(batch, (nw.DataFrame, nw.LazyFrame)):
            return cast("Frame", batch)
        return cast("Frame", nw.from_native(batch))

    def _run(self) -> None:
        """Background thread that processes the queue and flushes batches."""
        # Set the captured context as active in this thread
        # This is necessary because ContextVars don't propagate to child threads
        try:
            with self._graph.use(), self._config.use():
                # Accumulate batches per feature
                pending: dict[FeatureKey, list[Frame]] = defaultdict(list)
                pending_rows = 0
                last_flush = time.time()

                while True:
                    # Calculate timeout: min of time until next flush and a short poll interval
                    # Short poll interval ensures we respond quickly to stop signals
                    time_since_flush = time.time() - last_flush
                    time_until_flush = max(0, self._flush_interval - time_since_flush)
                    timeout = min(time_until_flush, 0.1)  # Check stop signal at least every 100ms

                    # Block efficiently waiting for data or timeout
                    try:
                        item = self._queue.get(timeout=timeout)
                        for feature_key, batch in item.items():
                            # Collect lazy frames if we need to count rows
                            if self._flush_batch_size is not None and isinstance(batch, nw.LazyFrame):
                                batch = batch.collect()
                            pending[feature_key].append(batch)
                            if self._flush_batch_size is not None:
                                pending_rows += len(batch)  # ty: ignore[invalid-argument-type]
                    except queue.Empty:
                        pass  # Timeout - check if we should flush

                    should_stop = self._should_stop.is_set()

                    # Exit if stopped and nothing left to process
                    if should_stop and self._queue.empty() and not pending:
                        break

                    # Check if we should flush
                    if self._should_flush(pending, pending_rows, last_flush, should_stop):
                        try:
                            rows_per_feature = self._flush(pending)
                            with self._lock:
                                for fk, count in rows_per_feature.items():
                                    self._num_written[fk] = self._num_written.get(fk, 0) + count
                            # Only clear pending data on successful flush
                            pending = defaultdict(list)
                            pending_rows = 0
                        except Exception as e:
                            # Set error so callers know data was lost
                            self._error = e
                            logger.exception("Error flushing batch")
                            # Don't clear pending - keep for potential recovery or inspection
                            # Break out of the loop since we're in an error state
                            break
                        last_flush = time.time()
        except BaseException as e:
            self._error = e
            logger.exception("Fatal error in background writer thread")
        finally:
            self._stopped.set()

    def _should_flush(
        self,
        pending: dict[FeatureKey, list[Frame]],
        pending_rows: int,
        last_flush: float,
        should_stop: bool,
    ) -> bool:
        """Determine if we should flush the pending batches."""
        if not pending:
            return False

        # Always flush on stop
        if should_stop:
            return True

        # Flush if batch size threshold reached (only if tracking rows)
        if self._flush_batch_size is not None and pending_rows >= self._flush_batch_size:
            return True

        # Flush if interval reached
        if time.time() - last_flush >= self._flush_interval:
            return True

        return False

    def _flush(self, pending: dict[FeatureKey, list[Frame]]) -> dict[FeatureKey, int]:
        """Flush accumulated batches to the store. Returns rows written per feature."""
        if not pending:
            return {}

        # Concatenate batches per feature
        combined: dict[Any, nw.DataFrame[Any]] = {}
        rows_per_feature: dict[FeatureKey, int] = {}

        for feature_key, batches in pending.items():
            if not batches:
                continue

            # Collect lazy frames and concatenate
            collected = [b.collect() if isinstance(b, nw.LazyFrame) else b for b in batches]
            if len(collected) == 1:
                combined_feature = collected[0]
            else:
                combined_feature = nw.concat(collected)

            combined[feature_key] = combined_feature
            rows_per_feature[feature_key] = len(combined_feature)

        if not combined:
            return {}

        # Write to store
        with self._store.open("write"):
            self._store.write_metadata_multi(combined)

        total_rows = sum(rows_per_feature.values())
        feature_count = len(combined)
        logger.debug("Flushed %d rows across %d features", total_rows, feature_count)

        return rows_per_feature

    def stop(self, timeout: float = 30.0) -> dict[FeatureKey, int]:
        """Signal stop and wait for flush to complete.

        This method signals the background thread to stop, waits for it to
        finish flushing any remaining data, and returns the number of rows
        written per feature.

        Args:
            timeout: Maximum seconds to wait for the background thread.
                Defaults to 30.0.

        Returns:
            Dict mapping feature keys to number of rows written.

        Raises:
            RuntimeError: If the background thread encountered an error during flush.
        """
        if not self._started or self._thread is None:
            return {}

        self._should_stop.set()
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            logger.warning("BatchedMetadataWriter did not stop within %.1fs", timeout)

        if self._error is not None:
            raise RuntimeError(f"Writer encountered an error: {self._error}") from self._error

        with self._lock:
            return dict(self._num_written)

    @property
    def num_written(self) -> dict[FeatureKey, int]:
        """Number of rows written so far per feature.

        This property is thread-safe and can be called while the writer
        is still running to check progress.

        Returns:
            Dict mapping feature keys to number of rows successfully flushed to the store.
        """
        with self._lock:
            return dict(self._num_written)

    @property
    def has_error(self) -> bool:
        """Check if the writer has encountered an error.

        Returns:
            True if the background thread encountered an error, False otherwise.
        """
        return self._error is not None

    def __enter__(self) -> BatchedMetadataWriter:
        """Enter context manager, starting the background thread."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, stopping the writer."""
        self.stop()
