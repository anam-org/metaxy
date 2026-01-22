from typing import Any

import pyarrow as pa
import ray

import metaxy as mx
from metaxy._public import public
from metaxy.utils.batched_writer import BatchedMetadataWriter


@public
class BatchedMetadataWriterActor:
    """Ray Data actor that writes metadata batches using [`BatchedMetadataWriter`][metaxy.BatchedMetadataWriter].

    !!! tip
        Records are queued and batched in an internal buffer, so this actor is safe
        to be used with both [`Dataset.map`][ray.data.Dataset.map] and [`Dataset.map_batches`][ray.data.Dataset.map_batches].

    !!! tip
        Ray setups tend to be highly custom. This actor is fairly simple and generic
        and should work in most cases. Feel free to customize it to your needs.

    !!! example "with `map`"

        ```python
        import metaxy as mx
        from metaxy.ext.ray import BatchedMetadataWriterActor

        cfg = mx.init_metaxy()
        dataset = ...  # a ray.data.Dataset

        dataset.map(
            BatchedMetadataWriterActor,
            fn_constructor_kwargs={
                "feature": "my/feature",
                "store": cfg.get_store(),
                "metaxy_config": cfg,
            },
        ).materialize()
        ```

    !!! example "with `map_batches`"

        ```python
        dataset.map_batches(
            BatchedMetadataWriterActor,
            fn_constructor_kwargs={
                "feature": "my/feature",
                "store": cfg.get_store(),
                "metaxy_config": cfg,
                "accept_batches": True,
            },
            batch_format=...,  # works with "pyarrow", "pandas", "numpy" and "default"
        ).materialize()
        ```

    Args:
        feature: Feature to write metadata for.

            !!! note
                In the future, this actor will support writing metadata for multiple features at once.

        store: Metadata store to write to.
        flush_interval: Interval in seconds between metadata flushes.
            Passed to [`BatchedMetadataWriter`][metaxy.BatchedMetadataWriter].
        flush_batch_size: Optional maximum number of records to flush at once.
            Passed to [`BatchedMetadataWriter`][metaxy.BatchedMetadataWriter].
        metaxy_config: Metaxy configuration. Will be auto-discovered by the actor if not provided.

            !!! warning
                Ensure the Ray environment is set up properly when not passing `metaxy_config` explicitly.
                This can be achieved by setting `METAXY_CONFIG` and other `METAXY_` environment variables.
                The best practice is to pass `metaxy_config` explicitly to the actor via `ray_remote_args` to avoid any surprises.

        accept_batches: Whether to expect batch inputs from [`map_batches`][ray.data.Dataset.map_batches].

            - `False`: Expects single-row dict input from [`Dataset.map`][ray.data.Dataset.map].

            - `True`: Accepts batch inputs (PyArrow tables, pandas DataFrames,
                or dicts of `numpy` arrays) from [`map_batches`][ray.data.Dataset.map_batches]
    """

    def __init__(
        self,
        feature: mx.CoercibleToFeatureKey,
        store: mx.MetadataStore,
        flush_interval: float = 2.0,
        flush_batch_size: int | None = None,
        metaxy_config: mx.MetaxyConfig | None = None,
        accept_batches: bool = False,
    ):
        mx.init_metaxy(metaxy_config)
        self._feature_key = mx.coerce_to_feature_key(feature)
        self._store = store
        self._accept_batches = accept_batches
        self._writer = BatchedMetadataWriter(
            store,
            flush_batch_size=flush_batch_size,
            flush_interval=flush_interval,
        )
        self._writer.start()

    def __call__(self, inputs: pa.Table | dict[str, Any]) -> pa.Table | dict[str, Any]:
        """Write an incoming batch to the metadata store.

        The input format depends on the `accept_batches` parameter:

        - `accept_batches=False`: Expects `dict[str, Any]` with scalar values (from `map`)
        - `accept_batches=True`: Accepts `pa.Table`, `pd.DataFrame`, or `dict[str, np.ndarray]` (from `map_batches`)

        The data is queued in an internal buffer and automatically flushed periodically.
        """
        table = self._to_table(inputs)
        self._writer.put({self._feature_key: table})
        return inputs

    def _to_table(self, inputs: pa.Table | dict[str, Any]) -> pa.Table:
        """Convert input batch to PyArrow table."""
        if self._accept_batches:
            # Batch input from map_batches - can be PyArrow, pandas, or numpy dict
            if isinstance(inputs, pa.Table):
                return inputs
            elif isinstance(inputs, dict):
                # dict[str, np.ndarray] from numpy batch format
                import numpy as np

                return pa.Table.from_pydict(
                    {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
                )
            else:
                # Assume pandas DataFrame
                return pa.Table.from_pandas(inputs)
        else:
            # Single row dict from map - wrap scalars in lists
            if not isinstance(inputs, dict):
                raise TypeError(f"Expected dict for accept_batches=False (map input), got {type(inputs)}")
            return pa.Table.from_pydict({k: [v] for k, v in inputs.items()})

    @ray.method
    def get_error(self) -> BaseException | None:
        """Get the error encountered by the writer, if any.

        Returns:
            The exception raised during the internal metadata writer startup, or None if no error occurred.
        """
        return self._writer._error

    @ray.method
    def stop(self):
        """A handle for stopping the metadata writer manually."""
        self._writer.stop()

    def __ray_shutdown__(self):
        """Automatic shutdown lifecycle hook for Ray.

        This method is called by Ray when the actor is being terminated.
        We explicitly re-raise any writer errors to ensure they are logged
        by Ray rather than silently swallowed.
        """
        try:
            self._writer.stop()
        except Exception:
            # Log and re-raise to ensure Ray sees the error
            import logging

            logging.getLogger(__name__).exception("Error during BatchedMetadataWriterActor shutdown")
            raise

    def __del__(self) -> None:
        """Cleanup hook for garbage collection.

        Ensures the writer is stopped even if __ray_shutdown__ wasn't called.
        Errors during stop are silently ignored since we're in __del__.
        """
        if hasattr(self, "_writer"):
            try:
                self._writer.stop()
            except Exception:
                # Ignore errors in __del__ to avoid breaking garbage collection
                pass
