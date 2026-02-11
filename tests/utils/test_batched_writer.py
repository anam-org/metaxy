"""Tests for BufferedMetadataWriter."""

import threading
import time
from typing import Any

import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from metaxy import (
    BaseFeature,
    BufferedMetadataWriter,
    FeatureKey,
    FeatureSpec,
    MetadataStore,
)
from metaxy.models.feature import FeatureGraph


def _make_test_data() -> dict[str, Any]:
    """Create test data as a dict."""
    return {
        "id": ["df1", "df2"],
        "value": [1, 2],
        "metaxy_provenance_by_field": [{"default": "hash_df1"}, {"default": "hash_df2"}],
    }


def _pandas_eager() -> pd.DataFrame:
    return pd.DataFrame(_make_test_data())


def _polars_eager() -> pl.DataFrame:
    return pl.DataFrame(_make_test_data())


def _polars_lazy() -> pl.LazyFrame:
    return pl.DataFrame(_make_test_data()).lazy()


def _pyarrow_table() -> pa.Table:
    return pa.table(_make_test_data())


def _narwhals_eager_pandas() -> nw.DataFrame[Any]:
    return nw.from_native(_pandas_eager())


def _narwhals_eager_polars() -> nw.DataFrame[Any]:
    return nw.from_native(_polars_eager())


def _narwhals_lazy_polars() -> nw.LazyFrame[Any]:
    return nw.from_native(_polars_lazy())


@pytest.fixture
def writer_feature(graph: FeatureGraph):
    """Create a test feature for batched writer tests."""

    class WriterFeature(
        BaseFeature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "batched_writer"]),
            id_columns=["id"],
        ),
    ):
        """Test feature for batched writer tests."""

        id: str
        value: int

    return WriterFeature


@pytest.fixture
def second_feature(graph: FeatureGraph):
    """Create a second test feature for multi-feature tests."""

    class SecondFeature(
        BaseFeature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "second_feature"]),
            id_columns=["id"],
        ),
    ):
        """Second test feature."""

        id: str
        count: int

    return SecondFeature


def test_batched_writer_basic(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test basic write functionality with manual start()."""
    with store.open("w"):
        writer = BufferedMetadataWriter(
            store,
            flush_batch_size=100,
            flush_interval=0.5,
        )
        writer.start()

        # Write a batch
        batch = pl.DataFrame(
            {
                "id": ["a", "b", "c"],
                "value": [1, 2, 3],
                "metaxy_provenance_by_field": [{"default": "hash1"}, {"default": "hash2"}, {"default": "hash3"}],
            }
        )
        writer.put({writer_feature: batch})

        # Stop and verify
        rows_written = writer.stop()
        assert rows_written[writer_feature.spec().key] == 3

        # Verify data was written
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 3
        assert set(result["id"].to_list()) == {"a", "b", "c"}


def test_batched_writer_context_manager(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test writer as context manager."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=100) as writer:
            batch = pl.DataFrame(
                {
                    "id": ["x", "y"],
                    "value": [10, 20],
                    "metaxy_provenance_by_field": [{"default": "hashx"}, {"default": "hashy"}],
                }
            )
            writer.put({writer_feature: batch})

        # Verify data was written after context exit
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 2


def test_batched_writer_batch_size_trigger(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that flush is triggered when batch size is reached."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=5, flush_interval=1.0) as writer:
            # Write batches totaling 10 rows (should trigger flush at 5)
            for i in range(10):
                batch = pl.DataFrame(
                    {
                        "id": [f"row{i}"],
                        "value": [i],
                        "metaxy_provenance_by_field": [{"default": f"hash{i}"}],
                    }
                )
                writer.put({writer_feature: batch})

            # Poll for flush completion with timeout (handles slow CI/Windows)
            feature_key = writer_feature.spec().key
            deadline = time.time() + 5.0  # 5 second timeout
            while time.time() < deadline:
                if writer.num_written.get(feature_key, 0) >= 5:
                    break
                time.sleep(0.05)

            assert writer.num_written.get(feature_key, 0) >= 5  # At least one batch should have been flushed

        # Verify all data was written after stop
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 10


def test_batched_writer_interval_trigger(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that flush is triggered after interval."""
    flush_interval = 0.1

    with store.open("w"):
        with BufferedMetadataWriter(
            store,
            flush_batch_size=1000,  # high threshold so batch size won't trigger
            flush_interval=flush_interval,
        ) as writer:
            # Write a small batch (won't trigger batch size)
            batch = pl.DataFrame(
                {
                    "id": ["interval_test"],
                    "value": [42],
                    "metaxy_provenance_by_field": [{"default": "hash_interval"}],
                }
            )
            writer.put({writer_feature: batch})

            # Poll for flush completion with timeout (handles slow CI/Windows)
            feature_key = writer_feature.spec().key
            deadline = time.time() + 5.0  # 5 second timeout
            while time.time() < deadline:
                if writer.num_written.get(feature_key, 0) == 1:
                    break
                time.sleep(0.05)

            # Should have flushed due to interval (not batch size)
            assert writer.num_written.get(feature_key, 0) == 1

        # Verify data was written
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 1


def test_batched_writer_no_batch_size(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that flush_batch_size=None only flushes on interval or stop."""
    with store.open("w"):
        # Use a long interval so it won't trigger during test
        with BufferedMetadataWriter(store, flush_interval=10.0) as writer:
            # Write many batches (should not trigger flush since no batch size)
            for i in range(20):
                batch = pl.DataFrame(
                    {
                        "id": [f"row{i}"],
                        "value": [i],
                        "metaxy_provenance_by_field": [{"default": f"hash{i}"}],
                    }
                )
                writer.put({writer_feature: batch})

            # Should NOT have flushed yet (no batch size trigger, interval is long)
            assert writer.num_written == {}

        # After stop, all data should be written
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 20


@pytest.mark.parametrize(
    "make_batch",
    [
        pytest.param(_pandas_eager, id="pandas"),
        pytest.param(_polars_eager, id="polars_eager"),
        pytest.param(_polars_lazy, id="polars_lazy"),
        pytest.param(_pyarrow_table, id="pyarrow"),
        pytest.param(_narwhals_eager_pandas, id="narwhals_eager_pandas"),
        pytest.param(_narwhals_eager_polars, id="narwhals_eager_polars"),
        pytest.param(_narwhals_lazy_polars, id="narwhals_lazy_polars"),
    ],
)
def test_batched_writer_accepts_dataframe_types(
    store: MetadataStore, writer_feature: type[BaseFeature], make_batch: Any
):
    """Test that writer accepts various dataframe types."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=100) as writer:
            batch = make_batch()
            writer.put({writer_feature: batch})

        # Verify data was written
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 2
        assert set(result["id"].to_list()) == {"df1", "df2"}


def test_batched_writer_num_written_property(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that num_written is updated correctly."""
    feature_key = writer_feature.spec().key
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=2, flush_interval=0.5) as writer:
            assert writer.num_written == {}

            # Write first batch (2 rows - should trigger flush)
            batch1 = pl.DataFrame(
                {
                    "id": ["nw1", "nw2"],
                    "value": [1, 2],
                    "metaxy_provenance_by_field": [{"default": "hash_nw1"}, {"default": "hash_nw2"}],
                }
            )
            writer.put({writer_feature: batch1})

            # Poll for flush completion
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if writer.num_written.get(feature_key, 0) == 2:
                    break
                time.sleep(0.05)
            assert writer.num_written[feature_key] == 2

            # Write second batch
            batch2 = pl.DataFrame(
                {
                    "id": ["nw3", "nw4"],
                    "value": [3, 4],
                    "metaxy_provenance_by_field": [{"default": "hash_nw3"}, {"default": "hash_nw4"}],
                }
            )
            writer.put({writer_feature: batch2})

            # Poll for flush completion
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if writer.num_written.get(feature_key, 0) == 4:
                    break
                time.sleep(0.05)
            assert writer.num_written[feature_key] == 4


def test_batched_writer_multiple_batches(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test writing multiple batches."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=100) as writer:
            for i in range(5):
                batch = pl.DataFrame(
                    {
                        "id": [f"multi_{i}_a", f"multi_{i}_b"],
                        "value": [i * 2, i * 2 + 1],
                        "metaxy_provenance_by_field": [
                            {"default": f"hash_multi_{i}_a"},
                            {"default": f"hash_multi_{i}_b"},
                        ],
                    }
                )
                writer.put({writer_feature: batch})

        # Verify all data was written
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 10


def test_batched_writer_put_after_stop_raises(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that put raises after stop is called."""
    with store.open("w"):
        writer = BufferedMetadataWriter(store, flush_batch_size=100)
        writer.start()
        writer.stop()

        batch = pl.DataFrame(
            {
                "id": ["after_stop"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "hash_after"}],
            }
        )

        with pytest.raises(RuntimeError, match="Cannot put data after writer has been stopped"):
            writer.put({writer_feature: batch})


def test_batched_writer_put_before_start_raises(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that put raises before start is called."""
    with store.open("w"):
        writer = BufferedMetadataWriter(store, flush_batch_size=100)

        batch = pl.DataFrame(
            {
                "id": ["before_start"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "hash_before"}],
            }
        )

        with pytest.raises(RuntimeError, match="Writer has not been started"):
            writer.put({writer_feature: batch})


def test_batched_writer_double_start_raises(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that calling start twice raises an error."""
    with store.open("w"):
        writer = BufferedMetadataWriter(store, flush_batch_size=100)
        writer.start()

        with pytest.raises(RuntimeError, match="Writer has already been started"):
            writer.start()

        writer.stop()


def test_batched_writer_has_error_property(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test has_error property when no error occurs."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=100) as writer:
            assert not writer.has_error

            batch = pl.DataFrame(
                {
                    "id": ["error_test"],
                    "value": [1],
                    "metaxy_provenance_by_field": [{"default": "hash_error"}],
                }
            )
            writer.put({writer_feature: batch})

        # Still no error after successful write
        assert not writer.has_error


def test_batched_writer_feature_key_string(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that writer accepts feature key as string."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=100) as writer:
            batch = pl.DataFrame(
                {
                    "id": ["str_key"],
                    "value": [99],
                    "metaxy_provenance_by_field": [{"default": "hash_str"}],
                }
            )
            # Use string key format
            writer.put({"test/batched_writer": batch})

        # Verify data was written
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 1


def test_batched_writer_thread_safety(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that writer handles concurrent puts safely."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=50, flush_interval=0.5) as writer:
            errors: list[Exception] = []

            def write_batch(thread_id: int):
                try:
                    for i in range(10):
                        batch = pl.DataFrame(
                            {
                                "id": [f"thread{thread_id}_row{i}"],
                                "value": [thread_id * 100 + i],
                                "metaxy_provenance_by_field": [{"default": f"hash_t{thread_id}_{i}"}],
                            }
                        )
                        writer.put({writer_feature: batch})
                except Exception as e:
                    errors.append(e)

            # Start multiple threads
            threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Errors during concurrent writes: {errors}"

        # Verify all data was written (5 threads * 10 rows each = 50 rows)
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 50


def test_batched_writer_multi_feature(
    store: MetadataStore, writer_feature: type[BaseFeature], second_feature: type[BaseFeature]
):
    """Test writing to multiple features in single put."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_batch_size=100) as writer:
            batch1 = pl.DataFrame(
                {
                    "id": ["multi_a", "multi_b"],
                    "value": [1, 2],
                    "metaxy_provenance_by_field": [{"default": "hash_a"}, {"default": "hash_b"}],
                }
            )
            batch2 = pl.DataFrame(
                {
                    "id": ["multi_x", "multi_y"],
                    "count": [10, 20],
                    "metaxy_provenance_by_field": [{"default": "hash_x"}, {"default": "hash_y"}],
                }
            )
            # Write to both features in one put
            writer.put({writer_feature: batch1, second_feature: batch2})

        # Verify both features have data
        result1 = store.read(writer_feature).collect().to_polars()
        result2 = store.read(second_feature).collect().to_polars()
        assert len(result1) == 2
        assert len(result2) == 2


def test_batched_writer_multi_feature_accumulated(
    store: MetadataStore, writer_feature: type[BaseFeature], second_feature: type[BaseFeature]
):
    """Test that multiple puts to different features are accumulated correctly."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_interval=0.5) as writer:
            # Write to first feature
            batch1 = pl.DataFrame(
                {
                    "id": ["acc_a"],
                    "value": [1],
                    "metaxy_provenance_by_field": [{"default": "hash_acc_a"}],
                }
            )
            writer.put({writer_feature: batch1})

            # Write to second feature
            batch2 = pl.DataFrame(
                {
                    "id": ["acc_x"],
                    "count": [10],
                    "metaxy_provenance_by_field": [{"default": "hash_acc_x"}],
                }
            )
            writer.put({second_feature: batch2})

            # Write more to first feature
            batch3 = pl.DataFrame(
                {
                    "id": ["acc_b"],
                    "value": [2],
                    "metaxy_provenance_by_field": [{"default": "hash_acc_b"}],
                }
            )
            writer.put({writer_feature: batch3})

        # Verify all data was written
        result1 = store.read(writer_feature).collect().to_polars()
        result2 = store.read(second_feature).collect().to_polars()
        assert len(result1) == 2  # acc_a and acc_b
        assert len(result2) == 1  # acc_x


def test_batched_writer_flush_error_sets_has_error(
    store: MetadataStore, writer_feature: type[BaseFeature], monkeypatch: pytest.MonkeyPatch
):
    """Test that has_error is True when flush fails."""
    from unittest.mock import MagicMock

    with store.open("w"):
        writer = BufferedMetadataWriter(store, flush_interval=0.05)
        writer.start()

        # Mock the _flush method to raise an exception
        mock_flush = MagicMock(side_effect=ValueError("Simulated flush error"))
        monkeypatch.setattr(writer, "_flush", mock_flush)

        # Write a batch
        batch = pl.DataFrame(
            {
                "id": ["error_test"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "hash_error"}],
            }
        )
        writer.put({writer_feature: batch})

        # Poll for error to be set
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if writer.has_error:
                break
            time.sleep(0.05)

        # Verify has_error is True
        assert writer.has_error

        # Verify stop raises the error
        with pytest.raises(RuntimeError, match="Writer encountered an error"):
            writer.stop()


def test_batched_writer_flush_error_prevents_further_puts(
    store: MetadataStore, writer_feature: type[BaseFeature], monkeypatch: pytest.MonkeyPatch
):
    """Test that put raises after a flush error."""
    from unittest.mock import MagicMock

    with store.open("w"):
        writer = BufferedMetadataWriter(store, flush_interval=0.05)
        writer.start()

        # Mock the _flush method to raise an exception
        mock_flush = MagicMock(side_effect=ValueError("Simulated flush error"))
        monkeypatch.setattr(writer, "_flush", mock_flush)

        # Write first batch
        batch = pl.DataFrame(
            {
                "id": ["error_test"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "hash_error"}],
            }
        )
        writer.put({writer_feature: batch})

        # Poll for error to be set
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if writer.has_error:
                break
            time.sleep(0.05)

        # Subsequent puts should raise
        with pytest.raises(RuntimeError, match="Writer encountered an error"):
            writer.put({writer_feature: batch})

        # Clean up - stop will also raise, catch it
        with pytest.raises(RuntimeError):
            writer.stop()


def test_batched_writer_stop_timeout_raises(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test that stop raises RuntimeError when thread doesn't finish in time."""
    from unittest.mock import patch

    with store.open("w"):
        writer = BufferedMetadataWriter(store, flush_interval=0.5)
        writer.start()

        original_thread = writer._thread
        assert original_thread is not None

        with patch.object(original_thread, "is_alive", return_value=True):
            with pytest.raises(RuntimeError, match="did not stop within"):
                writer.stop(timeout=0.01)


def test_batched_writer_flush(store: MetadataStore, writer_feature: type[BaseFeature]):
    """Test explicit flush writes pending data immediately."""
    with store.open("w"):
        with BufferedMetadataWriter(store, flush_interval=10.0) as writer:
            batch = pl.DataFrame(
                {
                    "id": ["flush_a", "flush_b"],
                    "value": [1, 2],
                    "metaxy_provenance_by_field": [{"default": "h1"}, {"default": "h2"}],
                }
            )
            writer.put({writer_feature: batch})

            # Without explicit flush, data would not be written yet (long interval)
            assert writer.num_written == {}

            writer.flush()

            assert writer.num_written[writer_feature.spec().key] == 2

        # Verify data was persisted
        result = store.read(writer_feature).collect().to_polars()
        assert len(result) == 2


def test_batched_writer_flush_error_propagates(
    store: MetadataStore, writer_feature: type[BaseFeature], monkeypatch: pytest.MonkeyPatch
):
    """Test that flush() propagates errors from the background thread."""
    from unittest.mock import MagicMock

    with store.open("w"):
        writer = BufferedMetadataWriter(store, flush_interval=10.0)
        writer.start()

        original_flush = writer._flush

        # Mock _flush to fail
        mock_flush = MagicMock(side_effect=ValueError("Simulated flush error"))
        monkeypatch.setattr(writer, "_flush", mock_flush)

        batch = pl.DataFrame(
            {
                "id": ["err"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "h"}],
            }
        )
        writer.put({writer_feature: batch})

        with pytest.raises(RuntimeError, match="Flush failed"):
            writer.flush()

        # Writer should still be running (explicit flush errors don't kill the thread)
        assert not writer.has_error

        # Restore real _flush so stop() can flush remaining data
        monkeypatch.setattr(writer, "_flush", original_flush)
        writer.stop()


def test_batched_writer_exit_does_not_mask_exception(
    store: MetadataStore, writer_feature: type[BaseFeature], monkeypatch: pytest.MonkeyPatch
):
    """Test that __exit__ does not mask the original exception when stop() also fails."""
    from unittest.mock import MagicMock

    with store.open("w"):
        # The original exception should propagate, not the stop() error
        with pytest.raises(ValueError, match="original"):
            with BufferedMetadataWriter(store, flush_interval=0.05) as writer:
                # Make _flush fail so stop() will raise
                mock_flush = MagicMock(side_effect=RuntimeError("flush boom"))
                monkeypatch.setattr(writer, "_flush", mock_flush)

                batch = pl.DataFrame(
                    {
                        "id": ["x"],
                        "value": [1],
                        "metaxy_provenance_by_field": [{"default": "h"}],
                    }
                )
                writer.put({writer_feature: batch})

                # Wait for the background flush error
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if writer.has_error:
                        break
                    time.sleep(0.05)

                raise ValueError("original")
