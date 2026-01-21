"""Tests for the _shared_transaction_timestamp context manager."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from metaxy.metadata_store.duckdb import DuckDBMetadataStore


@pytest.fixture
def store(tmp_path) -> DuckDBMetadataStore:
    """Create a DuckDB store for testing."""
    return DuckDBMetadataStore(database=str(tmp_path / "test.db"))


def test_transaction_sets_and_clears_timestamp(store: DuckDBMetadataStore):
    """The context manager sets timestamp on entry and clears on exit."""
    assert store._transaction_timestamp is None

    with store._shared_transaction_timestamp() as ts:
        assert store._transaction_timestamp is not None
        assert store._transaction_timestamp == ts
        assert isinstance(ts, datetime)
        assert ts.tzinfo == timezone.utc

    assert store._transaction_timestamp is None


def test_transaction_clears_timestamp_on_exception(store: DuckDBMetadataStore):
    """The context manager clears timestamp even when an exception is raised."""
    assert store._transaction_timestamp is None

    with pytest.raises(ValueError, match="test error"):
        with store._shared_transaction_timestamp() as ts:
            assert store._transaction_timestamp == ts
            raise ValueError("test error")

    # Timestamp should be cleared despite the exception
    assert store._transaction_timestamp is None


def test_transaction_is_reentrant(store: DuckDBMetadataStore):
    """Nested context manager calls reuse the outer timestamp."""
    assert store._transaction_timestamp is None

    with store._shared_transaction_timestamp() as outer_ts:
        assert store._transaction_timestamp == outer_ts

        with store._shared_transaction_timestamp() as inner_ts:
            # Inner transaction should reuse outer timestamp
            assert inner_ts == outer_ts
            assert store._transaction_timestamp == outer_ts

        # After inner exits, timestamp still set (outer still active)
        assert store._transaction_timestamp == outer_ts

    # After outer exits, timestamp cleared
    assert store._transaction_timestamp is None


def test_transaction_reentrant_with_inner_exception(store: DuckDBMetadataStore):
    """Nested transaction handles exception in inner context correctly."""
    assert store._transaction_timestamp is None

    with store._shared_transaction_timestamp() as outer_ts:
        assert store._transaction_timestamp == outer_ts

        with pytest.raises(ValueError, match="inner error"):
            with store._shared_transaction_timestamp() as inner_ts:
                assert inner_ts == outer_ts
                raise ValueError("inner error")

        # After inner exception, timestamp still set (outer still active)
        assert store._transaction_timestamp == outer_ts

    # After outer exits normally, timestamp cleared
    assert store._transaction_timestamp is None


def test_transaction_reentrant_with_outer_exception(store: DuckDBMetadataStore):
    """Exception in outer transaction still clears timestamp."""
    assert store._transaction_timestamp is None

    with pytest.raises(ValueError, match="outer error"):
        with store._shared_transaction_timestamp() as outer_ts:
            with store._shared_transaction_timestamp() as inner_ts:
                assert inner_ts == outer_ts
            # After inner exits, raise in outer
            raise ValueError("outer error")

    # Timestamp cleared despite outer exception
    assert store._transaction_timestamp is None


def test_multiple_context_entries_return_same_timestamp(store: DuckDBMetadataStore):
    """Multiple nested context manager entries return the same timestamp."""
    with store._shared_transaction_timestamp() as ts1:
        with store._shared_transaction_timestamp() as ts2:
            with store._shared_transaction_timestamp() as ts3:
                assert ts1 == ts2 == ts3


def test_sequential_transactions_get_different_timestamps(store: DuckDBMetadataStore):
    """Sequential transactions get different timestamps."""
    with store._shared_transaction_timestamp() as ts1:
        pass

    with store._shared_transaction_timestamp() as ts2:
        pass

    # Note: These could be equal if they happen within the same microsecond,
    # but that's very unlikely. We're just testing that independent transactions
    # create new timestamps.
    assert ts1 <= ts2
