"""Pytest fixtures for example snapshot testing."""

from typing import TYPE_CHECKING

import pytest
from syrupy import SnapshotAssertion
from syrupy.filters import props

if TYPE_CHECKING:
    pass


@pytest.fixture
def example_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Snapshot fixture for examples with timestamp filtering.

    Timestamps are excluded from snapshots so tests are deterministic,
    but they're preserved in the raw .example.result.json file.

    Snapshots are stored in standard __snapshots__/ directory.
    """
    # Configure snapshot to exclude timestamps - use standard syrupy location
    return snapshot.with_defaults(
        exclude=props("timestamp")  # Exclude all timestamp fields
    )
