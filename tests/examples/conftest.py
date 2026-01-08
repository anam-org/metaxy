"""Pytest fixtures for example snapshot testing."""

from typing import TYPE_CHECKING

import pytest
from syrupy import SnapshotAssertion
from syrupy.filters import props

if TYPE_CHECKING:
    pass


@pytest.fixture
def example_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Snapshot fixture for examples with filtering of non-deterministic fields.

    Excluded from snapshots:
    - timestamp: varies between runs
    - stderr: contains temp paths that vary between runs
    - stdout: can have non-deterministic ordering (e.g., dict iteration)

    These are preserved in the raw .example.result.json file.
    """
    return snapshot.with_defaults(exclude=props("timestamp", "stderr", "stdout"))
