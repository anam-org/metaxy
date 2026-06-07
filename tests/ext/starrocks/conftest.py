"""StarRocks test fixtures."""

from __future__ import annotations

import os
import uuid

import pytest


@pytest.fixture
def starrocks_db() -> str:
    dsn = os.environ.get("STARROCKS_TEST_DSN")
    if dsn is None:
        pytest.skip("Set STARROCKS_TEST_DSN=mysql://root@127.0.0.1:9030/metaxy to run StarRocks tests.")

    pytest.importorskip("MySQLdb")
    return dsn


@pytest.fixture
def starrocks_table_prefix() -> str:
    return f"sr_{uuid.uuid4().hex[:8]}__"
