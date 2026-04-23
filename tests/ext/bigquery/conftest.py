"""BigQuery test fixtures."""

import os
import uuid
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session")
def bigquery_project_id() -> str:
    """GCP project ID for BigQuery integration tests.

    Reads BIGQUERY_TEST_PROJECT_ID. Skips when unset.
    """
    import warnings

    project_id = os.environ.get("BIGQUERY_TEST_PROJECT_ID", "")
    if not project_id:
        warnings.warn("BIGQUERY_TEST_PROJECT_ID not set — skipping BigQuery tests", stacklevel=2)
        pytest.skip("BIGQUERY_TEST_PROJECT_ID not set")
    return project_id


@pytest.fixture(scope="function")
def bigquery_dataset(bigquery_project_id: str) -> Generator[str, None, None]:
    """Create an ephemeral BigQuery dataset for a single test, then drop it."""
    import ibis
    from google.cloud.bigquery import Dataset

    dataset_id = f"metaxy_test_{uuid.uuid4().hex[:8]}"
    conn = ibis.bigquery.connect(project_id=bigquery_project_id, dataset_id=dataset_id)
    conn.client.create_dataset(Dataset(f"{bigquery_project_id}.{dataset_id}"), exists_ok=True)

    yield dataset_id

    conn.client.delete_dataset(f"{bigquery_project_id}.{dataset_id}", delete_contents=True, not_found_ok=True)
