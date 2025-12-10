"""Vortex S3 integration tests.

Note: Remote storage (S3, GCS, Azure) is NOT yet supported by vortex-data.
The vortex Python library only supports local filesystem writes.
DuckDB's vortex extension has a bug with COPY TO S3.

These tests are kept as placeholders for when upstream support is added.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("vortex", reason="vortex-data not installed")

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="VortexMetadataStore requires Python 3.11+",
)


@pytest.mark.skip(
    reason=(
        "Vortex remote storage not yet supported. "
        "vortex.io.write() only supports local paths and "
        "DuckDB vortex extension COPY TO S3 has a bug. "
        "See https://github.com/vortex-data/vortex/issues"
    )
)
def test_vortex_s3_roundtrip_with_moto(
    s3_bucket_and_storage_options, test_features
) -> None:
    """Placeholder test for S3 support - enable when vortex adds remote storage."""
    pass
