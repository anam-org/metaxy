"""Feature definitions for the Alembic example."""

import metaxy as mx
from metaxy.ext.sqlmodel import BaseSQLModelFeature
from sqlmodel import Field


class AlembicDemoFeature(
    BaseSQLModelFeature,
    table=True,
    spec=mx.FeatureSpec(
        key="examples/alembic_demo",
        fields=["path"],
        id_columns=("sample_uid",),
    ),
):
    """Provenance columns use JSON (dynamic keys, no DDL churn on field changes).

    DuckDB 1.5+ VARIANT is the preferred type for DuckDB-only deployments.
    The env.py strips primary_key constraints for DuckLake compatibility.
    """

    sample_uid: str = Field(primary_key=True)
    path: str
