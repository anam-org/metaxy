"""Ibis implementation of upstream joiner."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan

from metaxy.data_versioning.joiners.base import UpstreamJoiner


class IbisJoiner(UpstreamJoiner["ir.Table"]):
    """Joins upstream features using Ibis table expressions.

    Type Parameters:
        TRef = ibis.expr.types.Table

    Strategy:
    - Starts with first upstream feature
    - Sequentially inner joins remaining upstream features on sample_id
    - Renames data_version columns to avoid conflicts
    - All operations are lazy (no execution until materialization)

    Attributes:
        backend: Optional Ibis backend connection for creating bound tables
    """

    def __init__(self, backend: "ir.BaseBackend"):
        """Initialize IbisJoiner with backend connection.

        Args:
            backend: Ibis backend connection for materializing tables
        """
        self.backend = backend

    def join_upstream(
        self,
        upstream_refs: dict[str, "ir.Table"],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
    ) -> tuple["ir.Table", dict[str, str]]:
        """Join upstream Ibis tables together.

        Args:
            upstream_refs: Dict of upstream feature key -> Ibis table
            feature_spec: Feature specification
            feature_plan: Feature plan

        Returns:
            (joined Ibis table, column mapping)
        """
        import polars as pl

        if not upstream_refs:
            # No upstream dependencies - source feature
            # Return empty table with just sample_id column
            # Use polars DataFrame to avoid pandas dependency
            # Explicitly set sample_id type to Utf8 to avoid NULL type issues with DuckDB
            empty_df = pl.DataFrame({"sample_id": []}, schema={"sample_id": pl.Utf8})

            # Materialize the table through backend to make it bound
            import uuid

            temp_name = f"_metaxy_empty_{uuid.uuid4().hex[:8]}"
            self.backend.create_table(temp_name, obj=empty_df, temp=True)  # type: ignore[attr-defined]
            return self.backend.table(temp_name), {}  # type: ignore[attr-defined]

        # Start with the first upstream feature
        upstream_keys = sorted(upstream_refs.keys())
        first_key = upstream_keys[0]

        # Start with first upstream, rename its data_version column
        col_name_first = f"__upstream_{first_key}__data_version"
        joined = upstream_refs[first_key].select(
            "sample_id", upstream_refs[first_key].data_version.name(col_name_first)
        )

        upstream_mapping = {first_key: col_name_first}

        # Join remaining upstream features
        for upstream_key in upstream_keys[1:]:
            col_name = f"__upstream_{upstream_key}__data_version"

            upstream_renamed = upstream_refs[upstream_key].select(
                "sample_id", upstream_refs[upstream_key].data_version.name(col_name)
            )

            joined = joined.join(
                upstream_renamed,
                "sample_id",
                how="inner",  # Only sample_ids present in ALL upstream
            )

            upstream_mapping[upstream_key] = col_name

        return joined, upstream_mapping
