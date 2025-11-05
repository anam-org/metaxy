import narwhals as nw
import metaxy as mx
from typing import Any
PROVENANCE_BY_KEY_COL = "metaxy_provenance_by_field"
PROVENANCE_COL = "metaxy_provenance"

def find_diff(
    feature_spec: mx.FeatureSpec,
    incoming: nw.DataFrame[Any],
    current: nw.DataFrame[Any],
):
    # identifies changes by looking at PROVENANCE_COL

    # new samples are rows from incoming that don't have matching id_columns in current

    # use polars API (Narwhals implements it)
    new = incoming.join(current.select(feature_spec.id_columns), on=feature_spec.id_columns, how="left_anti")

    # changed sample would have PROVENANCE_COL mismatch

    changed_ids = incoming.join(
        current.select(list(feature_spec.id_columns) + [PROVENANCE_COL]),
        on=feature_spec.id_columns,
        how="inner",
        suffix="_current"
    ).filter(
        nw.col(PROVENANCE_COL) != nw.col(PROVENANCE_COL + "_current")
    )

    changed = incoming.join(
        changed_ids.select(feature_spec.id_columns),
        on=feature_spec.id_columns,
        how="inner",
    )

    # deleted samples are rows from current that don't have matching id_columns in incoming

    deleted = current.join(
        incoming.select(feature_spec.id_columns),
        on=feature_spec.id_columns,
        how="left_anti"
    )

    return new


def resolve_update(
    feature_spec: mx.FeatureSpec,
    current_metadata: nw.DataFrame[Any],
    upstream_metadata: dict[mx.FeatureSpec, nw.DataFrame[Any]],
    sample: nw.DataFrame[Any] | None = None
):
    if len(upstream_metadata) == 0:
        # this is a root feature without dependencies
        # we must get a sample dataframe in this case
        # with metaxy_provenance and metaxy_provenance_by_field set
        # we should rely on metaxy_provenance to do the check

        assert sample is not None, "sample must be provided for root features"

        return find_diff(feature_spec, sample, current_metadata)

    assert len(upstream_metadata) > 0, "upstream_metadata must be provided for non-root features"
