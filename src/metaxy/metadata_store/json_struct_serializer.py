"""Helpers for JSON <-> flattened struct columns used by JSON-compatible stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
import polars as pl
from ibis.expr import types as ibis_types

from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.plan import FeaturePlan
from metaxy.versioning.flat_engine import FlatVersioningMixin

if TYPE_CHECKING:
    pass


class JsonStructSerializerMixin(ABC):
    """Shared helpers for JSON-compatible metadata stores."""

    @abstractmethod
    def _get_json_unpack_exprs(
        self,
        json_column: str,
        field_names: list[str],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_json_pack_expr(
        self,
        struct_name: str,
        field_columns: Mapping[str, str],
    ) -> Any:
        raise NotImplementedError

    def _get_field_names(
        self,
        feature_plan: FeaturePlan,
        *,
        include_dependencies: bool,
    ) -> list[str]:
        names = [f.key.to_struct_key() for f in feature_plan.feature.fields]
        if include_dependencies:
            if feature_plan.feature_deps:
                names += [dep.feature.to_struct_key() for dep in feature_plan.feature_deps]
            if feature_plan.feature.deps:
                names += [dep.feature.to_struct_key() for dep in feature_plan.feature.deps]
        return names

    def _get_flattened_field_columns(self, struct_name: str, field_names: list[str]) -> dict[str, str]:
        return {name: FlatVersioningMixin._get_flattened_column_name(struct_name, name) for name in field_names}

    def _select_flattened_feature_columns(self, columns: list[str], prefix: str) -> list[str]:
        return [col for col in columns if col.startswith(f"{prefix}__") and "__" not in col.split("__", 1)[1]]

    def _unpack_json_column(
        self,
        ibis_table: ibis_types.Table,
        json_column: str,
        field_names: list[str],
    ) -> ibis_types.Table:
        if json_column not in ibis_table.columns:
            return ibis_table
        unpack_exprs = self._get_json_unpack_exprs(json_column, field_names)
        return ibis_table.mutate(**unpack_exprs).drop(json_column)

    def _pack_json_column(
        self,
        ibis_table: ibis_types.Table,
        struct_name: str,
        field_names: list[str],
    ) -> ibis_types.Table:
        field_columns = self._get_flattened_field_columns(struct_name, field_names)
        if not any(col in ibis_table.columns for col in field_columns.values()):
            return ibis_table
        pack_expr = self._get_json_pack_expr(struct_name, field_columns)
        base_cols = [col for col in ibis_table.columns if col not in field_columns.values() and col != struct_name]
        return ibis_table.select(
            *[ibis_table[col] for col in base_cols],
            pack_expr.name(struct_name),
        )

    def _add_struct_from_flattened(
        self,
        ibis_table: ibis_types.Table,
        struct_name: str,
        field_names: list[str],
    ) -> ibis_types.Table:
        import ibis

        if struct_name in ibis_table.columns:
            return ibis_table
        struct_fields = {
            name: ibis_table[FlatVersioningMixin._get_flattened_column_name(struct_name, name)]
            for name in field_names
            if FlatVersioningMixin._get_flattened_column_name(struct_name, name) in ibis_table.columns
        }
        if not struct_fields:
            return ibis_table
        return ibis_table.mutate(**{struct_name: ibis.struct(struct_fields)})

    def _add_struct_from_flattened_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        struct_name: str,
        *,
        field_names: list[str] | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        columns = list(schema.names())
        flat_cols = self._select_flattened_feature_columns(columns, struct_name)
        if not flat_cols:
            return df

        if field_names is not None:
            allowed = {FlatVersioningMixin._get_flattened_column_name(struct_name, name) for name in field_names}
            flat_cols = [col for col in flat_cols if col in allowed]
            if not flat_cols:
                return df

        struct_fields = [pl.col(col).alias(col.split("__", 1)[1]) for col in sorted(flat_cols)]
        return df.with_columns(pl.struct(struct_fields).alias(struct_name))

    def _restore_struct_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        struct_name: str,
        *,
        field_names: list[str],
    ) -> pl.DataFrame | pl.LazyFrame:
        if not field_names:
            # Nothing to restore without known fields.
            return df
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        columns = list(schema.names())
        flat_cols = self._select_flattened_feature_columns(columns, struct_name)

        if struct_name in columns:
            dtype = schema[struct_name]
            if isinstance(dtype, pl.Struct):
                return df
            if dtype in (pl.String, pl.Utf8):
                if flat_cols:
                    return self._add_struct_from_flattened_polars(df, struct_name, field_names=field_names)
                return self._decode_json_struct_polars(
                    df,
                    struct_name,
                    field_names=field_names,
                )
            return df

        if not flat_cols:
            return df
        return self._add_struct_from_flattened_polars(df, struct_name, field_names=field_names)

    def _expand_struct_to_flattened_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        struct_name: str,
        field_names: list[str],
    ) -> pl.DataFrame | pl.LazyFrame:
        if struct_name not in df.columns:
            return df
        columns = set(df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns)
        exprs = []
        for name in field_names:
            flattened = FlatVersioningMixin._get_flattened_column_name(struct_name, name)
            if flattened in columns:
                continue
            exprs.append(pl.col(struct_name).struct.field(name).alias(flattened))
        return df.with_columns(exprs) if exprs else df

    def _prepare_polars_json_write_frame(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        *,
        plan: FeaturePlan,
    ) -> pl.DataFrame:
        """Prepare Polars frames for JSON-compatible writes."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        field_names = self._get_field_names(plan, include_dependencies=False)
        result = self._restore_struct_polars(
            df,
            METAXY_PROVENANCE_BY_FIELD,
            field_names=field_names,
        )
        result = self._restore_struct_polars(
            result,
            METAXY_DATA_VERSION_BY_FIELD,
            field_names=field_names,
        )

        if METAXY_MATERIALIZATION_ID in result.columns:
            result = result.with_columns(pl.col(METAXY_MATERIALIZATION_ID).cast(pl.String))

        drop_cols = [
            col
            for col in result.columns
            if col.startswith(f"{METAXY_PROVENANCE_BY_FIELD}__") or col.startswith(f"{METAXY_DATA_VERSION_BY_FIELD}__")
        ]
        if drop_cols:
            result = result.drop(drop_cols)

        # Safe to cast since we collected above and _restore_struct_polars
        # preserves frame type
        return cast(pl.DataFrame, result)

    def _decode_json_struct_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        struct_name: str,
        *,
        field_names: list[str],
    ) -> pl.DataFrame | pl.LazyFrame:
        if not field_names:
            # Skip decoding when no fields are expected.
            return df
        struct_fields_dict = {name: pl.String for name in field_names}
        struct_dtype = pl.Struct(struct_fields_dict)
        return df.with_columns(pl.col(struct_name).str.json_decode(struct_dtype).alias(struct_name))

    def _ensure_ibis_lazy_frame(self, frame: nw.LazyFrame[Any] | nw.DataFrame[Any]) -> nw.LazyFrame[Any]:
        """Convert non-Ibis frames to Ibis LazyFrames (used to keep operations lazy)."""
        if isinstance(frame, nw.LazyFrame) and frame.implementation == nw.Implementation.IBIS:
            return frame

        import polars as pl

        native = frame.to_native() if isinstance(frame, (nw.DataFrame, nw.LazyFrame)) else frame
        if isinstance(native, pl.LazyFrame):
            native = native.collect()
        if isinstance(native, pl.DataFrame):
            mem = _prepare_ibis_memtable_from_polars(native)
            frame_nw = nw.from_native(mem, eager_only=False)
            # Ensure we return a LazyFrame
            if isinstance(frame_nw, nw.DataFrame):
                return frame_nw.lazy()
            return frame_nw

        if isinstance(frame, nw.LazyFrame):
            return frame
        if isinstance(frame, nw.DataFrame):
            return frame.lazy()
        return nw.from_native(frame).lazy()

    def _unpack_json_columns(
        self,
        lazy_frame: nw.LazyFrame[Any],
        feature_plan: FeaturePlan,
    ) -> nw.LazyFrame[Any]:
        """Unpack JSON columns to flattened columns using backend-specific functions."""

        # Convert to Ibis for backend-specific operations
        ibis_table: ibis_types.Table = lazy_frame.to_native()

        # Get field names from feature spec (no dependency keys in struct fields)
        field_names = self._get_field_names(feature_plan, include_dependencies=False)

        ibis_table = self._unpack_json_column(
            ibis_table,
            METAXY_PROVENANCE_BY_FIELD,
            field_names,
        )
        ibis_table = self._unpack_json_column(
            ibis_table,
            METAXY_DATA_VERSION_BY_FIELD,
            field_names,
        )

        # Ensure all expected flattened data_version fields exist even if JSON column missing
        data_version_field_columns = self._get_flattened_field_columns(METAXY_DATA_VERSION_BY_FIELD, field_names)
        for flattened_name in data_version_field_columns.values():
            if flattened_name in ibis_table.columns:
                continue
            provenance_flat = flattened_name.replace(METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD, 1)
            if provenance_flat in ibis_table.columns:
                ibis_table = ibis_table.mutate(**{flattened_name: ibis_table[provenance_flat]})
            elif METAXY_DATA_VERSION in ibis_table.columns:
                ibis_table = ibis_table.mutate(**{flattened_name: ibis_table[METAXY_DATA_VERSION]})
        for parent_spec in feature_plan.deps or []:
            parent_name = parent_spec.key.to_struct_key()
            flattened_name = FlatVersioningMixin._get_flattened_column_name(METAXY_DATA_VERSION_BY_FIELD, parent_name)
            if flattened_name in ibis_table.columns:
                continue
            provenance_flat = flattened_name.replace(METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD, 1)
            if provenance_flat in ibis_table.columns:
                ibis_table = ibis_table.mutate(**{flattened_name: ibis_table[provenance_flat]})
            elif METAXY_DATA_VERSION in ibis_table.columns:
                ibis_table = ibis_table.mutate(**{flattened_name: ibis_table[METAXY_DATA_VERSION]})

        ibis_table = self._add_struct_from_flattened(ibis_table, METAXY_PROVENANCE_BY_FIELD, field_names)
        ibis_table = self._add_struct_from_flattened(ibis_table, METAXY_DATA_VERSION_BY_FIELD, field_names)

        # Convert back to Narwhals (stays lazy)
        return nw.from_native(ibis_table, eager_only=False)

    def _pack_json_columns(
        self,
        df: nw.LazyFrame[Any] | nw.DataFrame[Any],
        feature_plan: FeaturePlan,
    ) -> nw.LazyFrame[Any] | nw.DataFrame[Any]:
        """Pack flattened columns into JSON using backend-specific functions."""

        prov_const = METAXY_PROVENANCE_BY_FIELD
        data_ver_const = METAXY_DATA_VERSION_BY_FIELD

        # Get field names from feature spec
        field_names = self._get_field_names(feature_plan, include_dependencies=False)

        # Convert to native once so we can inspect columns
        native_df = df.to_native()

        # If only struct columns are present, expand them to flattened columns first (Polars path)
        if df.implementation == nw.Implementation.POLARS:
            if METAXY_PROVENANCE_BY_FIELD in native_df.columns:
                native_df = self._expand_struct_to_flattened_polars(native_df, METAXY_PROVENANCE_BY_FIELD, field_names)
            if METAXY_DATA_VERSION_BY_FIELD in native_df.columns:
                native_df = self._expand_struct_to_flattened_polars(
                    native_df, METAXY_DATA_VERSION_BY_FIELD, field_names
                )

        # If it's a Polars DataFrame, convert to an Ibis memtable so we can pack JSON
        if df.implementation == nw.Implementation.POLARS:
            import polars as pl

            if isinstance(native_df, pl.LazyFrame):
                native_df = native_df.collect()
            pl_df = cast(pl.DataFrame, native_df)
            ibis_table = _prepare_ibis_memtable_from_polars(
                pl_df,
            )
        else:
            # Handle Ibis tables
            ibis_table = native_df

        # Ensure materialization_id has a concrete string type (postgres dislikes NULL-typed columns)
        if METAXY_MATERIALIZATION_ID in ibis_table.columns:
            ibis_table = ibis_table.mutate(
                **{METAXY_MATERIALIZATION_ID: ibis_table[METAXY_MATERIALIZATION_ID].cast("string")}
            )

        # Cast any NULL-typed flattened provenance/data_version columns to string for Postgres
        null_casts_specific = {
            col: ibis_table[col].cast("string")
            for col, dtype in ibis_table.schema().items()
            if dtype.is_null()
            and (
                col.startswith(f"{METAXY_PROVENANCE_BY_FIELD}__") or col.startswith(f"{METAXY_DATA_VERSION_BY_FIELD}__")
            )
        }
        if null_casts_specific:
            ibis_table = ibis_table.mutate(**null_casts_specific)

        # Catch-all for any remaining null-typed columns (e.g., auto-generated right-hand columns)
        null_casts_any = {
            col: ibis_table[col].cast("string") for col, dtype in ibis_table.schema().items() if dtype.is_null()
        }
        if null_casts_any:
            ibis_table = ibis_table.mutate(**null_casts_any)

        ibis_table = self._pack_json_column(ibis_table, prov_const, field_names)
        ibis_table = self._pack_json_column(ibis_table, data_ver_const, field_names)

        # Ensure no flattened columns leak into the final insert (DuckDB insert fails
        # with a column/value mismatch if both JSON + flattened columns are present).
        pruned_columns = [
            col
            for col in ibis_table.columns
            if not col.startswith(f"{prov_const}__") and not col.startswith(f"{data_ver_const}__")
        ]
        ibis_table = ibis_table.select(*[ibis_table[col] for col in pruned_columns])

        # Convert back to Narwhals
        return nw.from_native(ibis_table, eager_only=False)


def _prepare_polars_for_ibis(df: pl.DataFrame) -> pl.DataFrame:
    import json

    import polars as pl

    from metaxy.models.constants import METAXY_MATERIALIZATION_ID

    conversions: list[pl.Expr] = []
    for name, dtype in df.schema.items():
        if dtype == pl.Object:
            conversions.append(
                pl.col(name)
                .map_elements(
                    lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v,
                    return_dtype=pl.String,
                )
                .alias(name)
            )
        elif dtype == pl.Null:
            conversions.append(pl.col(name).cast(pl.String).alias(name))

    if METAXY_MATERIALIZATION_ID in df.columns:
        conversions.append(pl.col(METAXY_MATERIALIZATION_ID).cast(pl.String).alias(METAXY_MATERIALIZATION_ID))

    return df.with_columns(conversions) if conversions else df


def _jsonify_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    import json

    import polars as pl

    # TODO: Consider vectorized JSON encoding if this becomes a hotspot.
    conversions = [
        pl.col(col)
        .map_elements(
            lambda v: json.dumps(v) if v is not None else None,
            return_dtype=pl.String,
        )
        .alias(col)
        for col in columns
        if col in df.columns
    ]
    return df.with_columns(conversions) if conversions else df


def _prepare_ibis_memtable_from_polars(
    df: pl.DataFrame,
    *,
    json_columns: list[str] | None = None,
) -> ibis_types.Table:
    import ibis

    pl_df = _prepare_polars_for_ibis(df)
    if json_columns is None:
        json_columns = _select_existing_columns(
            pl_df,
            [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
        )
    pl_df = _jsonify_columns(pl_df, json_columns)
    return ibis.memtable(pl_df.to_arrow())


def _select_existing_columns(df: pl.DataFrame, columns: Sequence[str]) -> list[str]:
    return [col for col in columns if col in df.columns]
