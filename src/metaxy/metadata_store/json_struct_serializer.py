"""Helpers for JSON <-> flattened struct columns used by JSON-compatible stores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import narwhals as nw

from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.plan import FeaturePlan
from metaxy.versioning.flat_engine import FlatVersioningMixin


class JsonStructSerializerMixin:
    """Shared helpers for JSON-compatible metadata stores."""

    def _get_field_names(
        self,
        feature_plan: FeaturePlan,
        *,
        include_dependencies: bool,
    ) -> list[str]:
        names = [f.key.to_struct_key() for f in feature_plan.feature.fields]
        if include_dependencies:
            if feature_plan.feature_deps:
                names += [
                    dep.feature.to_struct_key() for dep in feature_plan.feature_deps
                ]
            if feature_plan.feature.deps:
                names += [
                    dep.feature.to_struct_key() for dep in feature_plan.feature.deps
                ]
        return names

    def _get_flattened_field_columns(
        self, struct_name: str, field_names: list[str]
    ) -> dict[str, str]:
        return {
            name: FlatVersioningMixin._get_flattened_column_name(struct_name, name)
            for name in field_names
        }

    def _ensure_ibis_lazy_frame(
        self, frame: nw.LazyFrame[Any] | nw.DataFrame[Any]
    ) -> nw.LazyFrame[Any]:
        """Convert non-Ibis frames to Ibis LazyFrames (used to keep operations lazy)."""
        if (
            isinstance(frame, nw.LazyFrame)
            and frame.implementation == nw.Implementation.IBIS
        ):
            return frame

        import json

        import ibis
        import polars as pl

        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        native = frame.to_native() if hasattr(frame, "to_native") else frame
        if isinstance(native, pl.LazyFrame):
            native = native.collect()
        if isinstance(native, pl.DataFrame):
            pdf = native.to_pandas()

            # Avoid Postgres null-typed columns and dict payloads that Arrow can't coerce
            for col in pdf.columns:
                series = pdf[col]
                mask = series.map(lambda v: isinstance(v, (dict, list)))
                mask_any: bool = bool(mask.any())  # Explicit bool for type checkers
                if series.dtype == object and mask_any:
                    pdf[col] = series.apply(
                        lambda v: json.dumps(v) if v is not None else None
                    )

            if METAXY_MATERIALIZATION_ID in pdf.columns:
                pdf[METAXY_MATERIALIZATION_ID] = pdf[METAXY_MATERIALIZATION_ID].astype(
                    "string"
                )

            mem = ibis.memtable(pdf)
            null_casts = {
                col: mem[col].cast("string")
                for col, dtype in mem.schema().items()
                if dtype.is_null()
            }
            if METAXY_MATERIALIZATION_ID in mem.columns:
                null_casts[METAXY_MATERIALIZATION_ID] = mem[
                    METAXY_MATERIALIZATION_ID
                ].cast("string")
            if null_casts:
                mem = mem.mutate(**null_casts)
            frame_nw = nw.from_native(mem, eager_only=False)
            # Ensure we return a LazyFrame
            if isinstance(frame_nw, nw.DataFrame):
                return frame_nw.lazy()
            return frame_nw

        # If it's already a Narwhals LazyFrame (but not Ibis), return as-is to avoid surprises
        if isinstance(frame, nw.LazyFrame):
            return frame

        return frame.lazy()  # type: ignore[attr-defined]

    def _ensure_struct_from_flattened(
        self, lazy_frame: nw.LazyFrame[Any], struct_name: str
    ) -> nw.LazyFrame[Any]:
        """Add a struct column from flattened columns if missing."""
        import ibis
        from ibis.expr import datatypes as dt

        ibis_table: ibis.expr.types.Table = lazy_frame.to_native()  # type: ignore[assignment]
        if struct_name in ibis_table.columns:
            # Rebuild if the column exists but isn't a struct
            if not isinstance(ibis_table.schema()[struct_name], dt.Struct):
                ibis_table = ibis_table.drop(struct_name)
            else:
                return lazy_frame

        prefix = f"{struct_name}__"
        struct_fields = {
            col.split("__", 1)[1]: ibis_table[col]
            for col in ibis_table.columns
            if col.startswith(prefix)
            and "__" in col
            and "__" not in col.split("__", 1)[1]
        }
        if struct_fields:
            ibis_table = ibis_table.mutate(**{struct_name: ibis.struct(struct_fields)})
        return nw.from_native(ibis_table, eager_only=False)

    def _unpack_json_columns(
        self,
        lazy_frame: nw.LazyFrame[Any],
        feature_plan: FeaturePlan,
    ) -> nw.LazyFrame[Any]:
        """Unpack JSON columns to flattened columns using backend-specific functions."""
        import ibis

        # Convert to Ibis for backend-specific operations
        ibis_table: ibis.expr.types.Table = lazy_frame.to_native()  # type: ignore[assignment]

        # Get field names from feature spec
        field_names = self._get_field_names(feature_plan, include_dependencies=True)

        # Unpack metaxy_provenance_by_field if it exists
        if METAXY_PROVENANCE_BY_FIELD in ibis_table.columns:
            unpack_exprs = self._get_json_unpack_exprs(
                METAXY_PROVENANCE_BY_FIELD,
                field_names,
            )

            # Add unpacked columns
            ibis_table = ibis_table.mutate(**unpack_exprs)

            # Drop original JSON column
            ibis_table = ibis_table.drop(METAXY_PROVENANCE_BY_FIELD)

        # Also unpack metaxy_data_version_by_field if it exists
        if METAXY_DATA_VERSION_BY_FIELD in ibis_table.columns:
            data_version_unpack_exprs = self._get_json_unpack_exprs(
                METAXY_DATA_VERSION_BY_FIELD,
                field_names,
            )
            ibis_table = ibis_table.mutate(**data_version_unpack_exprs)
            ibis_table = ibis_table.drop(METAXY_DATA_VERSION_BY_FIELD)

        # Ensure all expected flattened data_version fields exist even if JSON column missing
        data_version_field_columns = self._get_flattened_field_columns(
            METAXY_DATA_VERSION_BY_FIELD, field_names
        )
        for flattened_name in data_version_field_columns.values():
            if (
                flattened_name not in ibis_table.columns
                and METAXY_DATA_VERSION in ibis_table.columns
            ):
                ibis_table = ibis_table.mutate(
                    **{flattened_name: ibis_table[METAXY_DATA_VERSION]}
                )
        for parent_spec in feature_plan.deps or []:
            parent_name = parent_spec.key.to_struct_key()
            flattened_name = FlatVersioningMixin._get_flattened_column_name(
                METAXY_DATA_VERSION_BY_FIELD, parent_name
            )
            if (
                flattened_name not in ibis_table.columns
                and METAXY_DATA_VERSION in ibis_table.columns
            ):
                ibis_table = ibis_table.mutate(
                    **{flattened_name: ibis_table[METAXY_DATA_VERSION]}
                )

        # Rebuild struct columns from flattened fields for consumers that expect them
        if METAXY_PROVENANCE_BY_FIELD not in ibis_table.columns:
            prov_struct_fields = {
                name: ibis_table[
                    FlatVersioningMixin._get_flattened_column_name(
                        METAXY_PROVENANCE_BY_FIELD, name
                    )
                ]
                for name in field_names
                if FlatVersioningMixin._get_flattened_column_name(
                    METAXY_PROVENANCE_BY_FIELD, name
                )
                in ibis_table.columns
            }
            if prov_struct_fields:
                ibis_table = ibis_table.mutate(
                    **{METAXY_PROVENANCE_BY_FIELD: ibis.struct(prov_struct_fields)}
                )

        if METAXY_DATA_VERSION_BY_FIELD not in ibis_table.columns:
            data_struct_fields = {
                name: ibis_table[
                    FlatVersioningMixin._get_flattened_column_name(
                        METAXY_DATA_VERSION_BY_FIELD, name
                    )
                ]
                for name in field_names
                if FlatVersioningMixin._get_flattened_column_name(
                    METAXY_DATA_VERSION_BY_FIELD, name
                )
                in ibis_table.columns
            }
            if data_struct_fields:
                ibis_table = ibis_table.mutate(
                    **{METAXY_DATA_VERSION_BY_FIELD: ibis.struct(data_struct_fields)}
                )

        # Convert back to Narwhals (stays lazy)
        return nw.from_native(ibis_table, eager_only=False)

    def _pack_json_columns(
        self,
        df: nw.LazyFrame[Any] | nw.DataFrame[Any],
        feature_plan: FeaturePlan,
    ) -> nw.LazyFrame[Any] | nw.DataFrame[Any]:
        """Pack flattened columns into JSON using backend-specific functions."""
        import ibis

        prov_const = METAXY_PROVENANCE_BY_FIELD
        data_ver_const = METAXY_DATA_VERSION_BY_FIELD

        # Get field names from feature spec
        field_names = self._get_field_names(feature_plan, include_dependencies=False)

        # Convert to native once so we can inspect columns
        native_df = df.to_native()

        # If only struct columns are present, expand them to flattened columns first (Polars path)
        if df.implementation == nw.Implementation.POLARS:
            import polars as pl

            if METAXY_PROVENANCE_BY_FIELD in native_df.columns:
                native_df = native_df.with_columns(
                    [
                        pl.col(METAXY_PROVENANCE_BY_FIELD)
                        .struct.field(name)
                        .alias(
                            FlatVersioningMixin._get_flattened_column_name(
                                METAXY_PROVENANCE_BY_FIELD, name
                            )
                        )
                        for name in field_names
                        if FlatVersioningMixin._get_flattened_column_name(
                            METAXY_PROVENANCE_BY_FIELD, name
                        )
                        not in native_df.columns
                    ]
                )

            if METAXY_DATA_VERSION_BY_FIELD in native_df.columns:
                native_df = native_df.with_columns(
                    [
                        pl.col(METAXY_DATA_VERSION_BY_FIELD)
                        .struct.field(name)
                        .alias(
                            FlatVersioningMixin._get_flattened_column_name(
                                METAXY_DATA_VERSION_BY_FIELD, name
                            )
                        )
                        for name in field_names
                        if FlatVersioningMixin._get_flattened_column_name(
                            METAXY_DATA_VERSION_BY_FIELD, name
                        )
                        not in native_df.columns
                    ]
                )

        # If it's a Polars DataFrame, convert to an Ibis memtable so we can pack JSON
        if df.implementation == nw.Implementation.POLARS:
            import json

            pdf = native_df.to_pandas()

            # Avoid NULL-typed columns in memtable: force all-null object columns to string dtype
            for col in pdf.columns:
                if pdf[col].dtype == object and pdf[col].isnull().all():
                    pdf[col] = pdf[col].astype("string")

            # Convert dict columns to JSON strings to avoid struct types in memtable
            for col in [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD]:
                if col in pdf.columns:
                    pdf[col] = pdf[col].apply(
                        lambda v: json.dumps(v) if v is not None else None
                    )

            if METAXY_MATERIALIZATION_ID in pdf.columns:
                pdf[METAXY_MATERIALIZATION_ID] = pdf[METAXY_MATERIALIZATION_ID].astype(
                    "string"
                )

            if (
                METAXY_DATA_VERSION_BY_FIELD in pdf.columns
                and pdf[METAXY_DATA_VERSION_BY_FIELD].dtype == object
            ):
                pdf[METAXY_DATA_VERSION_BY_FIELD] = pdf[
                    METAXY_DATA_VERSION_BY_FIELD
                ].apply(lambda v: json.dumps(v) if v is not None else None)

            if (
                METAXY_PROVENANCE_BY_FIELD in pdf.columns
                and pdf[METAXY_PROVENANCE_BY_FIELD].dtype == object
            ):
                pdf[METAXY_PROVENANCE_BY_FIELD] = pdf[METAXY_PROVENANCE_BY_FIELD].apply(
                    lambda v: json.dumps(v) if v is not None else None
                )

            ibis_table = ibis.memtable(pdf)
        else:
            # Handle Ibis tables
            ibis_table = native_df  # type: ignore[assignment]

        # Ensure materialization_id has a concrete string type (postgres dislikes NULL-typed columns)
        if METAXY_MATERIALIZATION_ID in ibis_table.columns:
            ibis_table = ibis_table.mutate(
                **{
                    METAXY_MATERIALIZATION_ID: ibis_table[
                        METAXY_MATERIALIZATION_ID
                    ].cast("string")
                }
            )

        # Cast any NULL-typed flattened provenance/data_version columns to string for Postgres
        null_casts_specific = {
            col: ibis_table[col].cast("string")
            for col, dtype in ibis_table.schema().items()
            if dtype.is_null()
            and (
                col.startswith(f"{METAXY_PROVENANCE_BY_FIELD}__")
                or col.startswith(f"{METAXY_DATA_VERSION_BY_FIELD}__")
            )
        }
        if null_casts_specific:
            ibis_table = ibis_table.mutate(**null_casts_specific)

        # Catch-all for any remaining null-typed columns (e.g., auto-generated right-hand columns)
        null_casts_any = {
            col: ibis_table[col].cast("string")
            for col, dtype in ibis_table.schema().items()
            if dtype.is_null()
        }
        if null_casts_any:
            ibis_table = ibis_table.mutate(**null_casts_any)

        # Pack metaxy_provenance_by_field if flattened columns exist
        prov_field_columns = self._get_flattened_field_columns(prov_const, field_names)

        has_prov_fields = any(
            col in ibis_table.columns for col in prov_field_columns.values()
        )

        if has_prov_fields:
            pack_expr = self._get_json_pack_expr(
                prov_const,
                prov_field_columns,
            )
            base_cols = [
                col
                for col in ibis_table.columns
                if col not in prov_field_columns.values()
            ]
            ibis_table = ibis_table.select(
                *[ibis_table[col] for col in base_cols],
                pack_expr.name(prov_const),
            )

        # Also pack metaxy_data_version_by_field if flattened columns exist
        data_version_field_columns = self._get_flattened_field_columns(
            data_ver_const, field_names
        )

        has_data_version_fields = any(
            col in ibis_table.columns for col in data_version_field_columns.values()
        )

        if has_data_version_fields:
            data_version_pack_expr = self._get_json_pack_expr(
                data_ver_const,
                data_version_field_columns,
            )
            base_cols = [
                col
                for col in ibis_table.columns
                if col not in data_version_field_columns.values()
            ]
            ibis_table = ibis_table.select(
                *[ibis_table[col] for col in base_cols],
                data_version_pack_expr.name(data_ver_const),
            )

        # Ensure no flattened columns leak into the final insert (DuckDB insert fails
        # with a column/value mismatch if both JSON + flattened columns are present).
        pruned_columns = [
            col
            for col in ibis_table.columns
            if not col.startswith(f"{prov_const}__")
            and not col.startswith(f"{data_ver_const}__")
        ]
        ibis_table = ibis_table.select(*[ibis_table[col] for col in pruned_columns])

        # Convert back to Narwhals
        return nw.from_native(ibis_table, eager_only=False)

    # The following abstract methods must be provided by concrete stores
    def _get_json_unpack_exprs(
        self,
        json_column: str,
        field_names: list[str],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def _get_json_pack_expr(
        self,
        struct_name: str,
        field_columns: Mapping[str, str],
    ) -> Any:
        raise NotImplementedError
