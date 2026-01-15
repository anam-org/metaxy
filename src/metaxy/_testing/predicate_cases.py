from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import narwhals as nw
import polars as pl


@dataclass(frozen=True)
class PredicateCase:
    name: str
    schema: nw.Schema
    exprs: list[nw.Expr]


def predicate_cases() -> list[PredicateCase]:
    schema = nw.from_native(
        pl.DataFrame(
            schema={
                "value": pl.Int64,
                "price": pl.Float64,
                "status": pl.Utf8,
                "ts_tz": pl.Datetime(time_zone="UTC"),
                "ts_naive": pl.Datetime(),
            }
        )
    ).collect_schema()

    return [
        PredicateCase(
            name="basic",
            schema=schema,
            exprs=[nw.col("price") > 5, nw.col("status") == "active"],
        ),
        PredicateCase(
            name="tz_timestamp",
            schema=schema,
            exprs=[nw.col("ts_tz") > datetime(2024, 1, 1, tzinfo=timezone.utc)],
        ),
        PredicateCase(
            name="naive_timestamp",
            schema=schema,
            exprs=[nw.col("ts_naive") > datetime(2024, 1, 1)],
        ),
        PredicateCase(
            name="is_in",
            schema=schema,
            exprs=[nw.col("status").is_in(["active", "pending"])],
        ),
        PredicateCase(
            name="range",
            schema=schema,
            exprs=[nw.col("value") >= 5, nw.col("value") < 8],
        ),
        PredicateCase(
            name="eq",
            schema=schema,
            exprs=[nw.col("value") == 3],
        ),
        PredicateCase(
            name="neq_and_gt",
            schema=schema,
            exprs=[nw.col("value") != 2, nw.col("value") > -1],
        ),
        PredicateCase(
            name="or_condition",
            schema=schema,
            exprs=[(nw.col("value") < 2) | (nw.col("value") > 7)],
        ),
        PredicateCase(
            name="tz_timestamp_and_price",
            schema=schema,
            exprs=[
                nw.col("price") > 5,
                nw.col("ts_tz") > datetime(2024, 1, 1, tzinfo=timezone.utc),
            ],
        ),
        PredicateCase(
            name="naive_timestamp_and_status",
            schema=schema,
            exprs=[
                nw.col("ts_naive") > datetime(2024, 1, 1),
                nw.col("status") == "active",
            ],
        ),
    ]
