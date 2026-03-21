"""Run configuration for selective sample processing."""

from __future__ import annotations

import random
from enum import Enum

import dagster as dg
import polars as pl

# --8<-- [start:run_config]


class RunMode(Enum):
    """Controls which samples are processed."""

    FULL = "full"
    KEYED = "keyed"
    SUBSAMPLE = "subsample"


class RunConfig(dg.Config):
    """Configuration for a pipeline run."""

    mode: RunMode = RunMode.FULL
    sample_uids: list[str] | None = None
    subsample_fraction: float = 1.0
    subsample_seed: int = 42


# --8<-- [end:run_config]

# --8<-- [start:select_samples]


def select_samples(
    df: pl.DataFrame,
    config: RunConfig,
    uid_column: str = "sample_uid",
) -> pl.DataFrame:
    """Filter a DataFrame based on the run configuration."""
    if config.mode is RunMode.KEYED:
        if config.sample_uids is None:
            msg = "sample_uids required for KEYED mode"
            raise ValueError(msg)
        return df.filter(pl.col(uid_column).is_in(config.sample_uids))

    if config.mode is RunMode.SUBSAMPLE:
        all_uids = df[uid_column].to_list()
        rng = random.Random(config.subsample_seed)
        k = max(1, int(len(all_uids) * config.subsample_fraction))
        selected = rng.sample(all_uids, k=min(k, len(all_uids)))
        return df.filter(pl.col(uid_column).is_in(selected))

    return df


# --8<-- [end:select_samples]
