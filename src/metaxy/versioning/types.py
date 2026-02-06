"""Hash algorithms supported for field provenance calculation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import narwhals as nw
import polars as pl

from metaxy._decorators import public
from metaxy._utils import lazy_frame_to_polars


@public
class HashAlgorithm(Enum):
    """Supported hash algorithms for field provenance calculation.

    These algorithms are chosen for:
    - Speed (non-cryptographic hashes preferred)
    - Cross-database availability
    - Good collision resistance for field provenance calculation
    """

    XXHASH64 = "xxhash64"  # Fast, available in DuckDB, ClickHouse, Polars
    XXHASH32 = "xxhash32"  # Faster for small data, less collision resistant
    WYHASH = "wyhash"  # Very fast, Polars-specific
    SHA256 = "sha256"  # Cryptographic, slower, universally available
    MD5 = "md5"  # Legacy, widely available, not recommended for new code
    FARMHASH = "farmhash"  # Better than MD5, available in BigQuery


@public
class PolarsChanges(NamedTuple):
    """Like [`Changes`][metaxy.versioning.types.Changes], but converted to Polars frames.

    Attributes:
        new: New samples from upstream not present in current metadata
        stale: Samples with provenance different to what was processed before
        orphaned: Samples that have been processed before but are no longer present in upstream
    """

    new: pl.DataFrame
    stale: pl.DataFrame
    orphaned: pl.DataFrame


@public
@dataclass(kw_only=True)
class PolarsLazyChanges:
    """Like [`LazyChanges`][metaxy.versioning.types.LazyChanges], but converted to Polars lazy frames.

    Attributes:
        new: New samples from upstream not present in current metadata
        stale: Samples with provenance different to what was processed before
        orphaned: Samples that have been processed before but are no longer present in upstream
        input: Joined upstream metadata with [`FeatureDep`][metaxy.models.feature_spec.FeatureDep] rules applied.
    """

    new: pl.LazyFrame
    stale: pl.LazyFrame
    orphaned: pl.LazyFrame
    input: pl.LazyFrame | None = None

    def collect(self, **kwargs: Any) -> PolarsChanges:
        """Collect into a [`PolarsChanges`][metaxy.versioning.types.PolarsChanges].

        !!! tip
            Leverages [`polars.collect_all`](https://docs.pola.rs/api/python/stable/reference/api/polars.collect_all.html)
            to optimize the collection process and take advantage of common subplan elimination.

        Args:
            **kwargs: backend-specific keyword arguments to pass to the collect method of the lazy frames.

        Returns:
            PolarsChanges: The collected increment.
        """
        added, changed, removed = pl.collect_all([self.new, self.stale, self.orphaned], **kwargs)
        return PolarsChanges(added, changed, removed)  # ty: ignore[invalid-argument-type]


@public
class Changes(NamedTuple):
    """Result of an incremental update containing eager dataframes.

    Attributes:
        new: New samples from upstream not present in current metadata
        stale: Samples with provenance different to what was processed before
        orphaned: Samples that have been processed before but are no longer present in upstream
    """

    new: nw.DataFrame[Any]
    stale: nw.DataFrame[Any]
    orphaned: nw.DataFrame[Any]

    def collect(self) -> "Changes":
        """Convenience method that's a no-op."""
        return self

    def to_polars(self) -> PolarsChanges:
        """Convert to Polars."""
        return PolarsChanges(
            new=self.new.to_polars(),
            stale=self.stale.to_polars(),
            orphaned=self.orphaned.to_polars(),
        )


@public
@dataclass(kw_only=True)
class LazyChanges:
    """Result of an incremental update containing lazy dataframes.

    Attributes:
        new: New samples from upstream not present in current metadata
        stale: Samples with provenance different to what was processed before
        orphaned: Samples that have been processed before but are no longer present in upstream
        input: Joined upstream metadata with [`FeatureDep`][metaxy.models.feature_spec.FeatureDep] rules applied.
    """

    new: nw.LazyFrame[Any]
    stale: nw.LazyFrame[Any]
    orphaned: nw.LazyFrame[Any]
    input: nw.LazyFrame[Any] | None = None

    def collect(self, **kwargs: Any) -> Changes:
        """Collect all lazy frames to eager DataFrames.

        !!! tip
            If all lazy frames are Polars frames, leverages
            [`polars.collect_all`](https://docs.pola.rs/api/python/stable/reference/api/polars.collect_all.html)
            to optimize the collection process and take advantage of common subplan elimination.

        Args:
            **kwargs: backend-specific keyword arguments to pass to the collect method of the lazy frames.

        Returns:
            Changes: The collected increment.
        """
        if (
            self.new.implementation
            == self.stale.implementation
            == self.orphaned.implementation
            == nw.Implementation.POLARS
        ):
            polars_eager_increment = PolarsLazyChanges(
                new=self.new.to_native(),
                stale=self.stale.to_native(),
                orphaned=self.orphaned.to_native(),
            ).collect(**kwargs)
            return Changes(
                new=nw.from_native(polars_eager_increment.new),
                stale=nw.from_native(polars_eager_increment.stale),
                orphaned=nw.from_native(polars_eager_increment.orphaned),
            )
        else:
            return Changes(
                new=self.new.collect(**kwargs),
                stale=self.stale.collect(**kwargs),
                orphaned=self.orphaned.collect(**kwargs),
            )

    def to_polars(self) -> PolarsLazyChanges:
        """Convert to Polars.

        !!! tip
            If the Narwhals lazy frames are already backed by Polars, this is a no-op.

        !!! warning
            If the Narwhals lazy frames are **not** backed by Polars, this will
            trigger a full materialization for them.
        """
        return PolarsLazyChanges(
            new=lazy_frame_to_polars(self.new),
            stale=lazy_frame_to_polars(self.stale),
            orphaned=lazy_frame_to_polars(self.orphaned),
            input=lazy_frame_to_polars(self.input) if self.input is not None else None,
        )
