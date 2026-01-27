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
class PolarsIncrement(NamedTuple):
    """Like [`Increment`][metaxy.versioning.types.Increment], but converted to Polars frames."""

    added: pl.DataFrame
    changed: pl.DataFrame
    removed: pl.DataFrame


@public
@dataclass(kw_only=True)
class PolarsLazyIncrement:
    """Like [`LazyIncrement`][metaxy.versioning.types.LazyIncrement], but converted to Polars lazy frames.

    Attributes:
        added: New samples from upstream not present in current metadata.
        changed: Samples with different provenance.
        removed: Samples in current metadata but not in upstream state.
        input: Joined upstream metadata with [`FeatureDep`][metaxy.models.feature_spec.FeatureDep] rules applied.
    """

    added: pl.LazyFrame
    changed: pl.LazyFrame
    removed: pl.LazyFrame
    input: pl.LazyFrame | None = None

    def collect(self, **kwargs: Any) -> PolarsIncrement:
        """Collect into a [`PolarsIncrement`][metaxy.versioning.types.PolarsIncrement].

        !!! tip
            Leverages [`polars.collect_all`](https://docs.pola.rs/api/python/stable/reference/api/polars.collect_all.html)
            to optimize the collection process and take advantage of common subplan elimination.

        Args:
            **kwargs: backend-specific keyword arguments to pass to the collect method of the lazy frames.

        Returns:
            PolarsIncrement: The collected increment.
        """
        added, changed, removed = pl.collect_all([self.added, self.changed, self.removed], **kwargs)
        return PolarsIncrement(added, changed, removed)


@public
class Increment(NamedTuple):
    """Result of an incremental update containing eager dataframes.

    Contains three sets of samples:

    - added: New samples from upstream not present in current metadata

    - changed: Samples with different provenance

    - removed: Samples in current metadata but not in upstream state
    """

    added: nw.DataFrame[Any]
    changed: nw.DataFrame[Any]
    removed: nw.DataFrame[Any]

    def collect(self) -> "Increment":
        """Convenience method that's a no-op."""
        return self

    def to_polars(self) -> PolarsIncrement:
        """Convert to Polars."""
        return PolarsIncrement(
            added=self.added.to_polars(),
            changed=self.changed.to_polars(),
            removed=self.removed.to_polars(),
        )


@public
@dataclass(kw_only=True)
class LazyIncrement:
    """Result of an incremental update containing lazy dataframes.

    Attributes:
        added: New samples from upstream not present in current metadata.
        changed: Samples with different provenance.
        removed: Samples in current metadata but not in upstream state.
        input: Joined upstream metadata with [`FeatureDep`][metaxy.models.feature_spec.FeatureDep] rules applied.
    """

    added: nw.LazyFrame[Any]
    changed: nw.LazyFrame[Any]
    removed: nw.LazyFrame[Any]
    input: nw.LazyFrame[Any] | None = None

    def collect(self, **kwargs: Any) -> Increment:
        """Collect all lazy frames to eager DataFrames.

        !!! tip
            If all lazy frames are Polars frames, leverages
            [`polars.collect_all`](https://docs.pola.rs/api/python/stable/reference/api/polars.collect_all.html)
            to optimize the collection process and take advantage of common subplan elimination.

        Args:
            **kwargs: backend-specific keyword arguments to pass to the collect method of the lazy frames.

        Returns:
            Increment: The collected increment.
        """
        if (
            self.added.implementation
            == self.changed.implementation
            == self.removed.implementation
            == nw.Implementation.POLARS
        ):
            polars_eager_increment = PolarsLazyIncrement(
                added=self.added.to_native(),
                changed=self.changed.to_native(),
                removed=self.removed.to_native(),
            ).collect(**kwargs)
            return Increment(
                added=nw.from_native(polars_eager_increment.added),
                changed=nw.from_native(polars_eager_increment.changed),
                removed=nw.from_native(polars_eager_increment.removed),
            )
        else:
            return Increment(
                added=self.added.collect(**kwargs),
                changed=self.changed.collect(**kwargs),
                removed=self.removed.collect(**kwargs),
            )

    def to_polars(self) -> PolarsLazyIncrement:
        """Convert to Polars.

        !!! tip
            If the Narwhals lazy frames are already backed by Polars, this is a no-op.

        !!! warning
            If the Narwhals lazy frames are **not** backed by Polars, this will
            trigger a full materialization for them.
        """
        return PolarsLazyIncrement(
            added=lazy_frame_to_polars(self.added),
            changed=lazy_frame_to_polars(self.changed),
            removed=lazy_frame_to_polars(self.removed),
            input=lazy_frame_to_polars(self.input) if self.input is not None else None,
        )
