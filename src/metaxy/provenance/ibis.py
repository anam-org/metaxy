"""Ibis implementation of ProvenanceTracker.

CRITICAL: This implementation NEVER materializes lazy expressions.
All operations stay in the lazy Ibis world for SQL execution.
"""

from collections.abc import Callable
from typing import Any, cast

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.plan import FeaturePlan
from metaxy.provenance.tracker import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm
from ibis import Expr as IbisExpr
from typing import Protocol


class IbisHashFn(Protocol):
    def __call__(self, expr: IbisExpr) -> IbisExpr:
        ...


class IbisProvenanceTracker(ProvenanceTracker):
    """Provenance tracker using Ibis for SQL databases.

    Only implements hash_string_column and build_struct_column.
    All logic lives in the base class.

    CRITICAL: This implementation NEVER leaves the lazy world.
    All operations stay as Ibis expressions that compile to SQL.
    """

    def __init__(
        self,
        plan: FeaturePlan,
        backend: Any,  # ibis.BaseBackend
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        """Initialize the Ibis tracker.

        Args:
            plan: Feature plan to track provenance for
            backend: Ibis backend instance (e.g., ibis.duckdb.connect())
            hash_functions: Mapping from HashAlgorithm to Ibis hash functions.
                Each function takes an Ibis expression and returns an Ibis expression.
        """
        super().__init__(plan)
        self.backend = backend
        self.hash_functions: dict[HashAlgorithm, IbisHashFn] = hash_functions

    def hash_string_column(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
    ) -> FrameT:
        """Hash a string column using Ibis hash functions.

        Args:
            df: Narwhals DataFrame backed by Ibis
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use

        Returns:
            Narwhals DataFrame with new hashed column added, backed by Ibis.
            The source column remains unchanged.
        """
        if hash_algo not in self.hash_functions:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported by this Ibis backend. "
                f"Supported: {list(self.hash_functions.keys())}"
            )

        # Import ibis lazily (module-level import restriction)
        import ibis
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Get hash function
        hash_fn = self.hash_functions[hash_algo]

        # Apply hash to source column
        # Hash functions are responsible for returning strings
        hashed = hash_fn(ibis_table[source_column])

        # Add new column with the hash
        result_table = ibis_table.mutate(**{target_column: hashed})

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    def build_struct_column(
        self,
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build a struct column from existing columns.

        Args:
            df: Narwhals DataFrame backed by Ibis
            struct_name: Name for the new struct column
            field_columns: Mapping from struct field names to column names

        Returns:
            Narwhals DataFrame with new struct column added, backed by Ibis.
            The source columns remain unchanged.
        """
        # Import ibis lazily
        import ibis
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Build struct expression - reference columns by name
        struct_expr = ibis.struct(
            {
                field_name: ibis_table[col_name]
                for field_name, col_name in field_columns.items()
            }
        )

        # Add struct column
        result_table = ibis_table.mutate(**{struct_name: struct_expr})

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))
