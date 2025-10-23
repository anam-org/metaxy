"""Ibis implementation of data version calculator.

Provides native SQL-based hash calculation using backend-specific SQL functions.
Hash functions are provided via SQL templates that can be customized per backend.
"""

from typing import TYPE_CHECKING, Protocol

import ibis
import ibis.expr.types as ir

from metaxy.data_versioning.calculators.base import DataVersionCalculator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


class HashSQLGenerator(Protocol):
    """Protocol for hash SQL generation functions.

    Takes a table and mapping of field keys to concat column names,
    returns SQL query that adds hash columns to the table.
    """

    def __call__(self, table: ir.Table, concat_columns: dict[str, str]) -> str:
        """Generate SQL query to compute hash columns.

        Args:
            table: Input Ibis table with concatenated columns
            concat_columns: Maps field_key -> concat_column_name

        Returns:
            SQL query string that selects all columns plus hash columns
        """
        ...


class IbisDataVersionCalculator(DataVersionCalculator[ir.Table]):
    """Calculates data versions for Ibis tables using SQL hash functions.

    Type Parameters:
        TRef = ir.Table

    This calculator uses raw SQL to apply hash functions, since different SQL
    backends have different hash function names and signatures. Hash functions
    are provided as SQL template generators that can be customized per backend.

    The calculator works by:
    1. Building concatenated string columns for each field (using Ibis)
    2. Writing the table to a temp table
    3. Applying SQL hash functions via conn.sql()
    4. Returning result as Ibis table

    Example SQL generators:
        DuckDB: lambda tbl, col: f"SELECT *, md5({col}) as hash FROM {{table}}"
        PostgreSQL: lambda tbl, col: f"SELECT *, md5({col}) as hash FROM {{table}}"
    """

    def __init__(
        self,
        hash_sql_generators: dict[HashAlgorithm, HashSQLGenerator],
    ):
        """Initialize Ibis calculator with backend-specific hash functions.

        Args:
            hash_sql_generators: Map from HashAlgorithm to SQL generator function
        """
        self._hash_sql_generators = hash_sql_generators

    @property
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """Algorithms supported by this calculator."""
        return list(self._hash_sql_generators.keys())

    @property
    def default_algorithm(self) -> HashAlgorithm:
        """Default hash algorithm.

        Subclasses can override this with backend-specific defaults.
        Base implementation returns XXHASH64 for performance.
        """
        return HashAlgorithm.XXHASH64

    def calculate_data_versions(
        self,
        joined_upstream: ir.Table,
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> ir.Table:
        """Calculate data_version using SQL hash functions.

        Args:
            joined_upstream: Ibis table with upstream data joined
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_column_mapping: Maps upstream key -> column name
            hash_algorithm: Hash to use

        Returns:
            Ibis table with data_version column added
        """
        algo = hash_algorithm or self.default_algorithm

        if algo not in self.supported_algorithms:
            raise ValueError(
                f"Hash algorithm {algo} not supported by {self.__class__.__name__}. "
                f"Supported: {self.supported_algorithms}"
            )

        # Get the hash SQL generator
        hash_sql_gen = self._hash_sql_generators[algo]

        # Build concatenated string columns for each field (using Ibis expressions)
        concat_columns = {}

        for field in feature_spec.fields:
            field_key_str = (
                field.key.to_string()
                if hasattr(field.key, "to_string")
                else "__".join(field.key)
            )

            field_deps = feature_plan.field_dependencies.get(field.key, {})

            # Build hash components (same structure as Polars)
            components = [
                ibis.literal(field_key_str),
                ibis.literal(str(field.code_version)),
            ]

            # Add upstream data versions in deterministic order
            for upstream_feature_key in sorted(field_deps.keys()):
                upstream_fields = field_deps[upstream_feature_key]
                upstream_key_str = (
                    upstream_feature_key.to_string()
                    if hasattr(upstream_feature_key, "to_string")
                    else "__".join(upstream_feature_key)
                )

                data_version_col_name = upstream_column_mapping.get(
                    upstream_key_str, "data_version"
                )

                for upstream_field in sorted(upstream_fields):
                    upstream_field_str = (
                        upstream_field.to_string()
                        if hasattr(upstream_field, "to_string")
                        else "__".join(upstream_field)
                    )

                    components.append(
                        ibis.literal(f"{upstream_key_str}/{upstream_field_str}")
                    )
                    # Access struct field for upstream field's hash
                    components.append(
                        joined_upstream[data_version_col_name][upstream_field_str]
                    )

            # Concatenate all components with separator
            concat_expr = components[0]
            for component in components[1:]:
                concat_expr = concat_expr.concat(ibis.literal("|")).concat(component)  # type: ignore[attr-defined]

            # Store concat column for this field
            concat_col_name = f"__concat_{field_key_str}"
            concat_columns[field_key_str] = concat_col_name
            joined_upstream = joined_upstream.mutate(**{concat_col_name: concat_expr})

        # Now apply hash functions via SQL
        # Get the backend connection
        backend = joined_upstream._find_backend()  # type: ignore[attr-defined]

        # Generate SQL for hashing all concat columns
        hash_sql = hash_sql_gen(joined_upstream, concat_columns)

        # Execute SQL to get table with hash columns
        result_table = backend.sql(hash_sql)  # type: ignore[attr-defined]

        # Build data_version struct from hash columns in a single select
        # This avoids the IntegrityError from trying to reference columns from result_table
        hash_col_names = [f"__hash_{k}" for k in concat_columns.keys()]
        field_keys = list(concat_columns.keys())

        # Create struct column from hash columns
        struct_fields = {
            field_key: result_table[f"__hash_{field_key}"] for field_key in field_keys
        }

        # Drop temp columns and add data_version in one select
        cols_to_keep = [
            c
            for c in result_table.columns
            if c not in concat_columns.values() and c not in hash_col_names
        ]

        result_table = result_table.select(
            *cols_to_keep, data_version=ibis.struct(struct_fields)
        )

        return result_table
