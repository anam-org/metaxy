"""Ibis-based field provenance calculator using native SQL hash functions.

This calculator uses Ibis to generate backend-specific SQL for hash computation,
executing entirely in the database without pulling data into memory.
"""

from typing import TYPE_CHECKING, Any, Protocol

import narwhals as nw

from metaxy.data_versioning.calculators.base import ProvenanceByFieldCalculator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.utils.hashing import get_hash_truncation_length

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types
    import ibis.expr.types.relations

    from metaxy.models.feature_spec import BaseFeatureSpecWithIDColumns
    from metaxy.models.plan import FeaturePlan


class HashSQLGenerator(Protocol):
    """Protocol for backend-specific hash SQL generation.

    Takes an Ibis table with concatenated columns and returns SQL that adds hash columns.
    """

    def __call__(
        self, table: "ibis.expr.types.Table", concat_columns: dict[str, str]
    ) -> str:
        """Generate SQL query to compute hash columns.

        Args:
            table: Input Ibis table with concatenated columns
            concat_columns: Maps field_key -> concat_column_name

        Returns:
            SQL query string that selects all columns plus hash columns
        """
        ...


class IbisProvenanceByFieldCalculator(ProvenanceByFieldCalculator):
    """Calculates provenance_by_field values using native SQL hash functions via Ibis.

    This calculator:
    1. Accepts Narwhals LazyFrame as input
    2. Converts to Ibis table internally
    3. Builds concatenated columns using Ibis expressions
    4. Applies backend-specific SQL hash functions
    5. Returns Narwhals LazyFrame

    Different SQL backends have different hash function names and signatures,
    so hash functions are provided as SQL template generators per backend.

    Example hash SQL generators:
        DuckDB: SELECT *, CAST(xxh64(concat_col) AS VARCHAR) as hash FROM table
        ClickHouse: SELECT *, CAST(xxHash64(concat_col) AS String) as hash FROM table
        PostgreSQL: SELECT *, MD5(concat_col) as hash FROM table
    """

    def __init__(
        self,
        backend: "ibis.BaseBackend",
        hash_sql_generators: dict[HashAlgorithm, HashSQLGenerator],
    ):
        """Initialize calculator with Ibis backend and hash SQL generators.

        Args:
            backend: Ibis backend connection for SQL execution
            hash_sql_generators: Map from HashAlgorithm to SQL generator function
        """
        self._backend = backend
        self._hash_sql_generators = hash_sql_generators

    @property
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """Algorithms supported by this calculator."""
        return list(self._hash_sql_generators.keys())

    @property
    def default_algorithm(self) -> HashAlgorithm:
        """Default hash algorithm.

        Base implementation returns XXHASH64 if available, otherwise first available.
        """
        if HashAlgorithm.XXHASH64 in self.supported_algorithms:
            return HashAlgorithm.XXHASH64
        return self.supported_algorithms[0]

    def calculate_provenance_by_field(
        self,
        joined_upstream: nw.LazyFrame[Any],
        feature_spec: "BaseFeatureSpecWithIDColumns",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> nw.LazyFrame[Any]:
        """Calculate provenance_by_field using SQL hash functions.

        Args:
            joined_upstream: Narwhals LazyFrame with upstream data joined
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_column_mapping: Maps upstream key -> column name
            hash_algorithm: Hash to use

        Returns:
            Narwhals LazyFrame with provenance_by_field column added
        """
        import ibis

        algo = hash_algorithm or self.default_algorithm

        if algo not in self.supported_algorithms:
            raise ValueError(
                f"Hash algorithm {algo} not supported by {self.__class__.__name__}. "
                f"Supported: {self.supported_algorithms}"
            )

        # Convert Narwhals LazyFrame to Ibis table
        import ibis.expr.types

        native = joined_upstream.to_native()

        # Validate that we have an Ibis table
        if not isinstance(native, ibis.expr.types.Table):
            # Not an Ibis table - this calculator only works with Ibis-backed data
            raise TypeError(
                f"IbisProvenanceByFieldCalculator requires Ibis-backed data. "
                f"Got {type(native)} instead. "
                f"This usually means the metadata store is not using Ibis tables. "
                f"Use PolarsProvenanceByFieldCalculator for non-Ibis stores."
            )

        ibis_table: ibis.expr.types.Table = native  # type: ignore[assignment]

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

            # Add upstream provenance values in deterministic order
            for upstream_feature_key in sorted(field_deps.keys()):
                upstream_fields = field_deps[upstream_feature_key]
                upstream_key_str = (
                    upstream_feature_key.to_string()
                    if hasattr(upstream_feature_key, "to_string")
                    else "__".join(upstream_feature_key)
                )

                provenance_col_name = upstream_column_mapping.get(
                    upstream_key_str, "provenance_by_field"
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
                        ibis_table[provenance_col_name][upstream_field_str]
                    )

            # Concatenate all components with separator
            concat_expr = components[0]
            for component in components[1:]:
                concat_expr = concat_expr.concat(ibis.literal("|")).concat(component)  # pyright: ignore[reportAttributeAccessIssue]

            # Store concat column for this field
            concat_col_name = f"__concat_{field_key_str}"
            concat_columns[field_key_str] = concat_col_name
            ibis_table = ibis_table.mutate(**{concat_col_name: concat_expr})

        # Generate SQL for hashing all concat columns
        hash_sql = hash_sql_gen(ibis_table, concat_columns)

        # Execute SQL to get table with hash columns
        result_table = self._backend.sql(hash_sql)  # pyright: ignore[reportAttributeAccessIssue]

        # Build provenance_by_field struct from hash columns
        hash_col_names = [f"__hash_{k}" for k in concat_columns.keys()]
        field_keys = list(concat_columns.keys())

        # Apply truncation if configured
        truncation_length = get_hash_truncation_length()
        if truncation_length is not None:
            # Apply substring to each hash column to truncate
            struct_fields = {}
            for field_key in field_keys:
                hash_col = result_table[f"__hash_{field_key}"]
                # Use substring to truncate the hash
                truncated_hash = hash_col.substr(0, truncation_length)
                struct_fields[field_key] = truncated_hash
        else:
            # Create struct column from hash columns (no truncation)
            struct_fields = {
                field_key: result_table[f"__hash_{field_key}"]
                for field_key in field_keys
            }

        # Drop temp columns and add provenance_by_field
        cols_to_keep = [
            c
            for c in result_table.columns
            if c not in concat_columns.values() and c not in hash_col_names
        ]

        result_table = result_table.select(
            *cols_to_keep, provenance_by_field=ibis.struct(struct_fields)
        )

        # Convert back to Narwhals LazyFrame
        return nw.from_native(result_table, eager_only=False)
