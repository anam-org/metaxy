"""SQLite metadata store - thin wrapper around IbisMetadataStore."""

from pathlib import Path
from typing import TYPE_CHECKING

import ibis.expr.types as ir
import polars as pl

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.calculators.ibis import (
    HashSQLGenerator,
    IbisDataVersionCalculator,
)
from metaxy.data_versioning.diff.ibis import IbisDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.ibis import IbisJoiner
from metaxy.metadata_store.ibis import IbisMetadataStore


class SQLiteMetadataStore(IbisMetadataStore):
    """
    SQLite metadata store using Ibis backend.

    Convenience wrapper that configures IbisMetadataStore for SQLite.

    Hash algorithm support:
    - MD5: Available (built-in SQLite function via extension)

    Components:
        - joiner: IbisJoiner | PolarsJoiner (based on prefer_native)
        - calculator: IbisDataVersionCalculator | PolarsDataVersionCalculator
        - diff_resolver: IbisDiffResolver | PolarsDiffResolver

    Examples:
        >>> # Local file database
        >>> with SQLiteMetadataStore("metadata.db") as store:
        ...     store.write_metadata(MyFeature, df)

        >>> # In-memory database
        >>> with SQLiteMetadataStore(":memory:") as store:
        ...     store.write_metadata(MyFeature, df)

        >>> # Explicit path
        >>> store = SQLiteMetadataStore(Path("/path/to/metadata.db"))
        >>> with store:
        ...     store.write_metadata(MyFeature, df)
    """

    def __init__(
        self,
        database: str | Path,
        *,
        fallback_stores: list["MetadataStore"] | None = None,
        **kwargs,
    ):
        """
        Initialize SQLite metadata store.

        Args:
            database: Database connection string or path.
                - File path: "metadata.db" or Path("metadata.db")
                - In-memory: ":memory:"

                Note: Parent directories are NOT created automatically. Ensure paths exist
                before initializing the store.
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Passed to IbisMetadataStore (e.g., hash_algorithm, prefer_native)
        """
        database_str = str(database)

        # Build connection params for Ibis SQLite backend
        connection_params = {"database": database_str}

        self.database = database_str

        # Initialize Ibis store with SQLite backend
        super().__init__(
            backend="sqlite",
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for SQLite stores.

        Uses MD5 which is universally supported in SQLite.
        """
        return HashAlgorithm.MD5

    def _supports_native_components(self) -> bool:
        """SQLite stores do not support native components.

        SQLite doesn't have built-in hash functions (MD5, SHA256, etc.),
        so we always use Polars components for data versioning.
        """
        return False

    def _create_native_components(self):
        """Create SQLite-specific native components."""
        from metaxy.data_versioning.calculators.base import DataVersionCalculator
        from metaxy.data_versioning.diff.base import MetadataDiffResolver
        from metaxy.data_versioning.joiners.base import UpstreamJoiner

        if self._conn is None:
            raise RuntimeError(
                "Cannot create native components: store is not open. "
                "Ensure store is used as context manager."
            )

        import ibis.expr.types as ir

        joiner: UpstreamJoiner[ir.Table] = IbisJoiner(backend=self._conn)
        calculator: DataVersionCalculator[ir.Table] = IbisDataVersionCalculator(
            hash_sql_generators=self.hash_sql_generators,
        )

        diff_resolver: MetadataDiffResolver[ir.Table] = IbisDiffResolver()

        return joiner, calculator, diff_resolver

    @property
    def hash_sql_generators(self) -> dict[HashAlgorithm, HashSQLGenerator]:
        """
        Build hash SQL generators for SQLite.

        Returns:
            Dictionary mapping HashAlgorithm to SQL generator functions
        """
        generators = {HashAlgorithm.MD5: self._generate_hash_sql(HashAlgorithm.MD5)}

        return generators

    def _generate_hash_sql(self, algorithm: HashAlgorithm) -> HashSQLGenerator:
        """
        Generate SQL hash function for SQLite.

        Creates a hash SQL generator that produces SQLite-specific SQL for
        computing hash values on concatenated columns.

        Args:
            algorithm: Hash algorithm to generate SQL for

        Returns:
            Hash SQL generator function

        Raises:
            ValueError: If algorithm is not supported by SQLite
        """
        # Map algorithm to SQLite function name
        if algorithm == HashAlgorithm.MD5:
            hash_function = "md5"
        else:
            from metaxy.metadata_store.exceptions import (
                HashAlgorithmNotSupportedError,
            )

            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm {algorithm} is not supported by SQLite. "
                f"Supported algorithms: MD5 (built-in)"
            )

        def generator(table: ir.Table, concat_columns: dict[str, str]) -> str:
            # Build SELECT clause with hash columns
            hash_selects: list[str] = []
            for container_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{container_key}"
                # Use lower() for MD5 to ensure consistent hex output
                hash_expr = f"LOWER(HEX({hash_function}({concat_col})))"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        return generator

    def _serialize_for_storage(self, df: pl.DataFrame) -> pl.DataFrame:
        """Serialize structs and arrays to JSON strings for SQLite storage.

        SQLite doesn't support struct or array types, so we convert them to JSON strings.

        Args:
            df: DataFrame with potential struct/array columns

        Returns:
            DataFrame with struct/array columns converted to JSON strings
        """
        # Convert struct and array columns to JSON strings
        for col_name in df.columns:
            dtype = df.schema[col_name]
            if isinstance(dtype, pl.Struct):
                # Convert struct to JSON string
                df = df.with_columns(
                    pl.col(col_name).struct.json_encode().alias(col_name)
                )
            elif isinstance(dtype, pl.List):
                # Convert array/list to JSON string
                # Note: Polars doesn't have native list.json_encode(), so we use map_elements
                import json

                df = df.with_columns(
                    pl.col(col_name)
                    .map_elements(
                        lambda x: None if x is None else json.dumps(x),
                        return_dtype=pl.Utf8,
                    )
                    .alias(col_name)
                )

        return df

    def _deserialize_from_storage(self, df: pl.DataFrame) -> pl.DataFrame:
        """Deserialize JSON strings back to structs and arrays.

        Converts JSON string columns back to their original struct/array types.

        Args:
            df: DataFrame with JSON string columns

        Returns:
            DataFrame with JSON strings converted back to structs/arrays
        """
        # Known struct and array columns with their expected dtypes
        # data_version is a struct, containers is a list of structs
        # Migration system columns: operation_ids, expected_steps (list of strings),
        #                          migration_yaml (struct), affected_features (list of strings)

        # Columns that need JSON deserialization with specific dtypes
        json_columns = {
            "data_version": None,  # Infer from data
            "migration_yaml": None,  # Infer from data
            "feature_spec": None,  # Infer from data (FeatureSpec struct)
            "operation_ids": pl.List(pl.Utf8),  # List of strings
            "expected_steps": pl.List(pl.Utf8),  # List of strings
            "affected_features": pl.List(pl.Utf8),  # List of strings
        }

        # Deserialize JSON columns
        for col_name, dtype in json_columns.items():
            if col_name in df.columns and df.schema[col_name] == pl.Utf8:
                if len(df) > 0:
                    if dtype is None:
                        # Infer dtype from sample value
                        sample_value = df[col_name].drop_nulls().head(1)
                        if len(sample_value) > 0:
                            inferred_series = sample_value.str.json_decode()
                            inferred_dtype = inferred_series.dtype
                            df = df.with_columns(
                                pl.col(col_name)
                                .str.json_decode(dtype=inferred_dtype)
                                .alias(col_name)
                            )
                    else:
                        # Use provided dtype
                        df = df.with_columns(
                            pl.col(col_name)
                            .str.json_decode(dtype=dtype)
                            .alias(col_name)
                        )

        return df
