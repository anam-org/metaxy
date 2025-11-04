"""BigQuery metadata store - thin wrapper around IbisMetadataStore."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.ibis import IbisMetadataStore


class BigQueryMetadataStore(IbisMetadataStore):
    """
    [BigQuery](https://cloud.google.com/bigquery) metadata store using [Ibis](https://ibis-project.org/) backend.

    Warning:
        It's on the user to set up infrastructure for Metaxy correctly.
        Make sure to have large tables partitioned as appropriate for your use case.

    Note:
        BigQuery automatically optimizes queries on partitioned tables.
        When tables are partitioned (e.g., by date or ingestion time with _PARTITIONTIME), BigQuery will
        automatically prune partitions based on WHERE clauses in queries, without needing
        explicit configuration in the metadata store.
        Make sure to use appropriate `filters` when calling [BigQueryMetadataStore.read_metadata][metaxy.metadata_store.bigquery.BigQueryMetadataStore.read_metadata].

    Example: Basic Connection
        ```py
        store = BigQueryMetadataStore(
            project_id="my-project",
            dataset_id="my_dataset",
        )
        ```

    Example: With Service Account
        ```py
        store = BigQueryMetadataStore(
            project_id="my-project",
            dataset_id="my_dataset",
            credentials_path="/path/to/service-account.json",
        )
        ```

    Example: With Location Configuration
        ```py
        store = BigQueryMetadataStore(
            project_id="my-project",
            dataset_id="my_dataset",
            location="EU",  # Specify data location
        )
        ```

    Example: With Custom Hash Algorithm
        ```py
        store = BigQueryMetadataStore(
            project_id="my-project",
            dataset_id="my_dataset",
            hash_algorithm=HashAlgorithm.SHA256,  # Use SHA256 instead of default FARMHASH
        )
        ```
    """

    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str | None = None,
        *,
        credentials_path: str | None = None,
        credentials: Any | None = None,
        location: str | None = None,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list["MetadataStore"] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize [BigQuery](https://cloud.google.com/bigquery) metadata store.

        Args:
            project_id: Google Cloud project ID containing the dataset.
                Can also be set via GOOGLE_CLOUD_PROJECT environment variable.
            dataset_id: BigQuery dataset name for storing metadata tables.
                If not provided, uses the default dataset for the project.
            credentials_path: Path to service account JSON file.
                Alternative to passing credentials object directly.
            credentials: Google Cloud credentials object.
                If not provided, uses default credentials from environment.
            location: Default location for BigQuery resources (e.g., "US", "EU").
                If not specified, BigQuery determines based on dataset location.
            connection_params: Additional Ibis BigQuery connection parameters.
                Overrides individual parameters if provided.
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Passed to [metaxy.metadata_store.ibis.IbisMetadataStore][]

        Raises:
            ImportError: If ibis-bigquery not installed
            ValueError: If neither project_id nor connection_params provided

        Note:
            Authentication priority:
            1. Explicit credentials or credentials_path
            2. Application Default Credentials (ADC)
            3. Google Cloud SDK credentials

            BigQuery automatically handles partition pruning when querying partitioned tables.
            If your tables are partitioned (e.g., by date or ingestion time), BigQuery will
            automatically optimize queries with appropriate WHERE clauses on the partition column.

        Example:
            ```py
            # Using environment authentication
            store = BigQueryMetadataStore(
                project_id="my-project",
                dataset_id="ml_metadata",
            )

            # Using service account
            store = BigQueryMetadataStore(
                project_id="my-project",
                dataset_id="ml_metadata",
                credentials_path="/path/to/key.json",
            )

            # With location specification
            store = BigQueryMetadataStore(
                project_id="my-project",
                dataset_id="ml_metadata",
                location="EU",
            )
            ```
        """
        # Build connection parameters if not provided
        if connection_params is None:
            connection_params = self._build_connection_params(
                project_id=project_id,
                dataset_id=dataset_id,
                credentials_path=credentials_path,
                credentials=credentials,
                location=location,
            )

        # Validate we have minimum required parameters
        if "project_id" not in connection_params and project_id is None:
            raise ValueError(
                "Must provide either project_id or connection_params with project_id. "
                "Example: project_id='my-project'"
            )

        # Store parameters for display
        self.project_id = project_id or connection_params.get("project_id")
        self.dataset_id = dataset_id or connection_params.get("dataset_id", "")

        # Initialize Ibis store with BigQuery backend
        super().__init__(
            backend="bigquery",
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    def _build_connection_params(
        self,
        project_id: str | None = None,
        dataset_id: str | None = None,
        credentials_path: str | None = None,
        credentials: Any | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Build connection parameters for Ibis BigQuery backend.

        This method centralizes the authentication logic, supporting:
        1. Explicit service account file (credentials_path)
        2. Explicit credentials object
        3. Application Default Credentials (automatic fallback)

        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset name
            credentials_path: Path to service account JSON file
            credentials: Pre-loaded credentials object
            location: BigQuery resource location

        Returns:
            Dictionary of connection parameters for Ibis
        """
        connection_params: dict[str, Any] = {}

        # Set core BigQuery parameters
        if project_id is not None:
            connection_params["project_id"] = project_id
        if dataset_id is not None:
            connection_params["dataset_id"] = dataset_id
        if location is not None:
            connection_params["location"] = location

        # Handle authentication - prioritize explicit credentials
        if credentials_path is not None:
            connection_params["credentials"] = self._load_service_account_credentials(
                credentials_path
            )
        elif credentials is not None:
            connection_params["credentials"] = credentials
        # Otherwise, Ibis will automatically use Application Default Credentials

        return connection_params

    def _load_service_account_credentials(self, credentials_path: str) -> Any:
        """Load service account credentials from a JSON file.

        Uses Google's recommended approach with google.oauth2.service_account
        instead of manually parsing JSON and constructing credentials.

        Args:
            credentials_path: Path to service account JSON file

        Returns:
            Google Cloud credentials object

        Raises:
            ImportError: If google-auth library not installed
            FileNotFoundError: If credentials file doesn't exist
            ValueError: If credentials file is invalid
        """
        try:
            from google.oauth2 import (
                service_account,  # pyright: ignore[reportMissingImports]
            )
        except ImportError as e:
            raise ImportError(
                "Google Cloud authentication libraries required for service account credentials. "
                "Install with: pip install google-auth"
            ) from e

        try:
            # Use Google's recommended method - it handles all edge cases
            return service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Service account credentials file not found: {credentials_path}"
            ) from e
        except Exception as e:
            # Catch JSON decode errors and other credential format issues
            raise ValueError(
                f"Invalid service account credentials file: {credentials_path}. "
                "Ensure it's a valid service account JSON key file."
            ) from e

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        # Should switch to FARM_FINGERPRINT64 once https://github.com/ion-elgreco/polars-hash/issues/49 is resolved
        return HashAlgorithm.MD5

    def _supports_native_components(self) -> bool:
        """BigQuery stores support native field provenance calculations when connection is open."""
        return self._conn is not None

    def _get_hash_sql_generators(self) -> dict[HashAlgorithm, "HashSQLGenerator"]:
        """Get hash SQL generators for BigQuery.

        BigQuery supports:
        - FARMHASH: Fast FARM_FINGERPRINT() function returning INT64 (BigQuery-specific)
        - MD5: Built-in MD5() function returning hex string
        - SHA256: Built-in SHA256() function returning bytes (needs TO_HEX)

        Returns:
            Dictionary mapping HashAlgorithm to SQL generator functions
        """

        def farmhash_generator(table, concat_columns: dict[str, str]) -> str:
            """Generate SQL to compute FARMHASH in BigQuery.

            BigQuery's FARM_FINGERPRINT() returns an INT64 value.
            We cast it to STRING for consistent string-based hashing across stores.
            """
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                # FARM_FINGERPRINT() returns INT64, cast to STRING for consistency
                hash_expr = f"CAST(FARM_FINGERPRINT({concat_col}) AS STRING)"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        def md5_generator(table, concat_columns: dict[str, str]) -> str:
            """Generate SQL to compute MD5 hashes in BigQuery.

            BigQuery's MD5() returns a hex string directly (lowercase).
            """
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                # MD5() in BigQuery returns hex string directly
                hash_expr = f"MD5({concat_col})"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        def sha256_generator(table, concat_columns: dict[str, str]) -> str:
            """Generate SQL to compute SHA256 hashes in BigQuery.

            BigQuery's SHA256() returns bytes, so we convert to hex string.
            """
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                # SHA256() returns bytes, convert to hex string
                hash_expr = f"TO_HEX(SHA256({concat_col}))"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        result = {
            HashAlgorithm.FARMHASH: farmhash_generator,
            HashAlgorithm.MD5: md5_generator,
            HashAlgorithm.SHA256: sha256_generator,
        }

        return result

    def display(self) -> str:
        """Display string for this store."""
        dataset_info = f"/{self.dataset_id}" if self.dataset_id else ""
        return f"BigQueryMetadataStore(project={self.project_id}{dataset_info})"
