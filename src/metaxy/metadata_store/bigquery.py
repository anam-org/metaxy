"""BigQuery metadata store - thin wrapper around IbisMetadataStore."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.ibis import IbisMetadataStore, IbisMetadataStoreConfig
from metaxy.versioning.types import HashAlgorithm


class BigQueryMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for BigQueryMetadataStore.

    Example:
        <!-- skip next -->
        ```python
        config = BigQueryMetadataStoreConfig(
            project_id="my-project",
            dataset_id="my_dataset",
            credentials_path="/path/to/service-account.json",
        )

        store = BigQueryMetadataStore.from_config(config)
        ```
    """

    project_id: str | None = Field(default=None, description="Google Cloud project ID containing the dataset.")
    dataset_id: str | None = Field(default=None, description="BigQuery dataset name for storing metadata tables.")
    credentials_path: str | None = Field(default=None, description="Path to service account JSON file.")
    credentials: Any | None = Field(default=None, description="Google Cloud credentials object.")
    location: str | None = Field(
        default=None,
        description="Default location for BigQuery resources (e.g., 'US', 'EU').",
    )


@public
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
        <!-- skip next -->
        ```py
        store = BigQueryMetadataStore(
            project_id="my-project",
            dataset_id="my_dataset",
        )
        ```

    Example: With Service Account
        <!-- skip next -->
        ```py
        store = BigQueryMetadataStore(
            project_id="my-project",
            dataset_id="my_dataset",
            credentials_path="/path/to/service-account.json",
        )
        ```

    Example: With Location Configuration
        <!-- skip next -->
        ```py
        store = BigQueryMetadataStore(
            project_id="my-project",
            dataset_id="my_dataset",
            location="EU",  # Specify data location
        )
        ```

    Example: With Custom Hash Algorithm
        <!-- skip next -->
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
            <!-- skip next -->
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
                "Must provide either project_id or connection_params with project_id. Example: project_id='my-project'"
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
            connection_params["credentials"] = self._load_service_account_credentials(credentials_path)
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
                service_account,
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
            raise FileNotFoundError(f"Service account credentials file not found: {credentials_path}") from e
        except Exception as e:
            # Catch JSON decode errors and other credential format issues
            raise ValueError(
                f"Invalid service account credentials file: {credentials_path}. "
                "Ensure it's a valid service account JSON key file."
            ) from e

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        # Should switch to FARM_FINGERPRINT64 once https://github.com/ion-elgreco/polars-hash/issues/49 is resolved
        return HashAlgorithm.MD5

    def _create_hash_functions(self):
        """Create BigQuery-specific hash functions for Ibis expressions.

        BigQuery supports FARM_FINGERPRINT, MD5, and SHA256 natively.
        """
        # Import ibis for wrapping built-in SQL functions
        import ibis

        # Use Ibis's builtin UDF decorator to wrap BigQuery's hash functions
        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:  # ty: ignore[invalid-return-type]
            """BigQuery MD5() function."""
            ...

        @ibis.udf.scalar.builtin
        def FARM_FINGERPRINT(x: str) -> str:  # ty: ignore[invalid-return-type]
            """BigQuery FARM_FINGERPRINT() function."""
            ...

        @ibis.udf.scalar.builtin
        def SHA256(x: str) -> str:  # ty: ignore[invalid-return-type]
            """BigQuery SHA256() function."""
            ...

        @ibis.udf.scalar.builtin
        def TO_HEX(x: str) -> str:  # ty: ignore[invalid-return-type]
            """BigQuery TO_HEX() function."""
            ...

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str:  # ty: ignore[invalid-return-type]
            """BigQuery LOWER() function."""
            ...

        # Create hash functions that use these wrapped SQL functions
        def md5_hash(col_expr):
            """Hash a column using BigQuery's MD5() function."""
            # MD5 returns bytes, convert to lowercase hex string
            return LOWER(TO_HEX(MD5(col_expr.cast(str))))

        def farmhash_hash(col_expr):
            """Hash a column using BigQuery's FARM_FINGERPRINT() function."""
            # FARM_FINGERPRINT returns INT64, cast to string
            return FARM_FINGERPRINT(col_expr).cast(str)

        def sha256_hash(col_expr):
            """Hash a column using BigQuery's SHA256() function."""
            # SHA256 returns bytes, convert to lowercase hex string
            return LOWER(TO_HEX(SHA256(col_expr)))

        hash_functions = {
            HashAlgorithm.MD5: md5_hash,
            HashAlgorithm.FARMHASH: farmhash_hash,
            HashAlgorithm.SHA256: sha256_hash,
        }

        return hash_functions

    def display(self) -> str:
        """Display string for this store."""
        dataset_info = f"/{self.dataset_id}" if self.dataset_id else ""
        return f"BigQueryMetadataStore(project={self.project_id}{dataset_info})"

    @classmethod
    def config_model(cls) -> type[BigQueryMetadataStoreConfig]:
        return BigQueryMetadataStoreConfig
