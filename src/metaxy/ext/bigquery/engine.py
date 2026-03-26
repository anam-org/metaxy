"""BigQuery compute engine using Ibis."""

from __future__ import annotations

from typing import Any

from metaxy.ext.bigquery.config import BigQueryMetadataStoreConfig
from metaxy.metadata_store.ibis_compute_engine import IbisComputeEngine
from metaxy.versioning.types import HashAlgorithm


class BigQueryEngine(IbisComputeEngine):
    """Compute engine for BigQuery backends using Ibis."""

    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str | None = None,
        *,
        credentials_path: str | None = None,
        credentials: Any | None = None,
        location: str | None = None,
        connection_params: dict[str, Any] | None = None,
        auto_create_tables: bool = False,
    ) -> None:
        if connection_params is None:
            connection_params = _build_bigquery_connection_params(
                project_id=project_id,
                dataset_id=dataset_id,
                credentials_path=credentials_path,
                credentials=credentials,
                location=location,
            )

        if "project_id" not in connection_params and project_id is None:
            raise ValueError(
                "Must provide either project_id or connection_params with project_id. Example: project_id='my-project'"
            )

        self.project_id = project_id or connection_params.get("project_id")
        self.dataset_id = dataset_id or connection_params.get("dataset_id", "")

        super().__init__(
            backend="bigquery",
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
        )

    def _create_hash_functions(self) -> dict:
        import ibis

        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def FARM_FINGERPRINT(x: str) -> str:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def SHA256(x: str) -> str:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def TO_HEX(x: str) -> str:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str:  # ty: ignore[empty-body]
            ...

        def md5_hash(col_expr):  # noqa: ANN001, ANN202
            return LOWER(TO_HEX(MD5(col_expr.cast(str))))

        def farmhash_hash(col_expr):  # noqa: ANN001, ANN202
            return FARM_FINGERPRINT(col_expr).cast(str)

        def sha256_hash(col_expr):  # noqa: ANN001, ANN202
            return LOWER(TO_HEX(SHA256(col_expr)))

        return {
            HashAlgorithm.MD5: md5_hash,
            HashAlgorithm.FARMHASH: farmhash_hash,
            HashAlgorithm.SHA256: sha256_hash,
        }

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.MD5

    def display(self) -> str:
        dataset_info = f"/{self.dataset_id}" if self.dataset_id else ""
        return f"BigQueryEngine(project={self.project_id}{dataset_info})"

    @classmethod
    def config_model(cls) -> type[BigQueryMetadataStoreConfig]:
        return BigQueryMetadataStoreConfig


def _build_bigquery_connection_params(
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
    credentials_path: str | None = None,
    credentials: Any | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Build connection parameters for the Ibis BigQuery backend."""
    connection_params: dict[str, Any] = {}

    if project_id is not None:
        connection_params["project_id"] = project_id
    if dataset_id is not None:
        connection_params["dataset_id"] = dataset_id
    if location is not None:
        connection_params["location"] = location

    if credentials_path is not None:
        try:
            from google.oauth2 import service_account
        except ImportError as e:
            raise ImportError(
                "Google Cloud authentication libraries required for service account credentials. "
                "Install with: pip install google-auth"
            ) from e

        try:
            connection_params["credentials"] = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Service account credentials file not found: {credentials_path}") from e
        except Exception as e:
            raise ValueError(
                f"Invalid service account credentials file: {credentials_path}. "
                "Ensure it's a valid service account JSON key file."
            ) from e
    elif credentials is not None:
        connection_params["credentials"] = credentials

    return connection_params
