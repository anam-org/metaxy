"""BigQuery metadata store — backward-compatible factory class."""

from __future__ import annotations

from typing import Any

from metaxy._decorators import public
from metaxy.ext.bigquery.config import BigQueryMetadataStoreConfig
from metaxy.ext.bigquery.engine import BigQueryEngine
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.ibis_compute_engine import IbisStorageConfig, IbisStoreBackcompat


@public
class BigQueryMetadataStore(IbisStoreBackcompat):
    """[BigQuery](https://cloud.google.com/bigquery) metadata store using [Ibis](https://ibis-project.org/) backend.

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
    """

    def __init__(
        self,
        project_id: str | None = None,  # noqa: ARG002
        dataset_id: str | None = None,  # noqa: ARG002
        *,
        credentials_path: str | None = None,  # noqa: ARG002
        credentials: Any | None = None,  # noqa: ARG002
        location: str | None = None,  # noqa: ARG002
        connection_params: dict[str, Any] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        table_prefix: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    @property
    def project_id(self) -> str | None:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("project_id")
        assert isinstance(self._engine, BigQueryEngine)
        return self._engine.project_id

    @property
    def dataset_id(self) -> str | None:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("dataset_id")
        assert isinstance(self._engine, BigQueryEngine)
        return self._engine.dataset_id

    def __new__(
        cls,
        project_id: str | None = None,
        dataset_id: str | None = None,
        *,
        credentials_path: str | None = None,
        credentials: Any | None = None,
        location: str | None = None,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        engine = BigQueryEngine(
            project_id=project_id,
            dataset_id=dataset_id,
            credentials_path=credentials_path,
            credentials=credentials,
            location=location,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
        )

        prefix = table_prefix or ""
        storage = [IbisStorageConfig(format="bigquery", location=project_id or "bigquery", table_prefix=prefix)]

        instance = IbisStoreBackcompat.__new__(cls)
        MetadataStore.__init__(
            instance,
            engine=engine,
            storage=storage,
            fallback_stores=fallback_stores,
            auto_create_tables=auto_create_tables,
            **kwargs,
        )
        return instance

    @classmethod
    def config_model(cls) -> type[BigQueryMetadataStoreConfig]:
        return BigQueryMetadataStoreConfig
