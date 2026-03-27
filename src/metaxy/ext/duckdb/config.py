"""DuckDB metadata store configuration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from metaxy._decorators import public
from metaxy.ext.duckdb.handlers.ducklake.config import DuckLakeConfig
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig


@public
class ExtensionSpec(BaseModel):
    """DuckDB extension specification accepted by DuckDBMetadataStore."""

    name: str
    repository: str = "core"
    """Extension repository: `"core"` for official extensions, `"community"` for community extensions."""
    init_sql: Sequence[str] = ()
    """SQL statements to execute immediately after loading the extension."""


@public
class DuckDBMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for DuckDBMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

        [stores.dev.config]
        database = "metadata.db"
        hash_algorithm = "xxhash64"
        ```
    """

    database: str | Path = Field(
        description="Database path (:memory:, file path, or md:database).",
    )
    config: dict[str, str] | None = Field(
        default=None,
        description="DuckDB configuration settings (e.g., {'threads': '4'}).",
    )
    extensions: Sequence[str | ExtensionSpec] | None = Field(
        default=None,
        description="DuckDB extensions to install and load on open. If only a string is provided, the `core` repository is assumed.",
    )
    ducklake: DuckLakeConfig | None = Field(
        default=None,
        description="DuckLake attachment configuration. Learn more [here](/integrations/metadata-stores/storage/ducklake.md).",
    )


def _normalise_extensions(
    extensions: Iterable[str | ExtensionSpec],
) -> list[ExtensionSpec]:
    """Coerce extension inputs into ExtensionSpec instances."""
    normalised: list[ExtensionSpec] = []
    for ext in extensions:
        if isinstance(ext, str):
            normalised.append(ExtensionSpec(name=ext))
        elif isinstance(ext, ExtensionSpec):
            normalised.append(ext)
        else:
            raise TypeError(f"DuckDB extensions must be strings or ExtensionSpec instances, got {type(ext).__name__}.")
    return normalised
