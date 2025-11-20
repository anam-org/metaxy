from pydantic import Field as PydanticField
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict
from typing_extensions import Self

from metaxy.config import PluginConfig


class SQLModelPluginConfig(PluginConfig):
    """Configuration for SQLModel"""

    model_config = SettingsConfigDict(env_prefix="METAXY_EXT__SQLMODEL_")

    infer_db_table_names: bool = PydanticField(
        default=True,
        description="Whether to automatically use `FeatureKey.table_name` for sqlalchemy's __tablename__ value.",
    )

    # Whether to use SQLModel definitions for system tables (for Alembic migrations)
    system_tables: bool = PydanticField(
        default=True,
        description="Whether to use SQLModel definitions for system tables (for Alembic migrations).",
    )

    inject_primary_key: bool = PydanticField(
        default=False,
        description="Whether to inject Metaxy composite primary key (id_columns + metaxy_created_at + metaxy_data_version) into SQLModel definitions.",
    )

    @model_validator(mode="after")
    def import_system_tables(self) -> Self:
        if self.system_tables:
            import metaxy.ext.sqlmodel.system_tables as system_tables

            assert system_tables
        return self
