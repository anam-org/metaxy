from typing import Any, TypeAlias

from pydantic.annotated_handlers import GetCoreSchemaHandler

# from pydantic_core import CoreSchema
from pydantic_core.core_schema import (
    CoreSchema,
    str_schema,
    tuple_schema,
)

# class Key(tuple):
#     def __get_pydantic_core_schema__(cls, handler) -> CoreSchema:
#         # breakpoint()
#         return list_schema(str_schema())


class FeatureKey(list):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_string(self) -> str:
        return "_".join(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return tuple_schema([str_schema()])


class ContainerKey(list):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_string(self) -> str:
        return "_".join(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return tuple_schema([str_schema()])


FeatureDepMetadata: TypeAlias = dict[str, Any]
