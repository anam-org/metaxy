"""Environment configuration source with stable handling of untyped dictionaries."""

from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo
from pydantic_settings import EnvSettingsSource


class MetaxyEnvSettingsSource(EnvSettingsSource):
    def next_field(
        self, field: FieldInfo | Any | None, key: str, case_sensitive: bool | None = None
    ) -> FieldInfo | None:
        annotation = field.annotation if isinstance(field, FieldInfo) else field
        if get_origin(annotation) is dict:
            # Preserve the value type, including Any, while traversing named stores.
            return FieldInfo(annotation=get_args(annotation)[-1])
        return super().next_field(field, key, case_sensitive)
