import pydantic


class FrozenBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
