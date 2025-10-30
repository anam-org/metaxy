import pydantic


class FrozenBaseModel(pydantic.BaseModel):
    # config class is deprecated
    model_config = pydantic.ConfigDict(frozen=True)
