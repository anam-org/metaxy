import pydantic


class FrozenBaseModel(pydantic.BaseModel):
    class Config:
        frozen = True
