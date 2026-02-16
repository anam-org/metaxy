import pydantic


class FrozenBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class VersioningEngineMismatchError(Exception):
    """Raised when versioning_engine='native' is requested but data has wrong implementation."""

    pass
