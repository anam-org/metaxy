from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from narwhals.typing import FrameT

# Context variable for suppressing feature_version warning in migrations
_suppress_feature_version_warning: ContextVar[bool] = ContextVar(
    "_suppress_feature_version_warning", default=False
)


@contextmanager
def allow_feature_version_override() -> Iterator[None]:
    """Context manager to suppress warnings when writing metadata with pre-existing metaxy_feature_version.

    This should only be used in migration code where writing historical feature versions
    is intentional and necessary.

    Example:
        ```py
        with allow_feature_version_override():
            # DataFrame already has metaxy_feature_version column from migration
            store.write_metadata(MyFeature, df_with_feature_version)
        ```
    """
    token = _suppress_feature_version_warning.set(True)
    try:
        yield
    finally:
        _suppress_feature_version_warning.reset(token)


# Helper to create empty DataFrame with correct schema and backend
#
def empty_frame_like(ref_frame: FrameT) -> FrameT:
    """Create an empty LazyFrame with the same schema as ref_frame."""
    return ref_frame.head(0)
