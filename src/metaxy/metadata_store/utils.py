from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlparse, urlunparse

from narwhals.typing import FrameT

# Context variable for suppressing feature_version warning in migrations
_suppress_feature_version_warning: ContextVar[bool] = ContextVar(
    "_suppress_feature_version_warning", default=False
)


def is_local_path(path: str) -> bool:
    """Return True when the path points to the local filesystem."""
    if path.startswith(("file://", "local://")):
        return True
    return "://" not in path


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
    return ref_frame.head(0)  # ty: ignore[invalid-argument-type]


def sanitize_uri(uri: str) -> str:
    """Sanitize URI to mask credentials.

    Replaces username and password in URIs with `***` to prevent credential exposure
    in logs, display strings, and error messages.

    Examples:
        >>> sanitize_uri("s3://bucket/path")
        's3://bucket/path'
        >>> sanitize_uri("db://user:pass@host/db")
        'db://***:***@host/db'
        >>> sanitize_uri("postgresql://admin:secret@host:5432/db")
        'postgresql://***:***@host:5432/db'
        >>> sanitize_uri("./local/path")
        './local/path'

    Args:
        uri: URI or path string that may contain credentials

    Returns:
        Sanitized URI with credentials masked as ***
    """
    # Try to parse as URI
    try:
        parsed = urlparse(uri)

        # If no scheme, it's likely a local path - return as-is
        if not parsed.scheme or parsed.scheme in ("file", "local"):
            return uri

        # Check if URI contains credentials (username or password)
        if parsed.username or parsed.password:
            # Replace credentials with ***
            username = "***" if parsed.username else ""
            password = "***" if parsed.password else ""
            credentials = f"{username}:{password}@" if username or password else ""
            # Reconstruct netloc without credentials
            host_port = parsed.netloc.split("@")[-1]
            masked_netloc = f"{credentials}{host_port}"

            # Reconstruct URI with masked credentials
            return urlunparse(
                (
                    parsed.scheme,
                    masked_netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
    except Exception:
        # If parsing fails, return as-is (likely a local path)
        pass

    return uri
