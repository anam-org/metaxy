from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import parse_qsl, quote, urlparse, urlunparse

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

    Replaces passwords in URIs (both in netloc and query parameters) with `***`
    to prevent credential exposure in logs, display strings, and error messages.
    Usernames are preserved for debugging purposes.

    Examples:
        >>> sanitize_uri("s3://bucket/path")
        's3://bucket/path'
        >>> sanitize_uri("db://user:pass@host/db")
        'db://user:***@host/db'
        >>> sanitize_uri("postgresql://admin:secret@host:5432/db")
        'postgresql://admin:***@host:5432/db'
        >>> sanitize_uri("postgresql://host/db?password=secret")
        'postgresql://host/db?password=***'
        >>> sanitize_uri("db://host?user=admin&pwd=secret")
        'db://host?user=admin&pwd=***'
        >>> sanitize_uri("./local/path")
        './local/path'

    Args:
        uri: URI or path string that may contain credentials

    Returns:
        Sanitized URI with passwords masked as ***
    """
    # Try to parse as URI
    try:
        parsed = urlparse(uri)

        # If no scheme, it's likely a local path - return as-is
        if not parsed.scheme or parsed.scheme in ("file", "local"):
            return uri

        # Sanitize netloc (username:password@host:port)
        sanitized_netloc = parsed.netloc
        if parsed.password:
            # Keep username visible, only mask password
            if "@" in sanitized_netloc:
                userinfo, hostinfo = sanitized_netloc.rsplit("@", 1)
                if ":" in userinfo:
                    username, _, _ = userinfo.partition(":")
                    userinfo = f"{username}:***"
                sanitized_netloc = f"{userinfo}@{hostinfo}"

        # Sanitize query parameters
        sanitized_query = parsed.query
        if parsed.query:
            query_params = parse_qsl(parsed.query, keep_blank_values=True)
            masked_parts = []
            changed = False
            for key, val in query_params:
                if key.lower() in {"password", "pwd", "pass"}:
                    # Use *** without URL encoding for readability
                    masked_parts.append(f"{quote(key, safe='')}=***")
                    changed = True
                else:
                    masked_parts.append(f"{quote(key, safe='')}={quote(val, safe='')}")
            if changed:
                sanitized_query = "&".join(masked_parts)

        # Only reconstruct if something changed
        if sanitized_netloc != parsed.netloc or sanitized_query != parsed.query:
            return urlunparse(
                (
                    parsed.scheme,
                    sanitized_netloc,
                    parsed.path,
                    parsed.params,
                    sanitized_query,
                    parsed.fragment,
                )
            )
    except Exception:
        # If parsing fails, return as-is (likely a local path)
        pass

    return uri
