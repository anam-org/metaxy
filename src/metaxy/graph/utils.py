"""Shared utilities for graph rendering and formatting."""

from metaxy.models.types import FeatureKey, FieldKey


def sanitize_mermaid_id(s: str) -> str:
    """Sanitize string for use as Mermaid node ID.

    Replaces characters that are invalid in Mermaid identifiers.

    Args:
        s: String to sanitize

    Returns:
        Sanitized string safe for use as Mermaid node ID
    """
    return s.replace("/", "_").replace("-", "_").replace("__", "_")


def format_hash(hash_str: str, length: int = 8) -> str:
    """Format hash string with optional truncation.

    Args:
        hash_str: Full hash string
        length: Number of characters to show (0 for full hash)

    Returns:
        Truncated or full hash string
    """
    if length == 0 or length >= len(hash_str):
        return hash_str
    return hash_str[:length]


def format_feature_key(key: FeatureKey) -> str:
    """Format feature key for display.

    Uses / separator for better readability.

    Args:
        key: Feature key

    Returns:
        Formatted string like "my/feature/key"
    """
    return "/".join(key)


def format_field_key(key: FieldKey) -> str:
    """Format field key for display.

    Args:
        key: Field key

    Returns:
        Formatted string like "field_name"
    """
    return "/".join(key)
