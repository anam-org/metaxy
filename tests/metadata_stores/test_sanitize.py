"""Tests for URI sanitization utilities."""


def test_sanitize_uri_without_credentials() -> None:
    """Test that URIs without credentials pass through unchanged."""
    from metaxy.metadata_store.utils import sanitize_uri

    # URIs without credentials should pass through unchanged
    assert sanitize_uri("s3://bucket/path") == "s3://bucket/path"
    assert sanitize_uri("db://database") == "db://database"


def test_sanitize_uri_local_paths() -> None:
    """Test that local paths pass through unchanged."""
    from metaxy.metadata_store.utils import sanitize_uri

    assert sanitize_uri("./local/path") == "./local/path"
    assert sanitize_uri("/absolute/path") == "/absolute/path"


def test_sanitize_uri_netloc_credentials() -> None:
    """Test that passwords in netloc are masked while usernames remain visible."""
    from metaxy.metadata_store.utils import sanitize_uri

    # URIs with credentials should mask passwords but keep usernames visible
    assert sanitize_uri("db://user:pass@host/db") == "db://user:***@host/db"
    assert (
        sanitize_uri("https://admin:secret@host:8000/api")
        == "https://admin:***@host:8000/api"
    )
    assert sanitize_uri("s3://key:secret@bucket/path") == "s3://key:***@bucket/path"
    assert (
        sanitize_uri("postgresql://admin:secret@host:5432/db")
        == "postgresql://admin:***@host:5432/db"
    )


def test_sanitize_uri_username_only() -> None:
    """Test that username-only URIs remain unchanged."""
    from metaxy.metadata_store.utils import sanitize_uri

    # Username only (no password) - unchanged
    assert sanitize_uri("db://user@host/db") == "db://user@host/db"


def test_sanitize_uri_query_parameters() -> None:
    """Test that password parameters in query strings are masked."""
    from metaxy.metadata_store.utils import sanitize_uri

    # Query parameter sanitization
    assert (
        sanitize_uri("postgresql://host/db?password=secret")
        == "postgresql://host/db?password=***"
    )
    assert (
        sanitize_uri("db://host?user=admin&pwd=secret")
        == "db://host?user=admin&pwd=***"
    )
    assert (
        sanitize_uri("db://host?user=admin&pass=secret123")
        == "db://host?user=admin&pass=***"
    )
    assert (
        sanitize_uri(
            "postgresql://host/db?sslmode=require&password=secret&connect_timeout=10"
        )
        == "postgresql://host/db?sslmode=require&password=***&connect_timeout=10"
    )


def test_sanitize_uri_combined_credentials() -> None:
    """Test sanitization when both netloc and query parameters contain credentials."""
    from metaxy.metadata_store.utils import sanitize_uri

    # Combined netloc and query parameter credentials
    assert (
        sanitize_uri("postgresql://user:pass@host/db?password=secret")
        == "postgresql://user:***@host/db?password=***"
    )


def test_sanitize_uri_case_insensitive_password_keys() -> None:
    """Test that password parameter names are case-insensitive."""
    from metaxy.metadata_store.utils import sanitize_uri

    # Case-insensitive password keys
    assert sanitize_uri("db://host?PASSWORD=secret") == "db://host?PASSWORD=***"
    assert sanitize_uri("db://host?Pwd=secret") == "db://host?Pwd=***"
