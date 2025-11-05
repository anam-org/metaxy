# Syrupy Examples

Practical examples of using syrupy for snapshot testing in various scenarios.

## Basic Examples

### Simple Value Snapshot

```python
def test_string_output(snapshot):
    result = "Hello, World!"
    assert result == snapshot


def test_dictionary_snapshot(snapshot):
    data = {"name": "John Doe", "age": 30, "email": "john@example.com"}
    assert data == snapshot


def test_list_snapshot(snapshot):
    items = ["apple", "banana", "cherry"]
    assert items == snapshot
```

### Complex Data Structures

```python
def test_nested_data(snapshot):
    user = {
        "id": 123,
        "profile": {
            "name": "Jane Smith",
            "preferences": {"theme": "dark", "notifications": True},
        },
        "tags": ["admin", "verified"],
    }
    assert user == snapshot
```

## Using Matchers

### Handling Dynamic Values with path_type

```python
from syrupy.matchers import path_type
import uuid
import datetime


def test_api_response_with_dynamic_fields(snapshot):
    response = {
        "id": uuid.uuid4(),
        "created_at": datetime.datetime.now(),
        "user": {"name": "Alice", "last_login": datetime.datetime.now()},
        "data": {"value": 42},
    }

    assert response == snapshot(
        matcher=path_type(
            {
                "id": (uuid.UUID,),
                "created_at": (datetime.datetime,),
                "user.last_login": (datetime.datetime,),
            }
        )
    )
```

### Replacing Values with path_value

```python
from syrupy.matchers import path_value


def test_sensitive_data_masking(snapshot):
    config = {
        "database_url": "postgresql://user:pass@localhost/db",
        "api_key": "sk_live_abc123def456",
        "settings": {"secret_token": "super-secret-value"},
    }

    assert config == snapshot(
        matcher=path_value(
            {
                "database_url": "REDACTED_URL",
                "api_key": "REDACTED_KEY",
                "settings.secret_token": "REDACTED_TOKEN",
            }
        )
    )
```

## Using Filters

### Excluding Properties

```python
from syrupy.filters import props, paths


def test_exclude_volatile_fields(snapshot):
    response = {
        "data": "important content",
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_abc123",
        "cache_key": "cache_xyz789",
    }

    # Exclude by property names
    assert response == snapshot(exclude=props("timestamp", "request_id", "cache_key"))


def test_exclude_nested_paths(snapshot):
    user_data = {
        "user": {
            "id": 1,
            "name": "Bob",
            "password": "hashed_password",
            "email": "bob@example.com",
        },
        "session": {"token": "session_token_123", "expires_at": "2024-12-31"},
    }

    # Exclude by full paths
    assert user_data == snapshot(
        exclude=paths("user.password", "session.token", "session.expires_at")
    )
```

### Including Only Specific Properties

```python
from syrupy.filters import props


def test_include_only_essential_fields(snapshot):
    full_response = {
        "id": 1,
        "type": "user",
        "data": {"name": "Charlie"},
        "metadata": {"created": "2024-01-01"},
        "debug_info": {"trace_id": "xyz"},
        "internal": {"cache": "data"},
    }

    # Include only what matters
    assert full_response == snapshot(include=props("id", "type", "data"))
```

## Extension Examples

### JSON Extension for API Testing

```python
from syrupy.extensions.json import JSONSnapshotExtension
import json


@pytest.fixture
def json_snapshot(snapshot):
    return snapshot.use_extension(JSONSnapshotExtension)


def test_api_endpoint(json_snapshot):
    api_response = {
        "status": "success",
        "data": {"users": [{"id": 1, "name": "User 1"}, {"id": 2, "name": "User 2"}]},
    }
    assert api_response == json_snapshot
```

### Single File Extension

```python
from syrupy.extensions.single_file import SingleFileSnapshotExtension


class TextFileExtension(SingleFileSnapshotExtension):
    @property
    def _file_extension(self):
        return "txt"


def test_text_output(snapshot):
    long_text = """
    This is a long text output
    that spans multiple lines
    and might be easier to review
    as a separate file.
    """
    assert long_text == snapshot(extension_class=TextFileExtension)
```

### Image Snapshot Testing

```python
from syrupy.extensions.image import PNGSnapshotExtension, SVGSnapshotExtension
from PIL import Image
import io


def test_png_image(snapshot):
    # Create a simple image
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    assert img_bytes.getvalue() == snapshot(extension_class=PNGSnapshotExtension)


def test_svg_output(snapshot):
    svg_content = """
    <svg width="100" height="100">
        <circle cx="50" cy="50" r="40" fill="blue" />
    </svg>
    """
    assert svg_content == snapshot(extension_class=SVGSnapshotExtension)
```

## Advanced Patterns

### Custom Snapshot Names

```python
def test_multiple_scenarios(snapshot):
    # Test different inputs with named snapshots
    inputs = [
        ("input1", "expected_output1"),
        ("input2", "expected_output2"),
        ("input3", "expected_output3"),
    ]

    for input_val, expected in inputs:
        result = process(input_val)
        assert result == snapshot(name=f"scenario_{input_val}")
```

### Persistent Configuration

```python
from syrupy.filters import props
from syrupy.extensions.json import JSONSnapshotExtension


def test_multiple_api_calls(snapshot):
    # Create snapshot with default configuration
    api_snapshot = snapshot.with_defaults(
        extension_class=JSONSnapshotExtension,
        exclude=props("timestamp", "request_id", "duration"),
    )

    # All assertions use the same configuration
    response1 = make_api_call("/endpoint1")
    assert response1 == api_snapshot

    response2 = make_api_call("/endpoint2")
    assert response2 == api_snapshot

    response3 = make_api_call("/endpoint3")
    assert response3 == api_snapshot
```

### Diff-Based Snapshots

```python
def test_configuration_changes(snapshot):
    base_config = {
        "app_name": "MyApp",
        "version": "1.0.0",
        "features": {"auth": True, "notifications": False},
    }

    # Apply changes and snapshot only the diff
    updated_config = base_config.copy()
    updated_config["version"] = "1.1.0"
    updated_config["features"]["notifications"] = True

    assert updated_config == snapshot(diff=base_config)
```

### Custom Extension Example

```python
from syrupy.extensions.base import AbstractSnapshotExtension
import yaml


class YAMLSnapshotExtension(AbstractSnapshotExtension):
    @property
    def _file_extension(self):
        return "yml"

    def serialize(self, data, **kwargs):
        return yaml.dump(data, default_flow_style=False)

    def matches(self, *, serialized_data, snapshot_data):
        return serialized_data == snapshot_data


def test_yaml_config(snapshot):
    config = {
        "database": {"host": "localhost", "port": 5432, "name": "testdb"},
        "cache": {"enabled": True, "ttl": 3600},
    }
    assert config == snapshot(extension_class=YAMLSnapshotExtension)
```

## Testing Patterns

### Parametrized Tests

```python
import pytest


@pytest.mark.parametrize(
    "input,expected_type",
    [("hello", str), (42, int), ([1, 2, 3], list), ({"key": "value"}, dict)],
)
def test_type_conversion(snapshot, input, expected_type):
    result = convert_to_type(input, expected_type)
    assert result == snapshot
```

### Class-Based Tests

```python
class TestUserAPI:
    def test_create_user(self, snapshot):
        user = create_user("John", "john@example.com")
        assert user == snapshot

    def test_update_user(self, snapshot):
        user = create_user("Jane", "jane@example.com")
        updated = update_user(user.id, name="Jane Doe")
        assert updated == snapshot

    def test_delete_user(self, snapshot):
        user = create_user("Bob", "bob@example.com")
        result = delete_user(user.id)
        assert result == snapshot
```

### Fixture-Based Setup

```python
@pytest.fixture
def sample_data():
    return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}


@pytest.fixture
def filtered_snapshot(snapshot):
    """Snapshot with common filters applied"""
    from syrupy.filters import paths

    return snapshot.with_defaults(exclude=paths("*.id", "*.created_at", "*.updated_at"))


def test_with_fixtures(filtered_snapshot, sample_data):
    processed = process_data(sample_data)
    assert processed == filtered_snapshot
```

## Real-World Examples

### Testing Data Transformations

```python
def test_data_pipeline(snapshot):
    raw_data = [
        {"name": "Alice", "score": 85},
        {"name": "Bob", "score": 92},
        {"name": "Charlie", "score": 78},
    ]

    # Test each step of the pipeline
    cleaned = clean_data(raw_data)
    assert cleaned == snapshot(name="cleaned_data")

    normalized = normalize_scores(cleaned)
    assert normalized == snapshot(name="normalized_data")

    final = calculate_rankings(normalized)
    assert final == snapshot(name="final_rankings")
```

### Testing CLI Output

```python
from click.testing import CliRunner


def test_cli_command_output(snapshot):
    runner = CliRunner()
    result = runner.invoke(my_cli_command, ["--verbose"])

    assert result.output == snapshot
    assert result.exit_code == 0
```

### Testing HTML/Template Generation

```python
from bs4 import BeautifulSoup


def test_html_generation(snapshot):
    html = generate_html_report(data)

    # Parse and normalize HTML for consistent snapshots
    soup = BeautifulSoup(html, "html.parser")
    pretty_html = soup.prettify()

    assert pretty_html == snapshot
```

### Testing Database Queries

```python
def test_query_results(snapshot, db_session):
    # Setup test data
    db_session.add_all(
        [
            User(name="Alice", age=30),
            User(name="Bob", age=25),
            User(name="Charlie", age=35),
        ]
    )
    db_session.commit()

    # Execute query
    results = db_session.query(User).filter(User.age > 25).all()

    # Snapshot the results (excluding IDs)
    from syrupy.filters import props

    assert [r.to_dict() for r in results] == snapshot(exclude=props("id", "created_at"))
```

## Error Handling Examples

### Testing Exception Messages

```python
def test_error_messages(snapshot):
    try:
        divide_by_zero()
    except ZeroDivisionError as e:
        assert str(e) == snapshot

    try:
        invalid_operation()
    except ValueError as e:
        assert str(e) == snapshot(name="value_error")
```

### Testing Validation Errors

```python
def test_validation_errors(snapshot):
    from syrupy.filters import paths

    invalid_data = {"email": "not-an-email", "age": -5, "name": ""}

    errors = validate_user_data(invalid_data)

    # Exclude line numbers that might change
    assert errors == snapshot(exclude=paths("*.line_number", "*.file_path"))
```

## Performance Testing

### Benchmarking with Snapshots

```python
import time


def test_performance_metrics(snapshot):
    metrics = []

    for size in [100, 1000, 10000]:
        start = time.time()
        process_items(size)
        duration = time.time() - start

        metrics.append(
            {"size": size, "duration_category": "fast" if duration < 0.1 else "slow"}
        )

    # Snapshot categories, not exact times
    assert metrics == snapshot
```

## Tips and Tricks

### Combining Matchers and Filters

```python
from syrupy.matchers import path_type
from syrupy.filters import paths
import datetime


def test_complex_filtering(snapshot):
    data = {
        "id": 123,
        "created": datetime.datetime.now(),
        "sensitive": "secret",
        "items": [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}],
    }

    assert data == snapshot(
        matcher=path_type({"created": (datetime.datetime,), "items.*.id": (int,)}),
        exclude=paths("sensitive"),
    )
```

### Debugging Failed Snapshots

```python
def test_with_debugging(snapshot):
    result = complex_calculation()

    # Add debug info to understand failures
    debug_info = {
        "result": result,
        "input_params": get_params(),
        "environment": get_environment(),
    }

    assert debug_info == snapshot
```

### Organizing Large Snapshots

```python
def test_large_structure(snapshot):
    data = generate_large_dataset()

    # Break into logical sections
    assert data["header"] == snapshot(name="header")
    assert data["body"] == snapshot(name="body")
    assert data["footer"] == snapshot(name="footer")
```
