import narwhals as nw
import polars as pl
import pytest
from metaxy import FeatureKey, MetadataQuery, parse_metadata_uri
from metaxy.models.metadata_query import MetadataQueryParseError


@pytest.mark.parametrize(
    ("query_string", "key", "version", "filter_count"),
    [
        ("my_feature", "my_feature", "current", 0),
        ("group/subgroup/feature:current", "group/subgroup/feature", "current", 0),
        ("my_feature:latest", "my_feature", "latest", 0),
        ("my_feature:local", "my_feature", "local", 0),
        ("my_feature:abc123def", "my_feature", "abc123def", 0),
        ("my_feature:latest?age>25&status='active'", "my_feature", "latest", 2),
    ],
)
def test_parse_metadata_uri(
    query_string: str,
    key: str,
    version: str,
    filter_count: int,
) -> None:
    query = parse_metadata_uri(query_string)

    assert query.key == FeatureKey(key)
    assert query.version == version
    assert len(query.filters) == filter_count


def test_metadata_query_filters_are_executable() -> None:
    query = MetadataQuery.from_uri("my_feature?age>25&status='active'")
    frame = nw.from_native(pl.DataFrame({"age": [20, 30, 40], "status": ["active", "active", "deleted"]}))

    result = frame.filter(*query.filter_expressions).to_native()

    assert result["age"].to_list() == [30]


def test_metadata_query_percent_encoded_delimiters_are_literals() -> None:
    query = MetadataQuery.from_uri("my_feature?label='a%26b%25'&note='x%3Fy'")
    frame = nw.from_native(pl.DataFrame({"label": ["a&b%", "a&b%"], "note": ["x?y", "z"]}))

    result = frame.filter(*query.filter_expressions).to_native()

    assert len(query.filters) == 2
    assert result["note"].to_list() == ["x?y"]


def test_metadata_query_model_is_immutable() -> None:
    query = MetadataQuery(key=FeatureKey("my_feature"))

    with pytest.raises(ValueError):
        query.version = "latest"


@pytest.mark.parametrize(
    "query_string",
    [
        "",
        " my_feature",
        "MyFeature",
        "my_feature:",
        "my_feature:abc-123",
        "my_feature?",
        "my_feature?age>25&",
        "my_feature:latest:extra",
        "my_feature?age>25?status='active'",
        "my_feature?label='%GG'",
        "my_feature?label='%FF'",
    ],
)
def test_invalid_metadata_query_raises(query_string: str) -> None:
    with pytest.raises(MetadataQueryParseError):
        parse_metadata_uri(query_string)
