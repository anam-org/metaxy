"""Parsing for metadata query strings."""

from __future__ import annotations

import re
from urllib.parse import unquote

import narwhals as nw
from pydantic import field_validator
from typing_extensions import Self

from metaxy._decorators import public
from metaxy.models.bases import FrozenBaseModel
from metaxy.models.filter_expression import NarwhalsFilter
from metaxy.models.types import FeatureKey

_METADATA_URI_PATTERN = re.compile(
    r"(?P<key>[^:?\s]+)(?::(?P<version>[A-Za-z0-9]+))?"
    r"(?:\?(?P<filters>[^\s?\r\n](?:[^?\r\n]*[^\s?\r\n])?))?"
)


@public
class MetadataQueryParseError(ValueError):
    """Raised when a metadata query string is invalid."""


@public
class MetadataQuery(FrozenBaseModel):
    """A parsed ``<key>:<version>?<filters>`` metadata query."""

    key: FeatureKey
    version: str = "current"
    filters: tuple[NarwhalsFilter, ...] = ()

    @field_validator("version")
    @classmethod
    def _validate_version(cls, version: str) -> str:
        if version not in {"current", "latest"} and re.fullmatch(r"[A-Za-z0-9]+", version) is None:
            raise ValueError("Version must be 'current', 'latest', or an alphanumeric version hash.")
        return version

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        """Parse ``<key>[:<version>][?<filter>[&<filter>...]]``."""
        match = _METADATA_URI_PATTERN.fullmatch(uri)
        if match is None:
            raise MetadataQueryParseError(f"Invalid metadata URI: {uri!r}")

        try:
            filter_string = match["filters"]
            if filter_string and re.search(r"%(?![0-9A-Fa-f]{2})", filter_string):
                raise ValueError("Invalid percent escape.")
            filters = (
                tuple(
                    NarwhalsFilter.model_validate(unquote(item, errors="strict")) for item in filter_string.split("&")
                )
                if filter_string
                else ()
            )
            return cls(
                key=FeatureKey(match["key"]),
                version=match["version"] or "current",
                filters=filters,
            )
        except ValueError as exc:
            raise MetadataQueryParseError(f"Invalid metadata URI {uri!r}: {exc}") from exc

    @property
    def filter_expressions(self) -> tuple[nw.Expr, ...]:
        """Return filters as Narwhals expressions."""
        return tuple(filter_.to_expr() for filter_ in self.filters)


@public
def parse_metadata_uri(uri: str) -> MetadataQuery:
    """Parse a metadata URI."""
    return MetadataQuery.from_uri(uri)


__all__ = [
    "MetadataQuery",
    "MetadataQueryParseError",
    "parse_metadata_uri",
]
