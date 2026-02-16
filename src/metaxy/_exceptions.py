"""Metaxy exceptions for external feature handling."""

from metaxy.utils.exceptions import MetaxyInvariantViolationError


class ExternalFeatureVersionMismatchError(MetaxyInvariantViolationError):
    """Error raised when external feature versions don't match and error mode is enabled."""
