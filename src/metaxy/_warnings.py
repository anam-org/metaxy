"""Metaxy warnings.

This module contains all warning classes used by the Metaxy framework.
"""

import warnings


class MovedMetadataStoreWarning(DeprecationWarning):
    """Warning raised when a metadata store has been moved to a different module."""


def _warn_deprecated_module(old: str, new: str) -> None:
    """Emit a DeprecationWarning for a relocated module."""
    warnings.warn(
        f"{old} is deprecated and will be removed in 0.2.0; use {new} instead.",
        MovedMetadataStoreWarning,
        stacklevel=3,
    )


class UnresolvedExternalFeatureWarning(UserWarning):
    """Warning raised when external features could not be resolved from the metadata store."""


class ExternalFeatureVersionMismatchWarning(UserWarning):
    """Warning raised when external feature versions don't match the metadata store."""


class InvalidStoredFeatureWarning(UserWarning):
    """Warning raised when a feature in the metadata store fails to validate."""
