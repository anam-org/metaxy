"""Metaxy warnings.

This module contains all warning classes used by the Metaxy framework.
"""


class UnresolvedExternalFeatureWarning(UserWarning):
    """Warning raised when external features could not be resolved from the metadata store."""


class ExternalFeatureVersionMismatchWarning(UserWarning):
    """Warning raised when external feature versions don't match the metadata store."""
