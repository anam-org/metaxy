"""Packaging utilities for Metaxy project detection.

This module provides functionality to detect which Metaxy project a Feature belongs to
based on its Python package location.
"""

import sys


def detect_project_from_package(module_name: str) -> str:
    """Detect project name from top-level Python package.

    Detection order:
    1. Check for __metaxy_project__ variable in the top-level package
    2. Use the top-level package name as project name

    If __metaxy_project__ is defined but not a string, raises TypeError.
    """
    top_level_package = module_name.split(".")[0]
    package_module = sys.modules.get(top_level_package)

    if package_module is None:
        return top_level_package

    project = getattr(package_module, "__metaxy_project__", None)

    if project is None:
        return top_level_package

    if not isinstance(project, str):
        raise TypeError(f"__metaxy_project__ in '{top_level_package}' must be a string, got {type(project).__name__}")

    return project
