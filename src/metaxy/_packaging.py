"""Packaging utilities for Metaxy project detection.

This module provides functionality to detect which Metaxy project a Feature belongs to
based on package metadata entry points.

Entry point format in pyproject.toml:
    [project.entry-points."metaxy.project"]
    my-project = "my_package.entrypoint"

The entry point name is the project name, and the value is the entrypoint module.
"""

from functools import lru_cache

try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points


def detect_project_from_entrypoints(module_name: str) -> str | None:
    """Detect project name from metaxy.project entry points.

    Checks if the module was loaded via a declared metaxy entry point.

    Entry point format:
        [project.entry-points."metaxy.project"]
        my-project = "my_package:init"

    The entry point should point to a function, but the function is never called.
    Metaxy only uses the module path portion for project detection.

    Args:
        module_name: Fully qualified module name (e.g., "my_package.features")

    Returns:
        Project name if found, None otherwise
    """
    # Use cached entry points to avoid repeated queries
    all_entrypoints = get_all_project_entrypoints()

    # Check if our module matches any entry point
    for project_name, ep_module in all_entrypoints.items():
        if module_name == ep_module or module_name.startswith(ep_module + "."):
            # The entry point name is the project name
            return project_name

    return None


@lru_cache(maxsize=1)
def get_all_project_entrypoints() -> dict[str, str]:
    """Get all declared metaxy.project entry points.

    This function is cached since entry points don't change during program execution.

    Returns:
        Dictionary mapping project names to entrypoint module paths (without :function)

    Raises:
        ValueError: If a package declares multiple entry points (only one per package is allowed)
    """
    result: dict[str, str] = {}

    # Track which top-level package each entry point belongs to
    package_to_projects: dict[str, list[str]] = {}

    # Query the metaxy.project group directly - no deprecation warning
    group_eps = entry_points(group="metaxy.project")

    # Map project name (entry point name) to entrypoint module (entry point value)
    for ep in group_eps:
        # Extract module path from entry point value (before ':' if present)
        ep_module = ep.value.split(":")[0] if ":" in ep.value else ep.value

        # Get the top-level package name (first component of module path)
        top_level_package = ep_module.split(".")[0]

        # Track this project for this package
        if top_level_package not in package_to_projects:
            package_to_projects[top_level_package] = []
        package_to_projects[top_level_package].append(ep.name)

        result[ep.name] = ep_module

    # Validate that each package only declares one entry point
    for package, projects in package_to_projects.items():
        if len(projects) > 1:
            # Format the entry points list
            entries = ", ".join(f"'{p}'" for p in projects)
            raise ValueError(
                f"Found multiple entries in `metaxy.project` entrypoints group: {entries}. "
                f"The key should be the Metaxy project name, thus only one entry is allowed."
            )

    return result
